use std::{collections::HashSet, num::NonZeroU32};

use egui::{Context, emath::OrderedFloat};
use nalgebra::Vector3;
use rand::{Rng, seq::SliceRandom};
use v4::{
    EngineDetails,
    builtin_actions::RegisterUiComponentAction,
    builtin_components::mesh_component::MeshComponent,
    component,
    ecs::{
        actions::ActionQueue,
        component::{Component, ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
        material::{ShaderAttachment, ShaderBufferAttachment},
    },
};
use wgpu::{Device, Queue};

use crate::{
    AREA_SIZE, ComputeEdge, Vertex, initialize_points,
    spatial_edge_hash::{Edge, SpatialEdgeHash},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct NetworkDetails {
    pub edge_orthogonality_lerp_factor: f32,
    pub branch_width_factor: f32,
    pub branch_length_factor: f32,
}

const INIT_PROBE_COUNT: usize = 50;

#[component]
pub struct NetworkGenerationComponent<T> {
    rng: T,
    #[default(INIT_PROBE_COUNT)]
    num_probes: usize,
    edge_map: SpatialEdgeHash,
    max_iter_count: usize,
    #[default(0)]
    current_iter: usize,
    network_parameters: NetworkDetails,
    #[default(40.0)]
    vessel_oxygen_transport_distance: f32,
    vessel_edges_component: ComponentId,
    display_vessel_edges_compute: ComponentId,
    #[default(vec![Vector3::zeros(); INIT_PROBE_COUNT])]
    probes: Vec<Vector3<f32>>,
}

impl<T: Rng> NetworkGenerationComponent<T> {
    /// Creates a low-discrepancy sequence using the N-rooks algorithm.
    /// https://blog.demofox.org/2017/05/29/when-random-numbers-are-too-random-low-discrepancy-sequences/
    fn distribute_probes(&mut self) {
        let mut values: Vec<u32> = (0..self.num_probes as u32).collect::<Vec<_>>();
        values.shuffle(&mut self.rng);
        let positions: Vec<Vector3<f32>> = values
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                Vector3::new(
                    i as f32 * AREA_SIZE as f32,
                    val as f32 * (473.0 - 39.0),
                    0.0,
                ) / self.num_probes as f32
                    + Vector3::new(0.0, 39.0, 0.0)
            })
            .collect();
        self.probes = positions;
    }

    fn update_buffers(
        all_edges: Vec<[Vector3<f32>; 2]>,
        mesh_component: &mut MeshComponent<Vertex>,
        visualization_buffer: &mut ShaderBufferAttachment,
        device: &Device,
        queue: &Queue,
    ) {
        mesh_component.update_vertices(
            all_edges
                .iter()
                .flat_map(|edge| {
                    edge.map(|position| Vertex {
                        pos: (position / 256.0 - Vector3::new(1.0, 1.0, 0.0)).into(),
                        color: [0.5, 0.0, 0.5, 1.0],
                    })
                })
                .collect::<Vec<_>>(),
            Some(0),
            device,
            queue,
            true,
        );
        let edges: Vec<ComputeEdge> = all_edges
            .iter()
            .map(|edge| ComputeEdge::new([edge[0].x, edge[0].y], [edge[1].x, edge[1].y]))
            .collect();

        let edge_count = edges.len();

        visualization_buffer.update_buffer(
            bytemuck::cast_slice(
                &edges
                    .into_iter()
                    .chain(vec![
                        ComputeEdge::default();
                        crate::BUFFER_SIZE - edge_count
                    ])
                    .collect::<Vec<_>>(),
            ),
            device,
            queue,
        );
    }

    /// Finite difference approximation for the gradient of the SDF. Used to direct the raycasting
    /// algorithm in the direction of highest oxygen change, which will send it to a nearby blood vessel
    fn sdf_field_gradient(&self, point: Vector3<f32>, h: f32) -> Vector3<f32> {
        let val_at_point = self.edge_map.eval_sdf_field(point).unwrap().0;
        [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0), // Temporary, while working in 2D
        ]
        .map(|offset| {
            offset * (val_at_point - self.edge_map.eval_sdf_field(point + offset * h).unwrap().0)
        })
        .into_iter()
        .sum::<Vector3<f32>>()
        .normalize()
    }

    /// Raycasting uses signed-distance fields to step through the scene in steps as large as
    /// possible while maintaining full accuracy. Used to find nearby target attachment points on
    /// other blood vessels
    fn raycast_in_dir(
        &self,
        origin: Vector3<f32>,
        dir: Vector3<f32>,
        max_dist: f32,
        max_iter_count: u32,
        min_dist: f32,
    ) -> (f32, usize, Vector3<f32>) {
        let mut pos = origin;
        let mut closest_edge = self.edge_map.eval_sdf_field(origin).unwrap().1;
        let mut distance = 0.0;
        for _ in 0..max_iter_count {
            if distance >= max_dist {
                break;
            }
            let (next_distance, edge_index) = self.edge_map.eval_sdf_field(pos).unwrap();
            pos += dir * next_distance;
            closest_edge = edge_index;
            distance += next_distance;
            if next_distance < min_dist {
                break;
            }
        }

        (distance, closest_edge, pos)
    }

    fn raycast_barrier_in_dir(
        &self,
        origin: Vector3<f32>,
        dir: Vector3<f32>,
        max_dist: f32,
        max_iter_count: u32,
        min_dist: f32,
    ) -> (usize, f32) {
        let mut pos = origin;
        let mut distance = 0.0;
        let mut closest_barrier = 0;
        for _ in 0..max_iter_count {
            if distance >= max_dist {
                break;
            }
            let (next_barrier, next_distance) = self.edge_map.eval_barrier_sdf_field(pos).unwrap();
            pos += dir * next_distance;
            distance += next_distance;
            closest_barrier = next_barrier;
            if next_distance < min_dist {
                break;
            }
        }

        (closest_barrier, distance)
    }

    /// Displays a line at the cursor in the direction of the sdf field with proper magnitude
    fn show_gizmo(
        &self,
        other_components: &mut [&mut Component],
        engine_details: &EngineDetails,
        device: &Device,
        queue: &Queue,
    ) {
        if let Some(component) = other_components
            .iter_mut()
            .find(|comp| comp.id() == self.vessel_edges_component)
        {
            let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
            let raw_cursor_pos = engine_details.cursor_position;
            let cursor_pos = Vector3::new(
                raw_cursor_pos.0 as f32 / engine_details.window_resolution.0 as f32,
                (engine_details.window_resolution.1 as f32 - raw_cursor_pos.1 as f32)
                    / engine_details.window_resolution.1 as f32,
                0.0,
            ) * AREA_SIZE as f32;
            let gradient = self.sdf_field_gradient(cursor_pos, 0.1);
            let strength = self.edge_map.eval_sdf_field(cursor_pos).unwrap().0;
            let end_pos = cursor_pos + gradient * strength;
            Self::update_gizmo(
                [cursor_pos, end_pos].map(|position| Vertex {
                    pos: (position / 256.0 - Vector3::new(1.0, 1.0, 0.0)).into(),
                    color: [0.0, 1.0, 0.0, 1.0],
                }),
                mesh_component,
                device,
                queue,
            );
        }
    }

    fn update_gizmo(
        gizmo: [Vertex; 2],
        mesh_component: &mut MeshComponent<Vertex>,
        device: &Device,
        queue: &Queue,
    ) {
        mesh_component.update_vertices(
            gizmo.to_vec(),
            if mesh_component.vertices().len() == 1 {
                None
            } else {
                Some(1)
            },
            device,
            queue,
            true,
        );
    }

    fn calc_saturation_at_point(point: Vector3<f32>, edges: &[Edge], oxygen_distance: f32) -> f32 {
        edges
            .iter()
            .map(|&[p0, p1]| {
                let projection = vector_project(p1 - p0, point - p0);
                oxygen_distance
                    - projection
                        .metric_distance(&(point - p0))
                        .min(oxygen_distance)
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    /// Calculates the total oxygen in the field along the current edge.
    /// Excludes oxygen calculation at edge endpoints, as it is quite uniform across all endpoints
    fn calc_saturation_along_edge(
        edge: [Vector3<f32>; 2],
        subdivisions: NonZeroU32,
        edges: &[Edge],
        oxygen_distance: f32,
    ) -> f32 {
        let subdivisions = subdivisions.get();
        (1..=subdivisions)
            .map(|i| {
                let t = (i as f32) / (subdivisions as f32 + 2.0);
                let pos = edge[0] + t * (edge[1] - edge[0]);

                Self::calc_saturation_at_point(pos, edges, oxygen_distance)
            })
            .sum()
    }

    /// Sweeps a single edge point across its vessel to find the edge with minimum oxygen
    fn find_minimum_oxygen_edge(
        &self,
        sweep_subdivisions: u32,
        edges: [[Vector3<f32>; 2]; 2],
        saturation_subdivisions: NonZeroU32,
    ) -> [Vector3<f32>; 2] {
        let nearby_edge_indices: HashSet<usize> = edges
            .iter()
            .flatten()
            .flat_map(|&point| self.edge_map.edges_in_cells_near_point(point))
            .collect();

        let nearby_edges: Vec<Edge> = nearby_edge_indices
            .into_iter()
            .map(|edge_index| self.edge_map.edge(edge_index))
            .collect();

        let (_, res) = async_scoped::TokioScope::scope_and_block(|scope| {
            for current_edge in (0..=(sweep_subdivisions + 1)).flat_map(|first_sweep_index| {
                let first_sweep_pos = lerp(
                    edges[0][0],
                    edges[0][1],
                    first_sweep_index as f32 / (sweep_subdivisions as f32 + 1.0),
                );
                (0..=sweep_subdivisions + 1)
                    .map(|second_sweep_index| {
                        let second_sweep_pos = lerp(
                            edges[1][0],
                            edges[1][1],
                            second_sweep_index as f32 / (sweep_subdivisions as f32 + 1.0),
                        );
                        [first_sweep_pos, second_sweep_pos]
                    })
                    .collect::<Vec<_>>()
            }) {
                let nearby_edges = &nearby_edges;
                let oxygen_distance = self.vessel_oxygen_transport_distance;
                scope.spawn(async move {
                    (
                        OrderedFloat(Self::calc_saturation_along_edge(
                            current_edge,
                            saturation_subdivisions,
                            nearby_edges,
                            oxygen_distance,
                        )),
                        current_edge,
                    )
                });
            }
        });

        res.into_iter()
            .flatten()
            .min_by_key(|(dist, _)| *dist)
            .unwrap()
            .1
    }

    /// Takes a straight edge and creates a branch within it.
    /// Before: ----, after: -<>-
    fn bifurcate_edge(&mut self, edge: Edge, branch_width_factor: f32, branch_length_factor: f32) {
        // TODO: Replace with a more parametric solution. This does not work very well in 3D
        let up = Vector3::<f32>::z();
        let edge_center = (edge[0] + edge[1]) / 2.0;

        let (sdf_distance_at_edge_center, _) = self.edge_map.eval_sdf_field(edge_center).unwrap();

        let sdf_distance_at_edge_center = sdf_distance_at_edge_center
            .min(self.edge_map.eval_barrier_sdf_field(edge_center).unwrap().1);

        for edge_point in edge {
            let main_dir = (edge_center - edge_point).normalize();
            let branch_dir =
                branch_width_factor * sdf_distance_at_edge_center * main_dir.cross(&up);

            let branch_apex_points = [edge_center + branch_dir, edge_center - branch_dir];
            let branch_point = lerp(edge_center, edge_point, branch_length_factor);

            for apex_point in branch_apex_points {
                self.edge_map.insert_edge([branch_point, apex_point]);
            }

            self.edge_map.insert_edge([edge_point, branch_point]);
        }
    }
}

impl<T: Rng + std::fmt::Debug + Send + Sync + 'static> ComponentSystem
    for NetworkGenerationComponent<T>
{
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        self.distribute_probes();
        self.set_initialized();
        vec![Box::new(RegisterUiComponentAction {
            component_id: self.id,
            text_component_properties: None,
        })]
    }

    fn update(
        &mut self,
        UpdateParams {
            other_components,
            device,
            queue,
            computes,
            engine_details,
            ..
        }: UpdateParams<'_, '_>,
    ) -> ActionQueue {
        self.show_gizmo(other_components, engine_details, device, queue);

        if self.current_iter >= self.max_iter_count {
            return Vec::new();
        }

        let a = std::time::Instant::now();
        let (probe_index, target_oxygen_probe) = self
            .probes
            .iter()
            .enumerate()
            .max_by_key(|(_, position)| {
                OrderedFloat(self.edge_map.eval_sdf_field(**position).unwrap().0)
            })
            .map(|(idx, probe)| (idx, *probe))
            .unwrap();

        let sdf_gradient = self.sdf_field_gradient(target_oxygen_probe, 0.01);

        let (first_ray_distance, first_edge_index, first_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, sdf_gradient, 1000.0, 50, 0.01);
        let (second_ray_distance, second_edge_index, second_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, -sdf_gradient, 1000.0, 50, 0.01);
        let raycast_edges = [first_raycast_pos, second_raycast_pos];

        let [
            (first_barrier_intersection_edge, first_barrier_distance),
            (second_barrier_intersection_edge, second_barrier_distance),
        ] = [1.0, -1.0].map(|c| {
            self.raycast_barrier_in_dir(target_oxygen_probe, c * sdf_gradient, 1000.0, 50, 0.01)
        });

        let first_edge = if first_ray_distance > first_barrier_distance {
            let barrier_edge = self.edge_map.barrier_edges()[first_barrier_intersection_edge];
            let intersection_point = target_oxygen_probe + first_barrier_distance * sdf_gradient;
            if barrier_edge[0].metric_distance(&intersection_point)
                < barrier_edge[1].metric_distance(&intersection_point)
            {
                [barrier_edge[0]; 2]
            } else {
                [barrier_edge[1]; 2]
            }
        } else {
            self.edge_map.edge(first_edge_index)
        };

        let second_edge = if second_ray_distance > second_barrier_distance {
            let barrier_edge = self.edge_map.barrier_edges()[second_barrier_intersection_edge];
            let intersection_point = target_oxygen_probe - second_barrier_distance * sdf_gradient;
            if barrier_edge[0].metric_distance(&intersection_point)
                < barrier_edge[1].metric_distance(&intersection_point)
            {
                [barrier_edge[0]; 2]
            } else {
                [barrier_edge[1]; 2]
            }
        } else {
            self.edge_map.edge(second_edge_index)
        };

        let min_oxygen_edge = self.find_minimum_oxygen_edge(
            10,
            [first_edge, second_edge],
            NonZeroU32::new(10).unwrap(),
        );
        println!("Ox time: {}", a.elapsed().as_millis());

        let new_edge_points = [0, 1].map(|i| {
            lerp(
                min_oxygen_edge[i],
                raycast_edges[i],
                self.network_parameters.edge_orthogonality_lerp_factor,
            )
        });

        /* // TODO: Temporary barrier detection, replace with more elegant detection and edge redirection
        if (new_edge_points[0].x > AREA_SIZE as f32 || new_edge_points[0].x < 0.0)
            || (new_edge_points[1].x > AREA_SIZE as f32 || new_edge_points[1].x < 0.0)
        {
            self.distribute_probes();
            return Vec::new();
        } */

        if first_edge[0] != first_edge[1] {
            println!("Edge: {first_edge:?}, split: {:?}", new_edge_points[0]);
            self.edge_map
                .split_edge_at_point(first_edge_index, new_edge_points[0]);
        }

        if second_edge[0] != second_edge[1] {
            println!("Edge: {second_edge:?}, split: {:?}", new_edge_points[1]);
            self.edge_map
                .split_edge_at_point(second_edge_index, new_edge_points[1]);
        }

        self.bifurcate_edge(
            new_edge_points,
            self.network_parameters.branch_width_factor,
            self.network_parameters.branch_length_factor,
        );

        if let Some(component) = other_components
            .iter_mut()
            .find(|comp| comp.id() == self.vessel_edges_component)
            && let Some(compute) = computes
                .iter_mut()
                .find(|comp| comp.id() == self.display_vessel_edges_compute)
            && let ShaderAttachment::Buffer(buf) = &mut compute.attachments_mut()[0]
        {
            let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
            Self::update_buffers(
                self.edge_map.raw_edges(),
                mesh_component,
                buf,
                device,
                queue,
            );
        }

        self.current_iter += 1;

        self.num_probes += 1;
        self.distribute_probes();

        /* println!(
            "frame time: {}",
            engine_details.last_frame_instant.elapsed().as_millis()
        ); */

        Vec::new()
    }

    fn ui_render(&mut self, ctx: &Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                egui::Frame::dark_canvas(&Default::default()).show(ui, |ui| {
                    let branch_width_factor_label = ui.label("Branch width factor");
                    let branch_width_factor_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.branch_width_factor,
                        0.0..=1.0,
                    ));

                    let branch_length_factor_label = ui.label("Branch length factor");
                    let branch_length_factor_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.branch_length_factor,
                        0.0..=1.0,
                    ));

                    let orthogonality_lerp_factor_label = ui.label("Orthogonality lerp factor");
                    let orthogonality_factor_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.edge_orthogonality_lerp_factor,
                        0.0..=1.0,
                    ));

                    let old_val = self.max_iter_count;
                    let iter_count = ui.add(
                        egui::DragValue::new(&mut self.max_iter_count)
                            .range(0..=200)
                            .update_while_editing(true),
                    );

                    if branch_width_factor_slider.changed()
                        || branch_length_factor_slider.changed()
                        || orthogonality_factor_slider.changed()
                        || (iter_count.changed() && self.max_iter_count < old_val)
                    {
                        self.current_iter = 0;
                        let (_, boundary) = initialize_points();
                        self.edge_map = SpatialEdgeHash::new(self.edge_map.cell_size(), boundary);
                    }

                    orthogonality_factor_slider.labelled_by(orthogonality_lerp_factor_label.id);
                    branch_width_factor_slider.labelled_by(branch_width_factor_label.id);
                    branch_length_factor_slider.labelled_by(branch_length_factor_label.id);
                });
            });
    }
}

fn vector_project(base: Vector3<f32>, target: Vector3<f32>) -> Vector3<f32> {
    base.dot(&target) / base.norm_squared() * base
}

fn lerp<T: std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>>(a: T, b: T, t: f32) -> T {
    a * (1.0 - t) + b * t
}

pub fn points_are_close(p1: Vector3<f32>, p2: Vector3<f32>) -> bool {
    p1.metric_distance(&p2) < 0.001
}

#[cfg(test)]
mod test {
    use nalgebra::Vector3;
    use rand::rng;

    use crate::{
        AREA_SIZE,
        network_generation_component::{NetworkGenerationComponent, points_are_close},
        spatial_edge_hash::SpatialEdgeHash,
    };

    #[test]
    fn sdf_test_single_edge() {
        let network = NetworkGenerationComponent::builder()
            .rng(rng())
            .edge_map(SpatialEdgeHash::new(
                40.0,
                vec![[[-1.0, -0.85], [1.0, -0.85]].map(|p| {
                    (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0)) * AREA_SIZE as f32
                        / 2.0
                })],
            ))
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_orthogonality_lerp_factor: 0.0,
                branch_length_factor: 0.5,
                branch_width_factor: 0.5,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();

        let test_point = Vector3::new(89.0, 426.0, 0.0);

        let (experimental, edge) = network.edge_map.eval_sdf_field(test_point).unwrap();
        let real = 387.6;
        assert!(
            (experimental - real).abs() < 0.001,
            "Expected sdf value {experimental} to equal {real}"
        );
        assert_eq!(edge, 0);
        let gradient = network.sdf_field_gradient(test_point, 0.01);
        assert!(points_are_close(gradient, Vector3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn sdf_test_two_edges() {
        let network = NetworkGenerationComponent::builder()
            .rng(rng())
            .edge_map(SpatialEdgeHash::new(
                40.0,
                [[[-1.0, -0.85], [1.0, -0.85]], [[1.0, 0.85], [-1.0, 0.85]]]
                    .map(|raw_edge| {
                        raw_edge.map(|p| {
                            (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0))
                                * AREA_SIZE as f32
                                / 2.0
                        })
                    })
                    .to_vec(),
            ))
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_orthogonality_lerp_factor: 0.0,
                branch_length_factor: 0.5,
                branch_width_factor: 0.5,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();
        let test_point = Vector3::new(89.0, 426.0, 0.0);
        let (experimental, edge) = network.edge_map.eval_sdf_field(test_point).unwrap();
        let real = 47.6;
        assert!(
            (experimental - real).abs() < 0.001,
            "Expected sdf value {experimental} to equal {real}"
        );
        assert_eq!(edge, 1);
        let gradient = network.sdf_field_gradient(test_point, 0.01);
        assert!(points_are_close(gradient, Vector3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn sdf_test_odd_edge_case() {
        let network = NetworkGenerationComponent::builder()
            .rng(rng())
            .edge_map(SpatialEdgeHash::new(
                50.0,
                [[[-1.0, -0.85], [1.0, -0.85]], [[1.0, 0.85], [-1.0, 0.85]]]
                    .map(|raw_edge| {
                        raw_edge.map(|p| {
                            (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0))
                                * AREA_SIZE as f32
                                / 2.0
                        })
                    })
                    .to_vec(),
            ))
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_orthogonality_lerp_factor: 0.0,
                branch_length_factor: 0.5,
                branch_width_factor: 0.5,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();

        let test_point = Vector3::new(460.8, 212.6, 0.0);
        let (experimental, edge) = network.edge_map.eval_sdf_field(test_point).unwrap();
        let real = 174.2;
        assert!(
            (experimental - real).abs() < 0.001,
            "Expected sdf value {experimental} to equal {real}"
        );
        assert_eq!(edge, 0);
    }
}
