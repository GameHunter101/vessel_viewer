use std::num::NonZeroU32;

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

use crate::{AREA_SIZE, ComputeEdge, Vertex, initialize_points};

#[derive(Debug, Clone, Copy, Default)]
pub struct NetworkDetails {
    pub edge_lerp_distance_to_length_factor: f32,
    pub edge_lerp_concentration_to_edge_perpendicular: f32,
}

const INIT_PROBE_COUNT: usize = 50;

#[component]
pub struct NetworkGenerationComponent<T> {
    rng: T,
    #[default(INIT_PROBE_COUNT)]
    num_probes: usize,
    boundary_verts: Vec<Vector3<f32>>,
    edges: Vec<[usize; 2]>,
    max_iter_count: usize,
    #[default(0)]
    current_iter: usize,
    network_parameters: NetworkDetails,
    #[default(40.0)]
    vessel_oxygen_transport_distance: f32,
    // non_edges: HashSet<[usize; 2]>,
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
        self.probes = positions.try_into().unwrap();
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

    /// The SDF function for an edge defined by two points.
    /// Sourced from https://iquilezles.org/articles/distfunctions2d/
    fn edge_sdf(&self, edge: [usize; 2], point: Vector3<f32>) -> f32 {
        let [a, b] = edge.map(|i| self.boundary_verts[i]);
        let pa = point - a;
        let ba = b - a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);

        (pa - ba * h).norm()
    }

    fn eval_sdf_field(&self, point: Vector3<f32>) -> (f32, usize) {
        let (distance, edge_index) = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, edge)| (OrderedFloat(self.edge_sdf(*edge, point)), i))
            .min_by_key(|(dist, _)| *dist)
            .unwrap();

        (distance.into_inner(), edge_index)
    }

    /// Finite difference approximation for the gradient of the SDF. Used to direct the raycasting
    /// algorithm in the direction of highest oxygen change, which will send it to a nearby blood vessel
    fn sdf_field_gradient(&self, point: Vector3<f32>, h: f32) -> Vector3<f32> {
        let val_at_point = self.eval_sdf_field(point).0;
        [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0), // Temporary, while working in 2D
        ]
        .map(|offset| offset * (val_at_point - self.eval_sdf_field(point + offset * h).0))
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
        let mut closest_edge = self.eval_sdf_field(origin).1;
        let mut distance = 0.0;
        for _ in 0..max_iter_count {
            if distance >= max_dist {
                break;
            }
            let (next_distance, edge_index) = self.eval_sdf_field(pos);
            pos += dir * next_distance;
            closest_edge = edge_index;
            distance += next_distance;
            if next_distance < min_dist {
                break;
            }
        }

        (distance, closest_edge, pos)
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
            .filter(|comp| comp.id() == self.vessel_edges_component)
            .next()
        {
            let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
            let raw_cursor_pos = engine_details.cursor_position;
            let cursor_pos = Vector3::new(
                raw_cursor_pos.0 as f32 / engine_details.window_resolution.0 as f32,
                (engine_details.window_resolution.1 - raw_cursor_pos.1) as f32
                    / engine_details.window_resolution.1 as f32,
                0.0,
            ) * AREA_SIZE as f32;
            let gradient = self.sdf_field_gradient(cursor_pos, 0.1);
            let strength = self.eval_sdf_field(cursor_pos).0;
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

    fn distance_to_edge(
        &self,
        edge_vertex_indices: [usize; 2],
        position: Vector3<f32>,
    ) -> OrderedFloat<f32> {
        let origin = self.boundary_verts[edge_vertex_indices[0]];
        let edge_vector = self.boundary_verts[edge_vertex_indices[1]] - origin;
        OrderedFloat(
            (vector_project(edge_vector, position - origin) - (position - origin)).norm_squared(),
        )
    }

    /// Splits an existing edge at a point, unless the point is too close to an endpoint of the
    /// edge to warrant a split. Returns the index of the split point in the `boundary_verts` vector
    fn split_edge_with_new_point(&mut self, edge: usize, new_point: Vector3<f32>) -> usize {
        let edge_indices = self.edges[edge];
        let edge_points = edge_indices.map(|i| self.boundary_verts[i]);
        if let Some((i, _)) = edge_points
            .into_iter()
            .enumerate()
            .filter(|(_, point)| {
                let res = points_are_close(*point, new_point);
                println!(
                    "Point: {:?}, edge point: {:?}, close: {res}",
                    (new_point.x, new_point.y),
                    (point.x, point.y)
                );
                res
            })
            .next()
        {
            return edge_indices[i];
        }

        let new_point_index = self.boundary_verts.len();
        self.boundary_verts.push(new_point);
        self.edges[edge] = [
            edge_indices[0].min(new_point_index),
            edge_indices[0].max(new_point_index),
        ];
        self.edges.push([
            edge_indices[1].min(new_point_index),
            edge_indices[1].max(new_point_index),
        ]);

        new_point_index
    }

    fn calc_saturation_at_point(&self, point: Vector3<f32>) -> f32 {
        self.edges
            .iter()
            .map(|edge| {
                let [p0, p1] = edge.map(|i| self.boundary_verts[i]);
                let projection = vector_project(p1 - p0, point - p0);
                self.vessel_oxygen_transport_distance
                    - projection
                        .metric_distance(&(point - p0))
                        .min(self.vessel_oxygen_transport_distance)
            })
            .sum()
    }

    /// Calculates the total oxygen in the field along the current edge.
    /// Excludes oxygen calculation at edge endpoints, as it is quite uniform across all endpoints
    fn calc_saturation_along_edge(&self, edge: [Vector3<f32>; 2], subdivisions: NonZeroU32) -> f32 {
        let subdivisions = subdivisions.get();
        (1..=subdivisions)
            .map(|i| {
                let t = (i as f32) / (subdivisions as f32 + 2.0);
                let pos = edge[0] + t * (edge[1] - edge[0]);
                let sat = self.calc_saturation_at_point(pos);
                return sat;
            })
            .sum()
    }

    /// Sweeps a single edge point across its vessel to find the edge with minimum oxygen
    fn find_minimum_oxygen_edge(
        &self,
        sweep_subdivisions: u32,
        edges: [[Vector3<f32>; 2]; 2],
        saturation_subdivisions: NonZeroU32,
        // perpendicular_lerp_factor: f32,
    ) -> [Vector3<f32>; 2] {
        (0..=(sweep_subdivisions + 1))
            .flat_map(|first_sweep_index| {
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
            })
            .min_by_key(|current_edge| {
                let edge_saturation =
                    self.calc_saturation_along_edge(*current_edge, saturation_subdivisions);
                // println!("polygon({:?}), saturation: {edge_saturation}", current_edge.map(|p| (p.x, p.y)));
                OrderedFloat(edge_saturation)

                /* let normalized_edge_dir = (current_edge[1] - current_edge[0]).normalize();

                OrderedFloat(lerp(
                    edge_saturation,
                    normalized_edge_dir
                        .dot(&(current_sweep_point - fixed_point).normalize())
                        .abs(),
                    perpendicular_lerp_factor,
                )) */
            })
            .unwrap()
    }
}

impl<T: Rng + std::fmt::Debug + Send + Sync + 'static> ComponentSystem
    for NetworkGenerationComponent<T>
{
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        self.distribute_probes();
        println!(
            "{:?}",
            self.probes.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>()
        );
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

        let target_oxygen_probe = *self
            .probes
            .iter()
            .max_by_key(|&&position| {
                let closest_edge = self
                    .edges
                    .iter()
                    .min_by_key(|edge_vertex_indices| {
                        self.distance_to_edge(**edge_vertex_indices, position)
                    })
                    .unwrap();
                self.distance_to_edge(*closest_edge, position)
            })
            .unwrap();

        let sdf_gradient = self.sdf_field_gradient(target_oxygen_probe, 0.01);

        let (_, first_edge_index, _first_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, sdf_gradient, 1000.0, 50, 0.01);
        let (_, second_edge_index, _second_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, -sdf_gradient, 1000.0, 50, 0.01);

        let min_oxygen_edge = self.find_minimum_oxygen_edge(
            10,
            [first_edge_index, second_edge_index]
                .map(|edge_index| self.edges[edge_index].map(|i| self.boundary_verts[i])),
            NonZeroU32::new(10).unwrap(),
        );

        // TODO: Temporary barrier detection, replace with more elegant detection and edge redirection
        if (min_oxygen_edge[0].x > AREA_SIZE as f32 || min_oxygen_edge[0].x < 0.0)
            || (min_oxygen_edge[1].x > AREA_SIZE as f32 || min_oxygen_edge[1].x < 0.0)
        {
            self.distribute_probes();
            return Vec::new();
        }

        let corrected_first_edge_index =
            self.split_edge_with_new_point(first_edge_index, min_oxygen_edge[0]);
        let corrected_second_edge_index =
            self.split_edge_with_new_point(second_edge_index, min_oxygen_edge[1]);

        self.edges
            .push([corrected_first_edge_index, corrected_second_edge_index]);

        if let Some(component) = other_components
            .iter_mut()
            .filter(|comp| comp.id() == self.vessel_edges_component)
            .next()
            && let Some(compute) = computes
                .iter_mut()
                .filter(|comp| comp.id() == self.display_vessel_edges_compute)
                .next()
            && let ShaderAttachment::Buffer(buf) = &mut compute.attachments_mut()[0]
        {
            let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
            Self::update_buffers(
                self.edges
                    .iter()
                    .map(|edge_indices| edge_indices.map(|i| self.boundary_verts[i]))
                    .collect(),
                mesh_component,
                buf,
                device,
                queue,
            );
        }

        self.current_iter += 1;

        self.num_probes += 1;
        self.distribute_probes();

        Vec::new()
    }

    fn ui_render(&mut self, ctx: &Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                egui::Frame::dark_canvas(&Default::default()).show(ui, |ui| {
                    let distance_to_length_factor_label =
                        ui.label("Lerp from edge distance to length");
                    let distance_to_length_factor_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.edge_lerp_distance_to_length_factor,
                        0.0..=1.0,
                    ));

                    let concentration_to_angle_factor_label =
                        ui.label("Lerp from ox concentration to perpendicularity");
                    let concentration_to_angle_factor_slider = ui.add(egui::Slider::new(
                        &mut self
                            .network_parameters
                            .edge_lerp_concentration_to_edge_perpendicular,
                        0.0..=1.0,
                    ));

                    let old_val = self.max_iter_count;
                    let iter_count = ui.add(
                        egui::DragValue::new(&mut self.max_iter_count)
                            .range(0..=200)
                            .update_while_editing(true),
                    );

                    if distance_to_length_factor_slider.changed()
                        || concentration_to_angle_factor_slider.changed()
                        || (iter_count.changed() && self.max_iter_count < old_val)
                    {
                        self.current_iter = 0;
                        let (_, boundary) = initialize_points();
                        self.boundary_verts = boundary;
                        self.edges = vec![[0, 1], [2, 3]];
                    }

                    distance_to_length_factor_slider
                        .labelled_by(distance_to_length_factor_label.id);
                    concentration_to_angle_factor_slider
                        .labelled_by(concentration_to_angle_factor_label.id);
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

fn clamp_vector_on_edge(vector: Vector3<f32>, edge: [Vector3<f32>; 2]) -> Vector3<f32> {
    Vector3::from_iterator(
        (0_usize..3)
            .map(|i| vector[i].clamp(edge[0][i].min(edge[1][i]), edge[0][i].max(edge[1][i]))),
    )
}

fn points_are_close(p1: Vector3<f32>, p2: Vector3<f32>) -> bool {
    p1.metric_distance(&p2) < 0.001
}

#[cfg(test)]
mod test {
    use nalgebra::Vector3;
    use rand::rng;

    use crate::{
        AREA_SIZE,
        network_generation_component::{NetworkGenerationComponent, points_are_close},
    };

    #[test]
    fn sdf_test_single_edge() {
        let network = NetworkGenerationComponent::builder()
            .rng(rng())
            .boundary_verts(
                [[-1.0, -0.85], [1.0, -0.85]]
                    .map(|p| {
                        (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0))
                            * AREA_SIZE as f32
                            / 2.0
                    })
                    .to_vec(),
            )
            .edges(vec![[0, 1]])
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_lerp_distance_to_length_factor: 0.0,
                edge_lerp_concentration_to_edge_perpendicular: 0.0,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();

        let test_point = Vector3::new(89.0, 426.0, 0.0);

        let (experimental, edge) = network.eval_sdf_field(test_point);
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
            .boundary_verts(
                [[-1.0, -0.85], [1.0, -0.85], [1.0, 0.85], [-1.0, 0.85]]
                    .map(|p| {
                        (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0))
                            * AREA_SIZE as f32
                            / 2.0
                    })
                    .to_vec(),
            )
            .edges(vec![[0, 1], [2, 3]])
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_lerp_distance_to_length_factor: 0.0,
                edge_lerp_concentration_to_edge_perpendicular: 0.0,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();
        let test_point = Vector3::new(89.0, 426.0, 0.0);
        let (experimental, edge) = network.eval_sdf_field(test_point);
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
    fn simple_edge_split() {
        let mut network = NetworkGenerationComponent::builder()
            .rng(rng())
            .boundary_verts(
                [[-1.0, -0.85], [1.0, -0.85]]
                    .map(|p| {
                        (Vector3::new(p[0], p[1], 0.0) + Vector3::new(1.0, 1.0, 0.0))
                            * AREA_SIZE as f32
                            / 2.0
                    })
                    .to_vec(),
            )
            .edges(vec![[0, 1]])
            .network_parameters(crate::network_generation_component::NetworkDetails {
                edge_lerp_distance_to_length_factor: 0.0,
                edge_lerp_concentration_to_edge_perpendicular: 0.0,
            })
            .vessel_edges_component(0)
            .display_vessel_edges_compute(0)
            .max_iter_count(0)
            .build();

        let split_index = network.split_edge_with_new_point(0, Vector3::new(128.0, 19.2, 0.0));

        assert_eq!(split_index, 2);
        assert_eq!(network.edges.len(), 2);
        assert_eq!(network.edges, [[0, 2], [1, 2]]);
    }
}
