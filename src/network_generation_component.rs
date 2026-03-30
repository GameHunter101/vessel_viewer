use std::num::NonZeroU32;

use egui::{Context, emath::OrderedFloat};
use nalgebra::Vector3;
use rand::{Rng, seq::SliceRandom};
use v4::{
    builtin_actions::RegisterUiComponentAction,
    builtin_components::mesh_component::MeshComponent,
    component,
    ecs::{
        actions::ActionQueue,
        component::{ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
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
    #[default(false)]
    reset: bool,
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

    fn update_buffers(
        new_edge: [Vector3<f32>; 2],
        mesh_component: &mut MeshComponent<Vertex>,
        visualization_buffer: &mut ShaderBufferAttachment,
        device: &Device,
        queue: &Queue,
        boundary_verts: Vec<Vertex>,
        reset: bool,
    ) {
        if reset {
            mesh_component.update_vertices(boundary_verts, Some(0), device, queue, true);
        }
        mesh_component.update_vertices(
            new_edge
                .map(|position| Vertex {
                    pos: (position / 256.0 - Vector3::new(1.0, 1.0, 0.0)).into(),
                    color: [0.5, 0.0, 0.5, 1.0],
                })
                .to_vec(),
            Some(0),
            device,
            queue,
            false,
        );
        let verts = &mesh_component.vertices()[0];
        let edges: Vec<ComputeEdge> = verts
            .chunks(2)
            .map(|chunk| {
                ComputeEdge::new(
                    [
                        (chunk[0].pos[0] + 1.0) * 256.0,
                        (chunk[0].pos[1] + 1.0) * 256.0,
                    ],
                    [
                        (chunk[1].pos[0] + 1.0) * 256.0,
                        (chunk[1].pos[1] + 1.0) * 256.0,
                    ],
                )
            })
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
                self.calc_saturation_at_point(pos)
            })
            .sum()
    }

    /// Sweeps a single edge point across its vessel to find the edge with minimum oxygen
    fn sweep_single_endpoint_for_lowest_oxygen(
        &self,
        sweep_subdivisions: u32,
        fixed_point: Vector3<f32>,
        sweep_edge: [Vector3<f32>; 2],
        saturation_subdivisions: NonZeroU32,
        perpendicular_lerp_factor: f32,
    ) -> Vector3<f32> {
        (0..=(sweep_subdivisions + 1))
            .map(|i| {
                sweep_edge[0]
                    + (sweep_edge[1] - sweep_edge[0]) * (i as f32)
                        / (sweep_subdivisions as f32 + 1.0)
            })
            .min_by_key(|current_sweep_point| {
                let edge_saturation = self.calc_saturation_along_edge(
                    [*current_sweep_point, fixed_point],
                    saturation_subdivisions,
                );

                OrderedFloat(lerp(
                    edge_saturation,
                    (current_sweep_point - sweep_edge[0])
                        .normalize()
                        .dot(&(current_sweep_point - fixed_point).normalize())
                        .abs(),
                    perpendicular_lerp_factor,
                ))
            })
            .unwrap()
    }

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

    fn sdf_field_gradient(&self, point: Vector3<f32>, h: f32) -> Vector3<f32> {
        let val_at_point = self.eval_sdf_field(point).0;
        Vector3::from(
            [
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ]
            .map(|offset| self.eval_sdf_field(point + offset * h).0 - val_at_point),
        )
        .normalize()
    }

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
        if let Some(component) = other_components
            .iter_mut()
            .filter(|comp| comp.id() == self.vessel_edges_component)
            .next()
        {
            let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
            let raw_cursor_pos = engine_details.cursor_position;
            let cursor_pos = Vector3::new(
                raw_cursor_pos.0 as f32,
                (engine_details.window_resolution.1 - raw_cursor_pos.1) as f32,
                0.0,
            );
            let gradient = self.sdf_field_gradient(cursor_pos, 0.1);
            let end_pos = cursor_pos + gradient * self.eval_sdf_field(cursor_pos).0;
            Self::update_gizmo(
                [cursor_pos, end_pos].map(|position| Vertex {
                    pos: (Vector3::new(
                        position.x / (engine_details.window_resolution.0 as f32 / 2.0),
                        position.y / (engine_details.window_resolution.1 as f32 / 2.0),
                        position.z,
                    ) - Vector3::new(1.0, 1.0, 0.0))
                    .into(),
                    color: [1.0, 1.0, 0.0, 1.0],
                }),
                mesh_component,
                device,
                queue,
            );
        }

        if self.current_iter >= self.max_iter_count {
            return Vec::new();
        }

        let distance_to_edge =
            |edge_vertex_indices: &&[usize; 2], position: Vector3<f32>| -> OrderedFloat<f32> {
                let origin = self.boundary_verts[edge_vertex_indices[0]];
                let edge_vector = self.boundary_verts[edge_vertex_indices[1]] - origin;
                OrderedFloat(
                    (vector_project(edge_vector, position - origin) - (position - origin))
                        .norm_squared(),
                )
            };

        let target_oxygen_probe = *self
            .probes
            .iter()
            .max_by_key(|&&position| {
                let closest_edge = self
                    .edges
                    .iter()
                    .min_by_key(|edge_vertex_indices| {
                        distance_to_edge(edge_vertex_indices, position)
                    })
                    .unwrap();
                distance_to_edge(&closest_edge, position)
            })
            .unwrap();

        let sdf_gradient = self.sdf_field_gradient(target_oxygen_probe, 0.01);

        // println!("Pos: {:?}, grad: {sdf_gradient}", (target_oxygen_probe.x, target_oxygen_probe.y));
        /* println!("-----------------------");
        for edge in &self.edges {
            println!(
                "polygon({:?})",
                edge.map(|i| {
                    let p = self.boundary_verts[i];
                    (p.x, p.y)
                })
            );
        }

        println!("?????????");

        for probe in &self.probes {
            let grad = self.sdf_field_gradient(*probe, 0.01);
            let other_point = probe - grad * self.eval_sdf_field(*probe).0;
            println!(
                "polygon({:?})",
                [(probe.x, probe.y), (other_point.x, other_point.y)]
            );
        } */

        let (_, first_edge_index, first_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, sdf_gradient, 1000.0, 50, 0.01);
        let (_, second_edge_index, second_raycast_pos) =
            self.raycast_in_dir(target_oxygen_probe, -sdf_gradient, 1000.0, 50, 0.01);

        /* let first_edge = self.edges[first_edge_index].map(|i| self.boundary_verts[i]);
        let second_edge = self.edges[second_edge_index].map(|i| self.boundary_verts[i]); */

        /* let first_projection = clamp_vector_on_edge(vector_project(
            first_edge[1] - first_edge[0],
            target_oxygen_probe - first_edge[0],
        ) + first_edge[0], first_edge);

        let second_projection = clamp_vector_on_edge(vector_project(
            second_edge[1] - second_edge[0],
            target_oxygen_probe - second_edge[0],
        ) + second_edge[0], second_edge); */

        // let new_edge = [first_projection, second_projection];

        /* let first_target_edge_indices = self
            .edges
            .iter()
            .min_by_key(|edge_vertex_indices| {
                let distance =
                    distance_to_edge(edge_vertex_indices, target_oxygen_probe).into_inner();
                let edge = edge_vertex_indices.map(|i| self.boundary_verts[i]);
                OrderedFloat(lerp(
                    distance,
                    AREA_SIZE as f32 - (edge[1] - edge[0]).norm(),
                    self.network_parameters.edge_lerp_distance_to_length_factor,
                ))
            })
            .unwrap();

        let first_target_edge = first_target_edge_indices.map(|i| self.boundary_verts[i]);

        let second_target_edge_indices = self
            .edges
            .iter()
            .filter(|edge_indices| {
                let [v0, v1] = edge_indices;
                *v0 != first_target_edge_indices[0] || *v1 != first_target_edge_indices[1]
            })
            .min_by_key(|edge_vertex_indices| {
                let distance =
                    distance_to_edge(edge_vertex_indices, target_oxygen_probe).into_inner();
                let edge = edge_vertex_indices.map(|i| self.boundary_verts[i]);
                OrderedFloat(lerp(
                    distance,
                    AREA_SIZE as f32 * std::f32::consts::SQRT_2 - (edge[1] - edge[0]).norm(),
                    self.network_parameters.edge_lerp_distance_to_length_factor,
                ))
            })
            .unwrap();

        let second_target_edge = second_target_edge_indices.map(|i| self.boundary_verts[i]); */

        /* let sweep_subdivision = 5;
        let saturation_subdivision = NonZeroU32::new(10).unwrap();
        let first_sweep_result = self.sweep_single_endpoint_for_lowest_oxygen(
            sweep_subdivision,
            second_edge[0],
            first_edge,
            saturation_subdivision,
            0.0,
        );

        let second_sweep_result = self.sweep_single_endpoint_for_lowest_oxygen(
            sweep_subdivision,
            first_sweep_result,
            second_edge,
            saturation_subdivision,
            self.network_parameters
                .edge_lerp_concentration_to_edge_perpendicular,
        ); */

        // let new_edge = [first_sweep_result, second_sweep_result];
        if (first_raycast_pos.x > AREA_SIZE as f32 || first_raycast_pos.x < 0.0/* || first_raycast_pos.y > AREA_SIZE as f32
        || first_raycast_pos.y < 0.0 */)
            || (second_raycast_pos.x > AREA_SIZE as f32 || second_raycast_pos.x < 0.0/* || second_raycast_pos.y > AREA_SIZE as f32
            || second_raycast_pos.y < 0.0 */)
        {
            self.distribute_probes();
            return Vec::new();
        }

        let new_edge = [first_raycast_pos, second_raycast_pos];
        let temp = target_oxygen_probe + sdf_gradient * self.eval_sdf_field(target_oxygen_probe).0;
        println!(
            "Target probe: {:?}, gradient: {:?}, raycast: {:?}",
            (target_oxygen_probe.x, target_oxygen_probe.y),
            (temp.x, temp.y),
            [
                (first_raycast_pos.x, first_raycast_pos.y),
                (second_raycast_pos.x, second_raycast_pos.y)
            ]
        );
        println!("New edge: polygon({:?})", new_edge.map(|p| (p.x, p.y)));

        self.edges
            .push([self.boundary_verts.len(), self.boundary_verts.len() + 1]);
        self.boundary_verts.extend_from_slice(&new_edge);

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
                new_edge,
                mesh_component,
                buf,
                device,
                queue,
                self.edges[0..2]
                    .iter()
                    .flatten()
                    .map(|&i| Vertex {
                        pos: (self.boundary_verts[i] / 256.0 - Vector3::new(1.0, 1.0, 0.0)).into(),
                        color: [0.5, 0.0, 0.5, 1.0],
                    })
                    .collect(),
                self.reset,
            );
        }

        self.current_iter += 1;

        self.num_probes += 1;

        self.distribute_probes();
        self.reset = false;

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

                    let iter_count = ui.add(
                        egui::DragValue::new(&mut self.max_iter_count)
                            .range(1..=100)
                            .update_while_editing(true),
                    );

                    if distance_to_length_factor_slider.changed()
                        || concentration_to_angle_factor_slider.changed()
                        || iter_count.changed()
                    {
                        self.current_iter = 0;
                        let (_, boundary) = initialize_points();
                        self.boundary_verts = boundary;
                        self.edges = vec![[0, 1], [2, 3]];
                        self.reset = true;
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
