use std::num::NonZeroU32;

use egui::{Context, emath::OrderedFloat};
use nalgebra::Vector3;
use rand::{SeedableRng, seq::SliceRandom};
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

const PROBE_COUNT: usize = 50;

#[component]
pub struct NetworkGenerationComponent {
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
    #[default([Vector3::zeros(); PROBE_COUNT])]
    probes: [Vector3<f32>; PROBE_COUNT],
}

impl NetworkGenerationComponent {
    /// Creates a low-discrepancy sequence using the N-rooks algorithm.
    /// https://blog.demofox.org/2017/05/29/when-random-numbers-are-too-random-low-discrepancy-sequences/
    fn distribute_probes(&mut self, seed: Option<u64>) {
        let mut values: Vec<u32> = (0..PROBE_COUNT as u32).collect::<Vec<_>>();
        if let Some(seed) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            values.shuffle(&mut rng);
        } else {
            values.shuffle(&mut rand::rng());
        };
        let positions: Vec<Vector3<f32>> = values
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                Vector3::new(i as f32, val as f32, 0.0) / PROBE_COUNT as f32 * AREA_SIZE as f32
            })
            .collect();
        self.probes = positions.try_into().unwrap();
    }

    fn update_buffers(
        new_edge: [Vector3<f32>; 2],
        current_iter: usize,
        mesh_component: &mut MeshComponent<Vertex>,
        visualization_buffer: &mut ShaderBufferAttachment,
        device: &Device,
        queue: &Queue,
    ) {
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
            current_iter == 0 && mesh_component.vertices()[0].len() > 4,
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
        visualization_buffer.update_buffer(bytemuck::cast_slice(&edges), device, queue);
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
}

impl ComponentSystem for NetworkGenerationComponent {
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        // self.recalculate_dcel();

        self.distribute_probes(None);
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
            ..
        }: UpdateParams<'_, '_>,
    ) -> ActionQueue {
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

        let target_oxygen_probe = self
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

        let first_target_edge_indices = self
            .edges
            .iter()
            .min_by_key(|edge_vertex_indices| {
                let distance =
                    distance_to_edge(edge_vertex_indices, *target_oxygen_probe).into_inner();
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
            .filter(|[v0, v1]| {
                *v0 != first_target_edge_indices[0] || *v1 != first_target_edge_indices[1]
            })
            .min_by_key(|edge_vertex_indices| {
                let distance =
                    distance_to_edge(edge_vertex_indices, *target_oxygen_probe).into_inner();
                let edge = edge_vertex_indices.map(|i| self.boundary_verts[i]);
                OrderedFloat(lerp(
                    distance,
                    AREA_SIZE as f32 * std::f32::consts::SQRT_2 - (edge[1] - edge[0]).norm(),
                    self.network_parameters.edge_lerp_distance_to_length_factor,
                ))
            })
            .unwrap();

        let second_target_edge = second_target_edge_indices.map(|i| self.boundary_verts[i]);

        let sweep_subdivision = 5;
        let saturation_subdivision = NonZeroU32::new(10).unwrap();
        let first_sweep_result = self.sweep_single_endpoint_for_lowest_oxygen(
            sweep_subdivision,
            second_target_edge[0],
            first_target_edge,
            saturation_subdivision,
            0.0,
        );

        let second_sweep_result = self.sweep_single_endpoint_for_lowest_oxygen(
            sweep_subdivision,
            first_sweep_result,
            second_target_edge,
            saturation_subdivision,
            self.network_parameters
                .edge_lerp_concentration_to_edge_perpendicular,
        );

        /* println!(
            "First edge: polygon({:?})",
            first_target_edge_indices.map(|i| (self.boundary_verts[i].x, self.boundary_verts[i].y))
        );
        println!(
            "Second edge: polygon({:?})",
            second_target_edge_indices
                .map(|i| (self.boundary_verts[i].x, self.boundary_verts[i].y))
        );
        println!(
            "Target: {:?}",
            (target_oxygen_probe.x, target_oxygen_probe.y)
        ); */

        let new_edge = [first_sweep_result, second_sweep_result];
        /* println!("New edge: polygon({:?})", new_edge.map(|p| (p.x, p.y)));
        println!(
            "Oxygen on edge: {}",
            self.calc_saturation_along_edge(new_edge, saturation_subdivision)
        ); */
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
                self.current_iter,
                mesh_component,
                buf,
                device,
                queue,
            );
        }

        self.current_iter += 1;

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

                    if distance_to_length_factor_slider.changed()
                        || concentration_to_angle_factor_slider.changed()
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
