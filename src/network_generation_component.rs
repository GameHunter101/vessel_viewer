use std::collections::{HashMap, HashSet};

use cool_utils::data_structures::dcel::DCEL;
use egui::Context;
use nalgebra::Vector3;
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

use crate::{ComputeEdge, Vertex, initialize_points};

#[derive(Debug, Clone, Copy, Default)]
pub struct NetworkDetails {
    pub prioritize_edge_length_weight: f32,
    pub prioritize_orthogonality_weight: f32,
    pub branch_dilation_factor: f32,
}

#[component]
pub struct NetworkGenerationComponent {
    boundary_verts: Vec<Vector3<f32>>,
    boundary_adjacency_list: HashMap<usize, HashSet<usize>>,
    #[default]
    dcel: DCEL,
    max_iter_count: usize,
    #[default(0)]
    current_iter: usize,
    network_parameters: NetworkDetails,
    #[default(40.0)]
    vessel_oxygen_transport_distance: f32,
    non_edges: HashSet<[usize; 2]>,
    vessel_edges_component: ComponentId,
    display_vessel_edges_compute: ComponentId,
}

impl NetworkGenerationComponent {
    fn simple_advect(&self, index_in_face: usize, face: &[usize]) -> Vector3<f32> {
        let face_center = face
            .iter()
            .map(|i| self.boundary_verts[*i])
            .sum::<Vector3<f32>>()
            / face.len() as f32;
        let real_index = face[index_in_face];
        let neighbors = [
            face[(index_in_face + 1) % face.len()],
            face[(index_in_face as i32 - 1).rem_euclid(face.len() as i32) as usize],
        ];
        let advection_directions = neighbors.map(|neighbor_index| {
            let edge = [
                real_index.min(neighbor_index),
                real_index.max(neighbor_index),
            ];
            if self.non_edges.contains(&edge) {
                Vector3::zeros()
            } else {
                let edge = edge.map(|i| self.boundary_verts[i]);
                let normal = (edge[1] - edge[0])
                    .cross(&Vector3::new(0.0, 0.0, -1.0))
                    .normalize();
                if normal.x.is_nan() {
                    println!(
                        "Nan normalizing: from {:?} to {:?}",
                        (edge[0].x, edge[0].y),
                        (edge[1].x, edge[1].y)
                    );
                    println!("Real: {real_index}, neighbors: {neighbors:?}");
                }
                let normal_aligned_inwards = (face_center - self.boundary_verts[real_index])
                    .dot(&normal)
                    .signum();
                normal * normal_aligned_inwards * self.vessel_oxygen_transport_distance
            }
        });
        self.boundary_verts[real_index] + advection_directions.into_iter().sum::<Vector3<f32>>()
    }

    fn clamp_point_on_edge(point: Vector3<f32>, edge: [Vector3<f32>; 2]) -> Vector3<f32> {
        [0, 1, 2]
            .map(|i| point[i].clamp(edge[0][i].min(edge[1][i]), edge[0][i].max(edge[1][i])))
            .into()
    }

    pub fn split_face_with_edge(
        mut face: Vec<usize>,
        connection_edges: [[usize; 2]; 2],
    ) -> ([Vec<usize>; 2], [usize; 2]) {
        let indices_of_connection_edges_within_face = connection_edges
            .map(|edge| edge.map(|vert_idx| face.iter().position(|&i| i == vert_idx).unwrap()));

        let indices_for_connection_insertions =
            indices_of_connection_edges_within_face.map(|[i0, i1]| {
                if i0.max(i1) - i0.min(i1) > 1 {
                    i0.min(i1)
                } else {
                    i0.max(i1)
                }
            });

        let min = indices_for_connection_insertions[0].min(indices_for_connection_insertions[1]);
        let max = indices_for_connection_insertions[0].max(indices_for_connection_insertions[1]);

        face.insert(max, usize::MAX);
        face.insert(min, usize::MAX - 1);

        let first_split = face[0..=min]
            .into_iter()
            .chain(&face[(max + 1)..])
            .copied()
            .collect();
        let second_split = face[min..max + 2].to_vec();

        (
            [first_split, second_split],
            [
                if min == indices_for_connection_insertions[0] {
                    usize::MAX - 1
                } else {
                    usize::MAX
                },
                if max == indices_for_connection_insertions[0] {
                    usize::MAX - 1
                } else {
                    usize::MAX
                },
            ],
        )
    }

    fn get_face_area(face: &[Vector3<f32>]) -> f32 {
        (0..face.len())
            .map(|i| face[i].cross(&face[(i + 1) % face.len()]))
            .sum::<Vector3<f32>>()
            .norm()
            / 2.0
    }

    fn split_face(
        &self,
        face: Vec<usize>,
        connection_edges: [[usize; 2]; 2],
        longest_edge_proj: Vector3<f32>,
        candidate_vert: Vector3<f32>,
    ) -> [Vec<Vector3<f32>>; 2] {
        let (splits, [first_val_replace, second_val_replace]) = Self::split_face_with_edge(
            face,
            connection_edges.map(|[i0, i1]| [i0.min(i1), i0.max(i1)]),
        );

        splits.map(|split| {
            split
                .into_iter()
                .map(|vert_index| {
                    if vert_index == first_val_replace {
                        longest_edge_proj
                    } else if vert_index == second_val_replace {
                        candidate_vert
                    } else {
                        self.boundary_verts[vert_index]
                    }
                })
                .collect::<Vec<Vector3<f32>>>()
        })
    }

    fn get_split_areas(
        &self,
        face: Vec<usize>,
        connection_edges: [[usize; 2]; 2],
        longest_edge_proj: Vector3<f32>,
        candidate_vert: Vector3<f32>,
    ) -> [f32; 2] {
        self.split_face(face, connection_edges, longest_edge_proj, candidate_vert)
            .map(|split| Self::get_face_area(&split))
    }

    fn calculate_connection_candidate_weight(
        &self,
        projected_candidate: Vector3<f32>,
        edge_index: usize,
        face: Vec<usize>,
        face_edges: &[[Vector3<f32>; 2]],
        face_edges_indices: &[[usize; 2]],
        longest_edge_projection: Vector3<f32>,
        longest_edge: &[Vector3<f32>; 2],
        longest_edge_index: usize,
    ) -> ((Vector3<f32>, usize), f32) {
        let v0 = face_edges[edge_index][0];
        let v1 = face_edges[edge_index][1];

        let edge_shrink_dir = (v1 - v0).normalize() * self.vessel_oxygen_transport_distance;
        let bounded_candidate = Self::clamp_point_on_edge(
            projected_candidate,
            [v0 + edge_shrink_dir, v1 - edge_shrink_dir],
        );

        let edge_ratio = (1.0
            - ((v0 - bounded_candidate).norm() - (v1 - bounded_candidate).norm()).abs()
                / (v1 - v0).norm())
            * lerp(
                1.0,
                (v1 - v0).norm(),
                self.network_parameters.prioritize_edge_length_weight,
            );

        let perpendicular = 1.0
            - (bounded_candidate - longest_edge_projection)
                .normalize()
                .dot(&(longest_edge[1] - longest_edge[0]).normalize())
                .abs();

        let connection_edges =
            [longest_edge_index, edge_index].map(|edge_index| face_edges_indices[edge_index]);

        let areas = self.get_split_areas(
            face.clone(),
            connection_edges,
            longest_edge_projection,
            bounded_candidate,
        );
        let original_area = Self::get_face_area(
            &face
                .iter()
                .map(|i| self.boundary_verts[*i])
                .collect::<Vec<_>>(),
        );
        assert!(
            (areas[0] + areas[1] - original_area).abs() < 0.05,
            "Failure on split, difference: {} | {:?}",
            (areas[0] + areas[1] - original_area).abs(),
            self.split_face(
                face.clone(),
                connection_edges,
                longest_edge_projection,
                bounded_candidate
            )
            .map(|face| face.iter().map(|v| (v.x, v.y)).collect::<Vec<_>>())
        );

        let ratio = areas[0].min(areas[1]) / areas[0].max(areas[1]);
        let weight = lerp(
            edge_ratio,
            perpendicular,
            self.network_parameters.prioritize_orthogonality_weight,
        ) * ratio;

        if weight.is_nan() {
            println!(
                "NaN weight: edge_ratio: {edge_ratio}, perp: {perpendicular}, v0: {:?}, v1: {:?}",
                (v0.x, v0.y),
                (v1.x, v1.y)
            );
        }

        ((bounded_candidate, edge_index), weight)
    }

    fn get_best_connection_points(
        &self,
        low_oxygen_point: Vector3<f32>,
        face_edges: &[[Vector3<f32>; 2]],
        face_edges_indices: &[[usize; 2]],
        face: Vec<usize>,
    ) -> [(Vector3<f32>, usize); 2] {
        let (longest_edge_index, longest_edge) = face_edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a[0].metric_distance(&a[1])
                    .total_cmp(&b[0].metric_distance(&b[1]))
            })
            .unwrap();
        let unclamped_longest_edge_projection = vector_project(
            longest_edge[1] - longest_edge[0],
            low_oxygen_point - longest_edge[0],
        ) + longest_edge[0];

        let longest_edge_projection =
            Self::clamp_point_on_edge(unclamped_longest_edge_projection, *longest_edge);

        let projection_of_central_point_on_edges: Vec<(Vector3<f32>, usize)> = face_edges
            .into_iter()
            .enumerate()
            .flat_map(|(i, [p0, p1])| {
                if i == longest_edge_index {
                    None
                } else {
                    let dir = p1 - p0;
                    let unclamped_projection =
                        dir.dot(&(low_oxygen_point - p0)) / dir.norm_squared() * dir + p0;
                    let clamped_projection =
                        Self::clamp_point_on_edge(unclamped_projection, [*p0, *p1]);
                    if clamped_projection.metric_distance(&longest_edge_projection) < 0.0001
                        || (clamped_projection - longest_edge_projection)
                            .normalize()
                            .dot(&(p1 - longest_edge_projection).normalize())
                            .abs()
                            < 0.0001
                    {
                        None
                    } else {
                        Some((clamped_projection, i))
                    }
                }
            })
            .collect();
        println!();

        let connection_candidate_weights: Vec<((Vector3<f32>, usize), f32)> =
            projection_of_central_point_on_edges
                .into_iter()
                .map(|(projected_candidate, edge_index)| {
                    self.calculate_connection_candidate_weight(
                        projected_candidate,
                        edge_index,
                        face.clone(),
                        face_edges,
                        face_edges_indices,
                        longest_edge_projection,
                        longest_edge,
                        longest_edge_index,
                    )
                })
                .collect();

        let other_connection = connection_candidate_weights
            .into_iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        [
            (longest_edge_projection, longest_edge_index),
            other_connection,
        ]
    }

    fn insert_edge_into_adjacency_list(
        &mut self,
        edge: [usize; 2],
        connection_edges: [[usize; 2]; 2],
    ) {
        for (i, &vertex_index) in edge.iter().enumerate() {
            for (j, &connection_vertex_index) in connection_edges[i].iter().enumerate() {
                let connection_vertex_neighbors = self
                    .boundary_adjacency_list
                    .get_mut(&connection_vertex_index)
                    .unwrap();
                if connection_vertex_index != vertex_index {
                    connection_vertex_neighbors.remove(&connection_edges[i][1 - j]);
                    connection_vertex_neighbors.insert(vertex_index);
                }
            }

            if let Some(neighbors) = self.boundary_adjacency_list.get_mut(&vertex_index) {
                let temp = connection_edges[i]
                    .into_iter()
                    .filter(|&i| i != vertex_index)
                    .chain([edge[1 - i]])
                    .collect::<Vec<_>>();
                neighbors.extend(temp);
            } else {
                let neighbors =
                    HashSet::from([edge[1 - i], connection_edges[i][0], connection_edges[i][1]]);
                self.boundary_adjacency_list.insert(vertex_index, neighbors);
            }
        }
    }

    fn update_adjacency_list(
        &mut self,
        low_oxygen_point: Vector3<f32>,
        connection_points: [(Vector3<f32>, usize); 2],
        edges_indices: &[[usize; 2]],
    ) {
        /* println!(
            "edge: {:?}, edge indices: {:?}",
            connection_points.map(|(p, _)| (p.x, p.y)),
            connection_points.map(|(_, i)| edges_indices[i])
        ); */
        let merged_connection_points = connection_points.map(|(connection_vertex, _)| {
            if let Some(existing_connection) = (0..self.boundary_verts.len())
                .filter(|&i| (self.boundary_verts[i] - connection_vertex).norm_squared() < 0.0001)
                .next()
            {
                existing_connection
            } else {
                self.boundary_verts.push(connection_vertex);
                self.boundary_verts.len() - 1
            }
        });

        let edges = [merged_connection_points]; //merged_connection_points.map(|p| [p, self.boundary_verts.len() - 1]);

        for edge in edges {
            self.insert_edge_into_adjacency_list(
                edge,
                connection_points.map(|(_, edge_index)| edges_indices[edge_index]),
            );
        }
    }

    fn recalculate_dcel(&mut self) {
        self.dcel = DCEL::new(
            &self
                .boundary_verts
                .iter()
                .map(|p| nalgebra::Vector2::from(p.xy()))
                .collect::<Vec<_>>(),
            &self.boundary_adjacency_list,
        );
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
            vec![new_edge]
                .into_flattened()
                .iter()
                .map(|position| Vertex {
                    pos: (position / 256.0 - Vector3::new(1.0, 1.0, 0.0)).into(),
                    color: [0.5, 0.0, 0.5, 1.0],
                })
                .collect(),
            Some(0),
            device,
            queue,
            current_iter == 0,
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
        let additional_edges = 128 - edges.len();
        visualization_buffer.update_buffer(
            bytemuck::cast_slice(
                &edges
                    .into_iter()
                    .chain(vec![ComputeEdge::default(); additional_edges])
                    .collect::<Vec<_>>(),
            ),
            device,
            queue,
        );
    }
}

impl ComponentSystem for NetworkGenerationComponent {
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        self.recalculate_dcel();

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
        /* println!("----------------------------------");
        for face in self.dcel.faces() {
            print!(
                "polygon({:?})",
                face.iter()
                    .map(|&face_idx| (
                        self.boundary_verts[face_idx].x,
                        self.boundary_verts[face_idx].y
                    ))
                    .collect::<Vec<_>>()
            );
        }
        println!(); */
        let face_areas = self
            .dcel
            .faces()
            .iter()
            .enumerate()
            .map(|(i, face)| {
                let area = face
                    .iter()
                    .enumerate()
                    .map(|(i, &vert_index)| {
                        let point_indices = [vert_index, face[(i + 1) % face.len()]];
                        let vectors = point_indices.map(|i| self.boundary_verts[i]);
                        vectors[0].cross(&vectors[1])
                    })
                    .sum::<Vector3<f32>>()
                    .norm()
                    / 2.0;
                (i, area)
            })
            .collect::<Vec<_>>();
        let face_index = face_areas
            .iter()
            .max_by(|(_, a_area), (_, b_area)| a_area.total_cmp(b_area))
            .unwrap()
            .0;

        let face = &self.dcel.faces()[face_index];
        let filtered_edges_indices: Vec<[usize; 2]> = self
            .dcel
            .edges_of_face(face_index)
            .iter()
            .filter(|edge| {
                !self
                    .non_edges
                    .contains(&[edge[0].min(edge[1]), edge[0].max(edge[1])])
            })
            .copied()
            .collect();

        let filtered_edges: Vec<[Vector3<f32>; 2]> = filtered_edges_indices
            .iter()
            .map(|edge| edge.map(|index| self.boundary_verts[index]))
            .collect();

        let central_lowest_oxygen_point = (0..face.len())
            .map(|i| self.simple_advect(i, &face))
            .sum::<Vector3<f32>>()
            / face.len() as f32;

        let connection_points = self.get_best_connection_points(
            central_lowest_oxygen_point,
            &filtered_edges,
            &filtered_edges_indices,
            face.clone(),
        );

        self.update_adjacency_list(
            central_lowest_oxygen_point,
            connection_points,
            &filtered_edges_indices,
        );

        let new_edge = connection_points.map(|(p, _)| p);

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
        self.recalculate_dcel();

        Vec::new()
    }

    fn ui_render(&mut self, ctx: &Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                egui::Frame::dark_canvas(&Default::default()).show(ui, |ui| {
                    let edge_length_weight_label = ui.label("Edge length weight");
                    let edge_length_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.prioritize_edge_length_weight,
                        0.0..=1.0,
                    ));

                    let orthogonality_weight_label = ui.label("Orthogonality weight");
                    let orthogonality_slider = ui.add(egui::Slider::new(
                        &mut self.network_parameters.prioritize_orthogonality_weight,
                        0.0..=1.0,
                    ));

                    if edge_length_slider.dragged() || orthogonality_slider.dragged() {
                        self.current_iter = 0;
                        let (_, boundary, boundary_adjacency_list) = initialize_points();
                        self.boundary_verts = boundary;
                        self.boundary_adjacency_list = boundary_adjacency_list;
                        self.recalculate_dcel();
                    }

                    edge_length_slider.labelled_by(edge_length_weight_label.id);
                    orthogonality_slider.labelled_by(orthogonality_weight_label.id);
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

#[cfg(test)]
mod test {
    use nalgebra::Vector3;

    use crate::network_generation_component::NetworkGenerationComponent;

    fn mock_comp(verts: Vec<Vector3<f32>>) -> NetworkGenerationComponent {
        NetworkGenerationComponent::builder()
            .boundary_verts(verts)
            .boundary_adjacency_list(Default::default())
            .display_vessel_edges_compute(0)
            .max_iter_count(1)
            .network_parameters(Default::default())
            .non_edges(Default::default())
            .vessel_edges_component(0)
            .build()
    }

    #[test]
    fn test_basic_face_split() {
        let ([first_split, second_split], _) =
            NetworkGenerationComponent::split_face_with_edge(vec![0, 1, 2, 3], [[0, 1], [1, 2]]);
        assert_eq!(first_split, vec![0, usize::MAX - 1, usize::MAX, 2, 3]);
        assert_eq!(second_split, vec![usize::MAX - 1, 1, usize::MAX]);
    }

    #[test]
    fn test_basic_face_split_orthogonal() {
        let ([first_split, second_split], _) =
            NetworkGenerationComponent::split_face_with_edge(vec![0, 1, 2, 3], [[0, 1], [2, 3]]);
        assert_eq!(first_split, vec![0, usize::MAX - 1, usize::MAX, 3]);
        assert_eq!(second_split, vec![usize::MAX - 1, 1, 2, usize::MAX]);
    }

    #[test]
    fn test_basic_face_split_on_array_edge() {
        let ([first_split, second_split], [first_replace, second_replace]) =
            NetworkGenerationComponent::split_face_with_edge(
                vec![10, 11, 12, 13],
                [[11, 12], [10, 13]],
            );
        assert_eq!(first_split, vec![usize::MAX - 1, usize::MAX, 12, 13]);
        assert_eq!(second_split, vec![usize::MAX - 1, 10, 11, usize::MAX]);
        assert_eq!(first_replace, usize::MAX);
        assert_eq!(second_replace, usize::MAX - 1);
    }

    #[test]
    fn test_area_calculation() {
        let verts = [
            Vector3::new(1.0, 6.0, 0.0),
            Vector3::new(3.0, 1.0, 0.0),
            Vector3::new(7.0, 2.0, 0.0),
            Vector3::new(4.0, 4.0, 0.0),
            Vector3::new(8.0, 5.0, 0.0),
        ];

        assert_eq!(NetworkGenerationComponent::get_face_area(&verts), 16.5);
    }

    #[test]
    fn test_basic_split_areas() {
        let verts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let comp = mock_comp(verts.clone());

        let areas = comp.get_split_areas(
            vec![0, 1, 2, 3],
            [[0, 1], [2, 3]],
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.5, 1.0, 0.0),
        );
        assert_eq!(
            areas[0] + areas[1],
            NetworkGenerationComponent::get_face_area(&verts)
        );
        assert_eq!(areas[0], 0.5);
        assert_eq!(areas[1], 0.5);
    }

    #[test]
    fn test_angled_split_areas() {
        let verts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let comp = mock_comp(verts.clone());

        let areas = comp.get_split_areas(
            vec![0, 1, 2, 3],
            [[0, 1], [0, 3]],
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
        );
        assert_eq!(
            areas[0] + areas[1],
            NetworkGenerationComponent::get_face_area(&verts)
        );
        assert_eq!(areas[0], 1.0 - (0.5 * 0.5 / 2.0));
        assert_eq!(areas[1], 0.5 * 0.5 / 2.0);
    }
}
