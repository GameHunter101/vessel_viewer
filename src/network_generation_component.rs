use std::collections::{HashMap, HashSet};

use cool_utils::data_structures::dcel::DCEL;
use nalgebra::Vector3;
use v4::{
    builtin_components::mesh_component::MeshComponent,
    component,
    ecs::{
        actions::ActionQueue,
        component::{ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
        material::ShaderAttachment,
    },
};
use wgpu::Device;

use crate::{ComputeEdge, Vertex};

#[component]
pub struct NetworkGenerationComponent {
    boundary_verts: Vec<Vector3<f32>>,
    boundary_adjacency_list: HashMap<usize, HashSet<usize>>,
    #[default]
    dcel: DCEL,
    max_iter_count: usize,
    #[default(0)]
    current_iter: usize,
    #[default(1.0)]
    branch_dilation_factor: f32,
    non_edges: HashSet<[usize; 2]>,
    vessel_edges_component: ComponentId,
    display_vessel_edges_compute: ComponentId,
}

impl NetworkGenerationComponent {
    fn calc_oxygen_gradient_at_point(
        point: Vector3<f32>,
        all_vessel_edges: &[[Vector3<f32>; 2]],
    ) -> Vector3<f32> {
        all_vessel_edges
            .iter()
            .map(|[p0, p1]| {
                let vessel_dir = p1 - p0;
                let offset_pos = point - p0;
                let proj =
                    offset_pos.dot(&vessel_dir) / vessel_dir.dot(&vessel_dir) * vessel_dir + p0;
                let cutoff = 0.08;
                (cutoff - point.metric_distance(&proj) / 512.0).max(0.0) / cutoff
                    * (point - proj).normalize()
            })
            .sum()
    }

    fn advect_boundary(
        &self,
        boundary_verts: &[Vector3<f32>],
        all_vessel_edges: &[[Vector3<f32>; 2]],
    ) -> Vec<Vector3<f32>> {
        let center = boundary_verts.iter().sum::<Vector3<f32>>() / boundary_verts.len() as f32;
        let mut new_boundary: Vec<Vector3<f32>> = boundary_verts
            .iter()
            .map(|pos| (pos - center) * 0.99 + center)
            .collect();

        for point in &mut new_boundary {
            let mut boundary_vector = Self::calc_oxygen_gradient_at_point(*point, all_vessel_edges);
            while boundary_vector.norm() > 0.0001 {
                *point += boundary_vector;
                boundary_vector = Self::calc_oxygen_gradient_at_point(*point, all_vessel_edges);
            }
        }
        new_boundary
    }

    fn get_best_connection_points(
        &self,
        low_oxygen_point: Vector3<f32>,
        all_edges: &[[Vector3<f32>; 2]],
    ) -> [(Vector3<f32>, usize); 2] {
        let mut projection_of_central_point_on_edges: Vec<(Vector3<f32>, usize)> = all_edges
            .into_iter()
            .enumerate()
            .map(|(i, [p0, p1])| {
                let dir = p1 - p0;
                let unclamped_projection =
                    dir.dot(&(low_oxygen_point - p0)) / dir.norm_squared() * dir + p0;
                (
                    Vector3::new(
                        unclamped_projection.x.clamp(p0.x.min(p1.x), p0.x.max(p1.x)),
                        unclamped_projection.y.clamp(p0.y.min(p1.y), p0.y.max(p1.y)),
                        unclamped_projection.z.clamp(p0.z.min(p1.z), p0.z.max(p1.z)),
                    ),
                    i,
                )
            })
            .collect();

        projection_of_central_point_on_edges.sort_by(|(a, a_edge_index), (b, b_edge_index)| {
            let a_edge_ratio = ((all_edges[*a_edge_index][0] - a).norm() / (all_edges[*a_edge_index][1] - a).norm()).abs();
            let b_edge_ratio = ((all_edges[*b_edge_index][0] - b).norm() / (all_edges[*b_edge_index][1] - b).norm()).abs();
            a_edge_ratio.total_cmp(&b_edge_ratio)
            /* (a - low_oxygen_point)
                .norm_squared()
                .total_cmp(&(b - low_oxygen_point).norm_squared()) */
        });

        [
            projection_of_central_point_on_edges[0],
            projection_of_central_point_on_edges[1],
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
                connection_vertex_neighbors.remove(&connection_edges[i][1 - j]);
                connection_vertex_neighbors.insert(vertex_index);
            }

            if let Some(neighbors) = self.boundary_adjacency_list.get_mut(&vertex_index) {
                neighbors.insert(edge[1 - i]);
                neighbors.extend(connection_edges[i]);
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

        // self.boundary_verts.push(low_oxygen_point);

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
}

impl ComponentSystem for NetworkGenerationComponent {
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        self.recalculate_dcel();

        self.set_initialized();
        Vec::new()
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

        for face_index in 0..self.dcel.faces().len() {
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

            let boundary_verts: Vec<Vector3<f32>> =
                face.iter().map(|i| self.boundary_verts[*i]).collect();
            let advected_boundary = self.advect_boundary(&boundary_verts, &filtered_edges);
            let central_lowest_oxygen_point =
                advected_boundary.iter().sum::<Vector3<f32>>() / advected_boundary.len() as f32;

            let connection_points =
                self.get_best_connection_points(central_lowest_oxygen_point, &filtered_edges);

            self.update_adjacency_list(
                central_lowest_oxygen_point,
                connection_points,
                &filtered_edges_indices,
            );

            let new_edge = connection_points.map(|(point, _)| point);

            if let Some(component) = other_components
                .iter_mut()
                .filter(|comp| comp.id() == self.vessel_edges_component)
                .next()
            {
                let mesh_component: &mut MeshComponent<Vertex> = component.downcast_mut().unwrap();
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
                );
                let verts = &mesh_component.vertices()[0];
                if let Some(compute) = computes
                    .iter_mut()
                    .filter(|comp| comp.id() == self.display_vessel_edges_compute)
                    .next()
                    && let ShaderAttachment::Buffer(buf) = &mut compute.attachments_mut()[0]
                {
                    let edges: Vec<ComputeEdge> = verts
                        .chunks(2)
                        .map(|chunk| {
                            ComputeEdge::new(
                                [(chunk[0].pos[0] + 1.0) * 256.0, (chunk[0].pos[1] + 1.0) * 256.0],
                                [(chunk[1].pos[0] + 1.0) * 256.0, (chunk[1].pos[1] + 1.0) * 256.0],
                            )
                        })
                        .collect();
                    buf.update_buffer(bytemuck::cast_slice(&edges), device, queue);
                }
            }
        }

        self.current_iter += 1;
        self.recalculate_dcel();

        Vec::new()
    }
}
