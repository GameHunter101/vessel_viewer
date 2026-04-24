use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;

use crate::network_generation_component::points_are_close;

pub type Edge = [Vector3<f32>; 2];

#[derive(Debug, Clone)]
pub struct SpatialEdgeHash {
    // edges: Vec<[usize; 2]>,
    map: HashMap<Vector3<u32>, HashSet<usize>>,
    verts: Vec<Vector3<f32>>,
    edge_indices: Vec<[usize; 2]>,
    cell_size: f32,
}

impl SpatialEdgeHash {
    pub fn new(spatial_subdivision: f32, edges: Vec<Edge>) -> Self {
        let mut edge_hash = Self {
            map: HashMap::new(),
            verts: Vec::new(),
            edge_indices: Vec::new(),
            cell_size: spatial_subdivision,
        };
        for edge in edges {
            edge_hash.insert_edge(edge);
        }

        edge_hash
    }

    fn floor_point_to_grid(&self, point: Vector3<f32>) -> Vector3<u32> {
        let scaled: [f32; 3] = (point / self.cell_size).into();
        scaled.map(|v| v as u32).into()
    }

    /// Detects which grid cells the edge intersects, updates cells with new edge.
    /// Algorithm adapted from [http://www.cse.yorku.ca/~amana/research/grid.pdf]
    fn find_cells_along_edge(&self, mut edge: Edge) -> HashSet<Vector3<u32>> {
        let mut edge_cells: HashSet<Vector3<u32>> = HashSet::new();
        let mut ray_point = edge[0];
        let mut edge_dir = (edge[1] - edge[0]).normalize();
        while !points_are_close(ray_point, edge[1]) {
            let grid_point = self.floor_point_to_grid(ray_point);
            edge_cells.insert(grid_point);
            // ray_point.x + t_delta_x * edge_dir.x = (grid_point.x + 1.0) * self.cell_size
            let t_delta_x = nan_to_inf(
                ((grid_point.x as f32 + 1.0) * self.cell_size - ray_point.x) / edge_dir.x,
            );
            let t_delta_y = nan_to_inf(
                ((grid_point.y as f32 + 1.0) * self.cell_size - ray_point.y) / edge_dir.y,
            );
            let t_delta_z = nan_to_inf(
                ((grid_point.z as f32 + 1.0) * self.cell_size - ray_point.z) / edge_dir.z,
            );

            let t_delta_min = if t_delta_x.abs() < t_delta_y.abs() {
                if t_delta_x.abs() < t_delta_z.abs() {
                    t_delta_x
                } else {
                    t_delta_z
                }
            } else if t_delta_y.abs() < t_delta_z.abs() {
                t_delta_y
            } else {
                t_delta_z
            };
            // t_delta_x.min(t_delta_y.min(t_delta_z));

            if t_delta_min < 0.0 {
                edge = [edge[1], edge[0]];
                ray_point = edge[0];
                edge_dir = (edge[1] - edge[0]).normalize();
                continue;
            }

            ray_point = clamp_vector_on_edge(ray_point + t_delta_min * edge_dir, edge);
        }

        edge_cells.insert(self.floor_point_to_grid(edge[1]));

        edge_cells
    }

    pub fn insert_edge(&mut self, edge: Edge) {
        let edge_index = self.edge_indices.len();
        let edge_indices = edge.map(|vert| {
            self.vert_in_map(vert).unwrap_or_else(|| {
                let idx = self.verts.len();
                self.verts.push(vert);
                idx
            })
        });
        self.edge_indices.push(edge_indices);
        let cells_to_modify = self.find_cells_along_edge(edge);
        for cell in cells_to_modify {
            if let Some(edges_in_cell) = self.map.get_mut(&cell) {
                edges_in_cell.insert(edge_index);
            } else {
                self.map.insert(cell, HashSet::from_iter([edge_index]));
            }
        }
    }

    /* pub fn remove_edge(&mut self, edge: Edge, edge_index: usize) {
        let cells_to_modify = self.find_cells_along_edge(edge);
        for cell in cells_to_modify {
            if let Some(edges_in_cell) = self.map.get_mut(&cell) {
                edges_in_cell.remove(&edge_index);
                if edges_in_cell.is_empty() {
                    self.map.remove(&cell);
                }
            }
        }

        /* if let Some(edge_remainder) = self.edge_remainder_after_deletion(edge_index, edge) {
            self.edge_vertices[edge_index] = edge_remainder;
            for vert in edge_remainder {
                let vert_cell = self.floor_point_to_grid(vert);
                if let Some(edges_in_cell) = self.map.get_mut(&vert_cell) {
                    edges_in_cell.insert(edge_index);
                } else {
                    self.map.insert(vert_cell, HashSet::from_iter([edge_index]));
                }
            }
        } else {
            self.edges.remove(edge_index);
        } */
    } */

    /// Takes an edge `edge_index` and splits it up into two new edges at `point`. Does nothing if
    /// `point` is near any of `edge_index`'s endpoints. Performs the operation by truncating the
    /// original edge to its new length, and creating a new edge to fill in the remaining endpoints.
    pub fn split_edge_at_point(&mut self, edge_index: usize, point: Vector3<f32>) {
        let edge_cells = self.find_cells_along_edge(self.edge(edge_index));

        let edge_indices = self.edge_indices[edge_index];
        if edge_indices
            .iter()
            .any(|&vert_idx| points_are_close(self.verts[vert_idx], point))
        {
            return;
        }

        // First index will correspond to the "old" edge, second index corresponds to the "new" edge from the split
        let cells_of_splits = edge_indices.map(|endpoint_index| {
            let split_edge = [point, self.verts[endpoint_index]];
            self.find_cells_along_edge(split_edge)
        });

        // println!("Edge: {:?}", self.edge(edge_index));
        /* assert_eq!(
            edge_cells,
            cells_of_splits[0]
                .clone()
                .union(&cells_of_splits[1].clone()).copied()
                .collect::<HashSet<Vector3<u32>>>()
        ); */

        for cell in cells_of_splits[1].difference(&cells_of_splits[0]) {
            self.map
                .get_mut(cell)
                .unwrap_or_else(|| panic!("Trying to access cell {cell}"))
                .remove(&edge_index);
        }

        for cell in &cells_of_splits[1] {
            self.map
                .get_mut(cell)
                .unwrap()
                .insert(self.edge_indices.len());
        }

        // Assume that there does not need to be any merging when subdividing an edge
        let split_point_index = self.verts.len();
        self.verts.push(point);

        let new_edges_indices =
            edge_indices.map(|endpoint_index| [endpoint_index, split_point_index]);
        self.edge_indices[edge_index] = new_edges_indices[0];
        self.edge_indices.push(new_edges_indices[1]);
    }

    pub fn edges_in_cells_near_point(&self, point: Vector3<f32>) -> HashSet<usize> {
        let point_cell = self.floor_point_to_grid(point);
        (-1..=1)
            .flat_map(|x| {
                (-1..=1).flat_map(move |y| {
                    (-1..=1).flat_map(move |z| {
                        let current_cell = Vector3::new(
                            point_cell.x.saturating_add_signed(x),
                            point_cell.y.saturating_add_signed(y),
                            point_cell.z.saturating_add_signed(z),
                        );
                        self.map.get(&current_cell)
                    })
                })
            })
            .flatten()
            .copied()
            .collect()
    }

    /// The SDF function for an edge defined by two points.
    /// Sourced from https://iquilezles.org/articles/distfunctions2d/
    fn edge_sdf(&self, edge_index: usize, point: Vector3<f32>) -> f32 {
        let [a, b] = self.edge(edge_index);
        let pa = point - a;
        let ba = b - a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);

        (pa - ba * h).norm()
    }

    pub fn eval_sdf_field(&self, point: Vector3<f32>) -> Option<(f32, usize)> {
        let point_cell = self.floor_point_to_grid(point);

        let mut cells_sorted_by_distance = Vec::from_iter(self.map.clone());
        cells_sorted_by_distance.sort_by_key(|&(cell, _)| {
            (0..2)
                .map(|i| (cell[i] as i32 - point_cell[i] as i32).abs())
                .sum::<i32>()
        });

        for (_, edge_indices) in cells_sorted_by_distance {
            if let Some(min_edge) = edge_indices
                .into_iter()
                .map(|edge_index| (self.edge_sdf(edge_index, point), edge_index))
                .min_by_key(|(dist, _)| egui::emath::OrderedFloat(*dist))
            {
                return Some(min_edge);
            }
        }

        None
    }

    /// Returns the index corresponding to a vertex if it is in the map
    pub fn vert_in_map(&self, vert: Vector3<f32>) -> Option<usize> {
        let nearby_edges = self.edges_in_cells_near_point(vert);

        nearby_edges
            .into_iter()
            .flat_map(|edge_index| {
                let edge_indices = self.edge_indices[edge_index];
                let edge = edge_indices.map(|vert_idx| self.verts[vert_idx]);
                if points_are_close(edge[0], vert) {
                    Some(edge_indices[0])
                } else if points_are_close(edge[1], vert) {
                    Some(edge_indices[1])
                } else {
                    None
                }
            })
            .next()
    }

    pub fn edge(&self, index: usize) -> Edge {
        self.edge_indices[index].map(|vert_idx| self.verts[vert_idx])
    }

    pub fn raw_edges(&self) -> Vec<Edge> {
        self.edge_indices
            .iter()
            .map(|edge_indices| edge_indices.map(|vert_index| self.verts[vert_index]))
            .collect()
    }

    pub fn edge_indices(&self) -> &[[usize; 2]] {
        &self.edge_indices
    }

    pub fn verts(&self) -> &[Vector3<f32>] {
        &self.verts
    }

    pub fn verts_mut(&mut self) -> &mut Vec<Vector3<f32>> {
        &mut self.verts
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn occupied_cells(&self) -> usize {
        self.map.len()
    }
}

fn clamp_vector_on_edge(vector: Vector3<f32>, edge: Edge) -> Vector3<f32> {
    Vector3::from_iterator(
        (0_usize..3)
            .map(|i| vector[i].clamp(edge[0][i].min(edge[1][i]), edge[0][i].max(edge[1][i]))),
    )
}

fn nan_to_inf(val: f32) -> f32 {
    if val.is_nan() || val == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        val
    }
}

#[cfg(test)]
mod test {
    use std::collections::{HashMap, HashSet};

    use nalgebra::Vector3;

    use crate::spatial_edge_hash::SpatialEdgeHash;

    macro_rules! set {
        ($($x:expr),+ $(,)?) => (
            HashSet::from_iter([$($x),+])
        );
    }

    #[test]
    fn basic_edge_insertion() {
        let edge_hash = SpatialEdgeHash::new(
            0.5,
            vec![[Vector3::new(0.1, 0.1, 0.0), Vector3::new(0.6, 0.1, 0.0)]],
        );

        assert_eq!(edge_hash.occupied_cells(), 2);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([
                (Vector3::new(0_u32, 0, 0), set![0]),
                (Vector3::new(1, 0, 0), set![0])
            ])
        );
    }

    #[test]
    fn complex_edge_insertion() {
        let edge_hash =
            SpatialEdgeHash::new(1.0, vec![[Vector3::zeros(), Vector3::new(2.9, 2.4, 0.0)]]);

        let index_set = set![0];

        assert_eq!(edge_hash.occupied_cells(), 5);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([
                (Vector3::new(0, 0, 0), index_set.clone()),
                (Vector3::new(1, 0, 0), index_set.clone()),
                (Vector3::new(1, 1, 0), index_set.clone()),
                (Vector3::new(2, 1, 0), index_set.clone()),
                (Vector3::new(2, 2, 0), index_set),
            ])
        );
    }

    #[test]
    fn same_cell_edge_insertion() {
        let edge_hash = SpatialEdgeHash::new(
            0.5,
            vec![[Vector3::new(0.5, 0.5, 0.5), Vector3::new(0.9, 0.8, 0.7)]],
        );

        assert_eq!(edge_hash.occupied_cells(), 1);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([(Vector3::new(1, 1, 1), set![0])])
        )
    }

    #[test]
    fn basic_edge_split() {
        let mut edge_hash = SpatialEdgeHash::new(
            0.5,
            vec![[Vector3::new(0.1, 0.1, 0.0), Vector3::new(0.6, 0.1, 0.0)]],
        );

        edge_hash.split_edge_at_point(0, Vector3::new(0.3, 0.1, 0.0));

        assert_eq!(edge_hash.occupied_cells(), 2);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([
                (Vector3::new(0, 0, 0), set![0, 1]),
                (Vector3::new(1, 0, 0), set![1]),
            ])
        );
        assert_eq!(edge_hash.verts.len(), 3);
    }

    #[test]
    fn complex_edge_split() {
        let mut edge_hash =
            SpatialEdgeHash::new(1.0, vec![[Vector3::zeros(), Vector3::new(2.9, 2.4, 0.0)]]);

        edge_hash.split_edge_at_point(0, Vector3::new(1.85, 1.53, 0.0));

        assert_eq!(edge_hash.occupied_cells(), 5);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([
                (Vector3::new(0, 0, 0), set![0]),
                (Vector3::new(1, 0, 0), set![0]),
                (Vector3::new(1, 1, 0), set![0, 1]),
                (Vector3::new(2, 1, 0), set![1]),
                (Vector3::new(2, 2, 0), set![1]),
            ])
        );
    }
}
