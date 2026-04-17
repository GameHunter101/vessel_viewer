use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;

use crate::network_generation_component::points_are_close;

pub type Edge = [Vector3<f32>; 2];

#[derive(Debug, Clone)]
pub struct SpatialEdgeHash {
    // edges: Vec<[usize; 2]>,
    map: HashMap<Vector3<u32>, HashSet<usize>>,
    edge_vertices: Vec<Edge>,
    cell_size: f32,
}

impl SpatialEdgeHash {
    pub fn new(spatial_subdivision: f32, edges: Vec<Edge>) -> Self {
        let mut edge_hash = Self {
            map: HashMap::new(),
            edge_vertices: Vec::new(),
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
    fn find_cells_along_edge(&self, edge: Edge) -> HashSet<Vector3<u32>> {
        let mut edge_cells: HashSet<Vector3<u32>> = HashSet::new();
        let mut ray_point = edge[0];
        let edge_dir = (edge[1] - edge[0]).normalize();
        while !points_are_close(ray_point, edge[1]) {
            let grid_point = self.floor_point_to_grid(ray_point);
            edge_cells.insert(grid_point);
            // ray_point.x + t_delta_x * edge_dir.x = grid_point.x + 1.0
            let t_delta_x = nan_to_inf((grid_point.x as f32 + 1.0 - ray_point.x) / edge_dir.x);
            let t_delta_y = nan_to_inf((grid_point.y as f32 + 1.0 - ray_point.y) / edge_dir.y);
            let t_delta_z = nan_to_inf((grid_point.z as f32 + 1.0 - ray_point.z) / edge_dir.z);

            ray_point = clamp_vector_on_edge(
                ray_point + t_delta_x.min(t_delta_y.min(t_delta_z)) * edge_dir,
                edge,
            );
        }

        edge_cells.insert(self.floor_point_to_grid(edge[1]));

        edge_cells
    }

    pub fn insert_edge(&mut self, edge: Edge) {
        let edge_index = self.edge_vertices.len();
        self.edge_vertices.push(edge);
        let cells_to_modify = self.find_cells_along_edge(edge);
        for cell in cells_to_modify {
            if let Some(edges_in_cell) = self.map.get_mut(&cell) {
                edges_in_cell.insert(edge_index);
            } else {
                self.map.insert(cell, HashSet::from_iter([edge_index]));
            }
        }
    }

    pub fn remove_edge(&mut self, edge: Edge, edge_index: usize) {
        let cells_to_modify = self.find_cells_along_edge(edge);
        for cell in cells_to_modify {
            if let Some(edges_in_cell) = self.map.get_mut(&cell) {
                edges_in_cell.remove(&edge_index);
                if edges_in_cell.is_empty() {
                    self.map.remove(&cell);
                }
            }
        }

        if let Some(edge_remainder) = self.edge_remainder_after_deletion(edge_index, edge) {
            self.edge_vertices[edge_index] = edge_remainder;
            for vert in edge_remainder {
                let vert_cell = self.floor_point_to_grid(vert);
                if let Some(edges_in_cell) = self.map.get_mut(&vert_cell) {
                    edges_in_cell.insert(edge_index);
                } else {
                    self.map.insert(vert_cell, HashSet::from_iter([edge_index]));
                }
            }
        }
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
                        self.map[&current_cell].clone()
                    })
                })
            })
            .collect()
    }

    pub fn occupied_cells(&self) -> usize {
        self.map.len()
    }

    fn edge_remainder_after_deletion(&self, edge: usize, removal: Edge) -> Option<Edge> {
        let full_edge = self.edge_vertices[edge];
        let original_endpoint = full_edge
            .into_iter()
            .filter(|v| !removal.iter().any(|r| points_are_close(*v, *r)))
            .next();

        original_endpoint.map(|point| {
            let other_endpoint = removal
                .into_iter()
                .filter(|r| !full_edge.iter().any(|v| points_are_close(*v, *r)))
                .next();

            [point, other_endpoint.unwrap()]
        })
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edge_vertices
    }
}

fn clamp_vector_on_edge(vector: Vector3<f32>, edge: Edge) -> Vector3<f32> {
    Vector3::from_iterator(
        (0_usize..3)
            .map(|i| vector[i].clamp(edge[0][i].min(edge[1][i]), edge[0][i].max(edge[1][i]))),
    )
}

fn nan_to_inf(val: f32) -> f32 {
    if val.is_nan() { f32::INFINITY } else { val }
}

#[cfg(test)]
mod test {
    use std::collections::{HashMap, HashSet};

    use nalgebra::Vector3;

    use crate::spatial_edge_hash::SpatialEdgeHash;

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
                (Vector3::new(0_u32, 0, 0), HashSet::from_iter(vec![0_usize])),
                (Vector3::new(1, 0, 0), HashSet::from_iter(vec![0]))
            ])
        );
    }

    #[test]
    fn complex_edge_insertion() {
        let edge_hash =
            SpatialEdgeHash::new(1.0, vec![[Vector3::zeros(), Vector3::new(2.9, 2.4, 0.0)]]);

        let index_set = HashSet::from_iter(vec![0_usize]);

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
            HashMap::from_iter([(Vector3::new(1, 1, 1), HashSet::from_iter([0]))])
        )
    }

    #[test]
    fn basic_edge_removal() {
        let edge = [Vector3::new(0.1, 0.1, 0.0), Vector3::new(0.6, 0.1, 0.0)];
        let mut edge_hash = SpatialEdgeHash::new(0.5, vec![edge]);

        edge_hash.remove_edge(edge, 0);

        assert_eq!(edge_hash.occupied_cells(), 0);
        assert_eq!(edge_hash.map, HashMap::new());
    }

    #[test]
    fn complex_edge_removal() {
        let mut edge_hash =
            SpatialEdgeHash::new(1.0, vec![[Vector3::zeros(), Vector3::new(2.9, 2.4, 0.0)]]);

        edge_hash.remove_edge(
            [Vector3::new(1.85, 1.53, 0.0), Vector3::new(2.9, 2.4, 0.0)],
            0,
        );

        let index_set = HashSet::from_iter(vec![0_usize]);

        assert_eq!(edge_hash.occupied_cells(), 3);
        assert_eq!(
            edge_hash.map,
            HashMap::from_iter([
                (Vector3::new(0, 0, 0), index_set.clone()),
                (Vector3::new(1, 0, 0), index_set.clone()),
                (Vector3::new(1, 1, 0), index_set.clone()),
            ])
        );
    }
}
