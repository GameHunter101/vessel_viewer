use std::collections::{HashMap, HashSet};

use egui::emath::OrderedFloat;
use nalgebra::{Vector2, Vector3};

use crate::network_generation_component::points_are_close;

pub type Edge = [Vector3<f32>; 2];

#[derive(Debug, Clone)]
pub struct SpatialEdgeHash {
    map: HashMap<Vector3<u32>, HashSet<usize>>,
    verts: Vec<Vector3<f32>>,
    edge_indices: Vec<[usize; 2]>,
    barrier_edges: Vec<Edge>,
    cell_size: f32,
}

impl SpatialEdgeHash {
    pub fn new(spatial_subdivision: f32, edges: Vec<Edge>) -> Self {
        let mut edge_hash = Self {
            map: HashMap::new(),
            verts: Vec::new(),
            edge_indices: Vec::new(),
            barrier_edges: vec![
                [Vector3::zeros(), Vector3::new(0.0, 512.0, 0.0)],
                [
                    Vector3::new(512.0, 0.0, 0.0),
                    Vector3::new(512.0, 512.0, 0.0),
                ],
            ],
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

    /// Calculates the parameter `t` for which the ray exits the "slab". A slab is a set of two
    /// parallel sides of the cube. [https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/03raytracing1.pdf]
    fn cube_slab_ray_exit(&self, ray: [Vector2<f32>; 2]) -> f32 {
        let d = ray[1] - ray[0];
        let a = ray[0];

        let t_x_min = nan_to_inf(-a.x / d.x);
        let t_x_max = nan_to_inf((self.cell_size - a.x) / d.x);
        let t_x_exit = t_x_min.max(t_x_max);

        let t_y_min = nan_to_inf(-a.y / d.y);
        let t_y_max = nan_to_inf((self.cell_size - a.y) / d.y);
        let t_y_exit = t_y_min.max(t_y_max);

        t_x_exit.min(t_y_exit)
    }

    fn cube_ray_intersection(&self, ray: Edge) -> Vector3<f32> {
        let t = [[0, 1], [1, 2], [0, 2]]
            .map(|[comp_1, comp_2]| {
                let current_ray = ray.map(|point| Vector2::new(point[comp_1], point[comp_2]));
                self.cube_slab_ray_exit(current_ray)
            })
            .into_iter()
            .min_by_key(|x| egui::emath::OrderedFloat(*x))
            .unwrap();

        ray[0] + t * (ray[1] - ray[0])
    }

    /// Detects which grid cells the edge intersects, updates cells with new edge.
    /// Algorithm adapted from [http://www.cse.yorku.ca/~amana/research/grid.pdf]
    fn find_cells_along_edge(&self, edge: Edge) -> HashSet<Vector3<u32>> {
        let mut edge_cells: HashSet<Vector3<u32>> = HashSet::new();
        let mut ray_point = edge[0];
        while !points_are_close(ray_point, edge[1]) {
            let grid_point = self.floor_point_to_grid(ray_point);
            edge_cells.insert(grid_point);

            let cell_origin = self.cell_size
                * Vector3::new(
                    grid_point.x as f32,
                    grid_point.y as f32,
                    grid_point.z as f32,
                );

            let intersection_point = self
                .cube_ray_intersection([ray_point - cell_origin, edge[1] - cell_origin])
                + cell_origin;

            // Introduce minor relaxation factor to nudge points off of cell boundaries
            ray_point =
                clamp_vector_on_edge(intersection_point + 0.001 * (edge[1] - edge[0]), edge);
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

    /// Takes an edge `edge_index` and splits it up into two new edges at `point`. Does nothing if
    /// `point` is near any of `edge_index`'s endpoints. Performs the operation by truncating the
    /// original edge to its new length, and creating a new edge to fill in the remaining endpoints.
    pub fn split_edge_at_point(&mut self, edge_index: usize, point: Vector3<f32>) {
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
    /// Sourced from [https://iquilezles.org/articles/distfunctions2d/]
    fn edge_sdf([a, b]: Edge, point: Vector3<f32>) -> f32 {
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
                .map(|edge_index| (Self::edge_sdf(self.edge(edge_index), point), edge_index))
                .min_by_key(|(dist, _)| egui::emath::OrderedFloat(*dist))
            {
                return Some(min_edge);
            }
        }

        None
    }

    pub fn eval_barrier_sdf_field(&self, point: Vector3<f32>) -> Option<f32> {
        self.barrier_edges
            .iter()
            .map(|edge| Self::edge_sdf(*edge, point))
            .min_by_key(|&x| OrderedFloat(x))
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

    // Autogenerated tests by ChatGPT

    fn approx_eq(a: Vector3<f32>, b: Vector3<f32>) -> bool {
        let eps = 1e-5;
        (a - b).abs().max() < eps
    }

    fn is_on_cube_surface(p: Vector3<f32>, cell: f32) -> bool {
        let eps = 1e-5;

        let in_bounds = |x: f32| x >= -eps && x <= cell + eps;
        let on_face = |x: f32| x.abs() < eps || (x - cell).abs() < eps;

        in_bounds(p.x)
            && in_bounds(p.y)
            && in_bounds(p.z)
            && (on_face(p.x) || on_face(p.y) || on_face(p.z))
    }

    fn make_edge(a: [f32; 3], b: [f32; 3]) -> super::Edge {
        [Vector3::from(a), Vector3::from(b)]
    }

    #[test]
    fn axis_aligned_positive_x() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [2.0, 0.5, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(cell, 0.5, 0.5)));
    }

    #[test]
    fn axis_aligned_negative_x() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [-1.0, 0.5, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(0.0, 0.5, 0.5)));
    }

    #[test]
    fn axis_aligned_positive_y() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.2, 0.3, 0.4], [0.2, 2.0, 0.4]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(0.2, cell, 0.4)));
    }

    #[test]
    fn axis_aligned_positive_z() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.2, 0.3, 0.4], [0.2, 0.3, 2.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(0.2, 0.3, cell)));
    }

    #[test]
    fn diagonal_hits_corner() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(cell, cell, cell)));
    }

    #[test]
    fn diagonal_hits_edge() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [2.0, 2.0, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(cell, cell, 0.5)));
    }

    #[test]
    fn arbitrary_direction() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.3, 0.4, 0.5], [1.5, 0.6, 0.7]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(is_on_cube_surface(hit, cell));
    }

    #[test]
    fn very_close_to_face() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.9999, 0.5, 0.5], [2.0, 0.5, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(cell, 0.5, 0.5)));
    }

    #[test]
    fn starting_near_corner() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.001, 0.001, 0.001], [-1.0, -1.0, -1.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_eq(hit, Vector3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn random_directions_stay_on_surface() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let origins = [[0.5, 0.5, 0.5], [0.2, 0.7, 0.3], [0.8, 0.1, 0.9]];

        let dirs = [
            [1.0, 0.3, 0.2],
            [-0.5, 1.0, 0.1],
            [0.2, -0.3, 1.0],
            [-1.0, -1.0, -1.0],
        ];

        for o in origins.iter() {
            for d in dirs.iter() {
                let start = Vector3::from(*o);
                let end = start + Vector3::from(*d);

                let ray = [start, end];
                let hit = cube.cube_ray_intersection(ray);

                assert!(is_on_cube_surface(hit, cell));
            }
        }
    }
}
