use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

use egui::emath::OrderedFloat;
use nalgebra::{Vector2, Vector3};

use crate::network_generation_component::{points_are_close, vector_project};

pub type Edge = [Vector3<f32>; 2];

const CACHE_SIZE: usize = 10;
type CellCache = Arc<RwLock<Vec<Option<(Vector3<u32>, HashSet<usize>)>>>>;

#[derive(Debug, Clone)]
pub struct SpatialEdgeHash {
    map: HashMap<Vector3<u32>, HashSet<usize>>,
    verts: Vec<Vector3<f32>>,
    edge_indices: Vec<[usize; 2]>,
    barrier_edges: Vec<Edge>,
    cell_size: f32,
    last_cells_memoization: CellCache,
}

#[allow(unused)]
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
            last_cells_memoization: Arc::new(RwLock::new(vec![None; CACHE_SIZE])),
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
                clamp_vector_on_edge(intersection_point + 0.0001 * (edge[1] - edge[0]), edge);
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

        let mut memo = self.last_cells_memoization.write().unwrap();
        *memo = vec![None; CACHE_SIZE];
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

        let edge_vertices = edge_indices.map(|vert_index| self.verts[vert_index]);

        let projection = vector_project(
            edge_vertices[1] - edge_vertices[0],
            point - edge_vertices[0],
        ) + edge_vertices[0];

        // First index will correspond to the "old" edge, second index corresponds to the "new" edge from the split
        let cells_of_splits = edge_indices.map(|endpoint_index| {
            let split_edge = [projection, self.verts[endpoint_index]];
            self.find_cells_along_edge(split_edge)
        });

        /* println!("Edge: {edge_vertices:?}, point: {point:?}, projection: {projection:?}");
        println!("Full: {:?}", self.find_cells_along_edge(edge_vertices));
        println!("Partial 0: {:?}", self.find_cells_along_edge([projection, edge_vertices[0]]));
        println!("Partial 1: {:?}", self.find_cells_along_edge([projection, edge_vertices[1]])); */

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
        self.verts.push(projection);

        let new_edges_indices =
            edge_indices.map(|endpoint_index| [endpoint_index, split_point_index]);
        self.edge_indices[edge_index] = new_edges_indices[0];
        self.edge_indices.push(new_edges_indices[1]);

        let mut memo = self.last_cells_memoization.write().unwrap();
        *memo = vec![None; CACHE_SIZE];
    }

    pub fn edges_in_cells_near_point(&self, point: Vector3<f32>) -> HashSet<usize> {
        let point_cell = self.floor_point_to_grid(point);
        if let Ok(cache) = self.last_cells_memoization.write().as_deref_mut()
            && let Some(cache_index) = cache.iter().position(|cell_res| {
                if let Some((cell, _)) = cell_res {
                    *cell == point_cell
                } else {
                    false
                }
            })
        {
            // println!("Memo hit");
            cache.swap(0, cache_index);
            return cache[0].clone().unwrap().1;
        }

        let nearby_edges: HashSet<usize> = [
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, 1, 0),
            (-1, 1, 1),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        .into_iter()
        .flat_map(|(x, y, z)| {
            let current_cell = Vector3::new(
                point_cell.x.saturating_add_signed(x),
                point_cell.y.saturating_add_signed(y),
                point_cell.z.saturating_add_signed(z),
            );
            self.map.get(&current_cell).cloned().unwrap_or_default()
        })
        .collect();

        let mut memo = self.last_cells_memoization.write().unwrap();
        memo.rotate_right(1);
        memo[0] = Some((point_cell, nearby_edges.clone()));

        nearby_edges
    }

    /// The SDF function for an edge defined by two points.
    /// Sourced from [https://iquilezles.org/articles/distfunctions2d/]
    fn edge_sdf([a, b]: Edge, point: Vector3<f32>) -> f32 {
        let pa = point - a;
        let ba = b - a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);
        if h.is_nan() {
            (point - a).norm()
        } else {
            (pa - ba * h).norm()
        }
    }

    fn closest_edge_in_cell(
        &self,
        cell: Vector3<u32>,
        point: Vector3<f32>,
        checked_edges: &mut HashSet<usize>,
    ) -> Option<(f32, usize)> {
        self.map.get(&cell).and_then(|cell_edges| {
            cell_edges
                .iter()
                .flat_map(|&edge_index| {
                    if checked_edges.contains(&edge_index) {
                        None
                    } else {
                        checked_edges.insert(edge_index);
                        Some((Self::edge_sdf(self.edge(edge_index), point), edge_index))
                    }
                })
                .min_by(|&(a, _), &(b, _)| a.total_cmp(&b))
        })
    }

    pub fn eval_sdf_field(&self, point: Vector3<f32>) -> Option<(f32, usize)> {
        let mut remaining_cells: HashSet<Vector3<u32>> = self.map.keys().copied().collect();

        let mut ring_distance = 0;
        let mut checked_edges = HashSet::new();

        while !remaining_cells.is_empty() && checked_edges.len() != self.edge_indices.len() {
            let mut minimum_edge: Option<(f32, usize)> = None;
            for cell in remaining_cells
                .iter()
                .filter(|&&cell| {
                    (point / self.cell_size
                        - Vector3::new(
                            cell.x as f32 + 0.5,
                            cell.y as f32 + 0.5,
                            cell.z as f32 + 0.5,
                        ))
                    .norm()
                        <= ring_distance as f32
                })
                .copied()
                .collect::<Vec<_>>()
            {
                remaining_cells.remove(&cell);

                if let Some(closest_edge) =
                    self.closest_edge_in_cell(cell, point, &mut checked_edges)
                {
                    if let Some(minimum_edge) = &mut minimum_edge {
                        if closest_edge.0 < minimum_edge.0 {
                            *minimum_edge = closest_edge;
                        }
                    } else {
                        minimum_edge = Some(closest_edge);
                    }
                }
            }

            if let Some(minimum_edge) = minimum_edge {
                return Some(minimum_edge);
            }

            ring_distance += 1;
        }

        None
    }

    pub fn eval_barrier_sdf_field(&self, point: Vector3<f32>) -> Option<(usize, f32)> {
        self.barrier_edges
            .iter()
            .enumerate()
            .map(|(edge_index, edge)| (edge_index, Self::edge_sdf(*edge, point)))
            .min_by_key(|(_, x)| OrderedFloat(*x))
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
        let mut memo = self.last_cells_memoization.write().unwrap();
        *memo = vec![None; CACHE_SIZE];
        &mut self.verts
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn occupied_cells(&self) -> usize {
        self.map.len()
    }

    pub fn barrier_edges(&self) -> &[Edge] {
        &self.barrier_edges
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

    use crate::spatial_edge_hash::{Edge, SpatialEdgeHash};

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

    // Autogenerated tests by ChatGPT for ray-cube intersection tests
    fn approx_vec(a: Vector3<f32>, b: Vector3<f32>) -> bool {
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

        assert!(approx_vec(hit, Vector3::new(cell, 0.5, 0.5)));
    }

    #[test]
    fn axis_aligned_negative_x() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [-1.0, 0.5, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(0.0, 0.5, 0.5)));
    }

    #[test]
    fn axis_aligned_positive_y() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.2, 0.3, 0.4], [0.2, 2.0, 0.4]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(0.2, cell, 0.4)));
    }

    #[test]
    fn axis_aligned_positive_z() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.2, 0.3, 0.4], [0.2, 0.3, 2.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(0.2, 0.3, cell)));
    }

    #[test]
    fn diagonal_hits_corner() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(cell, cell, cell)));
    }

    #[test]
    fn diagonal_hits_edge() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.5, 0.5, 0.5], [2.0, 2.0, 0.5]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(cell, cell, 0.5)));
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

        assert!(approx_vec(hit, Vector3::new(cell, 0.5, 0.5)));
    }

    #[test]
    fn starting_near_corner() {
        let cell = 1.0;
        let cube = SpatialEdgeHash::new(cell, Vec::new());

        let ray = make_edge([0.001, 0.001, 0.001], [-1.0, -1.0, -1.0]);
        let hit = cube.cube_ray_intersection(ray);

        assert!(approx_vec(hit, Vector3::new(0.0, 0.0, 0.0)));
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

    #[test]
    fn sensitivity_check() {
        let edge = [
            Vector3::new(395.63635, 408.31998, 0.0),
            Vector3::new(188.14307, 303.6836, 0.0),
        ];
        let cube = SpatialEdgeHash::new(60.0, Vec::new());

        let cells = cube.find_cells_along_edge(edge);

        assert_eq!(cells.len(), 5);
    }

    // Autogenerated tests by ChatGPT for sdf tests
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }
    /// Ground truth: point-to-segment distance
    fn point_segment_distance(p: Vector3<f32>, e: &Edge) -> f32 {
        let a = e[0];
        let b = e[1];
        let ab = b - a;
        let t = ((p - a).dot(&ab) / ab.dot(&ab)).clamp(0.0, 1.0);
        let closest = a + t * ab;
        (p - closest).norm()
    }

    fn build_field(edges: Vec<Edge>) -> SpatialEdgeHash {
        SpatialEdgeHash::new(60.0, edges)
    }

    #[test]
    fn empty_field_returns_none() {
        let field = build_field(vec![]);
        let result = field.eval_sdf_field(Vector3::new(0.0, 0.0, 0.0));
        assert!(result.is_none());
    }

    #[test]
    fn point_on_edge_returns_zero() {
        let edges = vec![[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)]];
        let field = build_field(edges);

        let p = Vector3::new(0.5, 0.0, 0.0);
        let (dist, idx) = field.eval_sdf_field(p).unwrap();

        assert!(approx_eq(dist, 0.0));
        assert_eq!(idx, 0);
    }

    #[test]
    fn point_near_middle_of_edge() {
        let edges = vec![[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)]];
        let field = build_field(edges);

        let p = Vector3::new(0.5, 1.0, 0.0);
        let (dist, idx) = field.eval_sdf_field(p).unwrap();

        assert!(approx_eq(dist, 1.0));
        assert_eq!(idx, 0);
    }

    #[test]
    fn point_closest_to_endpoint() {
        let edges = vec![[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)]];
        let field = build_field(edges);

        let p = Vector3::new(2.0, 1.0, 0.0);
        let (dist, _) = field.eval_sdf_field(p).unwrap();

        let expected = (Vector3::new(1.0, 0.0, 0.0) - p).norm();
        assert!(approx_eq(dist, expected));
    }

    #[test]
    fn picks_correct_edge() {
        let edges = vec![
            [Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)], // 0
            [Vector3::new(0.0, 2.0, 0.0), Vector3::new(1.0, 2.0, 0.0)], // 1
        ];
        let field = build_field(edges);

        let p = Vector3::new(0.5, 1.8, 0.0);
        let (_, idx) = field.eval_sdf_field(p).unwrap();

        assert_eq!(idx, 1);
    }

    #[test]
    fn tie_breaking_is_consistent() {
        let edges = vec![
            [Vector3::new(-1.0, 0.0, 0.0), Vector3::new(-1.0, 1.0, 0.0)], // 0
            [Vector3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0)],   // 1
        ];
        let field = build_field(edges);

        let p = Vector3::new(0.0, 0.5, 0.0);
        let (dist, idx) = field.eval_sdf_field(p).unwrap();

        assert!(approx_eq(dist, 1.0));
        assert!(idx == 0 || idx == 1); // depends on your policy
    }

    #[test]
    fn diagonal_edge_distance() {
        let edges = vec![[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0)]];
        let field = build_field(edges);

        let p = Vector3::new(1.0, 0.0, 0.0);
        let (dist, _) = field.eval_sdf_field(p).unwrap();

        let expected = (2.0f32).sqrt() / 2.0;
        assert!(approx_eq(dist, expected));
    }

    #[test]
    fn degenerate_edge_point() {
        let edges = vec![[Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0)]];
        let field = build_field(edges);

        let p = Vector3::new(2.0, 1.0, 1.0);
        let (dist, _) = field.eval_sdf_field(p).unwrap();

        assert!(approx_eq(dist, 1.0));
    }

    #[test]
    fn matches_bruteforce_for_multiple_edges() {
        let edges = vec![
            [Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)],
            [Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0)],
            [Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 1.0)],
        ];
        let field = build_field(edges.clone());

        let test_points = [
            Vector3::new(0.2, 0.3, 0.4),
            Vector3::new(1.5, 0.2, 0.1),
            Vector3::new(-0.5, 0.5, 0.5),
        ];

        for p in test_points {
            let (dist, idx) = field.eval_sdf_field(p).unwrap();

            let mut best = f32::MAX;
            let mut best_indices = vec![0];

            for (i, e) in edges.iter().enumerate() {
                let d = point_segment_distance(p, e);
                if d < best {
                    best = d;
                    best_indices = vec![i];
                } else if approx_eq(d, best) {
                    best_indices.push(i);
                }
            }

            assert!(approx_eq(dist, best));
            assert!(best_indices.contains(&idx));
        }
    }

    #[test]
    fn stability_near_edge() {
        let edges = vec![[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)]];
        let field = build_field(edges);

        let p1 = Vector3::new(0.5, 1e-6, 0.0);
        let p2 = Vector3::new(0.5, -1e-6, 0.0);

        let (d1, _) = field.eval_sdf_field(p1).unwrap();
        let (d2, _) = field.eval_sdf_field(p2).unwrap();

        assert!(approx_eq(d1, d2));
    }
}
