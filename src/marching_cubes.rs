use std::io::Write;

use nalgebra::Vector3;
use rand::rngs::StdRng;
use v4::{
    builtin_actions::RegisterUiComponentAction,
    component,
    ecs::{
        actions::ActionQueue,
        component::{ComponentDetails, ComponentSystem, UpdateParams},
        compute::Compute,
        scene::Id,
    },
};
use wgpu::{Device, Queue};

use crate::{
    AREA_SIZE, MAX_EDGES_IN_CELL, network_generation_component::NetworkGenerationComponent,
    spatial_edge_hash::SpatialEdgeHash,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchingCubesData {
    domain_padding: f32,
    vessel_thickness: f32,
    area_size: f32,
    padding_0: f32,
    domain_bottom_left: [f32; 3],
    padding_1: f32,
    domain_top_right: [f32; 3],
    cell_size: f32,
    cell_counts: [u32; 3],
    padding_2: f32,
}

impl Default for MarchingCubesData {
    fn default() -> Self {
        Self {
            domain_padding: 1.0,
            vessel_thickness: 1.0,
            area_size: AREA_SIZE as f32,
            padding_0: 0.0,
            domain_bottom_left: [-2.0, -2.0, -2.0],
            padding_1: 0.0,
            domain_top_right: [AREA_SIZE as f32 + 2.0, AREA_SIZE as f32 + 2.0, 2.0],
            cell_size: 40.0,
            cell_counts: [1; 3],
            padding_2: 0.0,
        }
    }
}

#[component]
pub struct MarchingCubesComponent {
    #[default(false)]
    execute_compute_step: bool,
    network_generation_component: Id,
    mesh_generation_computes: Vec<Id>,
    #[default((1, 1, 1))]
    marching_cubes_samples: (u32, u32, u32),
}

impl MarchingCubesComponent {
    fn update_buffers(
        &self,
        samples_compute: &mut Compute,
        raw_edge_map: &[[u32; MAX_EDGES_IN_CELL]],
        raw_edges: &[[[f32; 2]; 2]],
        parameters: MarchingCubesData,
        device: &Device,
        queue: &Queue,
    ) {
        let (x, y, z) = self.marching_cubes_samples;

        samples_compute.set_workgroup_counts(v4::ecs::compute::WorkgroupCounts::Static(x, y, z));

        for (i, buf) in [
            bytemuck::cast_slice(&[parameters]),
            bytemuck::cast_slice(raw_edge_map),
            bytemuck::cast_slice(raw_edges),
            bytemuck::cast_slice(&vec![0.0_f32; (x * y * z) as usize]),
        ]
        .into_iter()
        .enumerate()
        {
            samples_compute
                .update_buffer_attachment(i, buf, device, queue)
                .unwrap();
        }
    }

    fn vessel_sdf(edge_map: &SpatialEdgeHash, point: Vector3<f32>, thickness: f32) -> f32 {
        let nearby_edges = edge_map.edges_in_cells_near_point(point);

        let vector_min = |a: Vector3<f32>, b: Vector3<f32>| {
            Vector3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
        };
        let vector_max = |a: Vector3<f32>, b: Vector3<f32>| {
            Vector3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
        };
        // nearby_edges.iter().last().map(|x| *x as f32).unwrap_or(f32::INFINITY)
        // edge_map.temp(point)

        nearby_edges
            .into_iter()
            .map(|edge_index| {
                let [a, b] = edge_map.edge(edge_index);
                let projection =
                    crate::network_generation_component::vector_project(b - a, point - a) + a;

                let min_point = vector_min(a, b);
                let max_point = vector_max(a, b);
                let clamped_projection = vector_max(vector_min(projection, max_point), min_point);

                let dist = clamped_projection - point;

                dist.dot(&dist) - thickness * thickness
            })
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(f32::INFINITY)
    }

    fn generate_mesh(
        &self,
        computes: &mut [Compute],
        raw_edge_map: &[[u32; MAX_EDGES_IN_CELL]],
        raw_edges: &[[[f32; 2]; 2]],
        parameters: MarchingCubesData,
        device: &Device,
        queue: &Queue,
    ) {
        let mut computes_to_execute: Vec<&mut Compute> = computes
            .iter_mut()
            .filter(|compute| self.mesh_generation_computes.contains(&compute.id()))
            .collect();

        self.update_buffers(
            computes_to_execute[0],
            raw_edge_map,
            raw_edges,
            parameters,
            device,
            queue,
        );

        let mut encoder = device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("Marching Cubes Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Marching Cubes Pass"),
                timestamp_writes: None,
            });

            for compute in computes_to_execute {
                Compute::individual_compute_execution(
                    compute,
                    device,
                    queue,
                    Some(&mut compute_pass),
                )
                .unwrap();
            }
        }

        queue.submit(Some(encoder.finish()));

        /* let sdf: &(dyn Fn(Vector3<f32>) -> f32 + Sync) =
            &move |point: Vector3<f32>| Self::vessel_sdf(edge_map, point, parameters.vessel_thickness);
        // let tik = std::time::Instant::now();
        let marching_cubes = MarchingCubes::new(
            [
                Vector3::new(-1.0, -1.0, -1.0) * (parameters.vessel_thickness + parameters.domain_padding),
                Vector3::new(
                    AREA_SIZE as f32 + parameters.vessel_thickness + parameters.domain_padding,
                    AREA_SIZE as f32 + parameters.vessel_thickness + parameters.domain_padding,
                    parameters.vessel_thickness + parameters.domain_padding,
                ),
            ],
            &sdf,
            300,
            300,
            12,
        );
        println!("{:?}", marching_cubes.samples);
        let tok = std::time::Instant::now();
        marching_cubes.march_cubes(0.00001);
        println!(
            "Total: {}, grid gen: {}, polygonization: {}",
            tik.elapsed().as_secs(),
            (tok - tik).as_secs(),
            tok.elapsed().as_secs()
        ) */
    }
}

impl ComponentSystem for MarchingCubesComponent {
    fn initialize(&mut self, _device: &Device) -> ActionQueue {
        self.set_initialized();
        vec![Box::new(RegisterUiComponentAction {
            component_id: self.id,
            text_component_properties: None,
        })]
    }

    fn update(
        &mut self,
        UpdateParams {
            device,
            queue,
            other_components,
            computes,
            ..
        }: UpdateParams,
    ) -> ActionQueue {
        let Some(network_component) = other_components
            .iter()
            .find(|comp| comp.id() == self.network_generation_component)
            .and_then(|comp| comp.downcast_ref::<NetworkGenerationComponent<StdRng>>())
        else {
            return Vec::new();
        };

        let raw_edge_map = network_component.edge_map().raw_edge_map();

        let raw_edges: Vec<[[f32; 2]; 2]> = network_component
            .edge_map()
            .raw_edges()
            .iter()
            .map(|[v0, v1]| [[v0.x, v0.y], [v1.x, v1.y]])
            .collect();

        // if self.execute_compute_step {
        let parameters = MarchingCubesData {
            cell_size: network_component.edge_map().cell_size(),
            cell_counts: network_component.edge_map().cell_counts().map(|v| v as u32),
            ..Default::default()
        };

        self.generate_mesh(
            computes,
            &raw_edge_map,
            &raw_edges,
            parameters,
            device,
            queue,
        );

        if self.execute_compute_step {
            let sdf: &(dyn Fn(Vector3<f32>) -> f32 + Sync) = &move |point: Vector3<f32>| {
                Self::vessel_sdf(
                    network_component.edge_map(),
                    point,
                    parameters.vessel_thickness,
                )
            };
            // let tik = std::time::Instant::now();
            let marching_cubes = MarchingCubes::new(
                [
                    Vector3::new(-1.0, -1.0, -1.0)
                        * (parameters.vessel_thickness + parameters.domain_padding),
                    Vector3::new(
                        AREA_SIZE as f32 + parameters.vessel_thickness + parameters.domain_padding,
                        AREA_SIZE as f32 + parameters.vessel_thickness + parameters.domain_padding,
                        parameters.vessel_thickness + parameters.domain_padding,
                    ),
                ],
                &sdf,
                self.marching_cubes_samples.0 as usize,
                self.marching_cubes_samples.1 as usize,
                self.marching_cubes_samples.2 as usize,
                /* 300,
                300,
                12, */
            );
            println!("{:?}", marching_cubes.samples);
            self.execute_compute_step = false;
        }

        Vec::new()
    }

    fn ui_render(&mut self, ctx: &egui::Context) {
        egui::Area::new("sample counts area".into())
            .anchor(egui::Align2::RIGHT_TOP, [0.0, 0.0])
            .show(ctx, |ui| {
                egui::Frame::dark_canvas(&Default::default()).show(ui, |ui| {
                    let marching_cubes_x_axis_label = ui.label("Model X-axis samples");
                    let marching_cubes_x_axis_value = ui.add(
                        egui::DragValue::new(&mut self.marching_cubes_samples.0).range(2..=512),
                    );
                    let marching_cubes_y_axis_label = ui.label("Model Y-axis samples");
                    let marching_cubes_y_axis_value = ui.add(
                        egui::DragValue::new(&mut self.marching_cubes_samples.1).range(2..=512),
                    );
                    let marching_cubes_z_axis_label = ui.label("Model Z-axis samples");
                    let marching_cubes_z_axis_value = ui.add(
                        egui::DragValue::new(&mut self.marching_cubes_samples.2).range(2..=20),
                    );

                    marching_cubes_x_axis_value.labelled_by(marching_cubes_x_axis_label.id);
                    marching_cubes_y_axis_value.labelled_by(marching_cubes_y_axis_label.id);
                    marching_cubes_z_axis_value.labelled_by(marching_cubes_z_axis_label.id);

                    if ui.add(egui::Button::new("Generate STL")).clicked() {
                        self.execute_compute_step = true;
                    }
                });
            });
    }
}

/// My implementation of the marching cubes algorithm as described on
/// [https://paulbourke.net/geometry/polygonise]. The tables were also sourced from his website.
pub struct MarchingCubes<'a> {
    samples: Vec<Vec<Vec<f32>>>,
    sdf: &'a dyn Fn(Vector3<f32>) -> f32,
    domain: [Vector3<f32>; 2],
}

impl<'a> MarchingCubes<'a> {
    /// Initializes the marching cubes algorithm by sampling the grid along each axis.
    /// The domain is specified by two points: a lower-left corner and top-right
    /// corner.
    pub fn new(
        domain: [Vector3<f32>; 2],
        sdf: &'a (dyn Fn(Vector3<f32>) -> f32 + Sync),
        x_samples: usize,
        y_samples: usize,
        z_samples: usize,
    ) -> Self {
        assert!(x_samples >= 2);
        assert!(y_samples >= 2);
        assert!(z_samples >= 2);

        let samples = std::thread::scope(|scope| {
            let handles_grid: Vec<_> = (0..z_samples)
                .map(|z| {
                    (0..y_samples)
                        .map(|y| {
                            scope.spawn(move || {
                                (0..x_samples)
                                    .map(|x| {
                                        let domain_diff = domain[1] - domain[0];
                                        let point = Vector3::new(
                                            x as f32 / (x_samples - 1) as f32,
                                            y as f32 / (y_samples - 1) as f32,
                                            z as f32 / (z_samples - 1) as f32,
                                        )
                                        .component_mul(&domain_diff)
                                            + domain[0];
                                        sdf(point)
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            handles_grid
                .into_iter()
                .map(|handle_row| {
                    handle_row
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        Self {
            samples,
            sdf,
            domain,
        }
    }

    /// Each bit of the returned value describes whether its equivalent corner is inside or outside
    /// the SDF
    fn calculate_cube_index(cube: [f32; 8], surface_threshold: f32) -> u8 {
        cube.into_iter()
            .enumerate()
            .map(|(i, val)| (1 << i as u8) * (val < surface_threshold) as u8)
            .sum()
    }

    pub fn march_cubes(&self, surface_threshold: f32) {
        let z_samples = self.samples.len();
        let y_samples = self.samples[0].len();
        let x_samples = self.samples[0][0].len();

        let domain_diff = self.domain[1] - self.domain[0];
        let normalize_vector = domain_diff.component_div(&Vector3::new(
            (x_samples - 1) as f32,
            (y_samples - 1) as f32,
            (z_samples - 1) as f32,
        ));
        let cube_vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let position_offsets =
            cube_vertices.map(|offset| Vector3::from(offset).component_mul(&normalize_vector));

        let vertices_of_edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];

        let samples = &self.samples;
        let domain = self.domain;

        let tris = std::thread::scope(|scope| {
            (0..z_samples - 1)
                .map(|z| {
                    scope.spawn(move || {
                        (0..y_samples - 1)
                            .flat_map(|y| {
                                (0..x_samples - 1)
                                    .flat_map(|x| {
                                        let cube_bottom_left =
                                            Vector3::new(x as f32, y as f32, z as f32)
                                                .component_mul(&normalize_vector)
                                                + domain[0];

                                        let samples = cube_vertices.map(|[x_o, y_o, z_o]| {
                                            samples[z + z_o as usize][y + y_o as usize]
                                                [x + x_o as usize]
                                        });

                                        let cube_index =
                                            Self::calculate_cube_index(samples, surface_threshold);

                                        let points: [Vector3<f32>; 12] = (0..12)
                                            .map(|edge| {
                                                let edge_pos = vertices_of_edges[edge].map(|v| {
                                                    cube_bottom_left + position_offsets[v]
                                                });
                                                let edge_sdf =
                                                    vertices_of_edges[edge].map(|v| samples[v]);

                                                edge_pos[0]
                                                    - edge_sdf[0] * (edge_pos[1] - edge_pos[0])
                                                        / (edge_sdf[1] - edge_sdf[0])
                                            })
                                            .collect::<Vec<_>>()
                                            .try_into()
                                            .unwrap();

                                        let triangles: Vec<[Vector3<f32>; 3]> = TRIANGLE_TABLE
                                            [cube_index as usize]
                                            .chunks_exact(3)
                                            .flat_map(|chunk| {
                                                if chunk[0] != -1 {
                                                    let tri: [Vector3<f32>; 3] = chunk
                                                        .iter()
                                                        .map(|&idx| points[idx as usize])
                                                        .collect::<Vec<_>>()
                                                        .try_into()
                                                        .unwrap();
                                                    Some(tri)
                                                } else {
                                                    None
                                                }
                                            })
                                            .collect();

                                        triangles
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        let normals = tris
            .iter()
            .map(|tri| (tri[1] - tri[0]).cross(&(tri[2] - tri[0])).normalize())
            .collect::<Vec<_>>();

        let mut obj = std::fs::File::create("./vessels.obj").unwrap();

        let verts_str: String = tris
            .iter()
            .flatten()
            .map(|vert| format!("v {} {} {}\n", vert.x, vert.y, vert.z))
            .collect();
        obj.write_all(&verts_str.into_bytes()).unwrap();
        let normals_str: String = normals
            .iter()
            .flat_map(|normal| {
                (0..3).map(|_| format!("vn {} {} {}\n", normal.x, normal.y, normal.z))
            })
            .collect();
        obj.write_all(&normals_str.into_bytes()).unwrap();

        let faces_str: String = (0..tris.len())
            .map(|face_idx| {
                format!(
                    "f {} {} {}\n",
                    3 * face_idx + 1,
                    3 * face_idx + 2,
                    3 * face_idx + 3
                )
            })
            .collect();
        obj.write_all(&faces_str.into_bytes()).unwrap();
    }
}

/// Each value is an array describing how triangles should be generated for any cube. There are
/// maximum 5 triangles that can be generated. Each value in the subarray is the index of an edge.
/// Linear interpolation is used to determine the point along the edge in order to construct the
/// face.
const TRIANGLE_TABLE: [[i32; 15]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];
