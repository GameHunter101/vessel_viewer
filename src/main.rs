const BUFFER_SIZE: usize = 256;
const AREA_SIZE: usize = 512;

use image::EncodableLayout;
use nalgebra::Vector3;
use rand::{SeedableRng, rngs::StdRng};
use v4::{
    V4,
    builtin_components::mesh_component::{MeshComponent, VertexData, VertexDescriptor},
    ecs::{
        compute::Compute,
        material::{ShaderAttachment, ShaderBufferAttachment, ShaderTextureAttachment},
    },
    engine_support::texture_support::{TextureBundle, TextureProperties},
    scene,
};
use wgpu::vertex_attr_array;
use winit::window::WindowAttributes;

use crate::network_generation_component::{NetworkDetails, NetworkGenerationComponent};

mod network_generation_component;

#[tokio::main]
async fn main() {
    let mut engine = V4::builder()
        .features(
            wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        )
        .window_attributes(WindowAttributes::default().with_surface_size(
            winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(600, 600)),
        ))
        .build()
        .await;

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

    let (vessels, boundary) = initialize_points();

    let oxygen_compute_edges: Vec<ComputeEdge> = vessels
        .chunks(2)
        .map(|points| {
            ComputeEdge::new(
                [
                    (points[0].pos[0] + 1.0) * (AREA_SIZE / 2) as f32,
                    (points[0].pos[1] + 1.0) * (AREA_SIZE / 2) as f32,
                ],
                [
                    (points[1].pos[0] + 1.0) * (AREA_SIZE / 2) as f32,
                    (points[1].pos[1] + 1.0) * (AREA_SIZE / 2) as f32,
                ],
            )
        })
        .chain([ComputeEdge::default(); BUFFER_SIZE - 2])
        .collect();

    let img = image::Rgba32FImage::from_pixel(512, 512, image::Rgba([0.0, 0.0, 0.0, 1.0]));
    let bytes = img.as_bytes();

    let (oxygen_concentration_texture, oxygen_concentration_texture_bundle) =
        TextureBundle::from_bytes(
            bytes,
            (512, 512),
            device,
            queue,
            TextureProperties {
                format: wgpu::TextureFormat::Rgba32Float,
                storage_texture: Some(wgpu::StorageTextureAccess::ReadWrite),
                is_sampled: false,
                extra_usages: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
                ..Default::default()
            },
        );

    let oxygen_concentration_display_texture_bundle = TextureBundle::new(
        oxygen_concentration_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        TextureProperties {
            format: wgpu::TextureFormat::Rgba32Float,
            is_filtered: false,
            ..Default::default()
        },
    );

    scene! {
        scene: vessel_viewer,
        "oxygen_concentration" = {
            material: {
                pipeline: {
                    vertex_shader_path: "shaders/oxygen_display_vertex.wgsl",
                    fragment_shader_path: "shaders/oxygen_display_fragment.wgsl",
                    uses_camera: false,
                    vertex_layouts: [DisplayVertex::vertex_layout()],
                },
                attachments: [
                    Texture(
                        texture_bundle: oxygen_concentration_display_texture_bundle,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                    )
                ]
            },
            components: [
                MeshComponent(
                    vertices: vec![vec![
                        DisplayVertex {
                            pos: [-1.0, 3.0, 0.1],
                            tex_coords: [0.0, 2.0]
                        },
                        DisplayVertex {
                            pos: [-1.0, -1.0, 0.1],
                            tex_coords: [0.0, 0.0]
                        },
                        DisplayVertex {
                            pos: [3.0, -1.0, 0.1],
                            tex_coords: [2.0, 0.0]
                        },
                    ]],
                    enabled_models: vec![(0, None)]
                ),
                NetworkGenerationComponent(
                    rng: StdRng::seed_from_u64(0),
                    boundary_verts: boundary,
                    edges: vec![[0, 1], [2, 3]],
                    max_iter_count: 20,
                    network_parameters: NetworkDetails {
                        edge_lerp_distance_to_length_factor: 0.0,
                        edge_lerp_concentration_to_edge_perpendicular: 0.0,
                    },
                    vessel_edges_component: ident("vessel_edges"),
                    display_vessel_edges_compute: ident("vessel_compute"),
                )
            ],
            computes: [Compute(
                    attachments: vec![
                        ShaderAttachment::Buffer(ShaderBufferAttachment::new(
                            device,
                            bytemuck::cast_slice(&oxygen_compute_edges),
                            wgpu::BufferBindingType::Uniform,
                            wgpu::ShaderStages::COMPUTE,
                            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        )),
                        ShaderAttachment::Texture(ShaderTextureAttachment {
                            texture_bundle: oxygen_concentration_texture_bundle.clone(),
                            visibility: wgpu::ShaderStages::COMPUTE,
                        }),
                    ],
                shader_path: "shaders/oxygen_compute.wgsl",
                workgroup_counts: (512, 512, 1),
                ident: "vessel_compute"
            )],
        },
        "vessels" = {
            material: {
                pipeline: {
                    vertex_shader_path: "shaders/vessel_vertex.wgsl",
                    fragment_shader_path: "shaders/vessel_fragment.wgsl",
                    uses_camera: false,
                    vertex_layouts: [Vertex::vertex_layout()],
                    geometry_details: {
                        topology: wgpu::PrimitiveTopology::LineList,
                        polygon_mode: wgpu::PolygonMode::Line,
                    }
                }
            },
            components: [
                MeshComponent(
                    vertices: vec![vessels],
                    enabled_models: vec![(0, None)],
                    ident: "vessel_edges",
                )
            ],
        }
    }

    engine.attach_scene(vessel_viewer);

    engine.main_loop().await;
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pos: [f32; 3],
    color: [f32; 4],
}

impl VertexDescriptor for Vertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] =
        &vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn from_data(VertexData { pos, .. }: VertexData) -> Self {
        Self {
            pos,
            color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ComputeEdge {
    p0: [f32; 2],
    p1: [f32; 2],
    // enabled: u32,
    // padding: [u32; 2],
}

impl ComputeEdge {
    pub fn new(p0: [f32; 2], p1: [f32; 2]) -> Self {
        Self {
            p0,
            p1,
            // enabled: 1,
            // padding: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DisplayVertex {
    pos: [f32; 3],
    tex_coords: [f32; 2],
}

impl VertexDescriptor for DisplayVertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] =
        &vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn from_data(
        VertexData {
            pos, tex_coords, ..
        }: VertexData,
    ) -> Self {
        Self { pos, tex_coords }
    }
}

pub fn initialize_points() -> (Vec<Vertex>, Vec<Vector3<f32>>) {
    let vessels = vec![
        Vertex {
            pos: [-1.0, -0.85, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [1.0, -0.85, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [1.0, 0.85, 0.0],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        Vertex {
            pos: [-1.0, 0.85, 0.0],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    let boundary: Vec<Vector3<f32>> = vessels
        .iter()
        .map(|vert| {
            (Vector3::from(vert.pos) + Vector3::new(1.0, 1.0, 0.0)) * (AREA_SIZE as f32 / 2.0)
        })
        .collect();

    (vessels, boundary)
}
