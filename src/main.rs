use std::collections::{HashMap, HashSet};

use image::EncodableLayout;
use nalgebra::Vector3;
use v4::{
    V4,
    builtin_components::mesh_component::{MeshComponent, VertexData, VertexDescriptor},
    ecs::{
        component::ComponentSystem,
        compute::Compute,
        material::{ShaderAttachment, ShaderBufferAttachment, ShaderTextureAttachment},
    },
    engine_support::texture_support::{TextureBundle, TextureProperties},
    scene,
};
use wgpu::vertex_attr_array;

use crate::network_generation_component::NetworkGenerationComponent;

mod network_generation_component;

#[tokio::main]
async fn main() {
    let mut engine = V4::builder()
        .features(
            wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        )
        .build()
        .await;

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

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

    let oxygen_compute_edges: Vec<ComputeEdge> = vessels
        .chunks(2)
        .map(|points| ComputeEdge {
            p1: [
                (points[0].pos[0] + 1.0) * 256.0,
                (points[0].pos[1] + 1.0) * 256.0,
            ],
            p2: [
                (points[1].pos[0] + 1.0) * 256.0,
                (points[1].pos[1] + 1.0) * 256.0,
            ],
        })
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

    let mut compute = Compute::builder()
        .attachments(vec![
            ShaderAttachment::Buffer(ShaderBufferAttachment::new(
                device,
                bytemuck::cast_slice(&oxygen_compute_edges),
                wgpu::BufferBindingType::Uniform,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferUsages::empty(),
            )),
            ShaderAttachment::Texture(ShaderTextureAttachment {
                texture_bundle: oxygen_concentration_texture_bundle.clone(),
                visibility: wgpu::ShaderStages::COMPUTE,
            }),
        ])
        .shader_path("shaders/oxygen_compute.wgsl")
        .workgroup_counts((512, 512, 1))
        .build();

    compute.initialize(device);

    rendering_manager.individual_compute_execution(&[compute]);

    let boundary: Vec<Vector3<f32>> = vessels
        .iter()
        .map(|vert| (Vector3::from(vert.pos) + Vector3::new(1.0, 1.0, 0.0)) * 256.0)
        .collect();

    let boundary_adjacency_list: HashMap<usize, HashSet<usize>> = (0..boundary.len())
        .map(|i| {
            (
                i,
                HashSet::from_iter([
                    (i as i32 - 1).rem_euclid(boundary.len() as i32) as usize,
                    (i + 1) % boundary.len(),
                ]),
            )
        })
        .collect();

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
                    boundary_verts: boundary,
                    boundary_adjacency_list: boundary_adjacency_list,
                    max_iter_count: 3,
                    non_edges: HashSet::from([[0, 3], [1, 2]]),
                    vessel_edges_component: ident("vessel_edges"),
                )
            ]
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
struct ComputeEdge {
    p1: [f32; 2],
    p2: [f32; 2],
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
