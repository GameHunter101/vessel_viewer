use image::Rgba32FImage;
use nalgebra::{Vector3, Vector4};
use v4::{
    component,
    ecs::component::{ComponentSystem, UpdateParams},
    engine_support::texture_support::TextureBundle,
};
use wgpu::{Buffer, Device, Queue};

#[component]
pub struct NetworkGenerationComponent {
    concentration_texture_bundle: TextureBundle,
    boundary: Vec<Vector3<f32>>,
    buf: Buffer,
    max_iter_count: usize,
    #[default(0)]
    current_iter: usize,
    #[default(1.0)]
    branch_dilation_factor: f32,
}

impl NetworkGenerationComponent {
    async fn fetch_texture(&self, device: &Device, queue: &Queue) -> Rgba32FImage {
        let tex = self.concentration_texture_bundle.view().texture();
        let size = self
            .concentration_texture_bundle
            .properties()
            .format
            .theoretical_memory_footprint(tex.size());

        let mut encoder =
            device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size as u32 / tex.height()),
                    rows_per_image: Some(tex.height()),
                },
            },
            tex.size(),
        );
        queue.submit([encoder.finish()]);
        let slice = self.buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range();
        let bytes: &[f32] = bytemuck::cast_slice(&data);
        let img = Rgba32FImage::from_raw(tex.width(), tex.height(), bytes.to_vec())
            .expect("Failed to create CPU-side rgba32Float image");
        drop(data);
        img
    }

    fn advect_boundary(&self, device: &Device, queue: &Queue) -> Vec<Vector3<f32>> {
        let img = pollster::block_on(self.fetch_texture(device, queue));
        let center = self.boundary.iter().sum::<Vector3<f32>>() / self.boundary.len() as f32;
        let mut new_boundary: Vec<Vector3<f32>> = self
            .boundary
            .iter()
            .map(|pos| (pos - center) * 0.99 + center)
            .collect();

        for point in &mut new_boundary {
            let mut boundary_vector = Vector3::from(
                Vector4::from(
                    img.get_pixel((point.x as u32).min(511), (point.y as u32).min(511))
                        .0,
                )
                .xyz(),
            );
            while boundary_vector.norm() > 0.0001 {
                *point += boundary_vector;
                boundary_vector = Vector3::from(
                    Vector4::from(
                        img.get_pixel((point.x as u32).min(511), (point.y as u32).min(511))
                            .0,
                    )
                    .xyz(),
                );
            }
        }
        self.buf.unmap();
        new_boundary
    }

    fn get_best_connection_points(&self, low_oxygen_point: Vector3<f32>) -> [Vector3<f32>; 2] {
        let all_edges: Vec<[Vector3<f32>; 2]> = self
            .boundary
            .chunks(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let mut projection_of_central_point_on_edges: Vec<Vector3<f32>> = all_edges
            .into_iter()
            .map(|[p0, p1]| {
                let dir = p1 - p0;
                let unclamped_projection =
                    dir.dot(&(low_oxygen_point - p0)) / dir.norm_squared() * dir + p0;
                Vector3::new(
                    unclamped_projection.x.clamp(p0.x.min(p1.x), p0.x.max(p1.x)),
                    unclamped_projection.y.clamp(p0.y.min(p1.y), p0.y.max(p1.y)),
                    unclamped_projection.z.clamp(p0.z.min(p1.z), p0.z.max(p1.z)),
                )
            })
            .collect();

        projection_of_central_point_on_edges.sort_by(|a, b| {
            (a - low_oxygen_point)
                .norm_squared()
                .total_cmp(&(b - low_oxygen_point).norm_squared())
        });

        [projection_of_central_point_on_edges[0], projection_of_central_point_on_edges[1]]
    }
}

impl ComponentSystem for NetworkGenerationComponent {
    fn update(
        &mut self,
        UpdateParams { device, queue, .. }: UpdateParams<'_, '_>,
    ) -> v4::ecs::actions::ActionQueue {
        if self.current_iter >= self.max_iter_count {
            return Vec::new();
        }

        let advected_boundary = self.advect_boundary(device, queue);
        let central_lowest_oxygen_point =
            advected_boundary.iter().sum::<Vector3<f32>>() / advected_boundary.len() as f32;

        let connection_points = self.get_best_connection_points(central_lowest_oxygen_point);

        let edges: Vec<[Vector3<f32>; 2]> = connection_points.iter().flat_map(|connection| {
            let branch_point = connection.lerp(&central_lowest_oxygen_point, 0.5);
            let main_edge = [*connection, branch_point];
            let first_path_point = (central_lowest_oxygen_point - main_edge[1]).cross(&Vector3::new(0.0, 0.0, self.branch_dilation_factor)) + central_lowest_oxygen_point;

            let second_path_point = (central_lowest_oxygen_point - main_edge[1]).cross(&Vector3::new(0.0, 0.0, -self.branch_dilation_factor)) + central_lowest_oxygen_point;

            [main_edge].into_iter().chain([[branch_point, first_path_point], [branch_point, second_path_point]])
        }).collect();

        for edge in edges {
            println!("polygon(({}, {}), ({}, {}))", edge[0].x, edge[0].y, edge[1].x, edge[1].y);
        }

        self.current_iter += 1;

        Vec::new()
    }
}
