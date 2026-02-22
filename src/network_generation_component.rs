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
}

impl ComponentSystem for NetworkGenerationComponent {
    fn update(
        &mut self,
        UpdateParams { device, queue, .. }: UpdateParams<'_, '_>,
    ) -> v4::ecs::actions::ActionQueue {
        let advected_boundary = self.advect_boundary(device, queue);
        println!(
            "New boundary: {:?}",
            advected_boundary
                .iter()
                .map(|pos| (pos.x, pos.y))
                .collect::<Vec<_>>()
        );

        Vec::new()
    }
}
