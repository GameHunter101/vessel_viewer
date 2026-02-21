@group(0) @binding(0) var oxygen_tex: texture_2d<f32>;
@group(0) @binding(1) var oxygen_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(oxygen_tex, oxygen_sampler, in.tex_coords);
}
