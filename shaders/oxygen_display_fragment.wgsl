@group(0) @binding(0) var oxygen_tex: texture_2d<f32>;
@group(0) @binding(1) var oxygen_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let val = textureSample(oxygen_tex, oxygen_sampler, in.tex_coords).xyz;
    let pos = max(val, vec3f(0.0));
    let neg = cross(-min(val.xyz, vec3f(0.0)), vec3f(1.0));
    return vec4f(pos + neg, 1.0);
}
