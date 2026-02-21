struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4f(in.pos, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}
