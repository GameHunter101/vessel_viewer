struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4f(in.pos, 1.0);
    out.color = in.color;

    return out;
}
