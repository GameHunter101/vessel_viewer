@group(0) @binding(0) var<uniform> vessel_edges: array<VesselEdge, 2>;

@group(0) @binding(1) var oxygen_concentration: texture_storage_2d<rgba32float, read_write>;

struct VesselEdge {
    p1: vec2<f32>,
    p2: vec2<f32>,
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var res = vec2f(0.0);
    for (var i = 0; i < 2; i++) {
        let pos = vec2<f32>(id.xy);
        let vessel = vessel_edges[i];
        let vessel_dir = vessel.p2 - vessel.p1;
        let offset_pos = pos - vessel.p1;
        let proj = dot(offset_pos, vessel_dir) / dot(vessel_dir, vessel_dir) * vessel_dir + vessel.p1;
        let cutoff = 0.08;
        let dist = max(cutoff - distance(pos, proj) / 512.0, 0.0) / cutoff * normalize(pos - proj);
        res += dist;
    }
    textureStore(oxygen_concentration, id.xy, vec4f(res, 0.0, 1.0));
    // textureStore(oxygen_concentration, id.xy, vec4f(1.0));
}
