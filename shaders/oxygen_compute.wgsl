@group(0) @binding(0) var<uniform> vessel_edges: array<VesselEdge, 128>;

@group(0) @binding(1) var oxygen_concentration: texture_storage_2d<rgba32float, read_write>;

struct VesselEdge {
    p1: vec2<f32>,
    p2: vec2<f32>,
}

fn is_nan(val: f32) -> bool {
    let min: f32 = -20000000.0;
    return max(val, min) == min;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var res = vec2f(0.0);
    for (var i = 0; i < 128; i++) {
        let vessel = vessel_edges[i];
        if (vessel.p1.x == vessel.p2.x && vessel.p1.y == vessel.p2.y) {
            break;
        }

        let pos = vec2<f32>(id.xy);
        let vessel_dir = vessel.p2 - vessel.p1;
        let offset_pos = pos - vessel.p1;
        let low_bound = vec2f(min(vessel.p1.x, vessel.p2.x), min(vessel.p1.y, vessel.p2.y));
        let high_bound = vec2f(max(vessel.p1.x, vessel.p2.x), max(vessel.p1.y, vessel.p2.y));

        let proj = clamp(dot(offset_pos, vessel_dir) / dot(vessel_dir, vessel_dir) * vessel_dir + vessel.p1, low_bound, high_bound);
        let cutoff = 0.08;
        let dist = max(cutoff - distance(pos, proj) / 512.0, 0.0) / cutoff * normalize(pos - proj);
        if (any((pos - proj) != vec2f(0.0))) {
            let highval = 1000000.0;
            res += dist;
            if min(vessel.p1.x, highval) == highval {
                res = vec2f(50.0, 50.0);// vec2f(distance(vec2f(id.xy), vessel_edges[0].p1)) / 512.0;
                break;
            }
        }
        /* let prev = textureLoad(oxygen_concentration, id.xy);
        textureStore(oxygen_concentration, id.xy, prev + vec4f(dist, 0.0, 0.0)); */
    }
    textureStore(oxygen_concentration, id.xy, vec4f(res, 0.0, 0.0));
}
