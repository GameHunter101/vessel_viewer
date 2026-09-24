const SIZE = 2048u;

@group(0) @binding(0) var<uniform> parameters: MarchingCubesParameters;
@group(0) @binding(1) var<storage, read> edges_map: array<array<u32, 16>>;
@group(0) @binding(2) var<uniform> vessels: array<VesselEdge, SIZE>;
@group(0) @binding(3) var<storage, read_write> samples: array<f32>;

struct MarchingCubesParameters {
    padding: f32,
    thickness: f32,
    area_size: f32,
    domain_bl: vec3<f32>,
    domain_tr: vec3<f32>,
    cell_size: f32,
    cell_counts: vec3<u32>,
}

struct VesselEdge {
    p0: vec2<f32>,
    p1: vec2<f32>,
}

fn sdf_point(pos: vec3<f32>) -> f32 {
    let grid_pos = vec3u(pos / parameters.cell_size);

    return 0.0;
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) sample_counts: vec3<u32>) {
    let domain_diff = parameters.domain_tr - parameters.domain_bl;
    let point = (vec3f(id) / (vec3f(sample_counts) - 1.0)) * domain_diff + parameters.domain_bl;

    samples[id.z * (sample_counts.x * sample_counts.y) + id.y * sample_counts.x + id.x] = f32(id.x);
}
