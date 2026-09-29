const SIZE = 2048u;
const EDGE_MAP_SIZE = 16u;

@group(0) @binding(0) var<uniform> parameters: MarchingCubesParameters;
@group(0) @binding(1) var<storage, read> edges_map: array<array<u32, EDGE_MAP_SIZE>>;
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

fn cell_to_index(cell: vec3<u32>) -> u32 {
    return parameters.cell_counts.x * parameters.cell_counts.y * cell.z + parameters.cell_counts.x * cell.y + cell.x;
}

fn vector_project(line: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    return dot(line, point) / dot(line, line) * line;
}

fn sdf_point(pos: vec3<f32>) -> f32 {
    let grid_pos = vec3u(pos / parameters.cell_size);

    var sample: f32 = 0x7f800000; // Hex representation of infinity
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let cell_to_check = max(vec3i(grid_pos) + vec3i(x, y, z), vec3i(0));
                let index = min(max(0, cell_to_index(vec3u(cell_to_check))), arrayLength(&edges_map));
                let nearby_edges = edges_map[index];
                for (var i = 0u; i < EDGE_MAP_SIZE; i++) {
                    if (nearby_edges[i] == 0xffffffff) {
                        break;
                    }
                    let edge = vessels[nearby_edges[i]];
                    let p0 = vec3f(edge.p0, 0.0);
                    let p1 = vec3f(edge.p1, 0.0);
                    let projection = vector_project(p1 - p0, pos - p0) + p0;

                    let min_point = min(p0, p1);
                    let max_point = max(p0, p1);
                    let clamped_projection = max(min(projection, max_point), min_point);

                    let dist = clamped_projection - pos;

                    sample = min(sample, dot(dist, dist) - parameters.thickness * parameters.thickness);
                }
            }
        }
    }

    return sample;
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) sample_counts: vec3<u32>) {
    let domain_diff = parameters.domain_tr - parameters.domain_bl;
    let point = (vec3f(id) / (vec3f(sample_counts) - 1.0)) * domain_diff + parameters.domain_bl;

    samples[id.z * (sample_counts.x * sample_counts.y) + id.y * sample_counts.x + id.x] = sdf_point(point);
}
