@group(0) @binding(0) var oxygen_tex: texture_2d<f32>;
@group(0) @binding(1) var oxygen_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}


fn random(coord: vec2<u32>) -> f32 {
    let input = (coord.x + coord.y * 512u);
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / f32(0xffffffff);
}

fn bilerp(coord: vec2<f32>) -> f32 {
    let coord_floor = vec2u(coord);
    let bl = random(coord_floor);
    let br = random(coord_floor + vec2u(1u, 0u));
    let tl = random(coord_floor + vec2u(0u, 1u));
    let tr = random(coord_floor + vec2u(1u, 1u));

    return mix(mix(bl, br, fract(coord.x)), mix(tl, tr, fract(coord.x)), fract(coord.y));
}

fn lic(origin_pos: vec2<u32>, len: u32, tex: texture_2d<f32>) -> f32 {
    var sum = random(origin_pos);
    var pos_1 = vec2f(origin_pos) + vec2f(0.5, 0.5);
    var pos_2 = pos_1;

    let delta_s = 1.0;

    var i = 1u;

    for (; i < len; i++) {
        if (pos_1.x < 0.0 || pos_1.x > 512.0) || (pos_2.x < 0.0 || pos_2.x > 512.0) || 
            (pos_1.y < 0.0 || pos_1.y > 512.0) || (pos_2.y < 0.0 || pos_2.y > 512.0) {
            break;
        }

        let noise_fac = 0.1;
        let noise_1 = mix(-1.0, 1.0, random(vec2u(pos_1)));
        let noise_2 = mix(-1.0, 1.0, random(vec2u(pos_2)));

        sum += (bilerp(pos_1) + bilerp(pos_2)) + noise_fac * (noise_1 + noise_2);

        var pos_1_dir = textureLoad(tex, vec2u(pos_1), 0);
        var pos_2_dir = textureLoad(tex, vec2u(pos_2), 0);
        pos_1 += normalize(pos_1_dir.xy) * delta_s;
        pos_2 -= normalize(pos_2_dir.xy) * delta_s;
    }

    return sum / (2.0 * f32(i) + 1.0);
}

fn luminance(val: vec3<f32>) -> f32 {
    return dot(val, vec3f(0.299, 0.587, 0.114));
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let val = textureSample(oxygen_tex, oxygen_sampler, in.tex_coords).xyz;
    let pos = max(val, vec3f(0.0));
    let neg = cross(-min(val.xyz, vec3f(0.0)), vec3f(1.0));
    let brightness = luminance(abs(textureSample(oxygen_tex, oxygen_sampler, in.tex_coords).xyz));
    let oxygen_dir = lic(vec2u(in.tex_coords * 512.0), 5, oxygen_tex) * brightness;
    // return vec4f((pos + neg), 1.0);
    return vec4f(vec3f(oxygen_dir), 1.0);
}
