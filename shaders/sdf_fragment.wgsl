const SIZE = 2048;

@group(0) @binding(0) var<uniform> vessel_edges: array<VesselEdge, SIZE>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct VesselEdge {
    p1: vec2<f32>,
    p2: vec2<f32>,
}

fn sd_capped_cylinder(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let ba = b - a;
  let pa = p - a;
  let baba = dot(ba,ba);
  let paba = dot(pa,ba);
  let x = length(pa*baba-ba*paba) - r*baba;
  let y = abs(paba-baba*0.5)-baba*0.5;
  let x2 = x*x;
  let y2 = y*y*baba;
  let d = select(
  select(0.0, x2, x > 0.0) + select(0.0, y2, y > 0.0),
  -min(x2,y2),
  max(x,y)<0.0
  );
  return sign(d)*sqrt(abs(d))/baba;
}

fn distance_to_edge(p: vec3f, a: vec3f, b: vec3f) -> f32 {
    let line = b - a;
    let proj = clamp(dot(line, (p - a)) / dot(line, line) * line + a, min(a, b), max(a,b));

    return distance(p, proj);
}

fn smooth_union(a: f32, b: f32, smoothing: f32) -> f32 {
    let k = smoothing * 4.0;
    let h = max(k - abs(a - b), 0.0);
    return min(a, b) - h * h * 0.25/k;
}

fn map(p: vec3f, vessel_thickness: f32, smooth_factor: f32) -> f32 {
    var val = 0x1.fffffep+127;
    var numerator = 0.0;
    var denominator = 0.0;
    for (var i = 0; i < SIZE; i++) {
        let edge = vessel_edges[i];
        if all(edge.p1 == edge.p2) {
            break;
        }
        // let cylinder = sd_capped_cylinder(p, vec3f(edge.p1, 0.0), vec3f(edge.p2, 0.0), vessel_thickness) - vessel_thickness * smooth_factor;
        let cylinder = distance_to_edge(p, vec3f(edge.p1, 0.0), vec3f(edge.p2, 0.0)) - vessel_thickness;
        val = min(val, cylinder);
        // val = smooth_union(val, cylinder, vessel_thickness * smooth_factor);
    }

    return val;
}

fn getNormal(p: vec3f, vessel_thickness: f32, smooth_factor: f32) -> vec3f {
    let eps = 0.0001;
    let h = vec2f(eps, 0.0);
    return normalize(vec3f(
        map(p + h.xyy, vessel_thickness, smooth_factor) - map(p - h.xyy, vessel_thickness, smooth_factor),
        map(p + h.yxy, vessel_thickness, smooth_factor) - map(p - h.yxy, vessel_thickness, smooth_factor),
        map(p + h.yyx, vessel_thickness, smooth_factor) - map(p - h.yyx, vessel_thickness, smooth_factor),
    ));
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = 2.0 * (in.tex_coords - vec2f(0.5));
    let camera_pos = vec3f(0.0, 0.0, -1.0);

    let ray_origin = camera_pos;
    let ray_destination = vec3f(uv, 1.0);

    let ray_dir = normalize(ray_destination - ray_origin);

    var t = 0.0;

    let vessel_thickness = 0.01;
    let smooth_factor = vessel_thickness / 5.0;

    for (var i = 0; i < 100; i++) {
        if t > 100.0 {
            break;
        }

        let point = ray_origin + t * ray_dir;
        let dist = map(point, vessel_thickness, smooth_factor);
        if dist < 0.001 {
            return vec4f(vec3f(dot(getNormal(point, vessel_thickness, smooth_factor), vec3f(0.0, 1.0, 0.0))), 1.0);
        }
        t += dist;
    }

    return vec4f(0.0);
}
