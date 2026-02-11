#!/usr/bin/env python3
"""
Arc-based scratch hologram generator.

Goal: replicate HoloZens-style arc output from STL edges.
This script generates only arcs (no lines/profiles).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


EPS = 1e-12
NORMAL_TOLERANCE_DECIMALS = 4


@dataclass
class CameraConfig:
    po: tuple[float, float, float]
    pr: tuple[float, float, float]
    look_up: tuple[float, float, float]
    current_scale: float
    zf: float
    canvas_width: int
    canvas_height: int


@dataclass
class PipelineConfig:
    line_resolution: float
    auto_center: bool
    stroke_width: float
    dedupe_decimals: int
    min_arc_radius: float


@dataclass
class Edge:
    edge_id: int
    start_idx: int
    end_idx: int


@dataclass
class Arc:
    edge_id: int
    zero_coord: tuple[float, float, float]
    rect_x: int
    rect_y: int
    rect_w: int
    rect_h: int
    start_angle: float
    sweep_angle: float


def ensure_dependencies() -> None:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if trimesh is None:
        missing.append("trimesh")
    if missing:
        raise RuntimeError(
            f"Missing Python packages: {', '.join(missing)}. "
            "Install with: pip install -r requirements.txt"
        )


def normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n < EPS:
            raise ValueError("Cannot normalize a zero-length vector.")
        return arr / n

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, EPS, None)


def load_mesh(stl_path: Path) -> trimesh.Trimesh:
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    loaded = trimesh.load(stl_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in {stl_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input file does not contain a valid triangular mesh.")

    mesh = mesh.copy()

    # Keep compatible with older/newer trimesh APIs.
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())

    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_infinite_values"):
        mesh.remove_infinite_values()

    if mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has zero faces after cleanup.")

    return mesh


def build_model_vertices_and_edges(
    mesh: trimesh.Trimesh,
    auto_center: bool,
    decimals: int,
) -> tuple[np.ndarray, list[Edge]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Match HoloZens behavior: parser rounds coordinates to a global tolerance.
    vertices = np.round(vertices, decimals)

    if auto_center:
        max_z = float(np.max(vertices[:, 2]))
        vertices[:, 2] = np.round(vertices[:, 2] - max_z, decimals)

    # Build a stable set of unique vertices using rounded coordinates.
    coord_to_new_idx: dict[tuple[float, float, float], int] = {}
    unique_vertices: list[tuple[float, float, float]] = []
    remap = np.zeros(vertices.shape[0], dtype=np.int64)

    for old_idx, coord in enumerate(vertices):
        key = (float(coord[0]), float(coord[1]), float(coord[2]))
        existing = coord_to_new_idx.get(key)
        if existing is None:
            existing = len(unique_vertices)
            coord_to_new_idx[key] = existing
            unique_vertices.append(key)
        remap[old_idx] = existing

    # Build unique undirected edges, preserving creation order for stable IDs.
    # Keep per-edge adjacent face normals so we can skip coplanar triangulation edges
    # (equivalent to HoloZens "Merge Faces" behavior).
    edge_map: dict[tuple[int, int], Edge] = {}
    edge_adjacent_normals: dict[tuple[int, int], list[np.ndarray]] = {}
    next_edge_id = 0

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    for face_idx, tri in enumerate(faces):
        tri_new = [int(remap[int(tri[0])]), int(remap[int(tri[1])]), int(remap[int(tri[2])])]

        # Skip invalid/degenerate triangles after rounding/remap.
        if len(set(tri_new)) < 3:
            continue

        face_normal = normalize(face_normals[face_idx])

        tri_pairs = (
            (tri_new[0], tri_new[1]),
            (tri_new[1], tri_new[2]),
            (tri_new[2], tri_new[0]),
        )
        for a, b in tri_pairs:
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in edge_map:
                edge_adjacent_normals[key].append(face_normal)
                continue
            edge_map[key] = Edge(edge_id=next_edge_id, start_idx=a, end_idx=b)
            edge_adjacent_normals[key] = [face_normal]
            next_edge_id += 1

    normal_tol = 10.0 ** (-decimals)
    filtered_edges: list[Edge] = []
    for key, edge in edge_map.items():
        normals = edge_adjacent_normals.get(key, [])
        if len(normals) >= 2:
            n0 = normals[0]
            n1 = normals[1]
            if np.all(np.abs(n0 - n1) <= normal_tol):
                # Coplanar shared edge: skip to mimic merged faces.
                continue
        filtered_edges.append(edge)

    return np.asarray(unique_vertices, dtype=np.float64), filtered_edges


def build_model_to_window_matrix(camera: CameraConfig) -> np.ndarray:
    po = np.asarray(camera.po, dtype=np.float64)
    pr = np.asarray(camera.pr, dtype=np.float64)
    look_up = normalize(np.asarray(camera.look_up, dtype=np.float64))

    n = normalize(po - pr)
    u = np.cross(look_up, n)
    u_norm = float(np.linalg.norm(u))
    if u_norm < EPS:
        raise ValueError("Look-up vector is collinear with camera direction.")
    u = u / u_norm
    v = np.cross(n, u)

    model_to_view = np.eye(4, dtype=np.float64)
    model_to_view[0, 0:3] = u
    model_to_view[1, 0:3] = v
    model_to_view[2, 0:3] = n
    model_to_view[0, 3] = -float(np.dot(u, po))
    model_to_view[1, 3] = -float(np.dot(v, po))
    model_to_view[2, 3] = -float(np.dot(n, po))

    perspective = np.eye(4, dtype=np.float64)
    zf = camera.zf if abs(camera.zf) > EPS else 1e-5
    perspective[3, 2] = -1.0 / zf

    base_scale = float(camera.canvas_width) / 17.0
    scale_val = base_scale * camera.current_scale
    scale = np.eye(4, dtype=np.float64)
    scale[0, 0] = scale_val
    scale[1, 1] = scale_val
    scale[2, 2] = scale_val

    view_to_window = np.eye(4, dtype=np.float64)
    view_to_window[1, 1] = -1.0
    view_to_window[0, 3] = camera.canvas_width / 2.0
    view_to_window[1, 3] = camera.canvas_height / 2.0

    return view_to_window @ scale @ perspective @ model_to_view


def model_to_window(points_xyz: np.ndarray, model_to_window_mtx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack((points, ones))

    transformed = (model_to_window_mtx @ homo.T).T
    w = transformed[:, 3:4]
    w_safe = np.where(np.abs(w) < EPS, EPS, w)
    transformed[:, 0:3] = transformed[:, 0:3] / w_safe
    return transformed[:, 0:3]


def get_arc_square(point_zero: np.ndarray, n_view: float) -> tuple[int, int, int, int]:
    distance_from_canvas = float(point_zero[2] - n_view)
    center_x = float(point_zero[0])
    center_y = float(point_zero[1] - distance_from_canvas / 2.0)
    halfwidth = abs(center_y - float(point_zero[1]))

    length = max(int(halfwidth * 2.0 + 0.5), 1)
    rect_x = int(center_x - halfwidth + 0.5)
    rect_y = int(center_y - halfwidth + 0.5)
    return rect_x, rect_y, length, length


def get_edge_points(
    model_start: np.ndarray,
    model_end: np.ndarray,
    zero_start: np.ndarray,
    zero_end: np.ndarray,
    view_points_per_unit_length: float,
) -> np.ndarray:
    length_model = float(np.linalg.norm(model_end - model_start))
    num_points = max(2, int(view_points_per_unit_length * length_model))
    fractions = np.linspace(0.0, 1.0, num_points)
    return zero_start + (zero_end - zero_start) * fractions[:, None]


def generate_arcs(
    model_vertices: np.ndarray,
    zero_vertices: np.ndarray,
    edges: list[Edge],
    view_points_per_unit_length: float,
    n_view: float,
    dedupe_decimals: int,
    min_arc_radius: float,
) -> list[Arc]:
    seen_coords: set[str] = set()
    seen_arc_geom: set[tuple[int, int, int, int, int]] = set()
    arcs: list[Arc] = []

    for edge in edges:
        model_start = model_vertices[edge.start_idx]
        model_end = model_vertices[edge.end_idx]
        zero_start = zero_vertices[edge.start_idx]
        zero_end = zero_vertices[edge.end_idx]

        points = get_edge_points(
            model_start=model_start,
            model_end=model_end,
            zero_start=zero_start,
            zero_end=zero_end,
            view_points_per_unit_length=view_points_per_unit_length,
        )

        for point in points:
            coord_hash = (
                f"{point[0]:.{dedupe_decimals}f}:"
                f"{point[1]:.{dedupe_decimals}f}:"
                f"{point[2]:.{dedupe_decimals}f}"
            )
            if coord_hash in seen_coords:
                continue
            seen_coords.add(coord_hash)

            rect_x, rect_y, rect_w, rect_h = get_arc_square(point, n_view)
            radius = rect_w / 2.0
            if radius < min_arc_radius:
                continue

            start_angle = 0.0 if (float(point[2]) - n_view) > 0.0 else 180.0
            arc_hash = (rect_x, rect_y, rect_w, rect_h, int(start_angle))
            if arc_hash in seen_arc_geom:
                continue
            seen_arc_geom.add(arc_hash)

            arcs.append(
                Arc(
                    edge_id=edge.edge_id,
                    zero_coord=(float(point[0]), float(point[1]), float(point[2])),
                    rect_x=rect_x,
                    rect_y=rect_y,
                    rect_w=rect_w,
                    rect_h=rect_h,
                    start_angle=start_angle,
                    sweep_angle=180.0,
                )
            )

    arcs.sort(
        key=lambda a: (
            a.edge_id,
            round(a.zero_coord[0], 4),
            round(a.zero_coord[1], 4),
            a.start_angle,
            a.sweep_angle,
        )
    )
    return arcs


def build_simulation_edge_info(
    model_vertices: np.ndarray,
    edges: list[Edge],
    view_points_per_unit_length: float,
) -> list[tuple[int, int, int]]:
    sim_edges: list[tuple[int, int, int]] = []
    for edge in edges:
        start = model_vertices[edge.start_idx]
        end = model_vertices[edge.end_idx]
        edge_len = float(np.linalg.norm(end - start))
        num_points = max(2, int(view_points_per_unit_length * edge_len))
        sim_edges.append((int(edge.start_idx), int(edge.end_idx), int(num_points)))
    return sim_edges


def compute_sim_bounds_from_arcs(arcs: list[Arc]) -> tuple[float, float, float, float]:
    if not arcs:
        return 0.0, 0.0, 1.0, 1.0

    min_x = float(min(a.rect_x for a in arcs))
    min_y = float(min(a.rect_y for a in arcs))
    max_x = float(max(a.rect_x + a.rect_w for a in arcs))
    max_y = float(max(a.rect_y + a.rect_h for a in arcs))
    return min_x, min_y, max_x, max_y


def write_simulation_html(
    html_path: Path,
    model_vertices: np.ndarray,
    sim_edges: list[tuple[int, int, int]],
    camera: CameraConfig,
    arcs: list[Arc],
    min_arc_radius: float,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    min_x, min_y, max_x, max_y = compute_sim_bounds_from_arcs(arcs)
    padding = 24.0

    width = max(1, int(math.ceil((max_x - min_x) + 2.0 * padding)))
    height = max(1, int(math.ceil((max_y - min_y) + 2.0 * padding)))
    offset_x = padding - min_x
    offset_y = padding - min_y

    payload = {
        "minArcRadius": float(min_arc_radius),
        "camera": {
            "po": [float(camera.po[0]), float(camera.po[1]), float(camera.po[2])],
            "pr": [float(camera.pr[0]), float(camera.pr[1]), float(camera.pr[2])],
            "lookUp": [float(camera.look_up[0]), float(camera.look_up[1]), float(camera.look_up[2])],
            "currentScale": float(camera.current_scale),
            "zf": float(camera.zf),
            "canvasWidth": int(camera.canvas_width),
            "canvasHeight": int(camera.canvas_height),
        },
        "vertices": [
            [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
            for v in model_vertices
        ],
        "edges": [[e[0], e[1], e[2]] for e in sim_edges],
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\"/>\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
        "  <title>Scratch Hologram Simulation</title>\n"
        "  <style>\n"
        "    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #17181d; color: #f0f0f0; }\n"
        "    .wrap { padding: 12px; max-width: 1200px; margin: 0 auto; }\n"
        "    .toolbar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }\n"
        "    .toolbar label { font-size: 14px; }\n"
        "    .panel { background: #20222a; border: 1px solid #2d313a; border-radius: 8px; padding: 10px; }\n"
        "    canvas { background: #0f1014; display: block; border-radius: 6px; width: 100%; height: auto; }\n"
        "    input[type=range] { width: 190px; }\n"
        "    .small { opacity: 0.8; font-size: 12px; margin: 6px 2px 10px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"wrap\">\n"
        "    <div class=\"toolbar panel\">\n"
        "      <label>View angle: <input id=\"angle\" type=\"range\" min=\"-90\" max=\"90\" step=\"1\" value=\"0\"/></label>\n"
        "      <strong id=\"angleValue\">0 deg</strong>\n"
        "      <label>Cam yaw: <input id=\"camYaw\" type=\"range\" min=\"-45\" max=\"45\" step=\"1\" value=\"0\"/></label>\n"
        "      <strong id=\"camYawValue\">0 deg</strong>\n"
        "      <label>Cam pitch: <input id=\"camPitch\" type=\"range\" min=\"-45\" max=\"45\" step=\"1\" value=\"0\"/></label>\n"
        "      <strong id=\"camPitchValue\">0 deg</strong>\n"
        "      <label>Zoom: <input id=\"camZoom\" type=\"range\" min=\"60\" max=\"220\" step=\"1\" value=\"100\"/></label>\n"
        "      <strong id=\"camZoomValue\">100%</strong>\n"
        "      <label>View gain: <input id=\"viewGain\" type=\"range\" min=\"10\" max=\"300\" step=\"5\" value=\"100\"/></label>\n"
        "      <strong id=\"viewGainValue\">1.0x</strong>\n"
        "      <label>Arc stride: <input id=\"arcStride\" type=\"range\" min=\"1\" max=\"12\" step=\"1\" value=\"4\"/></label>\n"
        "      <strong id=\"arcStrideValue\">4</strong>\n"
        "      <label>Arc limit: <input id=\"arcLimit\" type=\"range\" min=\"200\" max=\"10000\" step=\"100\" value=\"2500\"/></label>\n"
        "      <strong id=\"arcLimitValue\">2500</strong>\n"
        "      <label>Arc min r: <input id=\"arcMinR\" type=\"range\" min=\"0\" max=\"20\" step=\"1\" value=\"1\"/></label>\n"
        "      <strong id=\"arcMinRValue\">1</strong>\n"
        "      <label>Arc alpha: <input id=\"arcAlpha\" type=\"range\" min=\"2\" max=\"80\" step=\"1\" value=\"20\"/></label>\n"
        "      <strong id=\"arcAlphaValue\">20%</strong>\n"
        "      <label><input id=\"showArcs\" type=\"checkbox\" checked/> Show arcs</label>\n"
        "      <label><input id=\"showWire\" type=\"checkbox\" checked/> Show simulated profile</label>\n"
        "    </div>\n"
        "    <div class=\"small\">Camera orbit is 3D-projected. Increase View gain if angle motion feels too subtle.</div>\n"
        "    <div class=\"panel\">\n"
        f"      <canvas id=\"sim\" width=\"{width}\" height=\"{height}\"></canvas>\n"
        "    </div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const data = {payload_json};\n"
        "    const canvas = document.getElementById('sim');\n"
        "    const ctx = canvas.getContext('2d');\n"
        "    const angleInput = document.getElementById('angle');\n"
        "    const angleValue = document.getElementById('angleValue');\n"
        "    const camYaw = document.getElementById('camYaw');\n"
        "    const camYawValue = document.getElementById('camYawValue');\n"
        "    const camPitch = document.getElementById('camPitch');\n"
        "    const camPitchValue = document.getElementById('camPitchValue');\n"
        "    const camZoom = document.getElementById('camZoom');\n"
        "    const camZoomValue = document.getElementById('camZoomValue');\n"
        "    const viewGain = document.getElementById('viewGain');\n"
        "    const viewGainValue = document.getElementById('viewGainValue');\n"
        "    const arcStride = document.getElementById('arcStride');\n"
        "    const arcStrideValue = document.getElementById('arcStrideValue');\n"
        "    const arcLimit = document.getElementById('arcLimit');\n"
        "    const arcLimitValue = document.getElementById('arcLimitValue');\n"
        "    const arcMinR = document.getElementById('arcMinR');\n"
        "    const arcMinRValue = document.getElementById('arcMinRValue');\n"
        "    const arcAlpha = document.getElementById('arcAlpha');\n"
        "    const arcAlphaValue = document.getElementById('arcAlphaValue');\n"
        "    const showArcs = document.getElementById('showArcs');\n"
        "    const showWire = document.getElementById('showWire');\n"
        "    const baseCamera = data.camera;\n"
        "    const vAdd = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];\n"
        "    const vSub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];\n"
        "    const vDot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];\n"
        "    const vCross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];\n"
        "    const vNorm = (a) => Math.sqrt(vDot(a,a));\n"
        "    const vNormalize = (a) => { const n = vNorm(a); return (n < 1e-12) ? [0,0,0] : [a[0]/n,a[1]/n,a[2]/n]; };\n"
        "    function rotY(v, ang) { const c=Math.cos(ang), s=Math.sin(ang); return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]; }\n"
        "    function rotX(v, ang) { const c=Math.cos(ang), s=Math.sin(ang); return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]; }\n"
        "    function buildCameraState() {\n"
        "      const yaw = Number(camYaw.value) * Math.PI / 180.0;\n"
        "      const pitch = Number(camPitch.value) * Math.PI / 180.0;\n"
        "      const zoom = Number(camZoom.value) / 100.0;\n"
        "      const pr = baseCamera.pr;\n"
        "      const dir = vSub(baseCamera.po, pr);\n"
        "      const look = baseCamera.lookUp;\n"
        "      const dirRot = rotX(rotY(dir, yaw), pitch);\n"
        "      const lookRot = vNormalize(rotX(rotY(look, yaw), pitch));\n"
        "      return {\n"
        "        po: vAdd(pr, dirRot),\n"
        "        pr: pr,\n"
        "        lookUp: lookRot,\n"
        "        currentScale: baseCamera.currentScale * zoom,\n"
        "        zf: baseCamera.zf,\n"
        "        canvasWidth: baseCamera.canvasWidth,\n"
        "        canvasHeight: baseCamera.canvasHeight,\n"
        "      };\n"
        "    }\n"
        "    function buildModelToWindow(cam) {\n"
        "      const n = vNormalize(vSub(cam.po, cam.pr));\n"
        "      let u = vCross(cam.lookUp, n);\n"
        "      u = vNormalize(u);\n"
        "      const v = vCross(n, u);\n"
        "      const m11=u[0], m12=u[1], m13=u[2], m14=-vDot(u,cam.po);\n"
        "      const m21=v[0], m22=v[1], m23=v[2], m24=-vDot(v,cam.po);\n"
        "      const m31=n[0], m32=n[1], m33=n[2], m34=-vDot(n,cam.po);\n"
        "      const p43 = -1.0 / ((Math.abs(cam.zf) < 1e-12) ? 1e-5 : cam.zf);\n"
        "      const baseScale = cam.canvasWidth / 17.0;\n"
        "      const scale = baseScale * cam.currentScale;\n"
        "      return {\n"
        "        m11,m12,m13,m14,m21,m22,m23,m24,m31,m32,m33,m34,\n"
        "        p43, scale, tx: cam.canvasWidth / 2.0, ty: cam.canvasHeight / 2.0\n"
        "      };\n"
        "    }\n"
        "    function projectPoint(p, m) {\n"
        "      const x = p[0], y = p[1], z = p[2];\n"
        "      const v1 = m.m11*x + m.m12*y + m.m13*z + m.m14;\n"
        "      const v2 = m.m21*x + m.m22*y + m.m23*z + m.m24;\n"
        "      const v3 = m.m31*x + m.m32*y + m.m33*z + m.m34;\n"
        "      const w = (m.p43 * v3) + 1.0;\n"
        "      if (!Number.isFinite(w) || Math.abs(w) < 1e-9) return null;\n"
        "      const vx = (v1 / w) * m.scale + m.tx;\n"
        "      const vy = -((v2 / w) * m.scale) + m.ty;\n"
        "      const vz = (v3 / w) * m.scale;\n"
        "      if (!Number.isFinite(vx) || !Number.isFinite(vy) || !Number.isFinite(vz)) return null;\n"
        "      return [vx, vy, vz];\n"
        "    }\n"
        "    function pointAtAngleProjected(p, deg, nView, gain) {\n"
        "      const dist = (p[2] - nView) * gain;\n"
        "      const cx = p[0];\n"
        "      const cy = p[1] - dist / 2.0;\n"
        "      const oy = dist / 2.0;\n"
        "      const t = deg * Math.PI / 180.0;\n"
        "      const x = cx - oy * Math.sin(t);\n"
        "      const y = cy + oy * Math.cos(t);\n"
        "      return [x, y];\n"
        "    }\n"
        "    function draw() {\n"
        "      const angle = Number(angleInput.value);\n"
        "      angleValue.textContent = `${angle.toFixed(0)} deg`;\n"
        "      camYawValue.textContent = `${Number(camYaw.value).toFixed(0)} deg`;\n"
        "      camPitchValue.textContent = `${Number(camPitch.value).toFixed(0)} deg`;\n"
        "      camZoomValue.textContent = `${Number(camZoom.value).toFixed(0)}%`;\n"
        "      viewGainValue.textContent = `${(Number(viewGain.value) / 100.0).toFixed(1)}x`;\n"
        "      arcStrideValue.textContent = `${Number(arcStride.value).toFixed(0)}`;\n"
        "      arcLimitValue.textContent = `${Number(arcLimit.value).toFixed(0)}`;\n"
        "      arcMinRValue.textContent = `${Number(arcMinR.value).toFixed(0)}`;\n"
        "      arcAlphaValue.textContent = `${Number(arcAlpha.value).toFixed(0)}%`;\n"
        "      ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
        "      const cam = buildCameraState();\n"
        "      const m = buildModelToWindow(cam);\n"
        "      const prView = projectPoint(cam.pr, m);\n"
        "      if (!prView) return;\n"
        "      const nView = prView[2];\n"
        "      const proj = new Array(data.vertices.length);\n"
        "      for (let i = 0; i < data.vertices.length; i++) {\n"
        "        proj[i] = projectPoint(data.vertices[i], m);\n"
        "      }\n"
        "      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;\n"
        "      for (const p of proj) {\n"
        "        if (!p) continue;\n"
        "        if (p[0] < minX) minX = p[0];\n"
        "        if (p[1] < minY) minY = p[1];\n"
        "        if (p[0] > maxX) maxX = p[0];\n"
        "        if (p[1] > maxY) maxY = p[1];\n"
        "      }\n"
        "      if (!Number.isFinite(minX) || !Number.isFinite(minY)) return;\n"
        "      const tx = (canvas.width / 2.0) - ((minX + maxX) / 2.0);\n"
        "      const ty = (canvas.height / 2.0) - ((minY + maxY) / 2.0);\n"
        "      const gain = Number(viewGain.value) / 100.0;\n"
        "      if (showArcs.checked) {\n"
        "        const stride = Math.max(1, Number(arcStride.value));\n"
        "        const limit = Math.max(100, Number(arcLimit.value));\n"
        "        const minR = Math.max(0, Number(arcMinR.value));\n"
        "        const alpha = Math.max(0.01, Number(arcAlpha.value) / 100.0);\n"
        "        ctx.strokeStyle = `rgba(190,190,190,${alpha.toFixed(3)})`;\n"
        "        ctx.lineWidth = 0.9;\n"
        "        let drawn = 0;\n"
        "        edgeLoop: for (const e of data.edges) {\n"
        "          const p0 = proj[e[0]];\n"
        "          const p1 = proj[e[1]];\n"
        "          if (!p0 || !p1) continue;\n"
        "          const samples = Math.max(2, e[2]);\n"
        "          for (let k = 0; k < samples; k += stride) {\n"
        "            const t = (samples <= 1) ? 0.0 : (k / (samples - 1));\n"
            "            const px = p0[0] + (p1[0] - p0[0]) * t;\n"
            "            const py = p0[1] + (p1[1] - p0[1]) * t;\n"
            "            const pz = p0[2] + (p1[2] - p0[2]) * t;\n"
        "            const dist = (pz - nView) * gain;\n"
            "            const r = Math.abs(dist) / 2.0;\n"
        "            if (r < minR || r < data.minArcRadius * 0.25) continue;\n"
        "            const cx = px + tx;\n"
        "            const cy = (py - dist / 2.0) + ty;\n"
            "            const start = (dist > 0) ? 0 : Math.PI;\n"
        "            ctx.beginPath();\n"
        "            ctx.arc(cx, cy, r, start, start + Math.PI, false);\n"
        "            ctx.stroke();\n"
        "            drawn += 1;\n"
        "            if (drawn >= limit) break edgeLoop;\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "      if (showWire.checked) {\n"
        "        ctx.strokeStyle = 'rgba(245,245,245,0.92)';\n"
        "        ctx.lineWidth = 1.5;\n"
        "        for (const edge of data.edges) {\n"
        "          const q1 = proj[edge[0]];\n"
        "          const q2 = proj[edge[1]];\n"
        "          if (!q1 || !q2) continue;\n"
        "          const p1 = pointAtAngleProjected(q1, angle, nView, gain);\n"
        "          const p2 = pointAtAngleProjected(q2, angle, nView, gain);\n"
        "          ctx.beginPath();\n"
        "          ctx.moveTo(p1[0] + tx, p1[1] + ty);\n"
        "          ctx.lineTo(p2[0] + tx, p2[1] + ty);\n"
        "          ctx.stroke();\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "    angleInput.addEventListener('input', draw);\n"
        "    camYaw.addEventListener('input', draw);\n"
        "    camPitch.addEventListener('input', draw);\n"
        "    camZoom.addEventListener('input', draw);\n"
        "    viewGain.addEventListener('input', draw);\n"
        "    arcStride.addEventListener('input', draw);\n"
        "    arcLimit.addEventListener('input', draw);\n"
        "    arcMinR.addEventListener('input', draw);\n"
        "    arcAlpha.addEventListener('input', draw);\n"
        "    showArcs.addEventListener('change', draw);\n"
        "    showWire.addEventListener('change', draw);\n"
        "    draw();\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    html_path.write_text(html, encoding="utf-8")


def arc_endpoints_svg(arc: Arc) -> tuple[float, float, float, float, float]:
    r = arc.rect_w / 2.0
    cx = arc.rect_x + arc.rect_w / 2.0
    cy = arc.rect_y + arc.rect_h / 2.0

    start_rad = math.radians(arc.start_angle)
    end_rad = math.radians(arc.start_angle + arc.sweep_angle)

    x1 = cx + r * math.cos(start_rad)
    y1 = cy + r * math.sin(start_rad)
    x2 = cx + r * math.cos(end_rad)
    y2 = cy + r * math.sin(end_rad)
    return x1, y1, x2, y2, r


def write_svg(svg_path: Path, arcs: list[Arc], stroke_width: float) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    max_x = 1.0
    max_y = 1.0
    path_cmds: list[str] = []

    for arc in arcs:
        x1, y1, x2, y2, r = arc_endpoints_svg(arc)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        path_cmds.append(f"M {x1:.6f} {y1:.6f}")
        path_cmds.append(f"A {r:.6f} {r:.6f} 0 0 1 {x2:.6f} {y2:.6f}")

    width = int(max_x)
    height = int(max_y)

    rows: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            'fill="none" xmlns="http://www.w3.org/2000/svg">'
        ),
    ]

    if path_cmds:
        rows.append(
            f'<path stroke="black" stroke-width="{stroke_width:.4f}" d="'
        )
        rows.extend(path_cmds)
        rows.append('"/>')

    rows.append("</svg>")
    svg_path.write_text("\n".join(rows), encoding="utf-8")


def write_debug_json(
    json_path: Path,
    camera: CameraConfig,
    pipeline: PipelineConfig,
    n_view: float,
    edges_count: int,
    arcs: list[Arc],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "camera": asdict(camera),
        "pipeline": asdict(pipeline),
        "n_view_coordinates": n_view,
        "stats": {
            "edges_count": edges_count,
            "arcs_count": len(arcs),
        },
        "arcs": [
            {
                "EdgeID": arc.edge_id,
                "ZeroCoord": {
                    "X": arc.zero_coord[0],
                    "Y": arc.zero_coord[1],
                    "Z": arc.zero_coord[2],
                },
                "ArcRect": f"{arc.rect_x}, {arc.rect_y}, {arc.rect_w}, {arc.rect_h}",
                "StartAngle": arc.start_angle,
                "SweepAngle": arc.sweep_angle,
            }
            for arc in arcs
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HoloZens-style arc SVG from STL edges"
    )

    parser.add_argument("--stl", type=Path, required=True, help="Input STL file")
    parser.add_argument("--svg", type=Path, required=True, help="Output SVG path")
    parser.add_argument("--json", type=Path, default=None, help="Optional debug JSON path")
    parser.add_argument(
        "--simulate-html",
        type=Path,
        default=None,
        help="Optional interactive HTML simulator with View Angle slider.",
    )

    parser.add_argument(
        "--line-resolution",
        type=float,
        default=0.63,
        help="Points per model unit along each edge (HoloZens-like line resolution).",
    )
    parser.add_argument("--stroke-width", type=float, default=0.2, help="SVG stroke width")
    parser.add_argument("--dedupe-decimals", type=int, default=6, help="Arc point dedupe precision")
    parser.add_argument(
        "--min-arc-radius",
        type=float,
        default=4.0,
        help="Filter out arcs with radius smaller than this value (in canvas pixels).",
    )

    parser.add_argument("--canvas-width", type=int, default=710, help="Camera canvas width in pixels")
    parser.add_argument("--canvas-height", type=int, default=549, help="Camera canvas height in pixels")
    parser.add_argument("--current-scale", type=float, default=0.7239993109254925, help="Camera scale")
    parser.add_argument("--zf", type=float, default=25.0, help="Perspective Zf")

    parser.add_argument("--po", nargs=3, type=float, default=(0.900, 9.008, 23.978), metavar=("PX", "PY", "PZ"))
    parser.add_argument("--pr", nargs=3, type=float, default=(0.937, 8.290, 0.472), metavar=("RX", "RY", "RZ"))
    parser.add_argument(
        "--look-up",
        nargs=3,
        type=float,
        default=(0.0, 1.0, 0.0),
        metavar=("UX", "UY", "UZ"),
        help="Camera look-up vector",
    )

    parser.add_argument(
        "--no-auto-center",
        action="store_true",
        help="Disable HoloZens-style Z auto-centering (front face to Z=0).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.line_resolution <= 0:
        raise ValueError("--line-resolution must be > 0")
    if args.stroke_width <= 0:
        raise ValueError("--stroke-width must be > 0")
    if args.min_arc_radius < 0:
        raise ValueError("--min-arc-radius must be >= 0")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("--canvas-width and --canvas-height must be > 0")
    if args.current_scale <= 0:
        raise ValueError("--current-scale must be > 0")
    if abs(args.zf) < EPS:
        raise ValueError("--zf must be non-zero")
    if args.dedupe_decimals < 0 or args.dedupe_decimals > 12:
        raise ValueError("--dedupe-decimals must be in [0, 12]")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    ensure_dependencies()

    camera = CameraConfig(
        po=(float(args.po[0]), float(args.po[1]), float(args.po[2])),
        pr=(float(args.pr[0]), float(args.pr[1]), float(args.pr[2])),
        look_up=(float(args.look_up[0]), float(args.look_up[1]), float(args.look_up[2])),
        current_scale=float(args.current_scale),
        zf=float(args.zf),
        canvas_width=int(args.canvas_width),
        canvas_height=int(args.canvas_height),
    )

    pipeline = PipelineConfig(
        line_resolution=float(args.line_resolution),
        auto_center=not bool(args.no_auto_center),
        stroke_width=float(args.stroke_width),
        dedupe_decimals=int(args.dedupe_decimals),
        min_arc_radius=float(args.min_arc_radius),
    )

    mesh = load_mesh(args.stl)
    model_vertices, edges = build_model_vertices_and_edges(
        mesh=mesh,
        auto_center=pipeline.auto_center,
        decimals=NORMAL_TOLERANCE_DECIMALS,
    )

    model_to_window_mtx = build_model_to_window_matrix(camera)
    zero_vertices = model_to_window(model_vertices, model_to_window_mtx)
    pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), model_to_window_mtx)[0]
    n_view = float(pr_view[2])

    arcs = generate_arcs(
        model_vertices=model_vertices,
        zero_vertices=zero_vertices,
        edges=edges,
        view_points_per_unit_length=pipeline.line_resolution,
        n_view=n_view,
        dedupe_decimals=pipeline.dedupe_decimals,
        min_arc_radius=pipeline.min_arc_radius,
    )

    write_svg(args.svg, arcs, stroke_width=pipeline.stroke_width)

    if args.simulate_html is not None:
        sim_edges = build_simulation_edge_info(
            model_vertices=model_vertices,
            edges=edges,
            view_points_per_unit_length=pipeline.line_resolution,
        )
        write_simulation_html(
            html_path=args.simulate_html,
            model_vertices=model_vertices,
            sim_edges=sim_edges,
            camera=camera,
            arcs=arcs,
            min_arc_radius=pipeline.min_arc_radius,
        )

    if args.json is not None:
        write_debug_json(
            json_path=args.json,
            camera=camera,
            pipeline=pipeline,
            n_view=n_view,
            edges_count=len(edges),
            arcs=arcs,
        )

    print(f"[OK] STL loaded: {args.stl}")
    print(f"[OK] Unique edges: {len(edges)}")
    print(f"[OK] Arcs generated: {len(arcs)}")
    print(f"[OK] SVG saved: {args.svg}")
    if args.simulate_html is not None:
        print(f"[OK] Simulation HTML saved: {args.simulate_html}")
    if args.json is not None:
        print(f"[OK] JSON saved: {args.json}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
