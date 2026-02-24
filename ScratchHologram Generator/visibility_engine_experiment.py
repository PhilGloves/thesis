#!/usr/bin/env python3
"""
Experimental visibility engine for scratch hologram arcs.

Goal:
- Keep current production files unchanged.
- Prototype a single visibility pass that can be reused by both preview/export later.
- Produce comparable SVG outputs from one consistent arc dataset.

This script intentionally lives outside the main app flow.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scratch_pipeline import (
    Arc,
    CameraConfig,
    build_model_to_window_matrix,
    build_model_vertices_and_edges,
    ensure_dependencies,
    generate_arcs,
    load_mesh,
    model_to_window,
    write_svg,
)


EPS = 1.0e-12


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class VisibilityConfig:
    """Parameters for interval-based arc visibility."""

    cull_strength: float
    grid_cell: float
    samples_per_arc: int
    min_sweep_deg: float
    inside_tolerance: float


@dataclass
class TriangleRecord:
    """Precomputed projected triangle values used in repeated point tests."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    denom: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def build_camera_for_experiment(
    model_vertices: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    current_scale: float,
    zf: float,
    canvas_width: int,
    canvas_height: int,
) -> CameraConfig:
    """
    Build the same orbit-style camera used by the desktop app.

    Keeping this consistent is important: if camera math differs,
    visibility will differ even if culling logic is identical.
    """
    v_min = np.min(model_vertices, axis=0)
    v_max = np.max(model_vertices, axis=0)
    center = (v_min + v_max) / 2.0
    diag = float(np.linalg.norm(v_max - v_min))

    # Same heuristic used in the UI for "comfortable" orbit distance.
    orbit_radius = max(10.0, diag * 2.6)

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    direction = np.asarray(
        (
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ),
        dtype=np.float64,
    )
    nrm = float(np.linalg.norm(direction))
    direction = direction / nrm if nrm > EPS else np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    pr = center
    po = pr + (direction * orbit_radius)

    # If camera direction is almost parallel to +Z, switch up vector.
    look_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(direction, look_up))) > 0.98:
        look_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

    return CameraConfig(
        po=(float(po[0]), float(po[1]), float(po[2])),
        pr=(float(pr[0]), float(pr[1]), float(pr[2])),
        look_up=(float(look_up[0]), float(look_up[1]), float(look_up[2])),
        current_scale=float(current_scale),
        zf=float(zf),
        canvas_width=max(1, int(canvas_width)),
        canvas_height=max(1, int(canvas_height)),
    )


def build_triangle_acceleration(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    grid_cell: float,
) -> tuple[list[TriangleRecord], dict[tuple[int, int], list[int]]]:
    """
    Build an acceleration grid in screen space.

    This reduces candidate triangles for each arc-sample query and keeps
    the prototype reasonably fast on dense models.
    """
    tri_data: list[TriangleRecord] = []
    grid: dict[tuple[int, int], list[int]] = {}

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = projected_vertices[i0]
        p1 = projected_vertices[i1]
        p2 = projected_vertices[i2]

        if not (
            np.isfinite(p0[0])
            and np.isfinite(p0[1])
            and np.isfinite(p0[2])
            and np.isfinite(p1[0])
            and np.isfinite(p1[1])
            and np.isfinite(p1[2])
            and np.isfinite(p2[0])
            and np.isfinite(p2[1])
            and np.isfinite(p2[2])
        ):
            continue

        x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
        x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
        x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])
        denom = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2))
        if abs(denom) < EPS:
            continue

        min_x = min(x0, x1, x2)
        max_x = max(x0, x1, x2)
        min_y = min(y0, y1, y2)
        max_y = max(y0, y1, y2)

        tri_idx = len(tri_data)
        tri_data.append(
            TriangleRecord(
                x0=x0,
                y0=y0,
                z0=z0,
                x1=x1,
                y1=y1,
                z1=z1,
                x2=x2,
                y2=y2,
                z2=z2,
                denom=denom,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
        )

        ix0 = int(math.floor(min_x / grid_cell))
        ix1 = int(math.floor(max_x / grid_cell))
        iy0 = int(math.floor(min_y / grid_cell))
        iy1 = int(math.floor(max_y / grid_cell))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                key = (ix, iy)
                if key in grid:
                    grid[key].append(tri_idx)
                else:
                    grid[key] = [tri_idx]

    return tri_data, grid


def arc_point_xy(arc: Arc, u: float) -> tuple[float, float]:
    """
    Compute one 2D point on the arc at param u in [0, 1].

    u=0 -> arc start, u=1 -> arc end.
    """
    u_clamped = clamp(float(u), 0.0, 1.0)
    rx = max(0.1, float(arc.rect_w) / 2.0)
    ry = max(0.1, float(arc.rect_h) / 2.0)
    cx = float(arc.rect_x) + (float(arc.rect_w) / 2.0)
    cy = float(arc.rect_y) + (float(arc.rect_h) / 2.0)
    theta_deg = float(arc.start_angle) + (float(arc.sweep_angle) * u_clamped)
    theta = math.radians(theta_deg)
    x = cx + (rx * math.cos(theta))
    y = cy - (ry * math.sin(theta))
    return x, y


def sample_surface_depths(
    x: float,
    y: float,
    tri_data: list[TriangleRecord],
    grid: dict[tuple[int, int], list[int]],
    grid_cell: float,
    inside_tolerance: float,
) -> list[float]:
    """
    Return depths of all triangles covering (x, y) in screen space.
    """
    ix = int(math.floor(x / grid_cell))
    iy = int(math.floor(y / grid_cell))

    candidate: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidate.extend(grid.get((ix + dx, iy + dy), []))

    if len(candidate) == 0:
        return []

    z_values: list[float] = []
    seen: set[int] = set()
    for tri_idx in candidate:
        if tri_idx in seen:
            continue
        seen.add(tri_idx)

        tri = tri_data[tri_idx]
        if x < (tri.min_x - 0.2) or x > (tri.max_x + 0.2) or y < (tri.min_y - 0.2) or y > (tri.max_y + 0.2):
            continue

        a = ((tri.y1 - tri.y2) * (x - tri.x2)) + ((tri.x2 - tri.x1) * (y - tri.y2))
        b = ((tri.y2 - tri.y0) * (x - tri.x2)) + ((tri.x0 - tri.x2) * (y - tri.y2))
        a /= tri.denom
        b /= tri.denom
        c = 1.0 - a - b
        if a < inside_tolerance or b < inside_tolerance or c < inside_tolerance:
            continue

        z_surf = (a * tri.z0) + (b * tri.z1) + (c * tri.z2)
        z_values.append(z_surf)

    return z_values


def is_sample_visible(
    arc: Arc,
    sample_xy: tuple[float, float],
    tri_data: list[TriangleRecord],
    grid: dict[tuple[int, int], list[int]],
    grid_cell: float,
    z_eps: float,
    cull_strength: float,
    inside_tolerance: float,
) -> bool:
    """
    Visibility test for one sample of one arc.

    Rule mirrors current conservative depth logic:
    - lower strength => keep more uncertain points
    - higher strength => hide more aggressively
    """
    x, y = sample_xy
    z_arc = float(arc.zero_coord[2])

    z_cover = sample_surface_depths(
        x=x,
        y=y,
        tri_data=tri_data,
        grid=grid,
        grid_cell=grid_cell,
        inside_tolerance=inside_tolerance,
    )
    if len(z_cover) == 0:
        return True

    z_front = max(z_cover)
    adaptive_margin = z_eps + ((1.0 - cull_strength) * (0.55 * float(arc.rect_w) + 2.0))
    if cull_strength >= 0.80:
        required_hits = 1
    elif cull_strength >= 0.45:
        required_hits = 2
    else:
        required_hits = 3

    hit_threshold = z_arc + (0.35 * adaptive_margin)
    occluding_hits = 0
    for z_surf in z_cover:
        if z_surf > hit_threshold:
            occluding_hits += 1
            if occluding_hits >= required_hits:
                break

    if z_front <= -1.0e29:
        return True
    if z_arc >= (z_front - adaptive_margin):
        return True
    if occluding_hits < required_hits:
        return True
    return False


def clip_arc_to_visible_segments(
    arc: Arc,
    cfg: VisibilityConfig,
    tri_data: list[TriangleRecord],
    grid: dict[tuple[int, int], list[int]],
    z_eps: float,
) -> list[Arc]:
    """
    Split one arc into visible sub-arcs by sampling along its angle interval.

    Output sub-arcs preserve Arc structure, so they can be exported with existing SVG/G-code writers.
    """
    samples = max(8, int(cfg.samples_per_arc))

    visible: list[bool] = []
    for i in range(samples + 1):
        u = i / samples
        pt = arc_point_xy(arc, u)
        vis = is_sample_visible(
            arc=arc,
            sample_xy=pt,
            tri_data=tri_data,
            grid=grid,
            grid_cell=cfg.grid_cell,
            z_eps=z_eps,
            cull_strength=cfg.cull_strength,
            inside_tolerance=cfg.inside_tolerance,
        )
        visible.append(vis)

    # Segment visibility on intervals [i, i+1].
    seg_visible = [visible[i] and visible[i + 1] for i in range(samples)]

    out: list[Arc] = []
    run_start: int | None = None
    for i, is_vis in enumerate(seg_visible):
        if is_vis and run_start is None:
            run_start = i
        if (not is_vis) and run_start is not None:
            u0 = run_start / samples
            u1 = i / samples
            sweep = float(arc.sweep_angle) * (u1 - u0)
            if abs(sweep) >= cfg.min_sweep_deg:
                out.append(
                    Arc(
                        edge_id=arc.edge_id,
                        zero_coord=arc.zero_coord,
                        rect_x=arc.rect_x,
                        rect_y=arc.rect_y,
                        rect_w=arc.rect_w,
                        rect_h=arc.rect_h,
                        start_angle=float(arc.start_angle) + (float(arc.sweep_angle) * u0),
                        sweep_angle=sweep,
                    )
                )
            run_start = None

    if run_start is not None:
        u0 = run_start / samples
        u1 = 1.0
        sweep = float(arc.sweep_angle) * (u1 - u0)
        if abs(sweep) >= cfg.min_sweep_deg:
            out.append(
                Arc(
                    edge_id=arc.edge_id,
                    zero_coord=arc.zero_coord,
                    rect_x=arc.rect_x,
                    rect_y=arc.rect_y,
                    rect_w=arc.rect_w,
                    rect_h=arc.rect_h,
                    start_angle=float(arc.start_angle) + (float(arc.sweep_angle) * u0),
                    sweep_angle=sweep,
                )
            )

    return out


def cull_arcs_interval_based(
    arcs: list[Arc],
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    cfg: VisibilityConfig,
) -> list[Arc]:
    """
    Apply interval clipping to all arcs.
    """
    if len(arcs) == 0 or len(faces) == 0:
        return arcs

    tri_data, grid = build_triangle_acceleration(
        projected_vertices=projected_vertices,
        faces=faces,
        grid_cell=cfg.grid_cell,
    )
    if len(tri_data) == 0:
        return arcs

    z_vals = projected_vertices[:, 2]
    z_span = float(np.max(z_vals) - np.min(z_vals)) if projected_vertices.shape[0] > 0 else 1.0
    z_eps = max(0.003 * z_span, 0.45)

    out: list[Arc] = []
    for arc in arcs:
        out.extend(
            clip_arc_to_visible_segments(
                arc=arc,
                cfg=cfg,
                tri_data=tri_data,
                grid=grid,
                z_eps=z_eps,
            )
        )

    # Fallback: never output an empty file by mistake in experiments.
    return out if len(out) > 0 else arcs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental single visibility engine (keeps production files unchanged)."
    )
    parser.add_argument("--stl", type=Path, required=True, help="Input STL file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out") / "visibility_experiment",
        help="Output directory for SVG and report files.",
    )

    # Arc generation settings (same semantic meaning as production pipeline).
    parser.add_argument("--line-resolution", type=float, default=3.28)
    parser.add_argument("--min-arc-radius", type=float, default=0.0)
    parser.add_argument("--arc-mode", choices=("semi", "elliptic"), default="semi")
    parser.add_argument("--ellipse-ratio", type=float, default=0.65)
    parser.add_argument("--stroke-width", type=float, default=0.15)

    # Camera settings.
    parser.add_argument("--yaw", type=float, default=-60.0)
    parser.add_argument("--pitch", type=float, default=20.0)
    parser.add_argument("--camera-scale", type=float, default=0.724)
    parser.add_argument("--zf", type=float, default=25.0)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=900)
    parser.add_argument("--auto-center", action="store_true", default=True)
    parser.add_argument("--no-auto-center", dest="auto_center", action="store_false")

    # Visibility engine settings.
    parser.add_argument("--cull-strength", type=float, default=0.45, help="0.0 keep-all, 1.0 aggressive.")
    parser.add_argument("--grid-cell", type=float, default=24.0)
    parser.add_argument("--samples-per-arc", type=int, default=28)
    parser.add_argument("--min-sweep-deg", type=float, default=2.0)
    parser.add_argument("--inside-tolerance", type=float, default=0.0)

    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    ensure_dependencies()

    stl_path = args.stl.resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"STL non trovato: {stl_path}")

    mesh = load_mesh(stl_path)
    model_vertices, edges, faces = build_model_vertices_and_edges(
        mesh=mesh,
        auto_center=bool(args.auto_center),
        decimals=4,
    )

    camera = build_camera_for_experiment(
        model_vertices=model_vertices,
        yaw_deg=float(args.yaw),
        pitch_deg=float(args.pitch),
        current_scale=float(args.camera_scale),
        zf=float(args.zf),
        canvas_width=int(args.canvas_width),
        canvas_height=int(args.canvas_height),
    )
    mtx = build_model_to_window_matrix(camera)
    projected_vertices = model_to_window(model_vertices, mtx)
    pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), mtx)[0]
    n_view = float(pr_view[2])

    arcs = generate_arcs(
        model_vertices=model_vertices,
        zero_vertices=projected_vertices,
        edges=edges,
        view_points_per_unit_length=float(args.line_resolution),
        n_view=n_view,
        dedupe_decimals=6,
        min_arc_radius=float(args.min_arc_radius),
        arc_mode=str(args.arc_mode),
        ellipse_ratio=float(args.ellipse_ratio),
    )

    vis_cfg = VisibilityConfig(
        cull_strength=clamp(float(args.cull_strength), 0.0, 1.0),
        grid_cell=max(4.0, float(args.grid_cell)),
        samples_per_arc=max(8, int(args.samples_per_arc)),
        min_sweep_deg=max(0.1, float(args.min_sweep_deg)),
        inside_tolerance=float(args.inside_tolerance),
    )
    visible_arcs = cull_arcs_interval_based(
        arcs=arcs,
        projected_vertices=projected_vertices,
        faces=faces,
        cfg=vis_cfg,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stl_path.stem
    svg_all = out_dir / f"{stem}_exp_all.svg"
    svg_visible = out_dir / f"{stem}_exp_visible.svg"
    report_path = out_dir / f"{stem}_exp_report.json"

    write_svg(svg_all, arcs, stroke_width=float(args.stroke_width))
    write_svg(svg_visible, visible_arcs, stroke_width=float(args.stroke_width))

    report = {
        "input_stl": str(stl_path),
        "camera": {
            "yaw": float(args.yaw),
            "pitch": float(args.pitch),
            "camera_scale": float(args.camera_scale),
            "zf": float(args.zf),
            "canvas_width": int(args.canvas_width),
            "canvas_height": int(args.canvas_height),
        },
        "arc_generation": {
            "line_resolution": float(args.line_resolution),
            "min_arc_radius": float(args.min_arc_radius),
            "arc_mode": str(args.arc_mode),
            "ellipse_ratio": float(args.ellipse_ratio),
        },
        "visibility_engine": {
            "cull_strength": vis_cfg.cull_strength,
            "grid_cell": vis_cfg.grid_cell,
            "samples_per_arc": vis_cfg.samples_per_arc,
            "min_sweep_deg": vis_cfg.min_sweep_deg,
            "inside_tolerance": vis_cfg.inside_tolerance,
        },
        "counts": {
            "edges": len(edges),
            "faces": int(faces.shape[0]),
            "arcs_all": len(arcs),
            "arcs_visible": len(visible_arcs),
            "culled": max(0, len(arcs) - len(visible_arcs)),
        },
        "outputs": {
            "svg_all": str(svg_all),
            "svg_visible": str(svg_visible),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Input STL: {stl_path}")
    print(f"[OK] Arcs all: {len(arcs)}")
    print(f"[OK] Arcs visible (interval engine): {len(visible_arcs)}")
    print(f"[OK] SVG all: {svg_all}")
    print(f"[OK] SVG visible: {svg_visible}")
    print(f"[OK] Report: {report_path}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
