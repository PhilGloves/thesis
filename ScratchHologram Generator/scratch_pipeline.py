#!/usr/bin/env python3
"""
Scratch hologram generator pipeline.

Pipeline steps:
1) Load STL mesh
2) Sample points on the 3D surface
3) Compute simplified specular reflection score
4) Project to XY plane
5) Generate short scratch segments
6) Export SVG and optional G-code
"""

from __future__ import annotations

import argparse
import json
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

EPS = 1e-9


@dataclass
class PipelineConfig:
    width_mm: float
    height_mm: float
    margin_mm: float
    samples: int
    shininess: float
    min_len_mm: float
    max_len_mm: float
    spec_threshold: float
    max_segments: int
    stroke_mm: float
    light_dir: tuple[float, float, float]
    view_dir: tuple[float, float, float]
    seed: int


def ensure_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if trimesh is None:
        missing.append("trimesh")
    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            f"Missing Python packages: {packages}. "
            "Install with: pip install -r requirements.txt"
        )


def normalize(v):
    arr = np.asarray(v, dtype=np.float64)

    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm < EPS:
            raise ValueError("Cannot normalize zero-length vector")
        return arr / norm

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, EPS, None)


def reflect(incident, normal):
    # Specular reflection: R = I - 2 * dot(I, N) * N
    dot = np.sum(incident * normal, axis=1, keepdims=True)
    return incident - 2.0 * dot * normal


def load_mesh(stl_path: Path):
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    loaded = trimesh.load(stl_path, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in STL: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input file does not contain a valid triangular mesh")

    mesh = mesh.copy()

    # trimesh API differs across versions; keep cleanup compatible.
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
        raise ValueError("Mesh has zero valid faces after cleanup")

    return mesh


def sample_surface_points(mesh, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)

    valid_face_idx = np.flatnonzero(areas > EPS)
    if valid_face_idx.size == 0:
        raise ValueError("Mesh has no faces with positive area")

    probs = areas[valid_face_idx]
    probs = probs / probs.sum()
    sampled_face_idx = rng.choice(valid_face_idx, size=n_samples, replace=True, p=probs)

    tri = triangles[sampled_face_idx]

    # Uniform barycentric sampling inside each triangle
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v

    points = (
        tri[:, 0] * w[:, None]
        + tri[:, 1] * u[:, None]
        + tri[:, 2] * v[:, None]
    )

    normals = np.asarray(mesh.face_normals[sampled_face_idx], dtype=np.float64)

    return points, normalize(normals)


def project_to_xy_canvas(points_xyz, width_mm: float, height_mm: float, margin_mm: float):
    xy = np.asarray(points_xyz[:, :2], dtype=np.float64)

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span_xy = np.maximum(max_xy - min_xy, EPS)

    xy_norm = (xy - min_xy) / span_xy

    usable_w = max(width_mm - 2.0 * margin_mm, EPS)
    usable_h = max(height_mm - 2.0 * margin_mm, EPS)

    x = margin_mm + xy_norm[:, 0] * usable_w
    y = margin_mm + (1.0 - xy_norm[:, 1]) * usable_h

    return np.column_stack((x, y))


def compute_segments(points, normals, cfg: PipelineConfig):
    light = normalize(np.array(cfg.light_dir, dtype=np.float64))
    view = normalize(np.array(cfg.view_dir, dtype=np.float64))

    n = normalize(normals)

    # Incident direction points from light to the surface point
    incident = np.broadcast_to(-light, (n.shape[0], 3))
    reflected = normalize(reflect(incident, n))

    # Phong-like simplified specular response
    spec = np.clip(np.sum(reflected * view, axis=1), 0.0, 1.0) ** cfg.shininess

    # Scratch orientation on XY plane.
    # We orient each segment perpendicular to projected reflection direction.
    rxy = reflected[:, :2]
    rxy_norm = np.linalg.norm(rxy, axis=1)

    dir_xy = np.zeros_like(rxy)

    ok = rxy_norm > EPS
    dir_xy[ok] = rxy[ok] / rxy_norm[ok, None]

    # Fallback when reflection projects to almost a point
    nxy = n[:, :2]
    nxy_norm = np.linalg.norm(nxy, axis=1)
    fallback = (~ok) & (nxy_norm > EPS)
    dir_xy[fallback] = nxy[fallback] / nxy_norm[fallback, None]

    # Final fallback
    dir_xy[~(ok | fallback)] = np.array([1.0, 0.0])

    tangent = np.column_stack((-dir_xy[:, 1], dir_xy[:, 0]))

    centers = project_to_xy_canvas(
        points_xyz=points,
        width_mm=cfg.width_mm,
        height_mm=cfg.height_mm,
        margin_mm=cfg.margin_mm,
    )

    lengths = cfg.min_len_mm + (cfg.max_len_mm - cfg.min_len_mm) * spec

    keep = np.flatnonzero(spec >= cfg.spec_threshold)
    if keep.size == 0:
        return np.empty((0, 4), dtype=np.float64), spec, keep

    if keep.size > cfg.max_segments:
        top_order = np.argsort(spec[keep])[::-1][: cfg.max_segments]
        keep = keep[top_order]

    # Stable ordering helps reproducible output files
    stable_order = np.lexsort((centers[keep, 0], centers[keep, 1]))
    keep = keep[stable_order]

    half = 0.5 * lengths[keep][:, None]
    start = centers[keep] - half * tangent[keep]
    end = centers[keep] + half * tangent[keep]

    start[:, 0] = np.clip(start[:, 0], 0.0, cfg.width_mm)
    start[:, 1] = np.clip(start[:, 1], 0.0, cfg.height_mm)
    end[:, 0] = np.clip(end[:, 0], 0.0, cfg.width_mm)
    end[:, 1] = np.clip(end[:, 1], 0.0, cfg.height_mm)

    segments = np.hstack((start, end))
    return segments, spec, keep


def write_svg(svg_path: Path, segments, cfg: PipelineConfig) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{cfg.width_mm:.3f}mm" height="{cfg.height_mm:.3f}mm" '
            f'viewBox="0 0 {cfg.width_mm:.6f} {cfg.height_mm:.6f}">'
        ),
        (
            f'<g fill="none" stroke="black" stroke-width="{cfg.stroke_mm:.4f}" '
            'stroke-linecap="round">'
        ),
    ]

    for x1, y1, x2, y2 in segments:
        rows.append(f'<path d="M {x1:.6f} {y1:.6f} L {x2:.6f} {y2:.6f}" />')

    rows.append("</g>")
    rows.append("</svg>")

    svg_path.write_text("\n".join(rows), encoding="utf-8")


def write_gcode(
    gcode_path: Path,
    segments,
    canvas_height_mm: float,
    safe_z: float,
    cut_z: float,
    feed_xy: float,
    feed_z: float,
    invert_y: bool,
) -> None:
    gcode_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "; Scratch hologram toolpath",
        "G21 ; millimeters",
        "G90 ; absolute",
        "G17 ; XY plane",
        "G94 ; feed per minute",
        f"G0 Z{safe_z:.3f}",
    ]

    for x1, y1, x2, y2 in segments:
        if invert_y:
            y1 = canvas_height_mm - y1
            y2 = canvas_height_mm - y2

        lines.append(f"G0 X{x1:.3f} Y{y1:.3f}")
        lines.append(f"G1 Z{cut_z:.3f} F{feed_z:.1f}")
        lines.append(f"G1 X{x2:.3f} Y{y2:.3f} F{feed_xy:.1f}")
        lines.append(f"G0 Z{safe_z:.3f}")

    lines.append("M2")

    gcode_path.write_text("\n".join(lines), encoding="utf-8")


def write_debug_json(
    json_path: Path,
    cfg: PipelineConfig,
    segments,
    spec,
    kept_idx,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": asdict(cfg),
        "stats": {
            "total_samples": int(spec.shape[0]),
            "kept_segments": int(segments.shape[0]),
            "spec_min": float(spec.min()) if spec.size else 0.0,
            "spec_max": float(spec.max()) if spec.size else 0.0,
            "spec_mean": float(spec.mean()) if spec.size else 0.0,
        },
        "kept_sample_indices": [int(i) for i in kept_idx.tolist()],
        "segments": [
            {
                "x1": float(seg[0]),
                "y1": float(seg[1]),
                "x2": float(seg[2]),
                "y2": float(seg[3]),
            }
            for seg in segments
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scratch hologram paths from STL to SVG (and optional G-code)"
    )

    parser.add_argument("--stl", type=Path, required=True, help="Input STL file path")
    parser.add_argument("--svg", type=Path, required=True, help="Output SVG file path")
    parser.add_argument("--gcode", type=Path, default=None, help="Optional output G-code path")
    parser.add_argument("--json", type=Path, default=None, help="Optional debug JSON path")

    parser.add_argument("--samples", type=int, default=8000, help="Surface samples count")
    parser.add_argument("--width-mm", type=float, default=120.0, help="Output width in mm")
    parser.add_argument("--height-mm", type=float, default=120.0, help="Output height in mm")
    parser.add_argument("--margin-mm", type=float, default=5.0, help="Inner margin in mm")

    parser.add_argument("--shininess", type=float, default=40.0, help="Specular exponent")
    parser.add_argument("--min-len-mm", type=float, default=0.30, help="Minimum scratch length")
    parser.add_argument("--max-len-mm", type=float, default=1.80, help="Maximum scratch length")
    parser.add_argument("--spec-threshold", type=float, default=0.10, help="Minimum specular value")
    parser.add_argument("--max-segments", type=int, default=12000, help="Maximum emitted segments")
    parser.add_argument("--stroke-mm", type=float, default=0.12, help="SVG stroke width")

    parser.add_argument(
        "--light",
        nargs=3,
        type=float,
        default=(0.4, 0.2, 1.0),
        metavar=("LX", "LY", "LZ"),
        help="Global light direction vector",
    )
    parser.add_argument(
        "--view",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("VX", "VY", "VZ"),
        help="Global viewer direction vector",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--safe-z", type=float, default=3.0, help="Rapid safe Z for G-code")
    parser.add_argument("--cut-z", type=float, default=-0.05, help="Scratch Z depth for G-code")
    parser.add_argument("--feed-xy", type=float, default=500.0, help="XY feedrate in mm/min")
    parser.add_argument("--feed-z", type=float, default=120.0, help="Z feedrate in mm/min")
    parser.add_argument(
        "--gcode-y-down",
        action="store_true",
        help="Use SVG Y axis in G-code without inversion",
    )

    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.width_mm <= 0 or args.height_mm <= 0:
        raise ValueError("--width-mm and --height-mm must be > 0")
    if args.min_len_mm <= 0 or args.max_len_mm <= 0:
        raise ValueError("Scratch lengths must be > 0")
    if args.min_len_mm > args.max_len_mm:
        raise ValueError("--min-len-mm must be <= --max-len-mm")
    if not (0.0 <= args.spec_threshold <= 1.0):
        raise ValueError("--spec-threshold must be in [0, 1]")
    if args.max_segments <= 0:
        raise ValueError("--max-segments must be > 0")

    return PipelineConfig(
        width_mm=float(args.width_mm),
        height_mm=float(args.height_mm),
        margin_mm=float(args.margin_mm),
        samples=int(args.samples),
        shininess=float(args.shininess),
        min_len_mm=float(args.min_len_mm),
        max_len_mm=float(args.max_len_mm),
        spec_threshold=float(args.spec_threshold),
        max_segments=int(args.max_segments),
        stroke_mm=float(args.stroke_mm),
        light_dir=(float(args.light[0]), float(args.light[1]), float(args.light[2])),
        view_dir=(float(args.view[0]), float(args.view[1]), float(args.view[2])),
        seed=int(args.seed),
    )


def main() -> int:
    args = parse_args()

    try:
        ensure_dependencies()

        cfg = build_config(args)
        mesh = load_mesh(args.stl)
        points, normals = sample_surface_points(mesh, cfg.samples, cfg.seed)
        segments, spec, kept_idx = compute_segments(points, normals, cfg)

        write_svg(args.svg, segments, cfg)

        if args.gcode is not None:
            write_gcode(
                gcode_path=args.gcode,
                segments=segments,
                canvas_height_mm=cfg.height_mm,
                safe_z=float(args.safe_z),
                cut_z=float(args.cut_z),
                feed_xy=float(args.feed_xy),
                feed_z=float(args.feed_z),
                invert_y=(not args.gcode_y_down),
            )

        if args.json is not None:
            write_debug_json(args.json, cfg, segments, spec, kept_idx)

        print(f"[OK] STL loaded: {args.stl}")
        print(f"[OK] Segments generated: {segments.shape[0]} from {cfg.samples} samples")
        print(f"[OK] SVG saved: {args.svg}")

        if args.gcode is not None:
            print(f"[OK] G-code saved: {args.gcode}")
        if args.json is not None:
            print(f"[OK] Debug JSON saved: {args.json}")

        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
