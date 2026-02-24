#!/usr/bin/env python3
"""
CNC Validation Suite for ScratchHologram.

Why this script exists:
- Generate a small, repeatable set of validation files (SVG + G-code).
- Produce a readable report with objective metrics before CAM simulation.
- Make thesis experiments reproducible with one command.

This script does NOT replace real CAM verification:
it prepares a consistent "test pack" for that verification step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from scratch_desktop_app import write_gcode
from scratch_pipeline import (
    NORMAL_TOLERANCE_DECIMALS,
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


DEFAULT_MODELS = (
    "basic_cube_-_10mm.stl",
    "trefoil.stl",
    "easiestcleanedtorus.stl",
)


@dataclass
class CaseSummary:
    model: str
    mode: str
    arcs: int
    edges: int
    svg_file: str
    gcode_file: str
    width_mm: float
    height_mm: float
    path_count: int
    segment_count: int
    cut_length_mm: float
    est_total_time_min: float
    cmd_g0: int
    cmd_g1: int
    cmd_g1_xy: int
    cmd_g2: int
    cmd_g3: int
    downsample_stride: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a repeatable CNC validation pack (SVG + G-code + report)."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        type=Path,
        default=None,
        help="Optional STL list. If omitted, a default trio is used when found in ../knots.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "cnc_validation",
        help="Output directory for generated test files and report.",
    )
    parser.add_argument("--line-resolution", type=float, default=3.28)
    parser.add_argument("--min-arc-radius", type=float, default=6.0)
    parser.add_argument("--stroke-width", type=float, default=0.15)
    parser.add_argument("--ellipse-ratio", type=float, default=0.42)
    parser.add_argument(
        "--max-arcs-per-case",
        type=int,
        default=120000,
        help="Soft cap for validation speed. If exceeded, arcs are uniformly downsampled.",
    )
    parser.add_argument("--target-width-mm", type=float, default=10.0)
    parser.add_argument("--safe-z-mm", type=float, default=3.0)
    parser.add_argument("--cut-z-mm", type=float, default=-0.08)
    parser.add_argument("--feed-xy-mm-min", type=float, default=700.0)
    parser.add_argument("--feed-z-mm-min", type=float, default=220.0)
    parser.add_argument("--max-segment-mm", type=float, default=0.20)
    parser.add_argument("--spindle-rpm", type=int, default=12000)
    parser.add_argument(
        "--invert-y",
        action="store_true",
        default=True,
        help="Mirror Y to match common CNC machine convention used in this project.",
    )
    parser.add_argument(
        "--rapid-feed-mm-min",
        type=float,
        default=3000.0,
        help="Assumed machine rapid feed for rough cycle-time estimate.",
    )
    return parser.parse_args()


def resolve_models(args_models: list[Path] | None) -> list[Path]:
    """
    Resolve model paths with robust defaults.
    """
    if args_models:
        resolved = [p.resolve() for p in args_models]
        missing = [p for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing STL files: {', '.join(str(m) for m in missing)}")
        return resolved

    knots = Path(__file__).resolve().parent.parent / "knots"
    models: list[Path] = []
    for name in DEFAULT_MODELS:
        p = knots / name
        if p.exists():
            models.append(p.resolve())
    if not models:
        raise FileNotFoundError(
            "No default models found. Pass --models with explicit STL paths."
        )
    return models


def default_camera() -> CameraConfig:
    """
    Keep camera defaults stable so experiments are reproducible.
    """
    return CameraConfig(
        po=(0.900, 9.008, 23.978),
        pr=(0.937, 8.290, 0.472),
        look_up=(0.0, 1.0, 0.0),
        current_scale=0.724,
        zf=25.0,
        canvas_width=710,
        canvas_height=549,
    )


def build_arcs_for_model(
    stl_path: Path,
    mode: str,
    line_resolution: float,
    min_arc_radius: float,
    ellipse_ratio: float,
) -> tuple[list[Arc], int]:
    """
    Compute arc geometry using the same production pipeline as the app.
    """
    mesh = load_mesh(stl_path)
    model_vertices, edges, _faces = build_model_vertices_and_edges(
        mesh=mesh,
        auto_center=True,
        decimals=NORMAL_TOLERANCE_DECIMALS,
    )

    cam = default_camera()
    mtx = build_model_to_window_matrix(cam)
    zero_vertices = model_to_window(model_vertices, mtx)
    pr_view = model_to_window(
        [[cam.pr[0], cam.pr[1], cam.pr[2]]],
        mtx,
    )[0]
    n_view = float(pr_view[2])

    arcs = generate_arcs(
        model_vertices=model_vertices,
        zero_vertices=zero_vertices,
        edges=edges,
        view_points_per_unit_length=float(line_resolution),
        n_view=n_view,
        dedupe_decimals=6,
        min_arc_radius=float(min_arc_radius),
        arc_mode=mode,
        ellipse_ratio=float(ellipse_ratio),
    )
    return arcs, len(edges)


def arc_bbox_mm(arcs: list[Arc], target_width_mm: float) -> tuple[float, float]:
    """
    Estimate physical XY size after scaling to target width.
    """
    if not arcs:
        return 0.0, 0.0
    min_x = min(float(a.rect_x) for a in arcs)
    max_x = max(float(a.rect_x + a.rect_w) for a in arcs)
    min_y = min(float(a.rect_y) for a in arcs)
    max_y = max(float(a.rect_y + a.rect_h) for a in arcs)
    width_u = max(max_x - min_x, 1.0e-9)
    height_u = max_y - min_y
    mm_per_u = float(target_width_mm) / width_u
    return width_u * mm_per_u, height_u * mm_per_u


def _line_tokens(line: str) -> dict[str, float]:
    """
    Parse compact G-code values like X1.23 Y-4.5 F700.
    """
    out: dict[str, float] = {}
    for k, v in re.findall(r"([XYZIJF])\s*([-+]?\d*\.?\d+)", line.upper()):
        out[k] = float(v)
    return out


def estimate_time_from_gcode(gcode_text: str, rapid_feed_mm_min: float) -> tuple[float, dict[str, int]]:
    """
    Rough cycle-time estimate based on command lengths and feed values.
    """
    x = 0.0
    y = 0.0
    z = 0.0
    feed = 1000.0
    total_min = 0.0
    counts = {"G0": 0, "G1": 0, "G1_XY": 0, "G2": 0, "G3": 0}

    for raw in gcode_text.splitlines():
        line = raw.strip().upper()
        if not line or line.startswith(";"):
            continue

        cmd = None
        for c in ("G0", "G1", "G2", "G3"):
            if re.match(rf"^{c}\b", line):
                cmd = c
                break
        if cmd is None:
            continue
        counts[cmd] += 1

        tok = _line_tokens(line)
        if "F" in tok:
            feed = max(tok["F"], 1.0)
        has_xy = ("X" in tok) or ("Y" in tok)
        if cmd == "G1" and has_xy:
            counts["G1_XY"] += 1

        nx = tok.get("X", x)
        ny = tok.get("Y", y)
        nz = tok.get("Z", z)

        if cmd in ("G0", "G1"):
            dist = math.sqrt((nx - x) ** 2 + (ny - y) ** 2 + (nz - z) ** 2)
            use_feed = rapid_feed_mm_min if cmd == "G0" else feed
            total_min += dist / max(use_feed, 1.0)
        else:
            # Arc in XY plane with center offsets I/J from start point.
            i = tok.get("I", 0.0)
            j = tok.get("J", 0.0)
            cx = x + i
            cy = y + j
            r = math.hypot(x - cx, y - cy)
            if r < 1.0e-9:
                dist = math.hypot(nx - x, ny - y)
            else:
                a0 = math.atan2(y - cy, x - cx)
                a1 = math.atan2(ny - cy, nx - cx)
                da = a1 - a0
                if cmd == "G2":  # clockwise
                    if da >= 0.0:
                        da -= 2.0 * math.pi
                else:  # G3 counter-clockwise
                    if da <= 0.0:
                        da += 2.0 * math.pi
                dist = abs(da) * r
            total_min += dist / max(feed, 1.0)

        x, y, z = nx, ny, nz

    return total_min, counts


def write_markdown_report(
    report_path: Path,
    cases: list[CaseSummary],
    config: dict[str, float | int | str],
) -> None:
    """
    Human-readable report for thesis appendix + CAM checklist.
    """
    lines: list[str] = []
    lines.append("# CNC Validation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("- Goal: validate direction, dimensions, continuity and machining time before physical CNC run.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in config.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Model | Mode | Edges | Arcs | DS | Size (mm) | Paths | Segments | G0 | G1 | G1-XY | G2 | G3 | Cut Len (mm) | Est. Time (min) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        size = f"{c.width_mm:.2f} x {c.height_mm:.2f}"
        lines.append(
            f"| `{c.model}` | `{c.mode}` | {c.edges} | {c.arcs} | x{c.downsample_stride} | {size} | "
            f"{c.path_count} | {c.segment_count} | {c.cmd_g0} | {c.cmd_g1} | {c.cmd_g1_xy} | {c.cmd_g2} | {c.cmd_g3} | "
            f"{c.cut_length_mm:.2f} | {c.est_total_time_min:.2f} |"
        )
    lines.append("")
    lines.append("## CAM Checklist")
    lines.append("")
    lines.append("1. Import each `.nc` in CAM and verify orientation (XY, Y inversion, clockwise/counter-clockwise arcs).")
    lines.append("2. Check final dimensions against report `Size (mm)`.")
    lines.append("3. Verify continuity at arc junctions (no unexpected jumps or retracts).")
    lines.append("4. Confirm no out-of-bounds travel or collisions in machine envelope.")
    lines.append("5. Compare CAM machining time with report estimate; document delta.")
    lines.append("6. For `semi`: confirm CAM preserves G2/G3 arcs.")
    lines.append("7. For `semi`: `G1-XY` should stay very low compared to G2/G3 (mostly plunges/repositions).")
    lines.append("8. For `elliptic`: confirm CAM handles segmented G1 toolpath cleanly.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Fill this section after CAM simulation with findings and corrections.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> int:
    args = parse_args()
    ensure_dependencies()

    models = resolve_models(args.models)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[CaseSummary] = []
    for stl in models:
        for mode in ("semi", "elliptic"):
            arcs, edge_count = build_arcs_for_model(
                stl_path=stl,
                mode=mode,
                line_resolution=float(args.line_resolution),
                min_arc_radius=float(args.min_arc_radius),
                ellipse_ratio=float(args.ellipse_ratio),
            )
            downsample_stride = 1
            if args.max_arcs_per_case > 0 and len(arcs) > int(args.max_arcs_per_case):
                downsample_stride = int(math.ceil(len(arcs) / float(args.max_arcs_per_case)))
                arcs = arcs[::downsample_stride]

            stem = stl.stem
            suffix = "semi" if mode == "semi" else "elliptic"
            svg_path = out_dir / f"{stem}__{suffix}.svg"
            nc_path = out_dir / f"{stem}__{suffix}.nc"

            write_svg(svg_path, arcs, stroke_width=float(args.stroke_width))
            path_count, segment_count, cut_len_mm = write_gcode(
                gcode_path=nc_path,
                arcs=arcs,
                target_width_mm=float(args.target_width_mm),
                safe_z_mm=float(args.safe_z_mm),
                cut_z_mm=float(args.cut_z_mm),
                feed_xy_mm_min=float(args.feed_xy_mm_min),
                feed_z_mm_min=float(args.feed_z_mm_min),
                max_segment_mm=float(args.max_segment_mm),
                spindle_rpm=int(args.spindle_rpm),
                invert_y=bool(args.invert_y),
                arc_mode=mode,
            )

            gcode_text = nc_path.read_text(encoding="utf-8")
            est_min, counts = estimate_time_from_gcode(
                gcode_text=gcode_text,
                rapid_feed_mm_min=float(args.rapid_feed_mm_min),
            )
            width_mm, height_mm = arc_bbox_mm(arcs, target_width_mm=float(args.target_width_mm))

            cases.append(
                CaseSummary(
                    model=stl.name,
                    mode=mode,
                    arcs=len(arcs),
                    edges=edge_count,
                    svg_file=svg_path.name,
                    gcode_file=nc_path.name,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    path_count=path_count,
                    segment_count=segment_count,
                    cut_length_mm=float(cut_len_mm),
                    est_total_time_min=float(est_min),
                    cmd_g0=int(counts["G0"]),
                    cmd_g1=int(counts["G1"]),
                    cmd_g1_xy=int(counts["G1_XY"]),
                    cmd_g2=int(counts["G2"]),
                    cmd_g3=int(counts["G3"]),
                    downsample_stride=downsample_stride,
                )
            )

    report_md = out_dir / "validation_report.md"
    report_json = out_dir / "validation_report.json"

    config = {
        "line_resolution": args.line_resolution,
        "min_arc_radius": args.min_arc_radius,
        "ellipse_ratio": args.ellipse_ratio,
        "target_width_mm": args.target_width_mm,
        "safe_z_mm": args.safe_z_mm,
        "cut_z_mm": args.cut_z_mm,
        "feed_xy_mm_min": args.feed_xy_mm_min,
        "feed_z_mm_min": args.feed_z_mm_min,
        "max_segment_mm": args.max_segment_mm,
        "spindle_rpm": args.spindle_rpm,
        "invert_y": args.invert_y,
        "rapid_feed_mm_min": args.rapid_feed_mm_min,
        "max_arcs_per_case": args.max_arcs_per_case,
    }

    write_markdown_report(report_md, cases, config=config)
    report_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config": config,
                "cases": [asdict(c) for c in cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Validation pack saved in: {out_dir}")
    print(f"[OK] Markdown report: {report_md.name}")
    print(f"[OK] JSON report: {report_json.name}")
    for c in cases:
        print(
            f" - {c.model} | {c.mode:8s} | arcs={c.arcs:5d} | "
            f"G2/G3={c.cmd_g2+c.cmd_g3:4d} | G1-XY={c.cmd_g1_xy:5d} | "
            f"ds=x{c.downsample_stride:2d} | "
            f"est={c.est_total_time_min:6.2f} min"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
