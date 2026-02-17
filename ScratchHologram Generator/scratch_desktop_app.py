#!/usr/bin/env python3
"""
Scratch Hologram Desktop App (single view, performance-oriented).

What this app does:
- Load STL.
- Orbit camera (drag + wheel).
- Show arc preview in one canvas.
- Reintroduce View Angle simulation overlay (profile lines).
- Export SVG from current view.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

from scratch_pipeline import (
    EPS,
    NORMAL_TOLERANCE_DECIMALS,
    Arc,
    CameraConfig,
    Edge,
    build_model_to_window_matrix,
    build_model_vertices_and_edges,
    ensure_dependencies,
    generate_arcs,
    get_edge_points,
    load_mesh,
    model_to_window,
    write_svg,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_at_angle_projected(
    p: np.ndarray | tuple[float, float, float],
    angle_deg: float,
    n_view: float,
    ellipse_ratio: float = 1.0,
) -> tuple[float, float]:
    dist = float(p[2]) - n_view
    cx = float(p[0])
    cy = float(p[1]) - dist / 2.0
    r = abs(dist) / 2.0
    if r < EPS:
        return cx, cy

    # Match preview arc parameterization: angle [-90,+90] -> theta on [start, start+180].
    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    start_deg = 0.0 if dist > 0.0 else 180.0
    theta = math.radians(start_deg + (u * 180.0))
    ry = r * clamp(float(ellipse_ratio), 0.20, 1.00)
    x = cx + r * math.cos(theta)
    y = cy - ry * math.sin(theta)
    return x, y


def point_on_arc_compatible(arc: Arc, angle_deg: float) -> tuple[float, float]:
    """
    Compute a point on the rendered arc using the same angular parameterization
    that proved visually consistent in preview:
    u in [0,1] -> theta in [start_angle, start_angle + sweep_angle].
    """
    rx = max(0.1, float(arc.rect_w) / 2.0)
    ry = max(0.1, float(arc.rect_h) / 2.0)
    cx = float(arc.rect_x) + (float(arc.rect_w) / 2.0)
    cy = float(arc.rect_y) + (float(arc.rect_h) / 2.0)

    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    theta_deg = arc.start_angle + (u * arc.sweep_angle)
    theta = math.radians(theta_deg)

    x = cx + rx * math.cos(theta)
    y = cy - ry * math.sin(theta)
    return x, y


def sample_arc_polyline(arc: Arc, max_segment_units: float) -> list[tuple[float, float]]:
    rx = max(float(arc.rect_w) / 2.0, 0.5)
    ry = max(float(arc.rect_h) / 2.0, 0.5)
    cx = float(arc.rect_x) + (float(arc.rect_w) / 2.0)
    cy = float(arc.rect_y) + (float(arc.rect_h) / 2.0)

    sweep = float(arc.sweep_angle)
    if abs(sweep) < EPS:
        sweep = 180.0

    mean_radius = math.sqrt(((rx * rx) + (ry * ry)) / 2.0)
    arc_len = abs(math.radians(sweep)) * mean_radius
    seg_len = max(float(max_segment_units), 0.05)
    seg_count = max(8, int(math.ceil(arc_len / seg_len)))

    points: list[tuple[float, float]] = []
    for i in range(seg_count + 1):
        t = i / seg_count
        theta_deg = float(arc.start_angle) + (sweep * t)
        theta = math.radians(theta_deg)
        x = cx + rx * math.cos(theta)
        # Match Tk canvas arc orientation used in preview.
        y = cy - ry * math.sin(theta)
        points.append((x, y))
    return points


def write_gcode(
    gcode_path: Path,
    arcs: list[Arc],
    target_width_mm: float,
    safe_z_mm: float,
    cut_z_mm: float,
    feed_xy_mm_min: float,
    feed_z_mm_min: float,
    max_segment_mm: float,
    spindle_rpm: int,
    invert_y: bool,
    arc_mode: str,
) -> tuple[int, int, float]:
    if len(arcs) == 0:
        raise ValueError("Nessun arco disponibile per esportare G-code.")
    if target_width_mm <= 0.0:
        raise ValueError("target_width_mm deve essere > 0.")
    if feed_xy_mm_min <= 0.0 or feed_z_mm_min <= 0.0:
        raise ValueError("I feedrate devono essere > 0.")
    if max_segment_mm <= 0.0:
        raise ValueError("max_segment_mm deve essere > 0.")

    min_x_u = min(float(a.rect_x) for a in arcs)
    max_x_u = max(float(a.rect_x + a.rect_w) for a in arcs)
    width_u = max(max_x_u - min_x_u, EPS)
    mm_per_unit = target_width_mm / width_u

    max_segment_units = max_segment_mm / mm_per_unit
    raw_paths = [sample_arc_polyline(arc, max_segment_units=max_segment_units) for arc in arcs]

    all_points = [pt for path in raw_paths for pt in path]
    if len(all_points) == 0:
        raise ValueError("Campionamento archi vuoto.")

    y_max_u = max(p[1] for p in all_points)
    scaled_paths: list[list[tuple[float, float]]] = []
    for path in raw_paths:
        scaled_path: list[tuple[float, float]] = []
        for x_u, y_u in path:
            y_src = (y_max_u - y_u) if invert_y else y_u
            x_mm = x_u * mm_per_unit
            y_mm = y_src * mm_per_unit
            scaled_path.append((x_mm, y_mm))
        if len(scaled_path) >= 2:
            scaled_paths.append(scaled_path)

    all_scaled = [pt for path in scaled_paths for pt in path]
    min_x_mm = min(p[0] for p in all_scaled)
    min_y_mm = min(p[1] for p in all_scaled)

    translated_paths: list[list[tuple[float, float]]] = []
    for path in scaled_paths:
        out_path: list[tuple[float, float]] = []
        for x_mm, y_mm in path:
            p = (x_mm - min_x_mm, y_mm - min_y_mm)
            out_path.append(p)
        translated_paths.append(out_path)

    def to_mm_xy(x_u: float, y_u: float) -> tuple[float, float]:
        y_src = (y_max_u - y_u) if invert_y else y_u
        return (x_u * mm_per_unit) - min_x_mm, (y_src * mm_per_unit) - min_y_mm

    gcode_lines = [
        "; ScratchHologram Generator - G-code",
        "G21 ; millimeters",
        "G90 ; absolute coordinates",
        "G17 ; XY plane",
        "G94 ; units/min feed",
        f"G0 Z{safe_z_mm:.4f}",
    ]
    if spindle_rpm > 0:
        gcode_lines.append(f"M3 S{int(spindle_rpm)}")

    mode = str(arc_mode).strip().lower()
    use_circular_commands = mode == "semi"

    segment_count = 0
    path_count = 0
    total_cut_length = 0.0
    for arc, path in zip(arcs, translated_paths):
        if len(path) < 2:
            continue

        rx_u = max(float(arc.rect_w) / 2.0, 0.0)
        ry_u = max(float(arc.rect_h) / 2.0, 0.0)
        near_circle = abs(rx_u - ry_u) <= max(1.0e-6, 1.0e-3 * max(rx_u, ry_u, 1.0))

        if use_circular_commands and near_circle:
            cx_u = float(arc.rect_x) + rx_u
            cy_u = float(arc.rect_y) + ry_u
            start_rad = math.radians(float(arc.start_angle))
            end_rad = math.radians(float(arc.start_angle) + float(arc.sweep_angle))

            sx_u = cx_u + rx_u * math.cos(start_rad)
            sy_u = cy_u - ry_u * math.sin(start_rad)
            ex_u = cx_u + rx_u * math.cos(end_rad)
            ey_u = cy_u - ry_u * math.sin(end_rad)

            sx, sy = to_mm_xy(sx_u, sy_u)
            ex, ey = to_mm_xy(ex_u, ey_u)
            cx, cy = to_mm_xy(cx_u, cy_u)

            ccw = float(arc.sweep_angle) >= 0.0
            if not invert_y:
                ccw = not ccw
            cmd = "G3" if ccw else "G2"

            gcode_lines.append(f"G0 X{sx:.4f} Y{sy:.4f}")
            gcode_lines.append(f"G1 Z{cut_z_mm:.4f} F{feed_z_mm_min:.2f}")
            gcode_lines.append(f"{cmd} X{ex:.4f} Y{ey:.4f} I{(cx - sx):.4f} J{(cy - sy):.4f} F{feed_xy_mm_min:.2f}")
            gcode_lines.append(f"G0 Z{safe_z_mm:.4f}")

            total_cut_length += abs(math.radians(float(arc.sweep_angle))) * (rx_u * mm_per_unit)
            segment_count += 1
            path_count += 1
            continue

        sx, sy = path[0]
        gcode_lines.append(f"G0 X{sx:.4f} Y{sy:.4f}")
        gcode_lines.append(f"G1 Z{cut_z_mm:.4f} F{feed_z_mm_min:.2f}")
        gcode_lines.append(f"G1 X{sx:.4f} Y{sy:.4f} F{feed_xy_mm_min:.2f}")
        prev = path[0]
        for x, y in path[1:]:
            gcode_lines.append(f"G1 X{x:.4f} Y{y:.4f}")
            total_cut_length += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            segment_count += 1
        gcode_lines.append(f"G0 Z{safe_z_mm:.4f}")
        path_count += 1

    if spindle_rpm > 0:
        gcode_lines.append("M5")
    gcode_lines.extend(
        [
            "G0 X0.0000 Y0.0000",
            f"G0 Z{safe_z_mm:.4f}",
            "M2",
            "",
        ]
    )

    gcode_path.parent.mkdir(parents=True, exist_ok=True)
    gcode_path.write_text("\n".join(gcode_lines), encoding="utf-8")
    return path_count, segment_count, total_cut_length


class ScratchDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ScratchHologram Generator Desktop")
        self.root.geometry("1500x940")

        self.stl_path: Path | None = None
        self.model_vertices: np.ndarray | None = None
        self.edges: list[Edge] = []
        self.faces: np.ndarray | None = None
        self.edge_lengths: np.ndarray | None = None
        self.total_edge_length: float = 0.0

        self.current_arcs: list[Arc] = []
        self.profile_paths: list[list[Arc]] = []
        self.last_sampled_edges: list[Edge] = []
        self.last_projected: np.ndarray | None = None
        self.last_n_view: float = 0.0
        self.last_edge_stride: int = 1
        self.last_preview_lr: float = 0.0
        self.last_mode: str = "IDLE"

        self.model_center = np.zeros(3, dtype=np.float64)
        self.base_orbit_radius = 25.0
        self.distance_scale = 1.0
        self.yaw_deg = -35.0
        self.pitch_deg = 20.0

        self.dragging = False
        self.drag_last_x = 0
        self.drag_last_y = 0
        self.pending_recompute: str | None = None

        self.line_resolution = tk.DoubleVar(value=3.28)
        self.min_arc_radius = tk.DoubleVar(value=6.0)
        self.stroke_width = tk.DoubleVar(value=0.15)
        self.current_scale = tk.DoubleVar(value=0.724)
        self.zf = tk.DoubleVar(value=25.0)
        self.arc_limit = tk.IntVar(value=6000)
        self.preview_quality = tk.IntVar(value=55)
        self.view_angle = tk.DoubleVar(value=0.0)
        self.arc_mode = tk.StringVar(value="Semicircle (CNC)")
        self.ellipse_ratio = tk.DoubleVar(value=0.65)
        self.show_arcs = tk.BooleanVar(value=True)
        self.show_profile = tk.BooleanVar(value=True)
        self.rigid_profile = tk.BooleanVar(value=True)
        self.auto_center = tk.BooleanVar(value=True)
        self.export_occlusion_cull = tk.BooleanVar(value=False)
        self.export_cull_strength = tk.IntVar(value=45)
        self.export_use_preview = tk.BooleanVar(value=True)

        self.gcode_target_width_mm = 10.0
        self.gcode_safe_z_mm = 3.0
        self.gcode_cut_z_mm = -0.08
        self.gcode_feed_xy_mm_min = 700.0
        self.gcode_feed_z_mm_min = 220.0
        self.gcode_max_segment_mm = 0.20
        self.gcode_spindle_rpm = 12000
        self.gcode_invert_y = True

        self.status_var = tk.StringVar(value="Apri un file STL per iniziare.")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Apri STL", command=self.open_stl_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Preview GPU", command=self.open_gpu_preview).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Esporta SVG", command=self.export_svg_dialog).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Esporta G-code", command=self.export_gcode_dialog).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Ricalcola", command=lambda: self.recompute_and_redraw(interactive=False)).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.path_label = ttk.Label(top, text="Nessun file caricato", width=105)
        self.path_label.pack(side=tk.LEFT, padx=(12, 0))

        frame = ttk.LabelFrame(self.root, text="Preview (arcs + simulated profile)", padding=4)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(frame, bg="#0a0c12", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda _: self.schedule_recompute(delay_ms=60))

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        self._make_slider(
            controls,
            row=0,
            col=0,
            label="Line resolution",
            variable=self.line_resolution,
            min_val=0.2,
            max_val=12.0,
            fmt="{:.2f}",
        )
        self._make_slider(
            controls,
            row=0,
            col=1,
            label="Min arc radius",
            variable=self.min_arc_radius,
            min_val=0.0,
            max_val=25.0,
            fmt="{:.1f}",
        )
        self._make_slider(
            controls,
            row=0,
            col=2,
            label="Stroke width",
            variable=self.stroke_width,
            min_val=0.05,
            max_val=1.00,
            fmt="{:.2f}",
            redraw_only=True,
        )
        self._make_slider(
            controls,
            row=1,
            col=0,
            label="Camera scale",
            variable=self.current_scale,
            min_val=0.2,
            max_val=2.0,
            fmt="{:.3f}",
        )
        self._make_slider(
            controls,
            row=1,
            col=1,
            label="Perspective zf",
            variable=self.zf,
            min_val=5.0,
            max_val=100.0,
            fmt="{:.1f}",
        )
        self._make_slider(
            controls,
            row=1,
            col=2,
            label="Preview arc limit",
            variable=self.arc_limit,
            min_val=200,
            max_val=20000,
            fmt="{:.0f}",
            integer=True,
            redraw_only=True,
        )
        self._make_slider(
            controls,
            row=2,
            col=0,
            label="Preview quality",
            variable=self.preview_quality,
            min_val=5,
            max_val=100,
            fmt="{:.0f}%",
            integer=True,
        )
        self._make_slider(
            controls,
            row=2,
            col=1,
            label="View angle",
            variable=self.view_angle,
            min_val=-90.0,
            max_val=90.0,
            fmt="{:.0f} deg",
            redraw_only=True,
        )
        self._make_dropdown(
            controls,
            row=5,
            col=0,
            label="Arc mode",
            variable=self.arc_mode,
            values=("Semicircle (CNC)", "Elliptic"),
        )
        self._make_slider(
            controls,
            row=5,
            col=2,
            label="Ellipse ratio",
            variable=self.ellipse_ratio,
            min_val=0.20,
            max_val=1.00,
            fmt="{:.2f}",
        )
        ttk.Checkbutton(
            controls,
            text="Show arcs",
            variable=self.show_arcs,
            command=self.redraw_from_cache,
        ).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Show simulated profile",
            variable=self.show_profile,
            command=self.redraw_from_cache,
        ).grid(row=3, column=1, padx=8, pady=4, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Auto-center Z",
            variable=self.auto_center,
            command=self.on_auto_center_toggle,
        ).grid(row=3, column=2, padx=8, pady=4, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Cull hidden arcs on export",
            variable=self.export_occlusion_cull,
        ).grid(row=4, column=0, padx=8, pady=2, sticky="w")
        self._make_slider(
            controls,
            row=4,
            col=1,
            label="Cull strength",
            variable=self.export_cull_strength,
            min_val=0,
            max_val=100,
            fmt="{:.0f}%",
            integer=True,
            redraw_only=True,
        )
        ttk.Checkbutton(
            controls,
            text="Export exactly preview",
            variable=self.export_use_preview,
        ).grid(row=4, column=2, padx=8, pady=2, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Rigid simulation",
            variable=self.rigid_profile,
            command=self.redraw_from_cache,
        ).grid(row=5, column=1, padx=8, pady=2, sticky="w")

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(10, 2), anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _make_slider(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        variable: tk.DoubleVar | tk.IntVar,
        min_val: float,
        max_val: float,
        fmt: str,
        integer: bool = False,
        redraw_only: bool = False,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=8, pady=4, sticky="ew")
        parent.columnconfigure(col, weight=1)

        ttk.Label(frame, text=label).pack(anchor="w")
        value_label = ttk.Label(frame, width=10)
        value_label.pack(side=tk.RIGHT)

        def on_change(_: str) -> None:
            if integer:
                value = float(int(round(float(variable.get()))))
                variable.set(int(value))
            else:
                value = float(variable.get())
            value_label.config(text=fmt.format(value))
            if redraw_only:
                self.redraw_from_cache()
            else:
                self.schedule_recompute(delay_ms=100)

        value = float(variable.get())
        value_label.config(text=fmt.format(value))

        scale = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=on_change,
        )
        scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 8))

    def _make_dropdown(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        redraw_only: bool = False,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=8, pady=4, sticky="ew")
        parent.columnconfigure(col, weight=1)

        ttk.Label(frame, text=label).pack(anchor="w")
        combo = ttk.Combobox(
            frame,
            textvariable=variable,
            values=values,
            state="readonly",
        )
        combo.pack(fill=tk.X, expand=True)

        def on_change(_: tk.Event) -> None:
            if redraw_only:
                self.redraw_from_cache()
            else:
                self.schedule_recompute(delay_ms=100)

        combo.bind("<<ComboboxSelected>>", on_change)

    def schedule_recompute(self, delay_ms: int = 120) -> None:
        if self.pending_recompute is not None:
            self.root.after_cancel(self.pending_recompute)
        self.pending_recompute = self.root.after(delay_ms, lambda: self.recompute_and_redraw(interactive=False))

    def _flush_pending_recompute(self) -> None:
        if self.pending_recompute is None:
            return
        try:
            self.root.after_cancel(self.pending_recompute)
        except Exception:
            pass
        self.pending_recompute = None
        self.recompute_and_redraw(interactive=False)

    def on_auto_center_toggle(self) -> None:
        if self.stl_path is not None:
            self.load_stl(self.stl_path)

    def open_stl_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Apri STL",
            initialdir=str(Path.cwd()),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if filename:
            self.load_stl(Path(filename))

    def open_gpu_preview(self) -> None:
        script = Path(__file__).resolve().parent / "scratch_gpu_preview.py"
        if not script.exists():
            messagebox.showerror("GPU preview", f"File mancante: {script}")
            return

        cmd = [sys.executable, str(script)]
        if self.stl_path is not None:
            cmd.extend(["--stl", str(self.stl_path)])

        try:
            subprocess.Popen(cmd, cwd=str(script.parent))
        except Exception as exc:
            messagebox.showerror("GPU preview", f"Errore avvio preview GPU:\n{exc}")

    def load_stl(self, stl_path: Path) -> None:
        try:
            ensure_dependencies()
            mesh = load_mesh(stl_path)
            model_vertices, edges, faces = build_model_vertices_and_edges(
                mesh=mesh,
                auto_center=bool(self.auto_center.get()),
                decimals=NORMAL_TOLERANCE_DECIMALS,
            )
        except Exception as exc:
            messagebox.showerror("Errore caricamento STL", str(exc))
            return

        self.stl_path = stl_path
        self.model_vertices = model_vertices
        self.edges = edges
        self.faces = faces
        self.current_arcs = []

        v_min = np.min(model_vertices, axis=0)
        v_max = np.max(model_vertices, axis=0)
        self.model_center = (v_min + v_max) / 2.0
        diag = float(np.linalg.norm(v_max - v_min))
        self.base_orbit_radius = max(10.0, diag * 2.6)
        self.distance_scale = 1.0
        self.gcode_target_width_mm = max(1.0, float(max(v_max[0] - v_min[0], v_max[1] - v_min[1])))

        lengths = []
        for edge in self.edges:
            p0 = model_vertices[edge.start_idx]
            p1 = model_vertices[edge.end_idx]
            lengths.append(float(np.linalg.norm(p1 - p0)))
        self.edge_lengths = np.asarray(lengths, dtype=np.float64) if lengths else np.zeros((0,), dtype=np.float64)
        self.total_edge_length = float(np.sum(self.edge_lengths)) if len(lengths) > 0 else 0.0

        self.path_label.config(text=str(stl_path))
        self.recompute_and_redraw(interactive=False)

    def on_mouse_down(self, event: tk.Event) -> None:
        self.dragging = True
        self.drag_last_x = int(event.x)
        self.drag_last_y = int(event.y)

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.model_vertices is None or not self.dragging:
            return
        x = int(event.x)
        y = int(event.y)
        dx = x - self.drag_last_x
        dy = y - self.drag_last_y
        self.drag_last_x = x
        self.drag_last_y = y

        self.yaw_deg += dx * 0.45
        self.pitch_deg = clamp(self.pitch_deg - dy * 0.35, -89.0, 89.0)

        # Fast render while dragging: no arc recomputation.
        self.recompute_and_redraw(interactive=True)
        self.schedule_recompute(delay_ms=180)

    def on_mouse_up(self, _: tk.Event) -> None:
        self.dragging = False
        self.recompute_and_redraw(interactive=False)

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if self.model_vertices is None:
            return
        if int(event.delta) > 0:
            self.distance_scale *= 0.90
        else:
            self.distance_scale *= 1.10
        self.distance_scale = clamp(self.distance_scale, 0.35, 4.5)
        self.recompute_and_redraw(interactive=True)
        self.schedule_recompute(delay_ms=150)

    def _build_camera(self, canvas_width: int, canvas_height: int) -> CameraConfig:
        if self.model_vertices is None:
            return CameraConfig(
                po=(0.9, 9.0, 24.0),
                pr=(0.93, 8.29, 0.47),
                look_up=(0.0, 1.0, 0.0),
                current_scale=float(self.current_scale.get()),
                zf=float(self.zf.get()),
                canvas_width=max(1, int(canvas_width)),
                canvas_height=max(1, int(canvas_height)),
            )

        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
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

        orbit_radius = self.base_orbit_radius * self.distance_scale
        pr = np.asarray(self.model_center, dtype=np.float64)
        po = pr + direction * orbit_radius

        look_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(direction, look_up))) > 0.98:
            look_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

        return CameraConfig(
            po=(float(po[0]), float(po[1]), float(po[2])),
            pr=(float(pr[0]), float(pr[1]), float(pr[2])),
            look_up=(float(look_up[0]), float(look_up[1]), float(look_up[2])),
            current_scale=float(self.current_scale.get()),
            zf=float(self.zf.get()),
            canvas_width=max(1, int(canvas_width)),
            canvas_height=max(1, int(canvas_height)),
        )

    def _preview_params(self, interactive: bool, full_quality: bool) -> tuple[list[Edge], float, int, int]:
        if self.model_vertices is None or len(self.edges) == 0:
            return [], float(self.line_resolution.get()), 1, 0

        base_lr = float(self.line_resolution.get())
        base_limit = int(self.arc_limit.get())
        quality_raw = clamp(float(self.preview_quality.get()) / 100.0, 0.05, 1.0)
        quality = quality_raw * quality_raw

        if full_quality:
            q = 1.0
        else:
            q = quality
            if interactive:
                q *= 0.15

        edge_count = len(self.edges)
        target_edges = max(4, int(edge_count * q))
        stride = max(1, int(math.ceil(edge_count / target_edges)))
        sampled_edges = self.edges[::stride]

        effective_lr = base_lr * q
        effective_lr = clamp(effective_lr, 0.08, base_lr)

        if full_quality:
            draw_limit = base_limit
        else:
            draw_limit = max(100, int(base_limit * q))
            if interactive:
                draw_limit = min(draw_limit, 1200)

        return sampled_edges, effective_lr, stride, draw_limit

    @staticmethod
    def _coord_key(p: np.ndarray | tuple[float, float, float], decimals: int = 6) -> str:
        return (
            f"{float(p[0]):.{decimals}f}:"
            f"{float(p[1]):.{decimals}f}:"
            f"{float(p[2]):.{decimals}f}"
        )

    def _current_arc_mode(self) -> str:
        if str(self.arc_mode.get()).strip().lower().startswith("elliptic"):
            return "elliptic"
        return "semi"

    def _current_ellipse_ratio(self) -> float:
        return clamp(float(self.ellipse_ratio.get()), 0.20, 1.00)

    def _build_profile_paths_from_arcs(
        self,
        zero_vertices: np.ndarray,
        sampled_edges: list[Edge],
        effective_lr: float,
    ) -> list[list[Arc]]:
        if len(self.current_arcs) == 0:
            return []
        if self.model_vertices is None:
            return []

        arc_by_coord: dict[str, Arc] = {}
        for arc in self.current_arcs:
            key = self._coord_key(arc.zero_coord, decimals=6)
            arc_by_coord[key] = arc

        paths: list[list[Arc]] = []
        for edge in sampled_edges:
            model_start = self.model_vertices[edge.start_idx]
            model_end = self.model_vertices[edge.end_idx]
            zero_start = zero_vertices[edge.start_idx]
            zero_end = zero_vertices[edge.end_idx]

            points = get_edge_points(
                model_start=model_start,
                model_end=model_end,
                zero_start=zero_start,
                zero_end=zero_end,
                view_points_per_unit_length=effective_lr,
            )

            arc_path: list[Arc] = []
            prev_hash: tuple[float, float, float, float, float] | None = None
            for p in points:
                key = self._coord_key(p, decimals=6)
                arc = arc_by_coord.get(key)
                if arc is None:
                    continue

                # Collapse consecutive duplicates (same rendered arc geometry).
                arc_hash = (
                    round(float(arc.rect_x), 4),
                    round(float(arc.rect_y), 4),
                    round(float(arc.rect_w), 4),
                    round(float(arc.rect_h), 4),
                    arc.start_angle,
                )
                if prev_hash is not None and arc_hash == prev_hash:
                    continue
                prev_hash = arc_hash
                arc_path.append(arc)

            if len(arc_path) >= 2:
                paths.append(arc_path)

        return paths

    def recompute_and_redraw(self, interactive: bool) -> None:
        self.pending_recompute = None
        self.canvas.delete("all")

        if self.model_vertices is None:
            self.canvas.create_text(
                20,
                20,
                text="Apri un file STL",
                anchor="nw",
                fill="#d0d0d0",
                font=("Segoe UI", 14),
            )
            return

        w = max(20, int(self.canvas.winfo_width()))
        h = max(20, int(self.canvas.winfo_height()))
        camera = self._build_camera(w, h)

        try:
            mtx = build_model_to_window_matrix(camera)
            projected = model_to_window(self.model_vertices, mtx)
            pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), mtx)[0]
            n_view = float(pr_view[2])
            self.last_projected = projected
            self.last_n_view = n_view
        except Exception as exc:
            self.status_var.set(f"Errore proiezione: {exc}")
            return

        sampled_edges, effective_lr, stride, draw_limit = self._preview_params(
            interactive=interactive,
            full_quality=False,
        )
        self.last_sampled_edges = sampled_edges
        self.last_edge_stride = stride
        self.last_preview_lr = effective_lr
        self.last_mode = "FAST" if interactive else "PREVIEW"

        try:
            zero_vertices = projected
            arcs = generate_arcs(
                model_vertices=self.model_vertices,
                zero_vertices=zero_vertices,
                edges=sampled_edges,
                view_points_per_unit_length=effective_lr,
                n_view=n_view,
                dedupe_decimals=6,
                min_arc_radius=float(self.min_arc_radius.get()),
                arc_mode=self._current_arc_mode(),
                ellipse_ratio=self._current_ellipse_ratio(),
            )
            self.current_arcs = arcs
            self.profile_paths = self._build_profile_paths_from_arcs(
                zero_vertices=projected,
                sampled_edges=sampled_edges,
                effective_lr=effective_lr,
            )
        except Exception as exc:
            self.status_var.set(f"Errore calcolo archi: {exc}")
            return

        self._draw_scene(draw_limit=draw_limit, interactive=interactive)

    def _draw_scene(self, draw_limit: int, interactive: bool) -> None:
        if self.last_projected is None or self.model_vertices is None:
            return

        stroke_preview = max(1.0, float(self.stroke_width.get()) * 4.0)

        if self.show_arcs.get():
            if interactive:
                shown_arcs = min(len(self.current_arcs), min(draw_limit, 800))
            else:
                shown_arcs = min(len(self.current_arcs), draw_limit)

            for arc in self.current_arcs[:shown_arcs]:
                x0 = arc.rect_x
                y0 = arc.rect_y
                x1 = arc.rect_x + arc.rect_w
                y1 = arc.rect_y + arc.rect_h
                self.canvas.create_arc(
                    x0,
                    y0,
                    x1,
                    y1,
                    start=arc.start_angle,
                    extent=arc.sweep_angle,
                    style=tk.ARC,
                    outline="#dedede",
                    width=stroke_preview,
                )
        else:
            shown_arcs = 0

        if self.show_profile.get():
            angle = float(self.view_angle.get())
            rigid = bool(self.rigid_profile.get())
            profile_ratio = self._current_ellipse_ratio() if self._current_arc_mode() == "elliptic" else 1.0

            if interactive:
                path_stride = 2
                point_stride = 2
            else:
                path_stride = 1
                point_stride = 1

            if rigid and self.last_projected is not None:
                edges_to_draw = self.last_sampled_edges[::path_stride] if self.last_sampled_edges else self.edges[::path_stride]
                for edge in edges_to_draw:
                    p1 = self.last_projected[edge.start_idx]
                    p2 = self.last_projected[edge.end_idx]
                    x1, y1 = point_at_angle_projected(p1, angle, self.last_n_view, ellipse_ratio=profile_ratio)
                    x2, y2 = point_at_angle_projected(p2, angle, self.last_n_view, ellipse_ratio=profile_ratio)
                    self.canvas.create_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#f4f4f4",
                        width=1.25,
                    )
            else:
                for path in self.profile_paths[::path_stride]:
                    arcs = path[::point_stride]
                    if len(arcs) < 2:
                        continue

                    coords: list[float] = []
                    for arc in arcs:
                        x, y = point_on_arc_compatible(arc, angle)
                        coords.append(x)
                        coords.append(y)

                    if len(coords) >= 4:
                        self.canvas.create_line(
                            *coords,
                            fill="#f4f4f4",
                            width=1.25,
                        )

        if self.stl_path is not None:
            arc_mode_name = self._current_arc_mode()
            if arc_mode_name == "elliptic":
                arc_mode_status = f"ELL({self._current_ellipse_ratio():.2f})"
            else:
                arc_mode_status = "SEMI"
            self.status_var.set(
                " | ".join(
                    [
                        f"STL: {self.stl_path.name}",
                        f"Spigoli: {len(self.edges)}",
                        f"Archi: {len(self.current_arcs)}",
                        f"Preview: {shown_arcs}",
                        f"Q: {int(self.preview_quality.get())}%",
                        f"LR eff: {self.last_preview_lr:.2f}",
                        f"Edge step: {self.last_edge_stride}",
                        f"Paths: {len(self.profile_paths)}",
                        f"Arc: {arc_mode_status}",
                        f"Sim: {'RIGID' if self.rigid_profile.get() else 'LEGACY'}",
                        f"Mode: {self.last_mode}",
                        f"Yaw/Pitch: {self.yaw_deg:.1f}/{self.pitch_deg:.1f}",
                    ]
                )
            )

    def redraw_from_cache(self) -> None:
        self.canvas.delete("all")
        if self.model_vertices is None:
            return
        _, _, _, draw_limit = self._preview_params(interactive=False, full_quality=False)
        self._draw_scene(draw_limit=draw_limit, interactive=self.dragging)

    def _compute_full_view_data(self) -> tuple[list[Arc], np.ndarray]:
        if self.model_vertices is None:
            raise ValueError("Nessun modello caricato.")

        w = max(20, int(self.canvas.winfo_width()))
        h = max(20, int(self.canvas.winfo_height()))
        camera = self._build_camera(w, h)

        mtx = build_model_to_window_matrix(camera)
        zero_vertices = model_to_window(self.model_vertices, mtx)
        pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), mtx)[0]
        n_view = float(pr_view[2])

        arcs_full = generate_arcs(
            model_vertices=self.model_vertices,
            zero_vertices=zero_vertices,
            edges=self.edges,
            view_points_per_unit_length=float(self.line_resolution.get()),
            n_view=n_view,
            dedupe_decimals=6,
            min_arc_radius=float(self.min_arc_radius.get()),
            arc_mode=self._current_arc_mode(),
            ellipse_ratio=self._current_ellipse_ratio(),
        )
        return arcs_full, zero_vertices

    def _compute_export_view_data(self) -> tuple[list[Arc], np.ndarray, str]:
        """
        Returns arcs + projected vertices for export and the source label:
        - "preview": export uses the same arc dataset currently previewed
        - "full": export recomputes full-resolution arcs
        """
        if self.model_vertices is None:
            raise ValueError("Nessun modello caricato.")

        # Ensure export uses the latest UI parameters (arc mode, ellipse ratio, etc.).
        self._flush_pending_recompute()

        if self.export_use_preview.get():
            if self.last_mode == "FAST":
                # Ensure export never uses temporary low-quality drag state.
                self.recompute_and_redraw(interactive=False)
            if self.last_projected is not None and len(self.current_arcs) > 0:
                return list(self.current_arcs), self.last_projected, "preview"

        arcs_full, projected_vertices = self._compute_full_view_data()
        return arcs_full, projected_vertices, "full"

    def _filter_occluded_arcs(self, arcs: list[Arc], projected_vertices: np.ndarray) -> list[Arc]:
        if len(arcs) == 0:
            return arcs
        if self.faces is None or len(self.faces) == 0:
            return arcs

        try:
            strength = clamp(float(self.export_cull_strength.get()) / 100.0, 0.0, 1.0)
        except Exception:
            strength = 1.0

        # Screen-space acceleration grid over projected triangles.
        cell = 24.0
        tri_data: list[tuple[float, ...]] = []
        grid: dict[tuple[int, int], list[int]] = {}

        for tri in self.faces:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            p0 = projected_vertices[i0]
            p1 = projected_vertices[i1]
            p2 = projected_vertices[i2]

            if not (
                np.isfinite(p0[0]) and np.isfinite(p0[1]) and np.isfinite(p0[2]) and
                np.isfinite(p1[0]) and np.isfinite(p1[1]) and np.isfinite(p1[2]) and
                np.isfinite(p2[0]) and np.isfinite(p2[1]) and np.isfinite(p2[2])
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
            tri_data.append((x0, y0, z0, x1, y1, z1, x2, y2, z2, denom, min_x, max_x, min_y, max_y))

            ix0 = int(math.floor(min_x / cell))
            ix1 = int(math.floor(max_x / cell))
            iy0 = int(math.floor(min_y / cell))
            iy1 = int(math.floor(max_y / cell))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = (ix, iy)
                    if key in grid:
                        grid[key].append(tri_idx)
                    else:
                        grid[key] = [tri_idx]

        if len(tri_data) == 0:
            return arcs

        z_vals = projected_vertices[:, 2]
        z_span = float(np.max(z_vals) - np.min(z_vals)) if projected_vertices.shape[0] > 0 else 1.0
        z_eps = max(0.003 * z_span, 0.45)
        inside_tol = 0.0

        kept: list[Arc] = []
        for arc in arcs:
            x = float(arc.zero_coord[0])
            y = float(arc.zero_coord[1])
            z = float(arc.zero_coord[2])
            ix = int(math.floor(x / cell))
            iy = int(math.floor(y / cell))

            candidate: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidate.extend(grid.get((ix + dx, iy + dy), []))

            if len(candidate) == 0:
                kept.append(arc)
                continue

            z_front = -1.0e30
            z_cover: list[float] = []
            seen: set[int] = set()
            for tri_idx in candidate:
                if tri_idx in seen:
                    continue
                seen.add(tri_idx)
                x0, y0, z0, x1, y1, z1, x2, y2, z2, denom, min_x, max_x, min_y, max_y = tri_data[tri_idx]
                if x < (min_x - 0.2) or x > (max_x + 0.2) or y < (min_y - 0.2) or y > (max_y + 0.2):
                    continue

                a = ((y1 - y2) * (x - x2)) + ((x2 - x1) * (y - y2))
                b = ((y2 - y0) * (x - x2)) + ((x0 - x2) * (y - y2))
                a /= denom
                b /= denom
                c = 1.0 - a - b
                if a < inside_tol or b < inside_tol or c < inside_tol:
                    continue

                z_surf = (a * z0) + (b * z1) + (c * z2)
                z_cover.append(z_surf)
                if z_surf > z_front:
                    z_front = z_surf

            # Conservative depth culling:
            # low strength keeps more arcs (especially in dense/self-occluded meshes).
            adaptive_margin = z_eps + ((1.0 - strength) * (0.55 * float(arc.rect_w) + 2.0))
            if strength >= 0.80:
                required_hits = 1
            elif strength >= 0.45:
                required_hits = 2
            else:
                required_hits = 3

            hit_threshold = z + (0.35 * adaptive_margin)
            occluding_hits = 0
            for zs in z_cover:
                if zs > hit_threshold:
                    occluding_hits += 1
                    if occluding_hits >= required_hits:
                        break

            if z_front <= -1.0e29 or z >= (z_front - adaptive_margin) or occluding_hits < required_hits:
                kept.append(arc)

        if len(kept) == 0:
            return arcs
        return kept

    def _prompt_gcode_settings(self) -> dict[str, float | int | bool] | None:
        parent = self.root

        target_width_mm = simpledialog.askfloat(
            "G-code: scala",
            "Larghezza finale incisione (mm):",
            parent=parent,
            initialvalue=float(self.gcode_target_width_mm),
            minvalue=0.1,
        )
        if target_width_mm is None:
            return None

        safe_z_mm = simpledialog.askfloat(
            "G-code: sicurezza",
            "Quota Z di sicurezza (mm):",
            parent=parent,
            initialvalue=float(self.gcode_safe_z_mm),
        )
        if safe_z_mm is None:
            return None

        cut_z_mm = simpledialog.askfloat(
            "G-code: incisione",
            "Quota Z di incisione (mm):",
            parent=parent,
            initialvalue=float(self.gcode_cut_z_mm),
        )
        if cut_z_mm is None:
            return None

        feed_xy_mm_min = simpledialog.askfloat(
            "G-code: feed XY",
            "Feed XY (mm/min):",
            parent=parent,
            initialvalue=float(self.gcode_feed_xy_mm_min),
            minvalue=1.0,
        )
        if feed_xy_mm_min is None:
            return None

        feed_z_mm_min = simpledialog.askfloat(
            "G-code: feed Z",
            "Feed Z (mm/min):",
            parent=parent,
            initialvalue=float(self.gcode_feed_z_mm_min),
            minvalue=1.0,
        )
        if feed_z_mm_min is None:
            return None

        max_segment_mm = simpledialog.askfloat(
            "G-code: segmentazione",
            "Lunghezza massima segmento (mm):",
            parent=parent,
            initialvalue=float(self.gcode_max_segment_mm),
            minvalue=0.01,
        )
        if max_segment_mm is None:
            return None

        spindle_rpm = simpledialog.askinteger(
            "G-code: mandrino",
            "RPM mandrino (0 = non emettere M3/M5):",
            parent=parent,
            initialvalue=int(self.gcode_spindle_rpm),
            minvalue=0,
        )
        if spindle_rpm is None:
            return None

        invert_y = messagebox.askyesnocancel(
            "G-code: asse Y",
            "Invertire asse Y nel G-code?\n(SI consigliato su CNC con Y verso l'alto)",
            parent=parent,
        )
        if invert_y is None:
            return None

        self.gcode_target_width_mm = float(target_width_mm)
        self.gcode_safe_z_mm = float(safe_z_mm)
        self.gcode_cut_z_mm = float(cut_z_mm)
        self.gcode_feed_xy_mm_min = float(feed_xy_mm_min)
        self.gcode_feed_z_mm_min = float(feed_z_mm_min)
        self.gcode_max_segment_mm = float(max_segment_mm)
        self.gcode_spindle_rpm = int(spindle_rpm)
        self.gcode_invert_y = bool(invert_y)

        return {
            "target_width_mm": float(target_width_mm),
            "safe_z_mm": float(safe_z_mm),
            "cut_z_mm": float(cut_z_mm),
            "feed_xy_mm_min": float(feed_xy_mm_min),
            "feed_z_mm_min": float(feed_z_mm_min),
            "max_segment_mm": float(max_segment_mm),
            "spindle_rpm": int(spindle_rpm),
            "invert_y": bool(invert_y),
        }

    def export_svg_dialog(self) -> None:
        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        try:
            arcs_full, projected_vertices, source = self._compute_export_view_data()
        except Exception as exc:
            messagebox.showerror("Errore export SVG", str(exc))
            return

        if self.export_occlusion_cull.get():
            arcs_export = self._filter_occluded_arcs(arcs_full, projected_vertices)
        else:
            arcs_export = arcs_full

        default_name = f"{self.stl_path.stem}_arcs.svg"
        save_path = filedialog.asksaveasfilename(
            title="Esporta SVG",
            defaultextension=".svg",
            initialfile=default_name,
            filetypes=[("SVG file", "*.svg"), ("All files", "*.*")],
        )
        if not save_path:
            return

        try:
            write_svg(Path(save_path), arcs_export, stroke_width=float(self.stroke_width.get()))
        except Exception as exc:
            messagebox.showerror("Errore scrittura SVG", str(exc))
            return

        self.status_var.set(
            (
                f"SVG esportato: {save_path} | Archi visibili: {len(arcs_export)}"
                f"/{len(arcs_full)} | Source: {source.upper()}"
            )
        )

    def export_gcode_dialog(self) -> None:
        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        try:
            arcs_full, projected_vertices, source = self._compute_export_view_data()
        except Exception as exc:
            messagebox.showerror("Errore calcolo archi", str(exc))
            return

        if self.export_occlusion_cull.get():
            arcs_export = self._filter_occluded_arcs(arcs_full, projected_vertices)
        else:
            arcs_export = arcs_full

        if len(arcs_export) == 0:
            messagebox.showwarning("Nessun arco", "Con i parametri attuali non ci sono archi da esportare.")
            return

        params = self._prompt_gcode_settings()
        if params is None:
            return

        default_name = f"{self.stl_path.stem}_arcs.nc"
        save_path = filedialog.asksaveasfilename(
            title="Esporta G-code",
            defaultextension=".nc",
            initialfile=default_name,
            filetypes=[
                ("G-code", "*.nc"),
                ("G-code", "*.gcode"),
                ("Text", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not save_path:
            return

        try:
            path_count, segment_count, cut_length = write_gcode(
                gcode_path=Path(save_path),
                arcs=arcs_export,
                target_width_mm=float(params["target_width_mm"]),
                safe_z_mm=float(params["safe_z_mm"]),
                cut_z_mm=float(params["cut_z_mm"]),
                feed_xy_mm_min=float(params["feed_xy_mm_min"]),
                feed_z_mm_min=float(params["feed_z_mm_min"]),
                max_segment_mm=float(params["max_segment_mm"]),
                spindle_rpm=int(params["spindle_rpm"]),
                invert_y=bool(params["invert_y"]),
                arc_mode=self._current_arc_mode(),
            )
        except Exception as exc:
            messagebox.showerror("Errore export G-code", str(exc))
            return

        self.status_var.set(
            (
                f"G-code esportato: {save_path} | Archi visibili: {len(arcs_export)}"
                f"/{len(arcs_full)} | "
                f"Path: {path_count} | Segmenti: {segment_count} | "
                f"Lunghezza taglio: {cut_length:.2f} mm | Source: {source.upper()}"
            )
        )


def main() -> int:
    root = tk.Tk()
    app = ScratchDesktopApp(root)

    default_cube = Path(__file__).resolve().parent.parent / "knots" / "basic_cube_-_10mm.stl"
    if default_cube.exists():
        app.load_stl(default_cube)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
