#!/usr/bin/env python3
"""
Scratch hologram desktop interface.

Pipeline stage
--------------
This module is the interactive front-end for the scratch hologram pipeline. It
loads a mesh, controls camera orbit and projection, previews edge sampling and
arc generation, and exports the current view as SVG or G-code.

Input / output
--------------
Input is a mesh selected by the user plus interactive parameters such as edge
sampling density, projection orientation, arc generation mode, and visibility
cull enable/disable. Output is an on-screen preview and exported SVG or G-code
derived from the same current camera view.

Key parameters
--------------
`line_resolution` controls edge sampling density, `view_angle` controls the
simulated profile overlay, `arc_mode` selects semicircular or elliptic arc
generation, and the camera yaw/pitch/zoom determine the projection.

Coordinate conventions
----------------------
Model coordinates are inherited from `scratch_pipeline.py` and remain in mesh
units. The preview canvas uses screen-space pixels with `x` rightward and `y`
downward. The camera view direction points from the camera origin to the mesh
center, while exported SVG and G-code reuse the same projected arc generation
geometry to avoid mismatches between preview and fabrication.
"""

from __future__ import annotations

import math
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
    build_sample_visibility_lookup,
    build_model_to_view_matrix,
    build_model_to_window_matrix,
    build_model_vertices_and_edges,
    ensure_dependencies,
    generate_arcs,
    load_mesh,
    model_to_window,
    write_svg,
)


def clamp(value: float, low: float, high: float) -> float:
    """
    Clamp a scalar to a closed interval.

    Parameters
    ----------
    value : float
        Input scalar, any unit.
    low : float
        Lower bound, same unit as `value`.
    high : float
        Upper bound, same unit as `value`.

    Returns
    -------
    float
        Clamped scalar.

    Notes
    -----
    This helper is reused for camera controls and UI normalization.

    Assumptions
    -----------
    `low <= high`.
    """

    return max(low, min(high, value))


def point_at_angle_projected(
    p: np.ndarray | tuple[float, float, float],
    angle_deg: float,
    n_view: float,
    ellipse_ratio: float = 1.0,
) -> tuple[float, float]:
    """
    Map one projected point onto the simulated scratch profile.

    Parameters
    ----------
    p : np.ndarray | tuple[float, float, float]
        Projected point `(x, y, z)` in screen coordinates.
    angle_deg : float
        Simulated profile angle in degrees.
    n_view : float
        Projected depth of the camera target point.
    ellipse_ratio : float, optional
        Vertical scaling factor for elliptic profile motion.

    Returns
    -------
    tuple[float, float]
        Simulated profile point in screen pixels.

    Notes
    -----
    The design decision is to move each projected point along the same arc
    family used for arc generation instead of re-projecting a rotated mesh; the
    trade-off is a faster, visually coherent preview rather than a strict 3-D
    rigid transformation.

    Assumptions
    -----------
    `angle_deg` is interpreted in the interval `[-90, 90]`.
    """
    dist = float(p[2]) - n_view
    cx = float(p[0])
    cy = float(p[1]) - dist / 2.0
    r = abs(dist) / 2.0
    if r < EPS:
        return cx, cy

    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    # Keep angle=0 as neutral pose (projected model shape),
    # while -90/+90 move to the arc endpoints.
    start_deg = 180.0 if dist > 0.0 else 0.0
    theta = math.radians(start_deg + (u * 180.0))
    ry = r * clamp(float(ellipse_ratio), 0.20, 1.00)
    x = cx + r * math.cos(theta)
    # Match Tk canvas arc orientation exactly.
    y = cy - ry * math.sin(theta)
    return x, y


def sample_arc_polyline(arc: Arc, max_segment_units: float) -> list[tuple[float, float]]:
    """
    Approximate one arc with a polyline for G-code export.

    Parameters
    ----------
    arc : Arc
        Screen-space arc to approximate.
    max_segment_units : float
        Maximum polyline segment length in screen units.

    Returns
    -------
    list[tuple[float, float]]
        Polyline vertices in screen coordinates.

    Notes
    -----
    Polyline sampling is used only when exact circular G-code is not suitable.
    The trade-off is larger G-code output in exchange for broader geometry
    support, especially for elliptic arc generation.

    Assumptions
    -----------
    `arc` uses the same angle convention as the Tk preview.
    """

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
    """
    Export the current arc set as planar G-code.

    Parameters
    ----------
    gcode_path : Path
        Destination file path.
    arcs : list[Arc]
        Screen-space arcs selected for export.
    target_width_mm : float
        Final engraved width in millimeters.
    safe_z_mm : float
        Clearance height in millimeters.
    cut_z_mm : float
        Cutting height in millimeters.
    feed_xy_mm_min : float
        XY feed rate in millimeters per minute.
    feed_z_mm_min : float
        Z feed rate in millimeters per minute.
    max_segment_mm : float
        Maximum segment length for polyline fallback in millimeters.
    spindle_rpm : int
        Spindle speed in revolutions per minute. `0` suppresses spindle codes.
    invert_y : bool
        If True, flips Y to match CNC axis conventions.
    arc_mode : str
        Arc generation mode used upstream.

    Returns
    -------
    tuple[int, int, float]
        Path count, motion segment count, and cumulative cutting length in
        millimeters.

    Notes
    -----
    Circular commands are emitted only for near-circular semicircular arc
    generation. The trade-off is simpler, shorter G-code when possible, with a
    polyline fallback for robustness and elliptic geometry.

    Assumptions
    -----------
    `arcs` already matches the current projection and visibility decisions.
    """

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
    """
    Interactive desktop controller for mesh loading, projection, and export.

    Parameters
    ----------
    root : tk.Tk
        Root Tk application object.

    Returns
    -------
    None
        The class manages stateful UI interaction.

    Notes
    -----
    The desktop app keeps preview, SVG export, and G-code export on the same
    projection and arc generation path. This design decision avoids mismatched
    outputs; the trade-off is that preview recomputation must stay efficient.

    Assumptions
    -----------
    Tkinter is available and running on the main thread.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize application state, fixed defaults, and the UI.

        Parameters
        ----------
        root : tk.Tk
            Root Tk application object.

        Returns
        -------
        None

        Notes
        -----
        Several controls are intentionally fixed to reduce UI complexity. The
        trade-off is fewer tuning knobs in exchange for a more reproducible
        preview-to-export workflow.

        Assumptions
        -----------
        `root` has not been destroyed before initialization completes.
        """

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
        self.last_sampled_edges: list[Edge] = []
        self.last_projected: np.ndarray | None = None
        self.last_n_view: float = 0.0
        self.last_edge_stride: int = 1
        self.last_preview_lr: float = 0.0
        self.last_mode: str = "IDLE"

        self.model_center = np.zeros(3, dtype=np.float64)
        self.base_orbit_radius = 25.0
        # Mouse wheel zoom is implemented as optical zoom (projection scale),
        # not camera dolly, to avoid strong perspective deformation when zooming in.
        self.zoom_scale = 1.0
        self.yaw_deg = -35.0
        self.pitch_deg = 20.0

        self.dragging = False
        self.drag_last_x = 0
        self.drag_last_y = 0
        self.pending_recompute: str | None = None

        self.line_resolution = tk.DoubleVar(value=3.28)
        # Keep these fixed to simplify the UI and avoid unnecessary tuning knobs.
        self.fixed_min_arc_radius = 0.0
        self.fixed_stroke_width = 0.05
        self.fixed_camera_scale = 0.724
        # Use a much higher zf to keep perspective almost orthographic and
        # avoid trapezoid-like deformation while zooming/orbiting.
        self.fixed_zf = 50.0
        self.arc_limit = tk.IntVar(value=6000)
        self.preview_quality = tk.IntVar(value=55)
        self.view_angle = tk.DoubleVar(value=0.0)
        self.arc_mode = tk.StringVar(value="Semicircle (CNC)")
        self.ellipse_ratio = tk.DoubleVar(value=0.65)
        self.show_arcs = tk.BooleanVar(value=True)
        self.show_profile = tk.BooleanVar(value=True)
        # Keep this behavior fixed: less UI noise, same practical result for this workflow.
        self.fixed_auto_center = True
        self.visibility_cull = tk.BooleanVar(value=False)

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
            row=1,
            col=0,
            label="View angle",
            variable=self.view_angle,
            min_val=-90.0,
            max_val=90.0,
            fmt="{:.0f} deg",
            redraw_only=True,
        )
        self._make_slider(
            controls,
            row=0,
            col=2,
            label="Preview quality",
            variable=self.preview_quality,
            min_val=5,
            max_val=100,
            fmt="{:.0f}%",
            integer=True,
        )
        self._make_dropdown(
            controls,
            row=1,
            col=1,
            label="Arc mode",
            variable=self.arc_mode,
            values=("Semicircle (CNC)", "Elliptic"),
        )
        self._make_slider(
            controls,
            row=1,
            col=2,
            label="Ellipse ratio",
            variable=self.ellipse_ratio,
            min_val=0.20,
            max_val=1.00,
            fmt="{:.2f}",
        )
        checkbox_row = ttk.Frame(controls)
        checkbox_row.grid(row=2, column=0, columnspan=3, padx=8, pady=4, sticky="w")
        ttk.Checkbutton(
            checkbox_row,
            text="Show arcs",
            variable=self.show_arcs,
            command=self.redraw_from_cache,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            checkbox_row,
            text="Show simulated profile",
            variable=self.show_profile,
            command=self.redraw_from_cache,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            checkbox_row,
            text="Visibility cull (Z-buffer)",
            variable=self.visibility_cull,
            command=lambda: self.schedule_recompute(delay_ms=80),
        ).pack(side=tk.LEFT)

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
        """
        Debounce preview recomputation after a UI interaction.

        Parameters
        ----------
        delay_ms : int, optional
            Delay before recomputation in milliseconds.

        Returns
        -------
        None

        Notes
        -----
        Debouncing is a design decision that keeps the UI responsive while
        dragging or adjusting projection controls; the trade-off is that the
        preview can lag slightly behind rapid input.

        Assumptions
        -----------
        The Tk event loop is active.
        """

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

    def open_stl_dialog(self) -> None:
        """
        Open a file dialog and load a mesh selected by the user.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The dialog is limited to STL by default because the downstream mesh
        pipeline is optimized for STL-based scratch hologram workflows.

        Assumptions
        -----------
        The current process has permission to read the selected file.
        """

        filename = filedialog.askopenfilename(
            title="Apri STL",
            initialdir=str(Path.cwd()),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if filename:
            self.load_stl(Path(filename))

    def load_stl(self, stl_path: Path) -> None:
        """
        Load a mesh and initialize derived state for preview and export.

        Parameters
        ----------
        stl_path : Path
            Path to the input mesh file.

        Returns
        -------
        None

        Notes
        -----
        The method caches mesh statistics used by edge sampling, projection, and
        G-code defaults. Doing this once is a design decision because it avoids
        repeated work during preview updates; the trade-off is more state kept in
        memory.

        Assumptions
        -----------
        The selected file is readable and contains a valid triangular mesh.
        """

        try:
            ensure_dependencies()
            mesh = load_mesh(stl_path)
            model_vertices, edges, faces = build_model_vertices_and_edges(
                mesh=mesh,
                auto_center=bool(self.fixed_auto_center),
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
        self.zoom_scale = 1.0
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
        """
        Start camera drag interaction.

        Parameters
        ----------
        event : tk.Event
            Mouse press event in canvas pixel coordinates.

        Returns
        -------
        None

        Notes
        -----
        The handler stores the last cursor position so later drag events can
        update the projection incrementally.

        Assumptions
        -----------
        The event originates from the preview canvas.
        """

        self.dragging = True
        self.drag_last_x = int(event.x)
        self.drag_last_y = int(event.y)

    def on_mouse_drag(self, event: tk.Event) -> None:
        """
        Update camera yaw and pitch during drag interaction.

        Parameters
        ----------
        event : tk.Event
            Mouse motion event in canvas pixel coordinates.

        Returns
        -------
        None

        Notes
        -----
        The handler uses a fast preview path while dragging. The design decision
        preserves responsiveness, while the trade-off is temporary reduction in
        edge sampling density and arc generation fidelity.

        Assumptions
        -----------
        Dragging has already been initialized by `on_mouse_down`.
        """

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
        """
        Finish camera drag interaction and trigger a full preview refresh.

        Parameters
        ----------
        _ : tk.Event
            Mouse release event (unused).

        Returns
        -------
        None

        Notes
        -----
        Releasing the mouse switches back from the fast preview path to the full
        projection and arc generation path.

        Assumptions
        -----------
        The release event corresponds to a prior drag interaction.
        """

        self.dragging = False
        self.recompute_and_redraw(interactive=False)

    def on_mouse_wheel(self, event: tk.Event) -> None:
        """
        Update the projection zoom in response to the mouse wheel.

        Parameters
        ----------
        event : tk.Event
            Mouse wheel event containing a signed delta.

        Returns
        -------
        None

        Notes
        -----
        Zoom is implemented as projection scaling rather than camera dolly. The
        design decision keeps the scratch pattern shape more stable, while the
        trade-off is a less physically faithful camera model.

        Assumptions
        -----------
        The event originates from the preview canvas.
        """

        if self.model_vertices is None:
            return
        if int(event.delta) > 0:
            self.zoom_scale *= 1.08
        else:
            self.zoom_scale *= 0.92
        self.zoom_scale = clamp(self.zoom_scale, 0.35, 4.5)
        self.recompute_and_redraw(interactive=True)
        self.schedule_recompute(delay_ms=150)

    def _build_camera(self, canvas_width: int, canvas_height: int) -> CameraConfig:
        if self.model_vertices is None:
            return CameraConfig(
                po=(0.9, 9.0, 24.0),
                pr=(0.93, 8.29, 0.47),
                look_up=(0.0, 1.0, 0.0),
                current_scale=float(self.fixed_camera_scale) * float(self.zoom_scale),
                zf=float(self.fixed_zf),
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

        # Keep orbit radius fixed to preserve shape coherence while rotating/zooming.
        orbit_radius = self.base_orbit_radius
        pr = np.asarray(self.model_center, dtype=np.float64)
        po = pr + direction * orbit_radius

        look_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(direction, look_up))) > 0.98:
            look_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

        return CameraConfig(
            po=(float(po[0]), float(po[1]), float(po[2])),
            pr=(float(pr[0]), float(pr[1]), float(pr[2])),
            look_up=(float(look_up[0]), float(look_up[1]), float(look_up[2])),
            current_scale=float(self.fixed_camera_scale) * float(self.zoom_scale),
            zf=float(self.fixed_zf),
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

    def _current_arc_mode(self) -> str:
        if str(self.arc_mode.get()).strip().lower().startswith("elliptic"):
            return "elliptic"
        return "semi"

    def _current_ellipse_ratio(self) -> float:
        return clamp(float(self.ellipse_ratio.get()), 0.20, 1.00)

    def recompute_and_redraw(self, interactive: bool) -> None:
        """
        Recompute projection, visibility, edge sampling, and arc generation.

        Parameters
        ----------
        interactive : bool
            If True, uses a reduced-quality preview path intended for drag/zoom
            interaction.

        Returns
        -------
        None

        Notes
        -----
        The fast interactive mode deliberately reduces edge sampling and draw
        load to preserve responsiveness. The trade-off is lower preview fidelity
        until the deferred full recomputation completes.

        Assumptions
        -----------
        A mesh has already been loaded when full processing is expected.
        """

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
            view_mtx = build_model_to_view_matrix(camera)
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
            sample_visibility = self._visibility_sample_lookup(
                edges_for_cull=sampled_edges,
                view_points_per_unit_length=effective_lr,
                camera=camera,
                write_debug_pngs=not interactive,
            )
            self.current_arcs = generate_arcs(
                model_vertices=self.model_vertices,
                zero_vertices=zero_vertices,
                edges=sampled_edges,
                view_points_per_unit_length=effective_lr,
                n_view=n_view,
                dedupe_decimals=6,
                min_arc_radius=float(self.fixed_min_arc_radius),
                arc_mode=self._current_arc_mode(),
                ellipse_ratio=self._current_ellipse_ratio(),
                model_to_window_mtx=mtx,
                model_to_view_mtx=view_mtx,
                sample_visibility_by_edge=sample_visibility,
            )
        except Exception as exc:
            self.status_var.set(f"Errore calcolo archi: {exc}")
            return

        self._draw_scene(draw_limit=draw_limit, interactive=interactive)

    def _draw_scene(self, draw_limit: int, interactive: bool) -> None:
        if self.last_projected is None or self.model_vertices is None:
            return

        stroke_preview = max(1.0, float(self.fixed_stroke_width) * 4.0)

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
            profile_ratio = self._current_ellipse_ratio() if self._current_arc_mode() == "elliptic" else 1.0

            if interactive:
                path_stride = 2
            else:
                path_stride = 1

            if self.last_projected is not None:
                edges_to_draw = self.last_sampled_edges[::path_stride] if self.last_sampled_edges else self.edges[::path_stride]
                for edge in edges_to_draw:
                    p1 = self.last_projected[edge.start_idx]
                    p2 = self.last_projected[edge.end_idx]
                    x1, y1 = point_at_angle_projected(
                        p1,
                        angle_deg=angle,
                        n_view=self.last_n_view,
                        ellipse_ratio=profile_ratio,
                    )
                    x2, y2 = point_at_angle_projected(
                        p2,
                        angle_deg=angle,
                        n_view=self.last_n_view,
                        ellipse_ratio=profile_ratio,
                    )
                    self.canvas.create_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="#f4f4f4",
                        width=1.25,
                    )

        if self.stl_path is not None:
            arc_mode_name = self._current_arc_mode()
            if arc_mode_name == "elliptic":
                arc_mode_status = f"ELL({self._current_ellipse_ratio():.2f})"
            else:
                arc_mode_status = "SEMI"
            if self.visibility_cull.get():
                cull_status = "ZBUF ON"
            else:
                cull_status = "OFF"
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
                        f"Arc: {arc_mode_status}",
                        f"Cull: {cull_status}",
                        "Sim: LIGHT",
                        f"Mode: {self.last_mode}",
                        f"Zoom: {self.zoom_scale:.2f}x",
                        f"Yaw/Pitch: {self.yaw_deg:.1f}/{self.pitch_deg:.1f}",
                    ]
                )
            )

    def redraw_from_cache(self) -> None:
        """
        Redraw the preview using cached projection and arc generation results.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This avoids recomputing projection or edge sampling when only display
        toggles change, which is faster but depends on cached state coherence.

        Assumptions
        -----------
        Cached preview data is still valid for the current UI state.
        """

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
        view_mtx = build_model_to_view_matrix(camera)
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
            min_arc_radius=float(self.fixed_min_arc_radius),
            arc_mode=self._current_arc_mode(),
            ellipse_ratio=self._current_ellipse_ratio(),
            model_to_window_mtx=mtx,
            model_to_view_mtx=view_mtx,
            sample_visibility_by_edge=self._visibility_sample_lookup(
                edges_for_cull=self.edges,
                view_points_per_unit_length=float(self.line_resolution.get()),
                camera=camera,
                write_debug_pngs=True,
            ),
        )
        return arcs_full, zero_vertices

    def _compute_export_view_data(self) -> tuple[list[Arc], np.ndarray, str]:
        """
        Returns arcs + projected vertices for export and the source label:
        - "preview": export uses the same arc dataset currently previewed
        - "full": fallback only if preview cache is unavailable
        """
        if self.model_vertices is None:
            raise ValueError("Nessun modello caricato.")

        # Ensure export uses the latest UI parameters (arc mode, ellipse ratio, etc.).
        self._flush_pending_recompute()

        # Always export from preview cache (single source of truth).
        if self.last_mode == "FAST":
            # Ensure export never uses temporary low-quality drag state.
            self.recompute_and_redraw(interactive=False)
        if self.last_projected is not None and len(self.current_arcs) > 0:
            return list(self.current_arcs), self.last_projected, "preview"

        arcs_full, projected_vertices = self._compute_full_view_data()
        return arcs_full, projected_vertices, "full"

    def _visibility_sample_lookup(
        self,
        *,
        edges_for_cull: list[Edge],
        view_points_per_unit_length: float,
        camera: CameraConfig,
        write_debug_pngs: bool,
    ) -> dict[int, np.ndarray] | None:
        if not bool(self.visibility_cull.get()):
            return None
        if self.model_vertices is None or self.faces is None or len(self.faces) == 0 or len(edges_for_cull) == 0:
            return None

        sample_visibility_by_edge, _stats = build_sample_visibility_lookup(
            model_vertices=self.model_vertices,
            edges=edges_for_cull,
            faces=self.faces,
            camera=camera,
            view_points_per_unit_length=view_points_per_unit_length,
            zbuffer_resolution=1024,
            coverage_dilation_size=3,
            sample_window_size=5,
            debug_output_prefix=(
                Path(__file__).resolve().with_name("visibility_debug")
                if write_debug_pngs
                else None
            ),
        )
        return sample_visibility_by_edge

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
        """
        Export the current view as SVG using the active arc generation settings.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Export reuses the current projection and visibility decisions so SVG
        matches the preview. The trade-off is that export depends on the same
        cached state management used by the UI.

        Assumptions
        -----------
        A mesh is loaded and the user selects a writable destination path.
        """

        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        try:
            arcs_full, projected_vertices, source = self._compute_export_view_data()
        except Exception as exc:
            messagebox.showerror("Errore export SVG", str(exc))
            return

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
            write_svg(Path(save_path), arcs_export, stroke_width=float(self.fixed_stroke_width))
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
        """
        Export the current view as G-code using the active arc generation settings.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        G-code export uses the same projected arc set as SVG export so
        fabrication matches the preview. The trade-off is that fabrication
        output remains tied to screen-space arc generation rather than direct
        machining of the mesh itself.

        Assumptions
        -----------
        A mesh is loaded, at least one arc is available, and the destination
        path is writable.
        """

        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        try:
            arcs_full, projected_vertices, source = self._compute_export_view_data()
        except Exception as exc:
            messagebox.showerror("Errore calcolo archi", str(exc))
            return

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
    """
    Launch the scratch hologram desktop application.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Process exit code.

    Notes
    -----
    A default cube mesh is loaded when available so the preview is immediately
    useful for debugging projection and arc generation.

    Assumptions
    -----------
    Tkinter can create a window in the current environment.
    """

    root = tk.Tk()
    app = ScratchDesktopApp(root)

    default_cube = Path(__file__).resolve().parent.parent / "knots" / "basic_cube_-_10mm.stl"
    if default_cube.exists():
        app.load_stl(default_cube)

    root.mainloop()
    return 0


# Thesis extraction notes
# -----------------------
# 1. Load the mesh once, then cache mesh topology for repeated projection updates.
# 2. Build an orbiting camera from yaw, pitch, and zoom to drive the current projection.
# 3. Reduce edge sampling density during interaction, then restore full quality after debounce.
# 4. Reuse the same projection path for preview, SVG export, and G-code export.
# 5. Apply visibility culling before arc generation so hidden edge samples do not emit arcs.
# 6. Generate screen-space arcs from visible edge sampling points only.
# 7. Draw the scratch pattern and a simulated profile overlay from the same projected data.
# 8. Export SVG directly from the current arc set to keep preview/export consistency.
# 9. Export G-code from the same arc set, using circular moves when arc generation is near-circular.
#
# Useful figures/snippets
# -----------------------
# - `_build_camera`: orbit camera parameterization for projection.
# - `_preview_params`: quality scaling for interactive edge sampling.
# - `recompute_and_redraw`: full preview pipeline orchestration.
# - `_visibility_sample_lookup`: bridge to the visibility cull in `scratch_pipeline.py`.
# - `export_svg_dialog` / `export_gcode_dialog`: identical source-of-truth export path.
#
if __name__ == "__main__":
    raise SystemExit(main())
