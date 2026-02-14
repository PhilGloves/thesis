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
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
    gain: float,
) -> tuple[float, float]:
    dist = (float(p[2]) - n_view) * gain
    cx = float(p[0])
    cy = float(p[1]) - dist / 2.0
    oy = dist / 2.0
    t = math.radians(angle_deg)
    x = cx - oy * math.sin(t)
    y = cy + oy * math.cos(t)
    return x, y


def point_on_arc_for_view_angle(arc: Arc, angle_deg: float) -> tuple[float, float]:
    """
    Compute a point on the exact canvas arc geometry for a given view angle.

    Mapping:
    - view angle in [-90, 90] maps to sweep fraction [0, 1]
    - theta is interpreted with Tk canvas convention:
      x = cx + r*cos(theta), y = cy - r*sin(theta)
    """
    r = arc.rect_w / 2.0
    cx = arc.rect_x + r
    cy = arc.rect_y + r

    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    theta_deg = arc.start_angle + (u * arc.sweep_angle)
    theta = math.radians(theta_deg)

    x = cx + r * math.cos(theta)
    y = cy - r * math.sin(theta)
    return x, y


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
        self.view_gain = tk.DoubleVar(value=1.0)
        self.show_arcs = tk.BooleanVar(value=True)
        self.show_profile = tk.BooleanVar(value=True)
        self.auto_center = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Apri un file STL per iniziare.")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Apri STL", command=self.open_stl_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Esporta SVG", command=self.export_svg_dialog).pack(side=tk.LEFT, padx=(8, 0))
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
            max_val=80.0,
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
        self._make_slider(
            controls,
            row=2,
            col=2,
            label="View gain",
            variable=self.view_gain,
            min_val=0.2,
            max_val=4.0,
            fmt="{:.2f}x",
            redraw_only=True,
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

    def schedule_recompute(self, delay_ms: int = 120) -> None:
        if self.pending_recompute is not None:
            self.root.after_cancel(self.pending_recompute)
        self.pending_recompute = self.root.after(delay_ms, lambda: self.recompute_and_redraw(interactive=False))

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
            prev_hash: tuple[int, int, int, int, float] | None = None
            for p in points:
                key = self._coord_key(p, decimals=6)
                arc = arc_by_coord.get(key)
                if arc is None:
                    continue

                # Collapse consecutive duplicates (same rendered arc geometry).
                arc_hash = (
                    arc.rect_x,
                    arc.rect_y,
                    arc.rect_w,
                    arc.rect_h,
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
            gain = float(self.view_gain.get())

            if interactive:
                path_stride = 2
                point_stride = 2
            else:
                path_stride = 1
                point_stride = 1

            for path in self.profile_paths[::path_stride]:
                arcs = path[::point_stride]
                if len(arcs) < 2:
                    continue

                coords: list[float] = []
                for arc in arcs:
                    x, y = point_on_arc_for_view_angle(arc, angle)
                    coords.append(x)
                    coords.append(y)

                if len(coords) >= 4:
                    self.canvas.create_line(
                        *coords,
                        fill="#f4f4f4",
                        width=1.25,
                    )

        if self.stl_path is not None:
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

    def export_svg_dialog(self) -> None:
        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        w = max(20, int(self.canvas.winfo_width()))
        h = max(20, int(self.canvas.winfo_height()))
        camera = self._build_camera(w, h)

        try:
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
            )
        except Exception as exc:
            messagebox.showerror("Errore export SVG", str(exc))
            return

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
            write_svg(Path(save_path), arcs_full, stroke_width=float(self.stroke_width.get()))
        except Exception as exc:
            messagebox.showerror("Errore scrittura SVG", str(exc))
            return

        self.status_var.set(f"SVG esportato: {save_path} | Archi full: {len(arcs_full)}")


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
