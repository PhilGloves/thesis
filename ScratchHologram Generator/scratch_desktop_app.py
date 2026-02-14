#!/usr/bin/env python3
"""
Scratch Hologram Desktop App (Tkinter, single-view).

Features:
- Load STL.
- Orbit camera with mouse directly on arc preview.
- Lightweight adaptive preview for dense meshes.
- Export SVG from current camera/parameters.
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
    load_mesh,
    model_to_window,
    write_svg,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ScratchDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ScratchHologram Generator Desktop")
        self.root.geometry("1450x920")

        self.stl_path: Path | None = None
        self.model_vertices: np.ndarray | None = None
        self.edges: list[Edge] = []
        self.faces: np.ndarray | None = None
        self.edge_lengths: np.ndarray | None = None
        self.total_edge_length: float = 0.0
        self.current_arcs: list[Arc] = []

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
        self.preview_quality = tk.IntVar(value=45)
        self.auto_center = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Apri un file STL per iniziare.")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Apri STL", command=self.open_stl_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Esporta SVG", command=self.export_svg_dialog).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="Ricalcola", command=self.recompute_and_redraw).pack(side=tk.LEFT, padx=(8, 0))

        self.path_label = ttk.Label(top, text="Nessun file caricato", width=108)
        self.path_label.pack(side=tk.LEFT, padx=(12, 0))

        main_frame = ttk.LabelFrame(self.root, text="Preview Archi (drag: orbit, wheel: zoom)", padding=4)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(main_frame, bg="#0a0c12", highlightthickness=0)
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

        ttk.Checkbutton(
            controls,
            text="Auto-center Z",
            variable=self.auto_center,
            command=self.on_auto_center_toggle,
        ).grid(row=2, column=1, padx=10, sticky="w")

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(10, 2),
            anchor="w",
        )
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
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=8, pady=4, sticky="ew")
        parent.columnconfigure(col, weight=1)

        ttk.Label(frame, text=label).pack(anchor="w")
        value_label = ttk.Label(frame, width=9)
        value_label.pack(side=tk.RIGHT)

        def on_change(_: str) -> None:
            if integer:
                value = float(int(round(float(variable.get()))))
                variable.set(int(value))
            else:
                value = float(variable.get())
            value_label.config(text=fmt.format(value))
            self.schedule_recompute()

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
        self.pending_recompute = self.root.after(delay_ms, self.recompute_and_redraw)

    def on_auto_center_toggle(self) -> None:
        if self.stl_path is None:
            return
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
        self.edge_lengths = np.asarray(lengths, dtype=np.float64)
        self.total_edge_length = float(np.sum(self.edge_lengths)) if len(lengths) > 0 else 0.0

        self.path_label.config(text=str(stl_path))
        self.status_var.set(
            f"Caricato: {stl_path.name} | Vertici: {len(model_vertices)} | Spigoli: {len(edges)}"
        )
        self.recompute_and_redraw()

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
        self.schedule_recompute(delay_ms=35)

    def on_mouse_up(self, _: tk.Event) -> None:
        self.dragging = False
        self.recompute_and_redraw()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if self.model_vertices is None:
            return
        if int(event.delta) > 0:
            self.distance_scale *= 0.90
        else:
            self.distance_scale *= 1.10
        self.distance_scale = clamp(self.distance_scale, 0.35, 4.5)
        self.schedule_recompute(delay_ms=45)

    def recompute_and_redraw(self) -> None:
        self.pending_recompute = None
        self.compute_and_draw_arcs(interactive=self.dragging, full_quality=False)

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
        direction_norm = float(np.linalg.norm(direction))
        direction = direction / direction_norm if direction_norm > EPS else np.asarray((1.0, 0.0, 0.0))

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

    def _preview_sampling(self, interactive: bool) -> tuple[list[Edge], float, int]:
        base_lr = float(self.line_resolution.get())
        total_edges = len(self.edges)
        if total_edges == 0:
            return [], base_lr, 1

        if self.edge_lengths is None:
            self.edge_lengths = np.ones(total_edges, dtype=np.float64)
            self.total_edge_length = float(total_edges)

        quality = clamp(float(self.preview_quality.get()) / 100.0, 0.05, 1.0)
        if interactive:
            quality *= 0.70

        # Keep small models full quality.
        if total_edges <= 1800:
            effective_lr = max(0.10, base_lr * (0.9 if interactive else 1.0))
            return self.edges, effective_lr, 1

        max_preview_edges = max(500, int(600 + 5400 * quality))
        stride = max(1, int(math.ceil(total_edges / max_preview_edges)))
        preview_edges = self.edges[::stride]

        # Adaptive point budget to avoid freezes on dense meshes.
        # approx_points ~= total_edge_length * line_resolution
        target_points = 60000.0 * quality
        if interactive:
            target_points *= 0.6

        effective_lr = base_lr
        est_points = self.total_edge_length * effective_lr / float(stride)
        if est_points > target_points and est_points > 1e-9:
            effective_lr *= target_points / est_points

        effective_lr = clamp(effective_lr, 0.08, base_lr)
        return preview_edges, effective_lr, stride

    def compute_and_draw_arcs(self, interactive: bool, full_quality: bool) -> None:
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

        if full_quality:
            preview_edges = self.edges
            effective_lr = float(self.line_resolution.get())
            edge_stride = 1
        else:
            preview_edges, effective_lr, edge_stride = self._preview_sampling(interactive=interactive)

        try:
            mtx = build_model_to_window_matrix(camera)
            zero_vertices = model_to_window(self.model_vertices, mtx)
            pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), mtx)[0]
            n_view = float(pr_view[2])

            arcs = generate_arcs(
                model_vertices=self.model_vertices,
                zero_vertices=zero_vertices,
                edges=preview_edges,
                view_points_per_unit_length=effective_lr,
                n_view=n_view,
                dedupe_decimals=6,
                min_arc_radius=float(self.min_arc_radius.get()),
            )
            self.current_arcs = arcs
        except Exception as exc:
            self.status_var.set(f"Errore calcolo archi: {exc}")
            return

        stroke_preview = max(1.0, float(self.stroke_width.get()) * 4.0)
        draw_limit = max(1, int(self.arc_limit.get()))
        if interactive and not full_quality:
            draw_limit = min(draw_limit, 2500)
        shown = min(len(self.current_arcs), draw_limit)

        for arc in self.current_arcs[:shown]:
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

        if self.stl_path is not None:
            mode = "FULL" if full_quality else ("FAST" if interactive else "PREVIEW")
            self.status_var.set(
                " | ".join(
                    [
                        f"STL: {self.stl_path.name}",
                        f"Spigoli: {len(self.edges)}",
                        f"Archi: {len(self.current_arcs)}",
                        f"Preview: {shown}",
                        f"LR eff: {effective_lr:.2f}",
                        f"Edge step: {edge_stride}",
                        f"Mode: {mode}",
                        f"Yaw/Pitch: {self.yaw_deg:.1f}/{self.pitch_deg:.1f}",
                    ]
                )
            )

    def export_svg_dialog(self) -> None:
        if self.model_vertices is None or self.stl_path is None:
            messagebox.showwarning("Nessun modello", "Carica prima un file STL.")
            return

        self.compute_and_draw_arcs(interactive=False, full_quality=True)
        if not self.current_arcs:
            messagebox.showwarning("Nessun arco", "Nessun arco disponibile da esportare.")
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
            write_svg(Path(save_path), self.current_arcs, stroke_width=float(self.stroke_width.get()))
        except Exception as exc:
            messagebox.showerror("Errore export SVG", str(exc))
            return

        self.status_var.set(f"SVG esportato: {save_path}")


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

