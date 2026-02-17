#!/usr/bin/env python3
#todo fix the visualization: the 3d model appears flat and the arcs are not shown at all
"""
GPU preview for ScratchHologram (Qt + OpenGL).

This app focuses on interactive preview performance for dense STL models.
It reuses the same core arc-generation pipeline used by SVG/G-code export.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets

from scratch_pipeline import (
    Arc,
    CameraConfig,
    Edge,
    NORMAL_TOLERANCE_DECIMALS,
    build_model_to_window_matrix,
    build_model_vertices_and_edges,
    ensure_dependencies,
    generate_arcs,
    get_edge_points,
    load_mesh,
    model_to_window,
    write_svg,
)


EPS = 1e-12


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

    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    start_deg = 0.0 if dist > 0.0 else 180.0
    theta = math.radians(start_deg + (u * 180.0))
    ry = r * clamp(float(ellipse_ratio), 0.20, 1.00)
    x = cx + r * math.cos(theta)
    y = cy - ry * math.sin(theta)
    return x, y


def point_on_arc_compatible(arc: Arc, angle_deg: float) -> tuple[float, float]:
    rx = max(0.1, float(arc.rect_w) / 2.0)
    ry = max(0.1, float(arc.rect_h) / 2.0)
    cx = float(arc.rect_x) + (float(arc.rect_w) / 2.0)
    cy = float(arc.rect_y) + (float(arc.rect_h) / 2.0)

    u = (clamp(angle_deg, -90.0, 90.0) + 90.0) / 180.0
    theta_deg = float(arc.start_angle) + (u * float(arc.sweep_angle))
    theta = math.radians(theta_deg)

    x = cx + rx * math.cos(theta)
    y = cy - ry * math.sin(theta)
    return x, y


def arc_polyline_2d(arc: Arc, max_steps: int = 24) -> np.ndarray:
    rx = max(0.1, float(arc.rect_w) / 2.0)
    ry = max(0.1, float(arc.rect_h) / 2.0)
    cx = float(arc.rect_x) + (float(arc.rect_w) / 2.0)
    cy = float(arc.rect_y) + (float(arc.rect_h) / 2.0)

    mean_radius = math.sqrt(((rx * rx) + (ry * ry)) / 2.0)
    approx_len = abs(math.radians(float(arc.sweep_angle))) * mean_radius
    steps = int(clamp(approx_len / 5.0, 8.0, float(max_steps)))
    steps = max(8, min(max_steps, steps))

    out = np.zeros((steps + 1, 2), dtype=np.float64)
    for i in range(steps + 1):
        t = i / steps
        theta_deg = float(arc.start_angle) + (float(arc.sweep_angle) * t)
        theta = math.radians(theta_deg)
        out[i, 0] = cx + rx * math.cos(theta)
        out[i, 1] = cy - ry * math.sin(theta)
    return out


def polyline_to_segments_3d(points_xyz: np.ndarray) -> np.ndarray:
    n = points_xyz.shape[0]
    if n < 2:
        return np.zeros((0, 3), dtype=np.float32)
    seg = np.zeros((2 * (n - 1), 3), dtype=np.float32)
    seg[0::2, :] = points_xyz[:-1, :]
    seg[1::2, :] = points_xyz[1:, :]
    return seg


class GpuScratchPreview(QtWidgets.QMainWindow):
    def __init__(self, initial_stl: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ScratchHologram GPU Preview")
        self.resize(1600, 980)

        self.stl_path: Path | None = None
        self.model_vertices: np.ndarray | None = None
        self.edges: list[Edge] = []
        self.faces: np.ndarray | None = None
        self.model_center = np.zeros(3, dtype=np.float64)
        self.base_orbit_radius = 25.0
        self.distance_scale = 1.0

        self.current_arcs: list[Arc] = []
        self.profile_paths: list[list[Arc]] = []
        self.last_projected: np.ndarray | None = None
        self.last_n_view: float = 0.0
        self.last_sampled_edges: list[Edge] = []
        self.last_preview_lr: float = 0.0
        self.last_edge_stride: int = 1

        self.yaw_deg = -35.0
        self.pitch_deg = 20.0

        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.timeout.connect(self.recompute)

        self._build_ui()

        if initial_stl is not None and initial_stl.exists():
            self.load_stl(initial_stl)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        btn_open = QtWidgets.QPushButton("Apri STL")
        btn_open.clicked.connect(self.open_stl_dialog)
        top.addWidget(btn_open)

        btn_export_svg = QtWidgets.QPushButton("Esporta SVG")
        btn_export_svg.clicked.connect(self.export_svg_dialog)
        top.addWidget(btn_export_svg)

        btn_recompute = QtWidgets.QPushButton("Ricalcola")
        btn_recompute.clicked.connect(self.recompute)
        top.addWidget(btn_recompute)

        self.path_label = QtWidgets.QLabel("Nessun file caricato")
        top.addWidget(self.path_label, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        controls_wrap = QtWidgets.QWidget()
        controls_wrap.setMinimumWidth(340)
        controls_wrap.setMaximumWidth(420)
        split.addWidget(controls_wrap)

        form = QtWidgets.QFormLayout(controls_wrap)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.line_resolution = self._spin_double(0.2, 12.0, 3.28, 2, form, "Line resolution")
        self.min_arc_radius = self._spin_double(0.0, 25.0, 6.0, 1, form, "Min arc radius")
        self.preview_quality = self._spin_int(5, 100, 55, form, "Preview quality (%)")
        self.preview_arc_limit = self._spin_int(200, 40000, 12000, form, "Preview arc limit")

        self.yaw = self._spin_double(-180.0, 180.0, -35.0, 1, form, "Yaw")
        self.pitch = self._spin_double(-89.0, 89.0, 20.0, 1, form, "Pitch")
        self.camera_scale = self._spin_double(0.2, 2.0, 0.724, 3, form, "Camera scale")
        self.zf = self._spin_double(5.0, 120.0, 25.0, 1, form, "Perspective zf")
        self.depth_gain = self._spin_double(0.2, 5.0, 1.8, 2, form, "Depth gain")

        self.view_angle = self._spin_double(-90.0, 90.0, 0.0, 0, form, "View angle")

        self.arc_mode = QtWidgets.QComboBox()
        self.arc_mode.addItems(["Semicircle (CNC)", "Elliptic"])
        self.arc_mode.currentIndexChanged.connect(self.schedule_recompute)
        form.addRow("Arc mode", self.arc_mode)

        self.ellipse_ratio = self._spin_double(0.20, 1.00, 0.65, 2, form, "Ellipse ratio")

        self.show_arcs = QtWidgets.QCheckBox("Show arcs")
        self.show_arcs.setChecked(True)
        self.show_arcs.stateChanged.connect(self.schedule_recompute)
        form.addRow(self.show_arcs)

        self.show_profile = QtWidgets.QCheckBox("Show simulated profile")
        self.show_profile.setChecked(True)
        self.show_profile.stateChanged.connect(self.schedule_recompute)
        form.addRow(self.show_profile)

        self.rigid_profile = QtWidgets.QCheckBox("Rigid simulation")
        self.rigid_profile.setChecked(True)
        self.rigid_profile.stateChanged.connect(self.schedule_recompute)
        form.addRow(self.rigid_profile)

        self.auto_center = QtWidgets.QCheckBox("Auto-center Z")
        self.auto_center.setChecked(True)
        self.auto_center.stateChanged.connect(self._reload_current_stl)
        form.addRow(self.auto_center)

        gl_wrap = QtWidgets.QWidget()
        gl_layout = QtWidgets.QVBoxLayout(gl_wrap)
        gl_layout.setContentsMargins(0, 0, 0, 0)
        split.addWidget(gl_wrap)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((10, 12, 18, 255))
        self.view.opts["fov"] = 45.0
        gl_layout.addWidget(self.view)

        self.arcs_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0.88, 0.88, 0.88, 0.65),
            width=1.0,
            antialias=False,
            mode="lines",
        )
        self.profile_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0.96, 0.96, 0.96, 0.95),
            width=1.2,
            antialias=False,
            mode="lines",
        )
        self.view.addItem(self.arcs_item)
        self.view.addItem(self.profile_item)
        self._set_gl_camera()

        self.status = QtWidgets.QLabel("Apri un file STL per iniziare.")
        root.addWidget(self.status)

    def _spin_double(
        self,
        low: float,
        high: float,
        value: float,
        decimals: int,
        form: QtWidgets.QFormLayout,
        label: str,
    ) -> QtWidgets.QDoubleSpinBox:
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(low, high)
        w.setDecimals(decimals)
        w.setValue(value)
        w.setSingleStep(max((high - low) / 200.0, 10.0 ** (-decimals)))
        w.valueChanged.connect(self.schedule_recompute)
        form.addRow(label, w)
        return w

    def _spin_int(
        self,
        low: int,
        high: int,
        value: int,
        form: QtWidgets.QFormLayout,
        label: str,
    ) -> QtWidgets.QSpinBox:
        w = QtWidgets.QSpinBox()
        w.setRange(low, high)
        w.setValue(value)
        w.valueChanged.connect(self.schedule_recompute)
        form.addRow(label, w)
        return w

    def _set_gl_camera(self) -> None:
        self.view.setCameraPosition(distance=260.0, elevation=20.0, azimuth=-35.0)

    def _reload_current_stl(self) -> None:
        if self.stl_path is not None:
            self.load_stl(self.stl_path)

    def schedule_recompute(self) -> None:
        self._recompute_timer.start(60)

    def open_stl_dialog(self) -> None:
        start_dir = str(Path.cwd())
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Apri STL",
            start_dir,
            "STL files (*.stl);;All files (*.*)",
        )
        if filename:
            self.load_stl(Path(filename))

    def load_stl(self, stl_path: Path) -> None:
        try:
            ensure_dependencies()
            mesh = load_mesh(stl_path)
            model_vertices, edges, faces = build_model_vertices_and_edges(
                mesh=mesh,
                auto_center=bool(self.auto_center.isChecked()),
                decimals=NORMAL_TOLERANCE_DECIMALS,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Errore caricamento STL", str(exc))
            return

        self.stl_path = stl_path
        self.model_vertices = model_vertices
        self.edges = edges
        self.faces = faces
        self.current_arcs = []
        self.profile_paths = []

        v_min = np.min(model_vertices, axis=0)
        v_max = np.max(model_vertices, axis=0)
        self.model_center = (v_min + v_max) / 2.0
        diag = float(np.linalg.norm(v_max - v_min))
        self.base_orbit_radius = max(10.0, diag * 2.6)
        self.distance_scale = 1.0

        self.path_label.setText(str(stl_path))
        self.recompute()

    def _current_arc_mode(self) -> str:
        txt = self.arc_mode.currentText().strip().lower()
        return "elliptic" if txt.startswith("elliptic") else "semi"

    def _current_ellipse_ratio(self) -> float:
        return clamp(float(self.ellipse_ratio.value()), 0.20, 1.00)

    def _build_camera(self, canvas_width: int, canvas_height: int) -> CameraConfig:
        if self.model_vertices is None:
            return CameraConfig(
                po=(0.9, 9.0, 24.0),
                pr=(0.93, 8.29, 0.47),
                look_up=(0.0, 1.0, 0.0),
                current_scale=float(self.camera_scale.value()),
                zf=float(self.zf.value()),
                canvas_width=max(1, int(canvas_width)),
                canvas_height=max(1, int(canvas_height)),
            )

        yaw = math.radians(float(self.yaw.value()))
        pitch = math.radians(float(self.pitch.value()))
        direction = np.asarray(
            (
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ),
            dtype=np.float64,
        )
        nrm = float(np.linalg.norm(direction))
        if nrm <= EPS:
            direction = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        else:
            direction = direction / nrm

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
            current_scale=float(self.camera_scale.value()),
            zf=float(self.zf.value()),
            canvas_width=max(1, int(canvas_width)),
            canvas_height=max(1, int(canvas_height)),
        )

    def _preview_params(self) -> tuple[list[Edge], float, int, int]:
        if self.model_vertices is None or len(self.edges) == 0:
            return [], float(self.line_resolution.value()), 1, 0

        base_lr = float(self.line_resolution.value())
        base_limit = int(self.preview_arc_limit.value())
        quality_raw = clamp(float(self.preview_quality.value()) / 100.0, 0.05, 1.0)
        q = quality_raw * quality_raw

        edge_count = len(self.edges)
        target_edges = max(4, int(edge_count * q))
        stride = max(1, int(math.ceil(edge_count / target_edges)))
        sampled_edges = self.edges[::stride]

        effective_lr = clamp(base_lr * q, 0.08, base_lr)
        draw_limit = max(100, int(base_limit * q))
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
        if len(self.current_arcs) == 0 or self.model_vertices is None:
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

                arc_hash = (
                    round(float(arc.rect_x), 4),
                    round(float(arc.rect_y), 4),
                    round(float(arc.rect_w), 4),
                    round(float(arc.rect_h), 4),
                    float(arc.start_angle),
                )
                if prev_hash is not None and arc_hash == prev_hash:
                    continue
                prev_hash = arc_hash
                arc_path.append(arc)

            if len(arc_path) >= 2:
                paths.append(arc_path)

        return paths

    def _collect_arc_segments(self, draw_limit: int) -> np.ndarray:
        if not self.show_arcs.isChecked():
            return np.zeros((0, 3), dtype=np.float32)

        segs: list[np.ndarray] = []
        for arc in self.current_arcs[:draw_limit]:
            poly2 = arc_polyline_2d(arc, max_steps=24)
            zv = float(arc.zero_coord[2])
            poly3 = np.column_stack((poly2[:, 0], -poly2[:, 1], np.full((poly2.shape[0],), zv, dtype=np.float64)))
            segs.append(polyline_to_segments_3d(poly3))
        if len(segs) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(segs).astype(np.float32, copy=False)

    def _collect_profile_segments(self) -> np.ndarray:
        if not self.show_profile.isChecked():
            return np.zeros((0, 3), dtype=np.float32)
        if self.last_projected is None:
            return np.zeros((0, 3), dtype=np.float32)

        angle = float(self.view_angle.value())
        rigid = bool(self.rigid_profile.isChecked())
        mode = self._current_arc_mode()
        ratio = self._current_ellipse_ratio() if mode == "elliptic" else 1.0

        segs: list[np.ndarray] = []
        if rigid:
            edges_to_draw = self.last_sampled_edges if self.last_sampled_edges else self.edges
            for edge in edges_to_draw:
                p1 = self.last_projected[edge.start_idx]
                p2 = self.last_projected[edge.end_idx]
                x1, y1 = point_at_angle_projected(p1, angle, self.last_n_view, ellipse_ratio=ratio)
                x2, y2 = point_at_angle_projected(p2, angle, self.last_n_view, ellipse_ratio=ratio)
                segs.append(
                    np.asarray(
                        [[x1, -y1, float(p1[2])], [x2, -y2, float(p2[2])]],
                        dtype=np.float32,
                    )
                )
        else:
            for path in self.profile_paths:
                if len(path) < 2:
                    continue
                pts = np.zeros((len(path), 3), dtype=np.float32)
                for i, arc in enumerate(path):
                    x, y = point_on_arc_compatible(arc, angle)
                    pts[i, 0] = float(x)
                    pts[i, 1] = -float(y)
                    pts[i, 2] = float(arc.zero_coord[2])
                segs.append(polyline_to_segments_3d(pts))

        if len(segs) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(segs).astype(np.float32, copy=False)

    def _normalize_segments(self, *segments: np.ndarray) -> tuple[np.ndarray, ...]:
        valid = [s for s in segments if s.shape[0] > 0]
        if len(valid) == 0:
            return segments

        xyz = np.vstack(valid)
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        center = (min_xyz + max_xyz) / 2.0
        span = float(
            max(
                max_xyz[0] - min_xyz[0],
                max_xyz[1] - min_xyz[1],
                max_xyz[2] - min_xyz[2],
                1.0,
            )
        )
        scale = 220.0 / span

        out: list[np.ndarray] = []
        for s in segments:
            if s.shape[0] == 0:
                out.append(s)
                continue
            t = s.copy()
            t[:, 0] = (t[:, 0] - center[0]) * scale
            t[:, 1] = (t[:, 1] - center[1]) * scale
            t[:, 2] = (t[:, 2] - center[2]) * scale * float(self.depth_gain.value())
            out.append(t)
        return tuple(out)

    def recompute(self) -> None:
        if self.model_vertices is None:
            return

        self.yaw_deg = float(self.yaw.value())
        self.pitch_deg = float(self.pitch.value())

        w = max(20, self.view.width())
        h = max(20, self.view.height())
        camera = self._build_camera(w, h)

        try:
            mtx = build_model_to_window_matrix(camera)
            projected = model_to_window(self.model_vertices, mtx)
            pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), mtx)[0]
            n_view = float(pr_view[2])
            self.last_projected = projected
            self.last_n_view = n_view
        except Exception as exc:
            self.status.setText(f"Errore proiezione: {exc}")
            return

        sampled_edges, effective_lr, stride, draw_limit = self._preview_params()
        self.last_sampled_edges = sampled_edges
        self.last_edge_stride = stride
        self.last_preview_lr = effective_lr

        try:
            arcs = generate_arcs(
                model_vertices=self.model_vertices,
                zero_vertices=projected,
                edges=sampled_edges,
                view_points_per_unit_length=effective_lr,
                n_view=n_view,
                dedupe_decimals=6,
                min_arc_radius=float(self.min_arc_radius.value()),
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
            self.status.setText(f"Errore calcolo archi: {exc}")
            return

        shown = min(len(self.current_arcs), draw_limit)
        arc_seg = self._collect_arc_segments(draw_limit=shown)
        profile_seg = self._collect_profile_segments()
        arc_seg, profile_seg = self._normalize_segments(arc_seg, profile_seg)

        self.arcs_item.setData(pos=arc_seg, mode="lines", color=(0.88, 0.88, 0.88, 0.65), width=1.0, antialias=False)
        self.profile_item.setData(
            pos=profile_seg,
            mode="lines",
            color=(0.96, 0.96, 0.96, 0.95),
            width=1.2,
            antialias=False,
        )

        mode_txt = "ELL" if self._current_arc_mode() == "elliptic" else "SEMI"
        if mode_txt == "ELL":
            mode_txt = f"{mode_txt}({self._current_ellipse_ratio():.2f})"
        self.status.setText(
            " | ".join(
                [
                    f"STL: {self.stl_path.name if self.stl_path else '-'}",
                    f"Spigoli: {len(self.edges)}",
                    f"Archi: {len(self.current_arcs)}",
                    f"Preview: {shown}",
                    f"Q: {int(self.preview_quality.value())}%",
                    f"LR eff: {self.last_preview_lr:.2f}",
                    f"Edge step: {self.last_edge_stride}",
                    f"Paths: {len(self.profile_paths)}",
                    f"Arc: {mode_txt}",
                    f"Yaw/Pitch: {self.yaw_deg:.1f}/{self.pitch_deg:.1f}",
                ]
            )
        )

    def export_svg_dialog(self) -> None:
        if self.model_vertices is None or self.stl_path is None:
            QtWidgets.QMessageBox.warning(self, "Nessun modello", "Carica prima un file STL.")
            return
        if len(self.current_arcs) == 0:
            self.recompute()
        if len(self.current_arcs) == 0:
            QtWidgets.QMessageBox.warning(self, "Nessun arco", "Nessun arco disponibile per l'export.")
            return

        default_name = f"{self.stl_path.stem}_arcs.svg"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Esporta SVG",
            str((Path.cwd() / default_name).resolve()),
            "SVG file (*.svg);;All files (*.*)",
        )
        if not save_path:
            return

        try:
            write_svg(Path(save_path), self.current_arcs, stroke_width=0.15)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Errore scrittura SVG", str(exc))
            return
        self.status.setText(f"SVG esportato: {save_path} | Archi: {len(self.current_arcs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScratchHologram GPU preview (Qt/OpenGL)")
    parser.add_argument("--stl", type=Path, default=None, help="Optional STL to open at startup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = GpuScratchPreview(initial_stl=args.stl)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
