#!/usr/bin/env python3
"""
Scratch hologram computational pipeline.

Pipeline stage
--------------
This module converts a triangular mesh into sampled scratch arcs, then exports
the resulting pattern as SVG, optional diagnostic JSON, optional simulation
HTML, and optional G-code through the desktop application layer.

Input / output
--------------
Input is a triangular mesh (typically STL) plus a camera configuration and arc
generation parameters. Output is a sampled set of scratch arcs in screen space,
with optional visibility diagnostics derived from a conservative Z-buffer.

Key parameters
--------------
`line_resolution` controls edge sampling density in samples per model unit.
`min_arc_radius` rejects small projected arcs to limit manufacturing noise.
`arc_mode` selects semicircular or elliptic arc generation.
`zf` and `current_scale` control the projection model and effective zoom.

Coordinate conventions
----------------------
Model coordinates are assumed to be right-handed and expressed in mesh units
(typically millimeters, but the pipeline treats them as generic metric units).
The camera target is `pr`, the camera origin is `po`, and the viewing direction
in model space points from `po` toward `pr`. View-space `+z` points backward
toward the camera, so visible geometry in front of the camera has negative
view-space `z`; screen-space depth therefore uses `-z_view` as a positive
distance-like quantity. Screen-space `x` grows rightward, screen-space `y`
grows downward after the window transform to match Tk/SVG raster conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
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


# WARNING: this epsilon is intentionally small because it is reused in matrix
# normalization and geometric predicates; increasing it improves robustness but
# risks hiding thin geometry.
EPS = 1e-12
# WARNING: rounding is a design decision used to merge nearly coincident mesh
# vertices; the trade-off is better edge stability at the cost of small detail.
NORMAL_TOLERANCE_DECIMALS = 4
# WARNING: the visibility cull uses a deliberately conservative orthographic
# depth tolerance; the trade-off is fewer holes on visible faces at the cost of
# some false positives near silhouettes.
ORTHO_DEPTH_EPSILON_FACTOR = 2.0e-2


@dataclass
class CameraConfig:
    """
    Camera parameters shared by projection and visibility stages.

    Parameters
    ----------
    po : tuple[float, float, float]
        Camera origin in model units.
    pr : tuple[float, float, float]
        Camera target point in model units.
    look_up : tuple[float, float, float]
        Approximate up vector used to build the camera basis.
    current_scale : float
        Screen scaling factor, dimensionless.
    zf : float
        Perspective denominator used by the projective transform, in model
        units. Large values approximate orthographic projection.
    canvas_width : int
        Output viewport width in pixels.
    canvas_height : int
        Output viewport height in pixels.

    Returns
    -------
    None
        Dataclass container.

    Notes
    -----
    The view direction points from `po` to `pr`.

    Assumptions
    -----------
    `look_up` must not be collinear with the viewing direction.
    """

    po: tuple[float, float, float]
    pr: tuple[float, float, float]
    look_up: tuple[float, float, float]
    current_scale: float
    zf: float
    canvas_width: int
    canvas_height: int


@dataclass
class PipelineConfig:
    """
    Scalar parameters controlling edge sampling, projection filtering, and export.

    Parameters
    ----------
    line_resolution : float
        Edge sampling density in samples per model unit.
    auto_center : bool
        If True, shifts the mesh so the front-most `z` plane is mapped to
        `z = 0` before projection.
    stroke_width : float
        SVG stroke width in screen units (pixels in the generated viewBox).
    dedupe_decimals : int
        Decimal precision used for projected-sample deduplication.
    min_arc_radius : float
        Minimum projected arc radius in pixels.
    arc_mode : str
        Arc generation mode: `"semi"` or `"elliptic"`.
    ellipse_ratio : float
        Height/width ratio for elliptic arc generation, dimensionless.

    Returns
    -------
    None
        Dataclass container.

    Notes
    -----
    Parameters are evaluated after mesh loading and before SVG generation.

    Assumptions
    -----------
    The caller validates ranges before execution.
    """

    line_resolution: float
    auto_center: bool
    stroke_width: float
    dedupe_decimals: int
    min_arc_radius: float
    arc_mode: str
    ellipse_ratio: float


@dataclass
class Edge:
    """
    Topological mesh edge used by edge sampling and arc generation.

    Parameters
    ----------
    edge_id : int
        Stable edge identifier.
    start_idx : int
        Start vertex index in the remapped mesh vertex array.
    end_idx : int
        End vertex index in the remapped mesh vertex array.
    normal : tuple[float, float, float], optional
        Averaged edge normal in model coordinates, dimensionless.

    Returns
    -------
    None
        Dataclass container.

    Notes
    -----
    The edge is treated as undirected for topology, but `start_idx` and
    `end_idx` preserve a stable ordering for reproducible edge sampling.

    Assumptions
    -----------
    Indices refer to the deduplicated vertex array returned by
    `build_model_vertices_and_edges`.
    """

    edge_id: int
    start_idx: int
    end_idx: int
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class Arc:
    """
    Screen-space scratch arc emitted by the arc generation stage.

    Parameters
    ----------
    edge_id : int
        Source mesh edge identifier.
    zero_coord : tuple[float, float, float]
        Projected sample position in screen coordinates and projected depth.
    rect_x, rect_y : float
        Upper-left corner of the Tk/SVG arc bounding box in pixels.
    rect_w, rect_h : float
        Arc bounding box size in pixels.
    start_angle : float
        Start angle in degrees using the Tk/SVG convention adopted here.
    sweep_angle : float
        Signed angular extent in degrees.

    Returns
    -------
    None
        Dataclass container.

    Notes
    -----
    Arc geometry is stored in the same convention used by the preview canvas so
    SVG export can reuse the same projection and arc generation decisions.

    Assumptions
    -----------
    Bounding-box values are already expressed in screen space.
    """

    edge_id: int
    zero_coord: tuple[float, float, float]
    rect_x: float
    rect_y: float
    rect_w: float
    rect_h: float
    start_angle: float
    sweep_angle: float


@dataclass
class ZBuffer:
    """
    Coarse screen-space depth buffer used by conservative visibility culling.

    Parameters
    ----------
    depth : np.ndarray
        Two-dimensional depth grid, in positive screen-space depth units.
        `+inf` denotes an uncovered cell.
    screen_width : int
        Width of the source viewport in pixels.
    screen_height : int
        Height of the source viewport in pixels.

    Returns
    -------
    None
        Dataclass container.

    Notes
    -----
    The depth grid resolution can differ from the viewport resolution; mapping
    is handled explicitly by `screen_to_zbuffer_coords`.

    Assumptions
    -----------
    Depth values represent the closest triangle sample written into each cell.
    """

    depth: np.ndarray
    screen_width: int
    screen_height: int


def ensure_dependencies() -> None:
    """
    Validate optional runtime dependencies before executing the pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This explicit check keeps the failure mode readable; the trade-off is a
    small amount of startup overhead before mesh processing.

    Assumptions
    -----------
    Missing packages should stop execution immediately.
    """

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
    """
    Normalize one vector or a batch of vectors.

    Parameters
    ----------
    v : np.ndarray
        Vector array of shape `(3,)` or `(N, 3)`, dimensionless or expressed in
        any linear unit.

    Returns
    -------
    np.ndarray
        Normalized vector array with the same shape as `v`.

    Notes
    -----
    A small epsilon is used to avoid division by zero in batched mode.

    Assumptions
    -----------
    Single-vector input must have non-zero norm.
    """

    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n < EPS:
            raise ValueError("Cannot normalize a zero-length vector.")
        return arr / n

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, EPS, None)


def load_mesh(stl_path: Path) -> trimesh.Trimesh:
    """
    Load and sanitize a triangular mesh from disk.

    Parameters
    ----------
    stl_path : Path
        Path to the mesh file. The current workflow expects STL.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh ready for edge extraction.

    Notes
    -----
    The cleanup stage removes degenerate faces, duplicate faces, and
    unreferenced vertices. The design decision improves stable edge sampling,
    while the trade-off is that malformed source topology is normalized instead
    of preserved verbatim.

    Assumptions
    -----------
    The file contains a triangular mesh or a scene that can be concatenated
    into one.
    """

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
) -> tuple[np.ndarray, list[Edge], np.ndarray]:
    """
    Build the deduplicated mesh arrays used by edge sampling and projection.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular mesh in model units.
    auto_center : bool
        If True, translates the mesh so the maximum `z` becomes `0`.
    decimals : int
        Decimal precision used to round vertices before topology extraction.

    Returns
    -------
    tuple[np.ndarray, list[Edge], np.ndarray]
        Deduplicated vertex array `(N, 3)` in model units, filtered edge list,
        and remapped face index array `(M, 3)`.

    Notes
    -----
    Vertex rounding is a deliberate design decision because it stabilizes mesh
    deduplication and edge sampling across noisy STL exports; the trade-off is
    that sub-tolerance details may collapse. Coplanar shared edges are dropped
    to approximate merged faces, which reduces scratch noise but hides internal
    triangulation.

    Assumptions
    -----------
    `mesh.face_normals` is available and aligned with `mesh.faces`.
    """

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

    remapped_faces: list[tuple[int, int, int]] = []
    seen_faces: set[tuple[int, int, int]] = set()

    for face_idx, tri in enumerate(faces):
        tri_new = [int(remap[int(tri[0])]), int(remap[int(tri[1])]), int(remap[int(tri[2])])]

        # Skip invalid/degenerate triangles after rounding/remap.
        if len(set(tri_new)) < 3:
            continue

        face_key = (tri_new[0], tri_new[1], tri_new[2])
        if face_key not in seen_faces:
            seen_faces.add(face_key)
            remapped_faces.append(face_key)

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
        normals_arr = np.asarray(normals, dtype=np.float64) if normals else np.zeros((0, 3), dtype=np.float64)
        if normals_arr.shape[0] == 0:
            edge_normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            edge_normal = np.sum(normals_arr, axis=0)
            if float(np.linalg.norm(edge_normal)) < EPS:
                edge_normal = normals_arr[0]
            edge_normal = normalize(edge_normal)

        filtered_edges.append(
            Edge(
                edge_id=edge.edge_id,
                start_idx=edge.start_idx,
                end_idx=edge.end_idx,
                normal=(float(edge_normal[0]), float(edge_normal[1]), float(edge_normal[2])),
            )
        )

    remapped_faces_arr = np.asarray(remapped_faces, dtype=np.int64)
    if remapped_faces_arr.size == 0:
        remapped_faces_arr = np.zeros((0, 3), dtype=np.int64)

    return np.asarray(unique_vertices, dtype=np.float64), filtered_edges, remapped_faces_arr


def build_model_to_view_matrix(camera: CameraConfig) -> np.ndarray:
    """
    Construct the rigid model-to-view transform.

    Parameters
    ----------
    camera : CameraConfig
        Camera configuration in model units and pixels.

    Returns
    -------
    np.ndarray
        Homogeneous transform matrix of shape `(4, 4)`.

    Notes
    -----
    The returned basis uses `u` right, `v` up, and `n` pointing from the target
    back toward the camera. This convention keeps `z_view` negative in front of
    the camera, which simplifies the later visibility cull.

    Assumptions
    -----------
    The camera basis is well-defined and non-degenerate.
    """

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

    out = np.eye(4, dtype=np.float64)
    out[0, 0:3] = u
    out[1, 0:3] = v
    out[2, 0:3] = n
    out[0, 3] = -float(np.dot(u, po))
    out[1, 3] = -float(np.dot(v, po))
    out[2, 3] = -float(np.dot(n, po))
    return out


def build_model_to_window_matrix(camera: CameraConfig) -> np.ndarray:
    """
    Construct the full model-to-screen projection matrix.

    Parameters
    ----------
    camera : CameraConfig
        Camera configuration in model units and pixels.

    Returns
    -------
    np.ndarray
        Homogeneous transform matrix of shape `(4, 4)` mapping model
        coordinates to screen coordinates.

    Notes
    -----
    `zf` controls the projective denominator. Large `zf` values approximate an
    orthographic projection, which is a design decision used by the desktop app
    to reduce perspective distortion; the trade-off is weaker depth cues.

    Assumptions
    -----------
    `camera.canvas_width` and `camera.canvas_height` are positive.
    """

    model_to_view = build_model_to_view_matrix(camera)

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
    """
    Project model coordinates into screen coordinates.

    Parameters
    ----------
    points_xyz : np.ndarray
        Point array of shape `(N, 3)` in model units.
    model_to_window_mtx : np.ndarray
        Homogeneous projection matrix of shape `(4, 4)`.

    Returns
    -------
    np.ndarray
        Projected array of shape `(N, 3)` where `x` and `y` are screen pixels
        and `z` is the projected depth-like coordinate after homogeneous divide.

    Notes
    -----
    Division by a small homogeneous `w` is clamped for floating-point
    robustness. The trade-off is finite output at the cost of very near-plane
    precision.

    Assumptions
    -----------
    `points_xyz` is finite and row-major.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack((points, ones))

    transformed = (model_to_window_mtx @ homo.T).T
    w = transformed[:, 3:4]
    w_safe = np.where(np.abs(w) < EPS, EPS, w)
    transformed[:, 0:3] = transformed[:, 0:3] / w_safe
    return transformed[:, 0:3]


def model_to_view(points_xyz: np.ndarray, model_to_view_mtx: np.ndarray) -> np.ndarray:
    """
    Transform model coordinates into camera view coordinates.

    Parameters
    ----------
    points_xyz : np.ndarray
        Point array of shape `(N, 3)` in model units.
    model_to_view_mtx : np.ndarray
        Homogeneous rigid transform matrix of shape `(4, 4)`.

    Returns
    -------
    np.ndarray
        View-space point array of shape `(N, 3)` in model units.

    Notes
    -----
    No perspective divide is applied here; this stage is used by the visibility
    cull and depth computations.

    Assumptions
    -----------
    `points_xyz` is finite and row-major.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack((points, ones))
    transformed = (model_to_view_mtx @ homo.T).T
    return transformed[:, 0:3]


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute unit normals for a triangular mesh.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex array of shape `(N, 3)` in model units.
    faces : np.ndarray
        Triangle index array of shape `(M, 3)`.

    Returns
    -------
    np.ndarray
        Unit face normals of shape `(M, 3)`.

    Notes
    -----
    Degenerate triangles return zero normals rather than raising, because the
    visibility cull prefers to skip invalid faces instead of aborting.

    Assumptions
    -----------
    `faces` indexes `vertices` correctly.
    """

    verts = np.asarray(vertices, dtype=np.float64)
    tri_idx = np.asarray(faces, dtype=np.int64)
    if tri_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    tri = verts[tri_idx]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = lengths > EPS
    if np.any(valid):
        normals[valid] = cross[valid] / lengths[valid, None]
    return normals


def backface_mask(
    face_normals: np.ndarray,
    view_dir: np.ndarray,
    epsilon: float = 1.0e-9,
) -> np.ndarray:
    """
    Compute a front-facing mask for a set of face normals.

    Parameters
    ----------
    face_normals : np.ndarray
        Face normals of shape `(M, 3)`, dimensionless.
    view_dir : np.ndarray
        Viewing direction pointing from camera origin to target, dimensionless.
    epsilon : float, optional
        Signed dot-product tolerance used to reject nearly grazing faces.

    Returns
    -------
    np.ndarray
        Boolean mask of shape `(M,)`, True for faces considered front-facing.

    Notes
    -----
    A tolerance is used because floating-point noise near silhouette triangles
    can flip the sign of the dot product.

    Assumptions
    -----------
    `view_dir` is non-zero.
    """

    normals = np.asarray(face_normals, dtype=np.float64)
    if normals.size == 0:
        return np.zeros((0,), dtype=bool)
    forward = normalize(np.asarray(view_dir, dtype=np.float64))
    dots = normals @ forward
    return dots < -abs(float(epsilon))


def project_to_screen(
    points_xyz: np.ndarray,
    camera: CameraConfig,
    *,
    model_to_window_mtx: np.ndarray | None = None,
    model_to_view_mtx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project model-space points into screen coordinates plus positive depth.

    Parameters
    ----------
    points_xyz : np.ndarray
        Point array of shape `(N, 3)` in model units.
    camera : CameraConfig
        Camera configuration used for projection.
    model_to_window_mtx : np.ndarray | None, optional
        Precomputed projection matrix. If omitted, it is built internally.
    model_to_view_mtx : np.ndarray | None, optional
        Precomputed rigid view matrix. If omitted, it is built internally.

    Returns
    -------
    np.ndarray
        Array of shape `(N, 3)` with `x`, `y` in screen pixels and `depth` as
        positive `-z_view` in model units.

    Notes
    -----
    Depth uses view-space distance rather than post-projection `z` because the
    visibility cull is conceptually orthographic; the trade-off is simpler and
    more stable comparisons, while perspective-specific depth precision is not
    modeled here.

    Assumptions
    -----------
    The provided matrices, if any, match the supplied camera.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    window_mtx = model_to_window_mtx if model_to_window_mtx is not None else build_model_to_window_matrix(camera)
    view_mtx = model_to_view_mtx if model_to_view_mtx is not None else build_model_to_view_matrix(camera)

    projected = model_to_window(points, window_mtx)
    view_points = model_to_view(points, view_mtx)

    screen = np.zeros((points.shape[0], 3), dtype=np.float64)
    screen[:, 0] = projected[:, 0]
    screen[:, 1] = projected[:, 1]
    screen[:, 2] = np.maximum(-view_points[:, 2], 0.0)
    return screen


def screen_to_zbuffer_coords(
    screen_xy: np.ndarray,
    zbuffer: ZBuffer,
    *,
    clamp_to_screen: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map screen pixel coordinates into the coarse Z-buffer grid.

    Parameters
    ----------
    screen_xy : np.ndarray
        Screen-space coordinates of shape `(N, 2)` in pixels.
    zbuffer : ZBuffer
        Coarse Z-buffer descriptor.
    clamp_to_screen : bool, optional
        If True, clamps coordinates to the viewport before resampling into the
        Z-buffer grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Fractional Z-buffer coordinates `(fx, fy)` in grid cells.

    Notes
    -----
    The explicit mapping is a design decision because triangles and edge
    sampling must share identical projection semantics; the trade-off is a
    slightly more verbose implementation in exchange for reproducibility.

    Assumptions
    -----------
    `zbuffer.screen_width` and `zbuffer.screen_height` describe the original
    viewport used to create the depth grid.
    """

    xy = np.asarray(screen_xy, dtype=np.float64)
    if xy.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty

    sx = xy[:, 0]
    sy = xy[:, 1]
    if clamp_to_screen:
        sx = np.clip(sx, 0.0, float(max(zbuffer.screen_width - 1, 0)))
        sy = np.clip(sy, 0.0, float(max(zbuffer.screen_height - 1, 0)))

    height, width = zbuffer.depth.shape
    if width <= 1:
        fx = np.zeros(xy.shape[0], dtype=np.float64)
    else:
        denom_x = max(float(zbuffer.screen_width - 1), 1.0)
        fx = sx * (float(width - 1) / denom_x)

    if height <= 1:
        fy = np.zeros(xy.shape[0], dtype=np.float64)
    else:
        denom_y = max(float(zbuffer.screen_height - 1), 1.0)
        fy = sy * (float(height - 1) / denom_y)

    return fx, fy


def assert_zbuffer_corner_mapping(zbuffer: ZBuffer) -> None:
    """
    Verify that viewport corners map exactly to Z-buffer corners.

    Parameters
    ----------
    zbuffer : ZBuffer
        Coarse Z-buffer descriptor.

    Returns
    -------
    None

    Notes
    -----
    This debug assertion guards against off-by-one projection errors, with
    particular attention to border cells where visibility holes are common.

    Assumptions
    -----------
    The Z-buffer shape is already allocated.
    """

    height, width = zbuffer.depth.shape
    corners = np.asarray(
        [
            (0.0, 0.0),
            (float(max(zbuffer.screen_width - 1, 0)), 0.0),
            (0.0, float(max(zbuffer.screen_height - 1, 0))),
            (
                float(max(zbuffer.screen_width - 1, 0)),
                float(max(zbuffer.screen_height - 1, 0)),
            ),
        ],
        dtype=np.float64,
    )
    fx, fy = screen_to_zbuffer_coords(corners, zbuffer, clamp_to_screen=True)
    expected_x = np.asarray(
        [0.0, float(max(width - 1, 0)), 0.0, float(max(width - 1, 0))],
        dtype=np.float64,
    )
    expected_y = np.asarray(
        [0.0, 0.0, float(max(height - 1, 0)), float(max(height - 1, 0))],
        dtype=np.float64,
    )
    if not np.allclose(fx, expected_x, atol=1.0e-9):
        raise AssertionError("screen_to_zbuffer_coords x mapping mismatch at screen corners")
    if not np.allclose(fy, expected_y, atol=1.0e-9):
        raise AssertionError("screen_to_zbuffer_coords y mapping mismatch at screen corners")


def build_zbuffer_conservative(
    screen_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    screen_width: int,
    screen_height: int,
    face_mask: np.ndarray | None = None,
    resolution: int = 1024,
    coverage_dilation_size: int = 3,
) -> ZBuffer:
    """
    Rasterize mesh faces into a conservative coarse Z-buffer.

    Parameters
    ----------
    screen_vertices : np.ndarray
        Vertex array of shape `(N, 3)` in screen coordinates and positive depth.
    faces : np.ndarray
        Triangle index array of shape `(M, 3)`.
    screen_width : int
        Width of the source viewport in pixels.
    screen_height : int
        Height of the source viewport in pixels.
    face_mask : np.ndarray | None, optional
        Boolean mask used to include only selected faces.
    resolution : int, optional
        Target longer-side resolution for the coarse Z-buffer, in grid cells.
    coverage_dilation_size : int, optional
        Square dilation size applied to covered cells after inside-triangle
        rasterization.

    Returns
    -------
    ZBuffer
        Coarse depth grid with `+inf` for uncovered cells.

    Notes
    -----
    Only barycentrically valid pixels are written, then dilated. This is a
    conservative design decision: it expands coverage to reduce holes during
    edge sampling, while the trade-off is a higher chance of false-positive
    occlusion near silhouettes. Depth starts at `+inf`, so coverage is defined
    strictly by `depth < +inf`.

    Assumptions
    -----------
    `screen_vertices` and `faces` are already expressed in the same projection
    convention returned by `project_to_screen`.
    """

    screen = np.asarray(screen_vertices, dtype=np.float64)
    tri_idx = np.asarray(faces, dtype=np.int64)
    screen_w = max(1, int(screen_width))
    screen_h = max(1, int(screen_height))
    target = max(64, int(resolution))

    if screen_w >= screen_h:
        width = target
        height = max(1, int(math.ceil(target * (float(screen_h) / float(screen_w)))))
    else:
        height = target
        width = max(1, int(math.ceil(target * (float(screen_w) / float(screen_h)))))

    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    if screen.shape[0] == 0 or tri_idx.size == 0:
        return ZBuffer(
            depth=zbuffer,
            screen_width=screen_w,
            screen_height=screen_h,
        )

    if face_mask is not None:
        mask = np.asarray(face_mask, dtype=bool)
        if mask.shape[0] == tri_idx.shape[0]:
            tri_idx = tri_idx[mask]
    if tri_idx.size == 0:
        return ZBuffer(
            depth=zbuffer,
            screen_width=screen_w,
            screen_height=screen_h,
        )

    zbuffer_meta = ZBuffer(depth=zbuffer, screen_width=screen_w, screen_height=screen_h)
    assert_zbuffer_corner_mapping(zbuffer_meta)
    dilation_radius = max(0, (int(coverage_dilation_size) - 1) // 2)

    for tri in tri_idx:
        tri_screen = screen[tri]
        p = tri_screen[:, :2]
        d = tri_screen[:, 2]
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(d)):
            continue
        if np.any(d <= 0.0):
            continue

        px, py = screen_to_zbuffer_coords(
            np.column_stack((p[:, 0], p[:, 1])),
            zbuffer_meta,
            clamp_to_screen=True,
        )
        if not np.all(np.isfinite(px)) or not np.all(np.isfinite(py)):
            continue
        vertex_ix = np.clip(np.rint(px).astype(np.int64), 0, width - 1)
        vertex_iy = np.clip(np.rint(py).astype(np.int64), 0, height - 1)
        check_fx, check_fy = screen_to_zbuffer_coords(
            p,
            zbuffer_meta,
            clamp_to_screen=True,
        )
        check_ix = np.clip(np.rint(check_fx).astype(np.int64), 0, width - 1)
        check_iy = np.clip(np.rint(check_fy).astype(np.int64), 0, height - 1)
        if not np.array_equal(vertex_ix, check_ix):
            raise AssertionError("Triangle vertex x pixel mapping diverged from raster mapping")
        if not np.array_equal(vertex_iy, check_iy):
            raise AssertionError("Triangle vertex y pixel mapping diverged from raster mapping")

        denom = ((py[1] - py[2]) * (px[0] - px[2])) + ((px[2] - px[1]) * (py[0] - py[2]))
        if abs(float(denom)) < EPS:
            continue

        ix0 = max(0, int(math.floor(float(np.min(px)))))
        ix1 = min(width - 1, int(math.ceil(float(np.max(px)))))
        iy0 = max(0, int(math.floor(float(np.min(py)))))
        iy1 = min(height - 1, int(math.ceil(float(np.max(py)))))
        if ix1 < ix0 or iy1 < iy0:
            continue

        xs = np.arange(ix0, ix1 + 1, dtype=np.float64)
        ys = np.arange(iy0, iy1 + 1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

        a = (((py[1] - py[2]) * (grid_x - px[2])) + ((px[2] - px[1]) * (grid_y - py[2]))) / denom
        b = (((py[2] - py[0]) * (grid_x - px[2])) + ((px[0] - px[2]) * (grid_y - py[2]))) / denom
        c = 1.0 - a - b
        inside = (a >= -1.0e-6) & (b >= -1.0e-6) & (c >= -1.0e-6)
        if not np.any(inside):
            continue

        tri_depth = (a * d[0]) + (b * d[1]) + (c * d[2])
        inside_y, inside_x = np.nonzero(inside)
        cell_x = ix0 + inside_x
        cell_y = iy0 + inside_y
        cell_depth = tri_depth[inside_y, inside_x]

        for dy in range(-dilation_radius, dilation_radius + 1):
            ny = np.clip(cell_y + dy, 0, height - 1)
            for dx in range(-dilation_radius, dilation_radius + 1):
                nx = np.clip(cell_x + dx, 0, width - 1)
                np.minimum.at(zbuffer, (ny, nx), cell_depth)

    return ZBuffer(
        depth=zbuffer,
        screen_width=screen_w,
        screen_height=screen_h,
    )


def write_png_image(
    png_path: Path,
    image: np.ndarray,
) -> None:
    """
    Write a grayscale or RGB PNG without external imaging dependencies.

    Parameters
    ----------
    png_path : Path
        Destination file path.
    image : np.ndarray
        Image array of shape `(H, W)` or `(H, W, 3)` with 8-bit values.

    Returns
    -------
    None

    Notes
    -----
    A tiny custom writer keeps the diagnostic path self-contained; the trade-off
    is limited format support compared with using Pillow.

    Assumptions
    -----------
    `image` is already quantized to `uint8` semantics.
    """

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim not in (2, 3):
        raise ValueError("PNG image must be HxW or HxWx3.")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("RGB PNG image must have exactly 3 channels.")

    height, width = pixels.shape[:2]
    if height <= 0 or width <= 0:
        return

    color_type = 0 if pixels.ndim == 2 else 2
    if pixels.ndim == 2:
        raw = b"".join(
            b"\x00" + bytes(pixels[row_idx].tolist())
            for row_idx in range(height)
        )
    else:
        raw = b"".join(
            b"\x00" + pixels[row_idx].tobytes()
            for row_idx in range(height)
        )
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(
        chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0),
        )
    )
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    Path(png_path).write_bytes(bytes(png))


def write_zbuffer_coverage_mask_png(
    png_path: Path,
    zbuffer: ZBuffer,
) -> None:
    """
    Export a binary visualization of Z-buffer coverage.

    Parameters
    ----------
    png_path : Path
        Destination file path.
    zbuffer : ZBuffer
        Coarse depth buffer to visualize.

    Returns
    -------
    None

    Notes
    -----
    Covered cells (`depth < +inf`) are written in white, uncovered cells in
    black, matching the visibility-cull convention.

    Assumptions
    -----------
    `zbuffer.depth` is a finite 2-D array or empty.
    """

    mask = np.where(np.isfinite(zbuffer.depth), 255, 0).astype(np.uint8)
    write_png_image(png_path, mask)


def write_sample_visibility_overlay_png(
    png_path: Path,
    zbuffer: ZBuffer,
    sample_pixels: np.ndarray,
    visible_mask: np.ndarray,
) -> None:
    """
    Export a diagnostic overlay of edge sampling decisions on the Z-buffer grid.

    Parameters
    ----------
    png_path : Path
        Destination file path.
    zbuffer : ZBuffer
        Coarse depth buffer used as the background mask.
    sample_pixels : np.ndarray
        Edge-sampling positions of shape `(N, 2)` in Z-buffer grid indices.
    visible_mask : np.ndarray
        Boolean visibility decisions of shape `(N,)`.

    Returns
    -------
    None

    Notes
    -----
    Kept samples are green and culled samples are red. This is intended for
    debugging projection and visibility mismatches, not for publication output.

    Assumptions
    -----------
    `sample_pixels` and `visible_mask` refer to the same edge-sampling order.
    """

    base = np.where(np.isfinite(zbuffer.depth), 255, 0).astype(np.uint8)
    image = np.repeat(base[:, :, None], 3, axis=2)

    coords = np.asarray(sample_pixels, dtype=np.int64)
    visible = np.asarray(visible_mask, dtype=bool)
    if coords.shape[0] > 0:
        valid = (
            np.isfinite(coords[:, 0])
            & np.isfinite(coords[:, 1])
        )
        if np.any(valid):
            clamped_x = np.clip(coords[valid, 0], 0, image.shape[1] - 1)
            clamped_y = np.clip(coords[valid, 1], 0, image.shape[0] - 1)
            valid_visible = visible[valid]

            culled_idx = np.where(~valid_visible)[0]
            if culled_idx.size > 0:
                image[clamped_y[culled_idx], clamped_x[culled_idx]] = np.asarray((255, 0, 0), dtype=np.uint8)

            kept_idx = np.where(valid_visible)[0]
            if kept_idx.size > 0:
                image[clamped_y[kept_idx], clamped_x[kept_idx]] = np.asarray((0, 255, 0), dtype=np.uint8)

    write_png_image(png_path, image)


def visible_mask_for_edge_samples(
    screen_samples: np.ndarray,
    zbuffer: ZBuffer,
    *,
    depth_epsilon: float,
    window_size: int = 5,
    return_empty_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Evaluate conservative visibility for projected edge sampling points.

    Parameters
    ----------
    screen_samples : np.ndarray
        Sample array of shape `(N, 3)` in screen coordinates and positive depth.
    zbuffer : ZBuffer
        Coarse depth grid built from the same projection.
    depth_epsilon : float
        Additive depth tolerance in model units.
    window_size : int, optional
        Square neighborhood size in Z-buffer cells.
    return_empty_mask : bool, optional
        If True, also returns a mask indicating samples that touched empty or
        out-of-bounds neighborhood cells.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        Visibility mask, optionally paired with the empty-neighborhood mask.

    Notes
    -----
    The rule is intentionally conservative: if any neighborhood cell is empty,
    the sample is kept. Otherwise the sample is kept when its depth is no
    greater than `min_depth + depth_epsilon`. The design decision reduces holes
    on visible faces, while the trade-off is extra false positives.

    Assumptions
    -----------
    `screen_samples` was produced by `project_to_screen`.
    """

    screen = np.asarray(screen_samples, dtype=np.float64)
    if screen.shape[0] == 0:
        empty = np.zeros((0,), dtype=bool)
        return (empty, empty) if return_empty_mask else empty

    depths = screen[:, 2]
    valid = (
        np.isfinite(screen[:, 0])
        & np.isfinite(screen[:, 1])
        & np.isfinite(depths)
        & (depths > 0.0)
    )
    if zbuffer.depth.size == 0:
        empty = valid.copy()
        return (valid.copy(), empty) if return_empty_mask else valid.copy()

    height, width = zbuffer.depth.shape
    fx, fy = screen_to_zbuffer_coords(screen[:, :2], zbuffer)

    in_bounds = (
        (fx >= -0.5)
        & (fx <= (width - 0.5))
        & (fy >= -0.5)
        & (fy <= (height - 0.5))
    )

    visible = valid.copy()
    empty_query = valid & (~in_bounds)
    idx = np.where(valid & in_bounds)[0]
    if idx.size > 0:
        ix = np.clip(np.rint(fx[idx]).astype(np.int64), 0, width - 1)
        iy = np.clip(np.rint(fy[idx]).astype(np.int64), 0, height - 1)
        window_radius = max(0, (int(window_size) - 1) // 2)
        neighbor_depths: list[np.ndarray] = []
        neighbor_oob: list[np.ndarray] = []
        for dy in range(-window_radius, window_radius + 1):
            raw_ny = iy + dy
            ny = np.clip(raw_ny, 0, height - 1)
            y_oob = (raw_ny < 0) | (raw_ny >= height)
            for dx in range(-window_radius, window_radius + 1):
                raw_nx = ix + dx
                nx = np.clip(raw_nx, 0, width - 1)
                x_oob = (raw_nx < 0) | (raw_nx >= width)
                neighbor_depths.append(zbuffer.depth[ny, nx])
                neighbor_oob.append(x_oob | y_oob)

        neighbor_stack = np.stack(neighbor_depths, axis=1)
        oob_stack = np.stack(neighbor_oob, axis=1)
        finite_neighbors = (~oob_stack) & np.isfinite(neighbor_stack)
        touches_empty = np.any(oob_stack | (~np.isfinite(neighbor_stack)), axis=1)
        depth_floor = np.min(
            np.where(finite_neighbors, neighbor_stack, np.inf),
            axis=1,
        )

        empty_query[idx] = touches_empty
        visible[idx] = touches_empty | (depths[idx] <= (depth_floor + float(depth_epsilon)))

    if return_empty_mask:
        return visible, empty_query
    return visible


def build_sample_visibility_lookup(
    model_vertices: np.ndarray,
    edges: list[Edge],
    faces: np.ndarray,
    camera: CameraConfig,
    *,
    view_points_per_unit_length: float,
    zbuffer_resolution: int = 1024,
    coverage_dilation_size: int = 3,
    sample_window_size: int = 5,
    debug_output_prefix: Path | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, float | str]]:
    """
    Build per-edge visibility masks for the current camera view.

    Parameters
    ----------
    model_vertices : np.ndarray
        Deduplicated mesh vertices of shape `(N, 3)` in model units.
    edges : list[Edge]
        Mesh edges used for edge sampling.
    faces : np.ndarray
        Triangle index array of shape `(M, 3)`.
    camera : CameraConfig
        Camera used for projection and visibility culling.
    view_points_per_unit_length : float
        Edge sampling density in samples per model unit.
    zbuffer_resolution : int, optional
        Coarse Z-buffer resolution along the longer viewport side.
    coverage_dilation_size : int, optional
        Dilation window used during conservative rasterization.
    sample_window_size : int, optional
        Neighborhood window used during the sample visibility test.
    debug_output_prefix : Path | None, optional
        If provided, writes diagnostic PNGs using the prefix stem.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, float | str]]
        Mapping from `edge_id` to a boolean visibility mask in edge-sampling
        order, plus aggregate visibility statistics.

    Notes
    -----
    The orthographic depth epsilon is scaled from the view-space mesh diagonal.
    This is a deliberate design decision because a global epsilon is easier to
    reason about across the pipeline; the trade-off is that extremely small and
    extremely large meshes may still require retuning.

    Assumptions
    -----------
    The same camera and edge sampling density will later be reused by arc
    generation.
    """

    verts = np.asarray(model_vertices, dtype=np.float64)
    if verts.shape[0] == 0 or len(edges) == 0 or len(faces) == 0:
        return {}, {
            "total_samples": 0.0,
            "visible_samples": 0.0,
            "percent_empty_zbuffer_queries": 100.0,
            "zbuffer_coverage_ratio": 0.0,
            "depth_epsilon": 0.0,
        }

    model_to_window_mtx = build_model_to_window_matrix(camera)
    model_to_view_mtx = build_model_to_view_matrix(camera)
    view_vertices = model_to_view(verts, model_to_view_mtx)
    bbox_diag_view = max(float(np.linalg.norm(np.ptp(view_vertices, axis=0))), EPS)
    depth_epsilon = max(bbox_diag_view * ORTHO_DEPTH_EPSILON_FACTOR, bbox_diag_view * 1.0e-7)

    screen_vertices = project_to_screen(
        verts,
        camera,
        model_to_window_mtx=model_to_window_mtx,
        model_to_view_mtx=model_to_view_mtx,
    )
    zbuffer = build_zbuffer_conservative(
        screen_vertices=screen_vertices,
        faces=faces,
        screen_width=int(camera.canvas_width),
        screen_height=int(camera.canvas_height),
        resolution=zbuffer_resolution,
        coverage_dilation_size=coverage_dilation_size,
    )

    sample_visibility_by_edge: dict[int, np.ndarray] = {}
    total_samples = 0
    visible_samples = 0
    empty_queries = 0
    debug_sample_pixels: list[np.ndarray] = []
    debug_sample_visible: list[np.ndarray] = []

    for edge in edges:
        model_points = sample_edge_model_points(
            model_vertices=model_vertices,
            edge=edge,
            view_points_per_unit_length=view_points_per_unit_length,
        )
        screen_samples = project_to_screen(
            model_points,
            camera,
            model_to_window_mtx=model_to_window_mtx,
            model_to_view_mtx=model_to_view_mtx,
        )
        visible_mask, empty_mask = visible_mask_for_edge_samples(
            screen_samples,
            zbuffer,
            depth_epsilon=depth_epsilon,
            window_size=sample_window_size,
            return_empty_mask=True,
        )
        sample_visibility_by_edge[int(edge.edge_id)] = np.asarray(visible_mask, dtype=bool)
        total_samples += int(screen_samples.shape[0])
        visible_samples += int(np.count_nonzero(visible_mask))
        empty_queries += int(np.count_nonzero(empty_mask))

        if debug_output_prefix is not None and screen_samples.shape[0] > 0:
            sample_fx, sample_fy = screen_to_zbuffer_coords(
                screen_samples[:, :2],
                zbuffer,
                clamp_to_screen=True,
            )
            sample_ix = np.clip(np.rint(sample_fx).astype(np.int64), 0, zbuffer.depth.shape[1] - 1)
            sample_iy = np.clip(np.rint(sample_fy).astype(np.int64), 0, zbuffer.depth.shape[0] - 1)
            debug_sample_pixels.append(np.column_stack((sample_ix, sample_iy)))
            debug_sample_visible.append(np.asarray(visible_mask, dtype=bool))

    coverage_ratio = 0.0
    if zbuffer.depth.size > 0:
        coverage_ratio = float(np.count_nonzero(np.isfinite(zbuffer.depth))) / float(zbuffer.depth.size)

    stats = {
        "total_samples": float(total_samples),
        "visible_samples": float(visible_samples),
        "percent_empty_zbuffer_queries": (100.0 * float(empty_queries) / float(total_samples)) if total_samples > 0 else 100.0,
        "zbuffer_coverage_ratio": coverage_ratio,
        "depth_epsilon": depth_epsilon,
    }

    if debug_output_prefix is not None:
        prefix = Path(debug_output_prefix)
        coverage_path = prefix.parent / f"{prefix.name}_coverage.png"
        samples_path = prefix.parent / f"{prefix.name}_samples.png"
        write_zbuffer_coverage_mask_png(coverage_path, zbuffer)
        if debug_sample_pixels:
            overlay_pixels = np.vstack(debug_sample_pixels)
            overlay_visible = np.concatenate(debug_sample_visible)
        else:
            overlay_pixels = np.zeros((0, 2), dtype=np.int64)
            overlay_visible = np.zeros((0,), dtype=bool)
        write_sample_visibility_overlay_png(samples_path, zbuffer, overlay_pixels, overlay_visible)
        stats["coverage_debug_png"] = str(coverage_path)
        stats["sample_debug_png"] = str(samples_path)

    return sample_visibility_by_edge, stats


def get_arc_square(point_zero: np.ndarray, n_view: float) -> tuple[float, float, float, float]:
    """
    Convert one projected sample into the bounding box of its scratch arc.

    Parameters
    ----------
    point_zero : np.ndarray
        Projected sample `(x, y, z)` in screen coordinates and projected depth.
    n_view : float
        Projected depth of the camera target point.

    Returns
    -------
    tuple[float, float, float, float]
        Arc bounding box `(rect_x, rect_y, rect_w, rect_h)` in pixels.

    Notes
    -----
    The arc diameter is proportional to the sample depth offset from `n_view`.

    Assumptions
    -----------
    `point_zero` follows the same convention returned by `model_to_window`.
    """

    distance_from_canvas = float(point_zero[2] - n_view)
    center_x = float(point_zero[0])
    center_y = float(point_zero[1] - distance_from_canvas / 2.0)
    halfwidth = abs(center_y - float(point_zero[1]))

    length = max(halfwidth * 2.0, 1.0)
    rect_x = center_x - halfwidth
    rect_y = center_y - halfwidth
    return rect_x, rect_y, length, length


def get_edge_points(
    model_start: np.ndarray,
    model_end: np.ndarray,
    zero_start: np.ndarray,
    zero_end: np.ndarray,
    view_points_per_unit_length: float,
) -> np.ndarray:
    """
    Interpolate projected samples along one edge in screen space.

    Parameters
    ----------
    model_start, model_end : np.ndarray
        Edge endpoints in model units.
    zero_start, zero_end : np.ndarray
        Corresponding projected endpoints in screen coordinates.
    view_points_per_unit_length : float
        Edge sampling density in samples per model unit.

    Returns
    -------
    np.ndarray
        Interpolated sample array of shape `(N, 3)` in screen coordinates.

    Notes
    -----
    This helper is retained for compatibility. The preferred path is model-space
    edge sampling followed by projection because that keeps projection and edge
    sampling coupled; the trade-off here is speed versus geometric fidelity.

    Assumptions
    -----------
    The projected endpoints correspond to the model endpoints.
    """

    length_model = float(np.linalg.norm(model_end - model_start))
    num_points = max(2, int(view_points_per_unit_length * length_model))
    fractions = np.linspace(0.0, 1.0, num_points)
    return zero_start + (zero_end - zero_start) * fractions[:, None]


def sample_edge_model_points(
    model_vertices: np.ndarray,
    edge: Edge,
    view_points_per_unit_length: float,
) -> np.ndarray:
    """
    Sample one mesh edge uniformly in model space.

    Parameters
    ----------
    model_vertices : np.ndarray
        Vertex array of shape `(N, 3)` in model units.
    edge : Edge
        Mesh edge to sample.
    view_points_per_unit_length : float
        Edge sampling density in samples per model unit.

    Returns
    -------
    np.ndarray
        Sample array of shape `(K, 3)` in model units.

    Notes
    -----
    Uniform model-space edge sampling is used so visibility and arc generation
    see the same samples. The trade-off is that long projected edges under
    foreshortening may still be under-sampled compared with adaptive sampling.

    Assumptions
    -----------
    `edge.start_idx` and `edge.end_idx` are valid vertex indices.
    """

    model_start = np.asarray(model_vertices[edge.start_idx], dtype=np.float64)
    model_end = np.asarray(model_vertices[edge.end_idx], dtype=np.float64)
    length_model = float(np.linalg.norm(model_end - model_start))
    num_points = max(2, int(float(view_points_per_unit_length) * length_model))
    fractions = np.linspace(0.0, 1.0, num_points)
    return model_start + ((model_end - model_start) * fractions[:, None])


def generate_arcs(
    model_vertices: np.ndarray,
    zero_vertices: np.ndarray,
    edges: list[Edge],
    view_points_per_unit_length: float,
    n_view: float,
    dedupe_decimals: int,
    min_arc_radius: float,
    arc_mode: str = "semi",
    ellipse_ratio: float = 0.65,
    model_to_window_mtx: np.ndarray | None = None,
    model_to_view_mtx: np.ndarray | None = None,
    sample_visibility_by_edge: dict[int, np.ndarray] | None = None,
) -> list[Arc]:
    """
    Convert sampled mesh edges into projected scratch arcs.

    Parameters
    ----------
    model_vertices : np.ndarray
        Vertex array of shape `(N, 3)` in model units.
    zero_vertices : np.ndarray
        Preprojected vertex array of shape `(N, 3)` in screen coordinates.
    edges : list[Edge]
        Mesh edges used for edge sampling.
    view_points_per_unit_length : float
        Edge sampling density in samples per model unit.
    n_view : float
        Projected depth of the camera target point.
    dedupe_decimals : int
        Decimal precision used to deduplicate projected samples.
    min_arc_radius : float
        Minimum projected arc radius in pixels.
    arc_mode : str, optional
        Arc generation mode: `"semi"` or `"elliptic"`.
    ellipse_ratio : float, optional
        Height/width ratio for elliptic arc generation.
    model_to_window_mtx : np.ndarray | None, optional
        Precomputed projection matrix used to project edge-sampling points.
    model_to_view_mtx : np.ndarray | None, optional
        Reserved for projection consistency with callers.
    sample_visibility_by_edge : dict[int, np.ndarray] | None, optional
        Optional visibility masks keyed by `edge_id`.

    Returns
    -------
    list[Arc]
        Sorted scratch arcs ready for SVG or G-code export.

    Notes
    -----
    Arc generation operates on edge sampling points, not on faces. Deduplication
    is intentionally applied both to projected sample positions and to arc
    geometry; the trade-off is deterministic SVG output at the cost of dropping
    coincident arcs that would otherwise overdraw the same path.

    Assumptions
    -----------
    If `sample_visibility_by_edge` is provided, its masks were computed with the
    same edge sampling density.
    """

    seen_coords: set[str] = set()
    seen_arc_geom: set[tuple[float, float, float, float, int]] = set()
    arcs: list[Arc] = []
    geom_decimals = max(3, dedupe_decimals - 2)
    mode = str(arc_mode).strip().lower()
    if mode not in ("semi", "elliptic"):
        mode = "semi"
    ell_ratio = max(0.15, min(1.0, float(ellipse_ratio)))

    for edge in edges:
        model_points = sample_edge_model_points(
            model_vertices=model_vertices,
            edge=edge,
            view_points_per_unit_length=view_points_per_unit_length,
        )
        if model_to_window_mtx is not None:
            points = model_to_window(model_points, model_to_window_mtx)
        else:
            zero_start = zero_vertices[edge.start_idx]
            zero_end = zero_vertices[edge.end_idx]
            points = get_edge_points(
                model_start=model_points[0],
                model_end=model_points[-1],
                zero_start=zero_start,
                zero_end=zero_end,
                view_points_per_unit_length=view_points_per_unit_length,
            )

        edge_visibility = None
        if sample_visibility_by_edge is not None:
            edge_visibility = sample_visibility_by_edge.get(int(edge.edge_id))

        for sample_idx, point in enumerate(points):
            if edge_visibility is not None and sample_idx < edge_visibility.shape[0] and not bool(edge_visibility[sample_idx]):
                continue
            coord_hash = (
                f"{point[0]:.{dedupe_decimals}f}:"
                f"{point[1]:.{dedupe_decimals}f}:"
                f"{point[2]:.{dedupe_decimals}f}"
            )
            if coord_hash in seen_coords:
                continue
            seen_coords.add(coord_hash)

            rect_x, rect_y, rect_w, rect_h = get_arc_square(point, n_view)
            if mode == "elliptic":
                cx = rect_x + rect_w / 2.0
                cy = rect_y + rect_h / 2.0
                rect_h = max(1.0, rect_w * ell_ratio)
                rect_y = cy - rect_h / 2.0
            radius = rect_w / 2.0
            if radius < min_arc_radius:
                continue

            # Convention aligned with desktop preview:
            # view-angle 0 corresponds to the neutral projected pose.
            start_angle = 180.0 if (float(point[2]) - n_view) > 0.0 else 0.0
            arc_hash = (
                round(rect_x, geom_decimals),
                round(rect_y, geom_decimals),
                round(rect_w, geom_decimals),
                round(rect_h, geom_decimals),
                int(start_angle),
            )
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
) -> list[tuple[int, int, int, float, float, float]]:
    """
    Prepare compact edge-sampling metadata for the HTML simulator.

    Parameters
    ----------
    model_vertices : np.ndarray
        Vertex array of shape `(N, 3)` in model units.
    edges : list[Edge]
        Mesh edges used for edge sampling.
    view_points_per_unit_length : float
        Edge sampling density in samples per model unit.

    Returns
    -------
    list[tuple[int, int, int, float, float, float]]
        Tuples containing edge endpoints, sample count, and averaged edge normal.

    Notes
    -----
    The simulator receives metadata rather than full edge-sampling points to
    keep the HTML payload smaller; the trade-off is re-sampling in JavaScript.

    Assumptions
    -----------
    The simulator will rebuild projection from the same mesh and camera data.
    """

    sim_edges: list[tuple[int, int, int, float, float, float]] = []
    for edge in edges:
        start = model_vertices[edge.start_idx]
        end = model_vertices[edge.end_idx]
        edge_len = float(np.linalg.norm(end - start))
        num_points = max(2, int(view_points_per_unit_length * edge_len))
        sim_edges.append(
            (
                int(edge.start_idx),
                int(edge.end_idx),
                int(num_points),
                float(edge.normal[0]),
                float(edge.normal[1]),
                float(edge.normal[2]),
            )
        )
    return sim_edges


def compute_sim_bounds_from_arcs(arcs: list[Arc]) -> tuple[float, float, float, float]:
    """
    Compute a conservative screen-space bounding box for a set of arcs.

    Parameters
    ----------
    arcs : list[Arc]
        Arc list in screen coordinates.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds `(min_x, min_y, max_x, max_y)` in pixels.

    Notes
    -----
    Empty input falls back to a unit box so downstream HTML layout stays valid.

    Assumptions
    -----------
    Arc bounding boxes are already expressed in screen coordinates.
    """

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
    model_faces: np.ndarray,
    sim_edges: list[tuple[int, int, int, float, float, float]],
    camera: CameraConfig,
    arcs: list[Arc],
    min_arc_radius: float,
) -> None:
    """
    Write a self-contained HTML viewer for projection and arc generation checks.

    Parameters
    ----------
    html_path : Path
        Destination file path.
    model_vertices : np.ndarray
        Vertex array of shape `(N, 3)` in model units.
    model_faces : np.ndarray
        Triangle index array of shape `(M, 3)`.
    sim_edges : list[tuple[int, int, int, float, float, float]]
        Edge-sampling metadata returned by `build_simulation_edge_info`.
    camera : CameraConfig
        Camera configuration used as the baseline projection.
    arcs : list[Arc]
        Screen-space arcs used for pattern preview.
    min_arc_radius : float
        Arc generation threshold in pixels.

    Returns
    -------
    None

    Notes
    -----
    The viewer duplicates parts of the projection logic in JavaScript because
    it is intended as an inspection aid. The trade-off is some code duplication
    in exchange for an interactive artifact that can support thesis figures.

    Assumptions
    -----------
    The generated HTML is used for diagnostics, not for production export.
    """

    html_path.parent.mkdir(parents=True, exist_ok=True)

    min_x, min_y, max_x, max_y = compute_sim_bounds_from_arcs(arcs)
    padding = 24.0

    width = max(1, int(math.ceil((max_x - min_x) + 2.0 * padding)))
    height = max(1, int(math.ceil((max_y - min_y) + 2.0 * padding)))
    offset_x = padding - min_x
    offset_y = padding - min_y

    face_payload: list[list[float | int]] = []
    for tri in model_faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = model_vertices[i0]
        p1 = model_vertices[i1]
        p2 = model_vertices[i2]
        n = np.cross(p1 - p0, p2 - p0)
        n_len = float(np.linalg.norm(n))
        if n_len < EPS:
            continue
        n = n / n_len
        face_payload.append(
            [
                i0,
                i1,
                i2,
                round(float(n[0]), 6),
                round(float(n[1]), 6),
                round(float(n[2]), 6),
            ]
        )

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
        "edges": [
            [e[0], e[1], e[2], round(e[3], 6), round(e[4], 6), round(e[5], 6)] for e in sim_edges
        ],
        "faces": face_payload,
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
        "      <label>Mode:\n"
        "        <select id=\"renderMode\">\n"
        "          <option value=\"combined\" selected>Combined</option>\n"
        "          <option value=\"pattern\">Pattern</option>\n"
        "          <option value=\"wire\">Wireframe</option>\n"
        "          <option value=\"opaque\">Opaque</option>\n"
        "        </select>\n"
        "      </label>\n"
        "      <label>Light: <input id=\"lightAngle\" type=\"range\" min=\"-180\" max=\"180\" step=\"1\" value=\"30\"/></label>\n"
        "      <strong id=\"lightAngleValue\">30 deg</strong>\n"
        "      <label>View gain: <input id=\"viewGain\" type=\"range\" min=\"10\" max=\"300\" step=\"5\" value=\"100\"/></label>\n"
        "      <strong id=\"viewGainValue\">1.0x</strong>\n"
        "      <label>Arc stride: <input id=\"arcStride\" type=\"range\" min=\"1\" max=\"12\" step=\"1\" value=\"4\"/></label>\n"
        "      <strong id=\"arcStrideValue\">4</strong>\n"
        "      <label>Arc limit: <input id=\"arcLimit\" type=\"range\" min=\"200\" max=\"10000\" step=\"100\" value=\"2500\"/></label>\n"
        "      <strong id=\"arcLimitValue\">2500</strong>\n"
        "      <label>Arc min r: <input id=\"arcMinR\" type=\"range\" min=\"0\" max=\"20\" step=\"1\" value=\"1\"/></label>\n"
        "      <strong id=\"arcMinRValue\">1</strong>\n"
        "      <label>Depth: <input id=\"depthMix\" type=\"range\" min=\"5\" max=\"100\" step=\"1\" value=\"70\"/></label>\n"
        "      <strong id=\"depthMixValue\">70%</strong>\n"
        "      <label>Thresh: <input id=\"thresh\" type=\"range\" min=\"0\" max=\"90\" step=\"1\" value=\"10\"/></label>\n"
        "      <strong id=\"threshValue\">10%</strong>\n"
        "      <label>Model alpha: <input id=\"modelAlpha\" type=\"range\" min=\"5\" max=\"95\" step=\"1\" value=\"30\"/></label>\n"
        "      <strong id=\"modelAlphaValue\">30%</strong>\n"
        "      <label>Arc alpha: <input id=\"arcAlpha\" type=\"range\" min=\"2\" max=\"80\" step=\"1\" value=\"20\"/></label>\n"
        "      <strong id=\"arcAlphaValue\">20%</strong>\n"
        "      <label><input id=\"showArcs\" type=\"checkbox\" checked/> Show arcs</label>\n"
        "      <label><input id=\"showWire\" type=\"checkbox\" checked/> Show simulated profile</label>\n"
        "    </div>\n"
        "    <div class=\"small\">Camera orbit is 3D-projected. Pattern uses depth sort + light threshold.</div>\n"
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
        "    const renderMode = document.getElementById('renderMode');\n"
        "    const lightAngle = document.getElementById('lightAngle');\n"
        "    const lightAngleValue = document.getElementById('lightAngleValue');\n"
        "    const viewGain = document.getElementById('viewGain');\n"
        "    const viewGainValue = document.getElementById('viewGainValue');\n"
        "    const arcStride = document.getElementById('arcStride');\n"
        "    const arcStrideValue = document.getElementById('arcStrideValue');\n"
        "    const arcLimit = document.getElementById('arcLimit');\n"
        "    const arcLimitValue = document.getElementById('arcLimitValue');\n"
        "    const arcMinR = document.getElementById('arcMinR');\n"
        "    const arcMinRValue = document.getElementById('arcMinRValue');\n"
        "    const depthMix = document.getElementById('depthMix');\n"
        "    const depthMixValue = document.getElementById('depthMixValue');\n"
        "    const thresh = document.getElementById('thresh');\n"
        "    const threshValue = document.getElementById('threshValue');\n"
        "    const modelAlpha = document.getElementById('modelAlpha');\n"
        "    const modelAlphaValue = document.getElementById('modelAlphaValue');\n"
        "    const arcAlpha = document.getElementById('arcAlpha');\n"
        "    const arcAlphaValue = document.getElementById('arcAlphaValue');\n"
        "    const showArcs = document.getElementById('showArcs');\n"
        "    const showWire = document.getElementById('showWire');\n"
        "    const baseCamera = data.camera;\n"
        "    const vAdd = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];\n"
        "    const vSub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];\n"
        "    const vScale = (a,s) => [a[0]*s, a[1]*s, a[2]*s];\n"
        "    const vDot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];\n"
        "    const vCross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];\n"
        "    const vNorm = (a) => Math.sqrt(vDot(a,a));\n"
        "    const vNormalize = (a) => { const n = vNorm(a); return (n < 1e-12) ? [0,0,0] : [a[0]/n,a[1]/n,a[2]/n]; };\n"
        "    const vClamp01 = (x) => Math.max(0.0, Math.min(1.0, x));\n"
        "    function reflect(i, n) { const k = 2.0 * vDot(i, n); return [i[0]-k*n[0], i[1]-k*n[1], i[2]-k*n[2]]; }\n"
        "    function rotY(v, ang) { const c=Math.cos(ang), s=Math.sin(ang); return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]; }\n"
        "    function rotX(v, ang) { const c=Math.cos(ang), s=Math.sin(ang); return [v[0], c*v[1]-s*v[2], s*v[1]+c*v[2]]; }\n"
        "    function buildLightDirection() {\n"
        "      const az = Number(lightAngle.value) * Math.PI / 180.0;\n"
        "      const elevation = 20.0 * Math.PI / 180.0;\n"
        "      const c = Math.cos(elevation);\n"
        "      const v = [Math.cos(az) * c, Math.sin(elevation), Math.sin(az) * c];\n"
        "      return vNormalize(v);\n"
        "    }\n"
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
        "      lightAngleValue.textContent = `${Number(lightAngle.value).toFixed(0)} deg`;\n"
        "      viewGainValue.textContent = `${(Number(viewGain.value) / 100.0).toFixed(1)}x`;\n"
        "      arcStrideValue.textContent = `${Number(arcStride.value).toFixed(0)}`;\n"
        "      arcLimitValue.textContent = `${Number(arcLimit.value).toFixed(0)}`;\n"
        "      arcMinRValue.textContent = `${Number(arcMinR.value).toFixed(0)}`;\n"
        "      depthMixValue.textContent = `${Number(depthMix.value).toFixed(0)}%`;\n"
        "      threshValue.textContent = `${Number(thresh.value).toFixed(0)}%`;\n"
        "      modelAlphaValue.textContent = `${Number(modelAlpha.value).toFixed(0)}%`;\n"
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
        "      const lightDir = buildLightDirection();\n"
        "      const mode = renderMode.value;\n"
        "      const wantPattern = showArcs.checked && (mode === 'combined' || mode === 'pattern');\n"
        "      const wantWire = showWire.checked && (mode === 'combined' || mode === 'wire');\n"
        "      const wantOpaque = (mode === 'combined' || mode === 'opaque');\n"
        "      if (wantOpaque && Array.isArray(data.faces) && data.faces.length > 0) {\n"
        "        const tris = [];\n"
        "        const mAlpha = Math.max(0.05, Number(modelAlpha.value) / 100.0);\n"
        "        for (const f of data.faces) {\n"
        "          const q0 = proj[f[0]];\n"
        "          const q1 = proj[f[1]];\n"
        "          const q2 = proj[f[2]];\n"
        "          if (!q0 || !q1 || !q2) continue;\n"
        "          let n = vNormalize([f[3], f[4], f[5]]);\n"
        "          if (vNorm(n) < 1e-9) continue;\n"
        "          const m0 = data.vertices[f[0]];\n"
        "          const m1 = data.vertices[f[1]];\n"
        "          const m2 = data.vertices[f[2]];\n"
        "          const mc = [(m0[0] + m1[0] + m2[0]) / 3.0, (m0[1] + m1[1] + m2[1]) / 3.0, (m0[2] + m1[2] + m2[2]) / 3.0];\n"
        "          const toEye = vNormalize(vSub(cam.po, mc));\n"
        "          if (vDot(n, toEye) < 0.0) n = vScale(n, -1.0);\n"
        "          const ndotl = Math.max(0.0, vDot(n, lightDir));\n"
        "          const shade = 0.15 + 0.85 * ndotl;\n"
        "          const z = (q0[2] + q1[2] + q2[2]) / 3.0;\n"
        "          tris.push({ z, q0, q1, q2, shade, alpha: mAlpha });\n"
        "        }\n"
        "        tris.sort((a, b) => a.z - b.z);\n"
        "        ctx.lineWidth = 0.6;\n"
        "        for (const t of tris) {\n"
        "          const c = Math.round(24 + 92 * t.shade);\n"
        "          const b = Math.round(58 + 140 * t.shade);\n"
        "          ctx.fillStyle = `rgba(${c},${c},${b},${t.alpha.toFixed(3)})`;\n"
        "          ctx.strokeStyle = `rgba(210,225,255,${Math.min(0.80, t.alpha + 0.16).toFixed(3)})`;\n"
        "          ctx.beginPath();\n"
        "          ctx.moveTo(t.q0[0] + tx, t.q0[1] + ty);\n"
        "          ctx.lineTo(t.q1[0] + tx, t.q1[1] + ty);\n"
        "          ctx.lineTo(t.q2[0] + tx, t.q2[1] + ty);\n"
        "          ctx.closePath();\n"
        "          ctx.fill();\n"
        "          ctx.stroke();\n"
        "        }\n"
        "      }\n"
        "      if (wantPattern) {\n"
        "        const stride = Math.max(1, Number(arcStride.value));\n"
        "        const limit = Math.max(100, Number(arcLimit.value));\n"
        "        const minR = Math.max(0, Number(arcMinR.value));\n"
        "        const baseAlpha = Math.max(0.01, Number(arcAlpha.value) / 100.0);\n"
        "        const depthKeep = Math.max(0.05, Number(depthMix.value) / 100.0);\n"
        "        const threshold = Math.max(0.0, Math.min(0.95, Number(thresh.value) / 100.0));\n"
        "        ctx.lineWidth = 0.9;\n"
        "        const sampleCap = Math.max(10000, limit * 20);\n"
        "        const candidates = [];\n"
        "        collectLoop: for (const e of data.edges) {\n"
        "          const p0 = proj[e[0]];\n"
        "          const p1 = proj[e[1]];\n"
        "          if (!p0 || !p1) continue;\n"
        "          const m0 = data.vertices[e[0]];\n"
        "          const m1 = data.vertices[e[1]];\n"
        "          const nEdgeBase = vNormalize([e[3], e[4], e[5]]);\n"
        "          const samples = Math.max(2, e[2]);\n"
        "          for (let k = 0; k < samples; k += stride) {\n"
        "            const t = (samples <= 1) ? 0.0 : (k / (samples - 1));\n"
        "            const px = p0[0] + (p1[0] - p0[0]) * t;\n"
        "            const py = p0[1] + (p1[1] - p0[1]) * t;\n"
        "            const pz = p0[2] + (p1[2] - p0[2]) * t;\n"
        "            const mx = m0[0] + (m1[0] - m0[0]) * t;\n"
        "            const my = m0[1] + (m1[1] - m0[1]) * t;\n"
        "            const mz = m0[2] + (m1[2] - m0[2]) * t;\n"
        "            const pm = [mx, my, mz];\n"
        "            const toEye = vNormalize(vSub(cam.po, pm));\n"
        "            let nEdge = nEdgeBase;\n"
        "            if (vDot(nEdge, toEye) < 0.0) nEdge = vScale(nEdge, -1.0);\n"
        "            const ndotl = Math.max(0.0, vDot(nEdge, lightDir));\n"
        "            const halfV = vNormalize(vAdd(lightDir, toEye));\n"
        "            const spec = Math.pow(Math.max(0.0, vDot(nEdge, halfV)), 20.0);\n"
        "            const lightScore = vClamp01(0.35 * ndotl + 0.90 * spec);\n"
        "            const dist = (pz - nView) * gain;\n"
        "            const r = Math.abs(dist) / 2.0;\n"
        "            if (r < minR || r < data.minArcRadius * 0.25) continue;\n"
        "            const cx = px + tx;\n"
        "            const cy = (py - dist / 2.0) + ty;\n"
        "            const start = (dist > 0) ? 0 : Math.PI;\n"
        "            candidates.push({ z: pz, cx, cy, r, start, light: lightScore });\n"
        "            if (candidates.length >= sampleCap) break collectLoop;\n"
        "          }\n"
        "        }\n"
        "        if (candidates.length > 0) {\n"
        "          candidates.sort((a, b) => a.z - b.z);\n"
        "          const zMin = candidates[0].z;\n"
        "          const zMax = candidates[candidates.length - 1].z;\n"
        "          const zSpan = Math.max(1e-9, zMax - zMin);\n"
        "          const depthGate = 1.0 - depthKeep;\n"
        "          const depthDen = Math.max(1e-9, depthKeep);\n"
        "          let drawn = 0;\n"
        "          for (const c of candidates) {\n"
        "            const depthNorm = (c.z - zMin) / zSpan;\n"
        "            if (depthNorm < depthGate) continue;\n"
        "            const depthScore = vClamp01((depthNorm - depthGate) / depthDen);\n"
        "            const score = depthScore * c.light;\n"
        "            if (score < threshold) continue;\n"
        "            const a = Math.max(0.01, Math.min(1.0, baseAlpha * (0.12 + 0.88 * score)));\n"
        "            const rr = Math.round(130 + 90 * c.light);\n"
        "            const gg = Math.round(165 + 80 * c.light);\n"
        "            const bb = Math.round(70 + 60 * (1.0 - c.light));\n"
        "            ctx.strokeStyle = `rgba(${rr},${gg},${bb},${a.toFixed(3)})`;\n"
        "            ctx.beginPath();\n"
        "            ctx.arc(c.cx, c.cy, c.r, c.start, c.start + Math.PI, false);\n"
        "            ctx.stroke();\n"
        "            drawn += 1;\n"
        "            if (drawn >= limit) break;\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "      if (wantWire) {\n"
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
        "    renderMode.addEventListener('change', draw);\n"
        "    lightAngle.addEventListener('input', draw);\n"
        "    viewGain.addEventListener('input', draw);\n"
        "    arcStride.addEventListener('input', draw);\n"
        "    arcLimit.addEventListener('input', draw);\n"
        "    arcMinR.addEventListener('input', draw);\n"
        "    depthMix.addEventListener('input', draw);\n"
        "    thresh.addEventListener('input', draw);\n"
        "    modelAlpha.addEventListener('input', draw);\n"
        "    arcAlpha.addEventListener('input', draw);\n"
        "    showArcs.addEventListener('change', draw);\n"
        "    showWire.addEventListener('change', draw);\n"
        "    draw();\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    html_path.write_text(html, encoding="utf-8")


def arc_endpoints_svg(arc: Arc) -> tuple[float, float, float, float, float, float]:
    """
    Convert one arc record into SVG endpoint and radius parameters.

    Parameters
    ----------
    arc : Arc
        Screen-space arc description.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Start point `(x1, y1)`, end point `(x2, y2)`, and ellipse radii
        `(rx, ry)` in pixels.

    Notes
    -----
    The same angular convention as the Tk preview is preserved so SVG export
    matches the previewed arc generation exactly.

    Assumptions
    -----------
    `arc.rect_w` and `arc.rect_h` are non-negative.
    """

    rx = arc.rect_w / 2.0
    ry = arc.rect_h / 2.0
    cx = arc.rect_x + arc.rect_w / 2.0
    cy = arc.rect_y + arc.rect_h / 2.0

    start_rad = math.radians(arc.start_angle)
    end_rad = math.radians(arc.start_angle + arc.sweep_angle)

    # Keep the same angular convention used by Tk canvas preview:
    # y = cy - r * sin(theta)
    x1 = cx + rx * math.cos(start_rad)
    y1 = cy - ry * math.sin(start_rad)
    x2 = cx + rx * math.cos(end_rad)
    y2 = cy - ry * math.sin(end_rad)
    return x1, y1, x2, y2, rx, ry


def write_svg(svg_path: Path, arcs: list[Arc], stroke_width: float) -> None:
    """
    Export the arc set as one SVG path collection.

    Parameters
    ----------
    svg_path : Path
        Destination file path.
    arcs : list[Arc]
        Screen-space arc list.
    stroke_width : float
        SVG stroke width in screen units (pixels in the viewBox).

    Returns
    -------
    None

    Notes
    -----
    Arcs are emitted as one `<path>` with repeated move and arc commands. This
    is a design decision because it keeps the SVG compact; the trade-off is that
    individual arcs are not separately addressable in downstream editors.

    Assumptions
    -----------
    `arcs` is already ordered in the desired export sequence.
    """

    svg_path.parent.mkdir(parents=True, exist_ok=True)

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    path_cmds: list[str] = []

    for arc in arcs:
        x1, y1, x2, y2, rx, ry = arc_endpoints_svg(arc)
        # Use the full ellipse rect as a conservative SVG bounds estimate.
        # Using only the endpoints can clip the apex of semicircle/elliptic arcs.
        min_x = min(min_x, arc.rect_x, x1, x2)
        min_y = min(min_y, arc.rect_y, y1, y2)
        max_x = max(max_x, arc.rect_x + arc.rect_w, x1, x2)
        max_y = max(max_y, arc.rect_y + arc.rect_h, y1, y2)
        path_cmds.append(f"M {x1:.6f} {y1:.6f}")
        # Keep arc orientation consistent with Tk canvas preview:
        # use sweep-flag 0 to match extent direction in preview.
        path_cmds.append(f"A {rx:.6f} {ry:.6f} 0 0 0 {x2:.6f} {y2:.6f}")

    if path_cmds:
        pad = max(1.0, stroke_width)
        view_x = min_x - pad
        view_y = min_y - pad
        view_w = max(1.0, (max_x - min_x) + (2.0 * pad))
        view_h = max(1.0, (max_y - min_y) + (2.0 * pad))
    else:
        view_x = 0.0
        view_y = 0.0
        view_w = 1.0
        view_h = 1.0

    width = int(math.ceil(view_w))
    height = int(math.ceil(view_h))

    rows: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="{view_x:.6f} {view_y:.6f} {view_w:.6f} {view_h:.6f}" '
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
    """
    Write a structured JSON snapshot of the current pipeline state.

    Parameters
    ----------
    json_path : Path
        Destination file path.
    camera : CameraConfig
        Camera configuration used for projection.
    pipeline : PipelineConfig
        Arc generation parameters.
    n_view : float
        Projected depth of the camera target point.
    edges_count : int
        Number of mesh edges after filtering.
    arcs : list[Arc]
        Generated screen-space arcs.

    Returns
    -------
    None

    Notes
    -----
    The JSON is intended for debugging and thesis inspection, not for strict
    interchange. Field names preserve some legacy naming for compatibility.

    Assumptions
    -----------
    The caller wants a human-readable debug artifact.
    """

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
                "ArcRect": f"{arc.rect_x:.6f}, {arc.rect_y:.6f}, {arc.rect_w:.6f}, {arc.rect_h:.6f}",
                "StartAngle": arc.start_angle,
                "SweepAngle": arc.sweep_angle,
            }
            for arc in arcs
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """
    Define and parse the command-line interface for the pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    Notes
    -----
    The CLI exposes mesh loading, edge sampling, projection, SVG export, and
    optional diagnostics in one entry point.

    Assumptions
    -----------
    Validation is deferred to `validate_args`.
    """

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
    parser.add_argument(
        "--arc-mode",
        type=str,
        default="semi",
        choices=("semi", "elliptic"),
        help="Arc geometry mode.",
    )
    parser.add_argument(
        "--ellipse-ratio",
        type=float,
        default=0.65,
        help="Ellipse height/width ratio when --arc-mode=elliptic.",
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
    """
    Validate the numeric ranges of CLI parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    None

    Notes
    -----
    Validation is explicit rather than relying on downstream failures because it
    produces clearer errors for batch runs and thesis experiments.

    Assumptions
    -----------
    `args` was produced by `parse_args`.
    """

    if args.line_resolution <= 0:
        raise ValueError("--line-resolution must be > 0")
    if args.stroke_width <= 0:
        raise ValueError("--stroke-width must be > 0")
    if args.min_arc_radius < 0:
        raise ValueError("--min-arc-radius must be >= 0")
    if args.arc_mode not in ("semi", "elliptic"):
        raise ValueError("--arc-mode must be 'semi' or 'elliptic'")
    if args.ellipse_ratio <= 0 or args.ellipse_ratio > 1.0:
        raise ValueError("--ellipse-ratio must be in (0, 1]")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("--canvas-width and --canvas-height must be > 0")
    if args.current_scale <= 0:
        raise ValueError("--current-scale must be > 0")
    if abs(args.zf) < EPS:
        raise ValueError("--zf must be non-zero")
    if args.dedupe_decimals < 0 or args.dedupe_decimals > 12:
        raise ValueError("--dedupe-decimals must be in [0, 12]")


def run(args: argparse.Namespace) -> int:
    """
    Execute the full mesh-to-SVG pipeline for one command-line invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Validated command-line arguments.

    Returns
    -------
    int
        Process exit code (`0` on success).

    Notes
    -----
    The function computes all arcs first, then recomputes visible arcs with the
    visibility cull. Keeping both counts is a design decision because it makes
    debugging easier; the trade-off is one extra arc-generation pass.

    Assumptions
    -----------
    Runtime dependencies are installed and the input mesh is readable.
    """

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
        arc_mode=str(args.arc_mode),
        ellipse_ratio=float(args.ellipse_ratio),
    )

    mesh = load_mesh(args.stl)
    model_vertices, edges, model_faces = build_model_vertices_and_edges(
        mesh=mesh,
        auto_center=pipeline.auto_center,
        decimals=NORMAL_TOLERANCE_DECIMALS,
    )

    model_to_window_mtx = build_model_to_window_matrix(camera)
    model_to_view_mtx = build_model_to_view_matrix(camera)
    zero_vertices = model_to_window(model_vertices, model_to_window_mtx)
    pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), model_to_window_mtx)[0]
    n_view = float(pr_view[2])

    arcs_all = generate_arcs(
        model_vertices=model_vertices,
        zero_vertices=zero_vertices,
        edges=edges,
        view_points_per_unit_length=pipeline.line_resolution,
        n_view=n_view,
        dedupe_decimals=pipeline.dedupe_decimals,
        min_arc_radius=pipeline.min_arc_radius,
        arc_mode=pipeline.arc_mode,
        ellipse_ratio=pipeline.ellipse_ratio,
        model_to_window_mtx=model_to_window_mtx,
        model_to_view_mtx=model_to_view_mtx,
    )
    sample_visibility_by_edge, _vis_stats = build_sample_visibility_lookup(
        model_vertices=model_vertices,
        edges=edges,
        faces=model_faces,
        camera=camera,
        view_points_per_unit_length=pipeline.line_resolution,
        zbuffer_resolution=1024,
    )
    arcs = generate_arcs(
        model_vertices=model_vertices,
        zero_vertices=zero_vertices,
        edges=edges,
        view_points_per_unit_length=pipeline.line_resolution,
        n_view=n_view,
        dedupe_decimals=pipeline.dedupe_decimals,
        min_arc_radius=pipeline.min_arc_radius,
        arc_mode=pipeline.arc_mode,
        ellipse_ratio=pipeline.ellipse_ratio,
        model_to_window_mtx=model_to_window_mtx,
        model_to_view_mtx=model_to_view_mtx,
        sample_visibility_by_edge=sample_visibility_by_edge,
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
            model_faces=model_faces,
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
    print(f"[OK] Arcs generated: {len(arcs_all)}")
    print(f"[OK] Arcs visible after cull: {len(arcs)}")
    print(f"[OK] SVG saved: {args.svg}")
    if args.simulate_html is not None:
        print(f"[OK] Simulation HTML saved: {args.simulate_html}")
    if args.json is not None:
        print(f"[OK] JSON saved: {args.json}")
    return 0


def unit_cube_visibility_cull_smoke_test() -> dict[str, int]:
    """
    Run a deterministic visibility-cull smoke test on a unit cube.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, int]
        Summary counters for edge sampling and visible arc generation.

    Notes
    -----
    The test is intended as a regression check for the visibility cull and
    diagnostic PNG generation, not as a formal unit-test suite.

    Assumptions
    -----------
    `numpy` is available and the current working tree is writable for debug PNGs.
    """

    if np is None:
        raise RuntimeError("numpy is required for the visibility smoke test.")

    cube_vertices = np.asarray(
        [
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (0.5, 0.5, 0.5),
            (0.5, -0.5, 0.5),
            (-0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, 0.5, 0.5),
            (-0.5, -0.5, 0.5),
        ],
        dtype=np.float64,
    )
    cube_faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (1, 5, 6),
            (1, 6, 2),
            (0, 3, 7),
            (0, 7, 4),
            (3, 2, 6),
            (3, 6, 7),
            (0, 4, 5),
            (0, 5, 1),
        ],
        dtype=np.int64,
    )
    cube_edges = [
        Edge(edge_id=0, start_idx=0, end_idx=1),
        Edge(edge_id=1, start_idx=1, end_idx=2),
        Edge(edge_id=2, start_idx=2, end_idx=3),
        Edge(edge_id=3, start_idx=3, end_idx=0),
        Edge(edge_id=4, start_idx=4, end_idx=5),
        Edge(edge_id=5, start_idx=5, end_idx=6),
        Edge(edge_id=6, start_idx=6, end_idx=7),
        Edge(edge_id=7, start_idx=7, end_idx=4),
        Edge(edge_id=8, start_idx=0, end_idx=4),
        Edge(edge_id=9, start_idx=1, end_idx=5),
        Edge(edge_id=10, start_idx=2, end_idx=6),
        Edge(edge_id=11, start_idx=3, end_idx=7),
    ]
    camera = CameraConfig(
        po=(4.0, 0.0, 0.0),
        pr=(0.0, 0.0, 0.0),
        look_up=(0.0, 0.0, 1.0),
        current_scale=1.0,
        zf=8.0,
        canvas_width=640,
        canvas_height=640,
    )

    model_to_window_mtx = build_model_to_window_matrix(camera)
    model_to_view_mtx = build_model_to_view_matrix(camera)
    projected = model_to_window(cube_vertices, model_to_window_mtx)
    pr_view = model_to_window(np.asarray([camera.pr], dtype=np.float64), model_to_window_mtx)[0]
    n_view = float(pr_view[2])

    arcs_before = generate_arcs(
        model_vertices=cube_vertices,
        zero_vertices=projected,
        edges=cube_edges,
        view_points_per_unit_length=20.0,
        n_view=n_view,
        dedupe_decimals=6,
        min_arc_radius=0.0,
        arc_mode="semi",
        ellipse_ratio=1.0,
        model_to_window_mtx=model_to_window_mtx,
        model_to_view_mtx=model_to_view_mtx,
    )
    sample_visibility_by_edge, vis_stats = build_sample_visibility_lookup(
        model_vertices=cube_vertices,
        edges=cube_edges,
        faces=cube_faces,
        camera=camera,
        view_points_per_unit_length=20.0,
        zbuffer_resolution=1024,
        coverage_dilation_size=3,
        sample_window_size=5,
        debug_output_prefix=Path(__file__).resolve().with_name("unit_cube_visibility_debug"),
    )
    arcs_after = generate_arcs(
        model_vertices=cube_vertices,
        zero_vertices=projected,
        edges=cube_edges,
        view_points_per_unit_length=20.0,
        n_view=n_view,
        dedupe_decimals=6,
        min_arc_radius=0.0,
        arc_mode="semi",
        ellipse_ratio=1.0,
        model_to_window_mtx=model_to_window_mtx,
        model_to_view_mtx=model_to_view_mtx,
        sample_visibility_by_edge=sample_visibility_by_edge,
    )

    back_edge_ids = {4, 5, 6, 7}
    back_before = sum(1 for arc in arcs_before if arc.edge_id in back_edge_ids)
    back_after = sum(1 for arc in arcs_after if arc.edge_id in back_edge_ids)

    print(
        "Unit cube visibility test | "
        f"edges={len(cube_edges)} | samples before={len(arcs_before)} | samples after={len(arcs_after)}"
    )
    print(f"Before: {len(arcs_before)} arcs; After cull: {len(arcs_after)} arcs")
    print(
        "Stats | "
        f"total_samples={int(vis_stats['total_samples'])} | "
        f"visible_samples={int(vis_stats['visible_samples'])} | "
        f"percent_empty_zbuffer_queries={vis_stats['percent_empty_zbuffer_queries']:.2f}% | "
        f"zbuffer_coverage_ratio={vis_stats['zbuffer_coverage_ratio']:.4f}"
    )
    if "coverage_debug_png" in vis_stats and "sample_debug_png" in vis_stats:
        print(
            "Debug PNGs | "
            f"coverage={vis_stats['coverage_debug_png']} | "
            f"samples={vis_stats['sample_debug_png']}"
        )
    print(f"Far-side back-face samples kept: {back_after}/{back_before}")

    if back_before <= 0:
        raise AssertionError("Cube smoke test did not generate any back-face edge samples.")
    if back_after != 0:
        raise AssertionError("Far-side cube edges should be fully culled in the front-on view.")
    if len(arcs_after) <= 0 or len(arcs_after) >= len(arcs_before):
        raise AssertionError("Visibility culling should keep some arcs but remove occluded ones.")

    return {
        "edge_count": len(cube_edges),
        "arcs_before": len(arcs_before),
        "arcs_after": len(arcs_after),
        "back_before": back_before,
        "back_after": back_after,
        "total_samples": int(vis_stats["total_samples"]),
        "visible_samples": int(vis_stats["visible_samples"]),
    }


def main() -> int:
    """
    Command-line entry point for the scratch hologram pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Process exit code.

    Notes
    -----
    Exceptions are converted into a non-zero exit code after a readable stderr
    message, which is preferable for batch fabrication workflows.

    Assumptions
    -----------
    The script is executed in a standard Python process.
    """

    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# Thesis extraction notes
# -----------------------
# 1. Load and sanitize the mesh, then remove invalid topology before edge sampling.
# 2. Round vertices and merge coplanar triangulation edges to stabilize the mesh graph.
# 3. Build a camera basis, then apply projection from model space to screen space.
# 4. Sample each mesh edge uniformly in model space using a user-defined sampling density.
# 5. Rasterize projected mesh faces into a conservative coarse Z-buffer.
# 6. Evaluate visibility per edge-sampling point with a neighborhood-based depth test.
# 7. Convert visible projected samples into semicircular or elliptic arc geometry.
# 8. Deduplicate coincident projected samples and repeated arc geometry for stable SVG output.
# 9. Export the final arc set as SVG, optional debug JSON, and optional simulation HTML.
# 10. Reuse the same arc geometry for downstream G-code generation in the desktop application.
#
# Useful figures/snippets
# -----------------------
# - `build_model_vertices_and_edges`: mesh cleanup and topology extraction.
# - `build_model_to_view_matrix` + `build_model_to_window_matrix`: projection model.
# - `build_zbuffer_conservative`: conservative screen-space rasterization.
# - `visible_mask_for_edge_samples`: neighborhood-based visibility rule.
# - `generate_arcs`: transition from edge sampling to arc generation.
