import bpy
import math
from pathlib import Path
from mathutils import Vector

# =========================================================
# WORKFLOW FLAGS
# =========================================================
CLEAN_SCENE = True
DO_SETUP_SCENE = True
DO_ANIMATE_SUN = True
DO_RENDER = False                 # False = setup/preview, True = render finale
RENDER_MODE = 'ANIMATION'         # 'STILL' oppure 'ANIMATION'

# =========================================================
# PATHS
# Cambia solo questo nome file:
# =========================================================
BASE_DIR = Path(r"C:\Users\filip\Documents\thesis\Blender\knots")
SVG_FILENAME = "CubeElliptic.svg"

SVG_PATH = BASE_DIR / SVG_FILENAME
FILE_STEM = SVG_PATH.stem

# Ogni SVG ha la sua cartella output
OUT_DIR = BASE_DIR / FILE_STEM
STILL_RENDER_PATH = OUT_DIR / f"{FILE_STEM}.png"
ANIM_RENDER_PREFIX = OUT_DIR / f"{FILE_STEM}_"

# =========================================================
# SCENE / IMPORT
# =========================================================
PLANE_SIZE = 2.0
FIT_MARGIN = 0.82
Z_OFFSET_CURVES = 0.0001

# =========================================================
# CURVES / MATERIALS
# =========================================================
CURVE_BEVEL_DEPTH = 0.00015
CURVE_BEVEL_RES = 2

PLANE_ROUGHNESS = 0.05
CURVE_ROUGHNESS = 0.07

WORLD_BG = (0.02, 0.02, 0.02, 1.0)   # quasi nero

# =========================================================
# CAMERA
# =========================================================
CAM_LOC = (0.0, -0.48, 2.54)
CAM_ROT_DEG = (10, 0, 0)

CAMERA_TYPE = 'PERSP'   # 'PERSP' oppure 'ORTHO'
ORTHO_SCALE = 2.2

# =========================================================
# SUN
# =========================================================
SUN_STRENGTH = 5.0
SUN_ROT_DEG = (50, 0, 0)   # orientamento iniziale

START_FRAME = 1
END_FRAME = 10

SUN_X_FIXED = 50
SUN_Y_FIXED = 0
SUN_Z_START = -50
SUN_Z_END = 50

# =========================================================
# RENDER
# =========================================================
RES_X = 2048
RES_Y = 2048
FPS = 30

SAMPLES_STILL = 256
SAMPLES_ANIM = 64
USE_DENOISE = True

# =========================================================
# HELPERS
# =========================================================
def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            pass

def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_cycles(samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = USE_DENOISE
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

def try_enable_gpu():
    scene = bpy.context.scene
    prefs = bpy.context.preferences

    try:
        cycles_prefs = prefs.addons["cycles"].preferences
    except Exception:
        print("⚠️ Impossibile accedere alle preferenze Cycles. Uso le impostazioni correnti.")
        return

    backends = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
    enabled = False

    for backend in backends:
        try:
            cycles_prefs.compute_device_type = backend
            cycles_prefs.get_devices()

            has_gpu = False
            for dev_group in cycles_prefs.devices:
                for dev in dev_group:
                    if dev.type != 'CPU':
                        dev.use = True
                        has_gpu = True
                    else:
                        dev.use = False

            if has_gpu:
                scene.cycles.device = 'GPU'
                print(f"✅ GPU attivata con backend {backend}")
                enabled = True
                break
        except Exception:
            pass

    if not enabled:
        scene.cycles.device = 'CPU'
        print("⚠️ GPU non attivata automaticamente. Uso CPU.")

def ensure_world_background(color=(0.02, 0.02, 0.02, 1.0), strength=1.0):
    scene = bpy.context.scene

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    world = scene.world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    bg = nodes.get("Background")
    out = nodes.get("World Output")

    if bg is None:
        bg = nodes.new(type="ShaderNodeBackground")
    if out is None:
        out = nodes.new(type="ShaderNodeOutputWorld")

    already_linked = False
    for link in links:
        if link.from_node == bg and link.to_node == out:
            already_linked = True
            break

    if not already_linked:
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    bg.inputs["Color"].default_value = color
    bg.inputs["Strength"].default_value = strength

def make_metal_material(name="Metal", rough=0.05):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)

    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = rough

    return mat

def assign_material(obj, mat):
    if obj.type not in {'MESH', 'CURVE'}:
        return

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def add_plane(material):
    existing = bpy.data.objects.get("HoloPlane")
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "HoloPlane"
    assign_material(plane, material)
    return plane

def add_sun():
    existing = bpy.data.objects.get("Sun")
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 2))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = SUN_STRENGTH

    rx, ry, rz = [math.radians(d) for d in SUN_ROT_DEG]
    sun.rotation_euler = (rx, ry, rz)
    return sun

def add_camera():
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.camera_add(location=CAM_LOC)
    cam = bpy.context.active_object
    cam.name = "Camera"

    rx, ry, rz = [math.radians(d) for d in CAM_ROT_DEG]
    cam.rotation_euler = (rx, ry, rz)

    cam.data.type = CAMERA_TYPE
    if CAMERA_TYPE == 'ORTHO':
        cam.data.ortho_scale = ORTHO_SCALE

    bpy.context.scene.camera = cam
    return cam

def import_svg(svg_path):
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG non trovato: {svg_path}")

    before_names = set(obj.name for obj in bpy.data.objects)

    deselect_all()
    bpy.ops.import_curve.svg(filepath=str(svg_path))

    after_objects = [obj for obj in bpy.data.objects if obj.name not in before_names]
    curves = [obj for obj in after_objects if obj.type == 'CURVE']

    if not curves:
        curves = [obj for obj in bpy.context.selected_objects if obj.type == 'CURVE']

    if not curves:
        raise RuntimeError("Nessuna curva trovata dopo l'import dell'SVG.")

    return curves

def setup_curve_bevel(curves, bevel_depth, bevel_res, material=None):
    for c in curves:
        c.data.dimensions = '3D'
        c.data.bevel_depth = bevel_depth
        c.data.bevel_resolution = bevel_res
        if material is not None:
            assign_material(c, material)

def world_bbox(objects):
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, wc.x)
            mins.y = min(mins.y, wc.y)
            mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x)
            maxs.y = max(maxs.y, wc.y)
            maxs.z = max(maxs.z, wc.z)

    return mins, maxs

def center_and_fit_objects_xy(objects, target_size, margin=0.82):
    bpy.context.view_layer.update()

    mins, maxs = world_bbox(objects)
    size = maxs - mins
    max_dim = max(size.x, size.y)

    if max_dim <= 0:
        return

    desired_dim = target_size * margin
    scale_factor = desired_dim / max_dim

    for obj in objects:
        obj.scale *= scale_factor

    bpy.context.view_layer.update()

    mins, maxs = world_bbox(objects)
    center = (mins + maxs) * 0.5

    for obj in objects:
        obj.location.x -= center.x
        obj.location.y -= center.y

    bpy.context.view_layer.update()

def lift_curves(curves, dz=0.0001):
    for c in curves:
        c.location.z += dz

def parent_curves_to_empty(curves, empty_name="ImportedSVG"):
    existing = bpy.data.objects.get(empty_name)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = empty_name

    for c in curves:
        c.parent = empty

    return empty

def get_or_create_sun():
    sun = bpy.data.objects.get("Sun")
    if sun is None:
        sun = add_sun()
    return sun

def set_keyframes_linear_compatible(obj):
    """
    Prova a impostare i keyframe su interpolazione lineare.
    Compatibile sia con versioni vecchie sia con Blender 5.x.
    Se non riesce, non blocca lo script.
    """
    anim = obj.animation_data
    if not anim or not anim.action:
        return

    action = anim.action

    # Blender <= 4.x
    if hasattr(action, "fcurves"):
        try:
            for fcurve in action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'LINEAR'
            return
        except Exception:
            pass

    # Blender 5.x
    try:
        slot = getattr(anim, "action_slot", None)
        if slot is None:
            return

        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbag"):
                    channelbag = strip.channelbag(slot, ensure=False)
                    if channelbag:
                        for fcurve in channelbag.fcurves:
                            for kp in fcurve.keyframe_points:
                                kp.interpolation = 'LINEAR'
    except Exception as e:
        print("⚠️ Non sono riuscito a impostare LINEAR:", e)

def animate_sun(sun):
    scene = bpy.context.scene
    scene.frame_start = START_FRAME
    scene.frame_end = END_FRAME
    scene.render.fps = FPS

    sun.animation_data_clear()

    def set_rot(x_deg, y_deg, z_deg):
        sun.rotation_euler[0] = math.radians(x_deg)
        sun.rotation_euler[1] = math.radians(y_deg)
        sun.rotation_euler[2] = math.radians(z_deg)

    scene.frame_set(START_FRAME)
    set_rot(SUN_X_FIXED, SUN_Y_FIXED, SUN_Z_START)
    sun.keyframe_insert(data_path="rotation_euler", index=0)
    sun.keyframe_insert(data_path="rotation_euler", index=1)
    sun.keyframe_insert(data_path="rotation_euler", index=2)

    scene.frame_set(END_FRAME)
    set_rot(SUN_X_FIXED, SUN_Y_FIXED, SUN_Z_END)
    sun.keyframe_insert(data_path="rotation_euler", index=0)
    sun.keyframe_insert(data_path="rotation_euler", index=1)
    sun.keyframe_insert(data_path="rotation_euler", index=2)

    set_keyframes_linear_compatible(sun)
    scene.frame_set(START_FRAME)

def render_still(path):
    scene = bpy.context.scene
    ensure_cycles(SAMPLES_STILL)
    ensure_output_dir()
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)

def render_animation(path_prefix):
    scene = bpy.context.scene
    ensure_cycles(SAMPLES_ANIM)
    ensure_output_dir()
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(path_prefix)
    bpy.ops.render.render(animation=True)

# =========================================================
# MAIN
# =========================================================
if CLEAN_SCENE:
    delete_all_objects()

ensure_cycles(SAMPLES_STILL)
try_enable_gpu()
ensure_world_background(WORLD_BG, strength=1.0)
ensure_output_dir()

if DO_SETUP_SCENE:
    plane_mat = make_metal_material(name="PlaneMetal", rough=PLANE_ROUGHNESS)
    curve_mat = make_metal_material(name="CurveMetal", rough=CURVE_ROUGHNESS)

    add_plane(plane_mat)
    add_camera()
    add_sun()

    curves = import_svg(SVG_PATH)
    setup_curve_bevel(curves, CURVE_BEVEL_DEPTH, CURVE_BEVEL_RES, material=curve_mat)
    center_and_fit_objects_xy(curves, target_size=PLANE_SIZE, margin=FIT_MARGIN)
    lift_curves(curves, dz=Z_OFFSET_CURVES)
    parent_curves_to_empty(curves, empty_name="ImportedSVG")

    print(f"✅ Importate {len(curves)} curve da SVG")
    print(f"✅ SVG: {SVG_PATH}")
    print(f"✅ Output folder: {OUT_DIR}")

if DO_ANIMATE_SUN:
    sun = get_or_create_sun()
    animate_sun(sun)

    print("✅ Animazione sole creata")
    print(f"   X fissa = {SUN_X_FIXED}°")
    print(f"   Z da {SUN_Z_START}° a {SUN_Z_END}°")
    print(f"   Frame {START_FRAME} -> {END_FRAME}")

if DO_RENDER:
    if RENDER_MODE == 'STILL':
        render_still(STILL_RENDER_PATH)
        print("✅ Still render salvato in:")
        print(STILL_RENDER_PATH)

    elif RENDER_MODE == 'ANIMATION':
        render_animation(ANIM_RENDER_PREFIX)
        print("✅ Frame animazione salvati in:")
        print(OUT_DIR)

    else:
        raise ValueError("RENDER_MODE deve essere 'STILL' oppure 'ANIMATION'.")

else:
    print("👀 DO_RENDER = False")
    print("Nessun render eseguito. Controlla viewport, camera e riflessi.")