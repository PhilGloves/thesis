### Dopo aver importato lo script ho cambiato due cose:
-scale per aumentare la dimensione dei graffi
-Cambiato posizione sole
-scurito il mondo
-tolto la luce di default
-cambiato posizione camera

-cambiato material>roughness in modo da avere il piano a 0.05 e linee a 0.08


import bpy
import math
from mathutils import Vector

# ---------- USER SETTINGS ----------
SVG_PATH = r"C:\Users\filip\Documents\thesis\Blender\basic_cube_blender_arcs.svg"
RENDER_PATH = r"C:\Users\filip\Documents\thesis\Blender\hologram.png"
CURVE_BEVEL_DEPTH = 0.00015   # meters: try 0.00005 .. 0.0005 depending on scale
CURVE_BEVEL_RES = 2           # 0..4
SUN_STRENGTH = 5.0
SUN_ROT_DEG = (55, 0, 35)     # (X,Y,Z) degrees, tweak for highlight direction
PLANE_SIZE = 2.0              # meters
CAM_LOC = (0.0, -2.4, 1.2)
CAM_ROT_DEG = (65, 0, 0)
# ----------------------------------

def deselect_all():
    for o in bpy.context.selected_objects:
        o.select_set(False)

def ensure_cycles():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # basic noise control
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True  # render denoise (works in many versions)
    # viewport is separate; we keep it simple here

def make_metal_material(name="HoloMetal", rough=0.05):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = rough
    return mat

def add_plane(material):
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0,0,0))
    plane = bpy.context.active_object
    plane.name = "HoloPlane"
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
    return plane

def add_sun():
    bpy.ops.object.light_add(type='SUN', location=(0,0,2))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = SUN_STRENGTH
    rx, ry, rz = [math.radians(d) for d in SUN_ROT_DEG]
    sun.rotation_euler = (rx, ry, rz)
    return sun

def add_camera():
    bpy.ops.object.camera_add(location=CAM_LOC)
    cam = bpy.context.active_object
    cam.name = "Camera"
    rx, ry, rz = [math.radians(d) for d in CAM_ROT_DEG]
    cam.rotation_euler = (rx, ry, rz)
    bpy.context.scene.camera = cam
    return cam

def import_svg(svg_path):
    deselect_all()
    # Official operator
    bpy.ops.import_curve.svg(filepath=svg_path)  # :contentReference[oaicite:1]{index=1}

    # Imported curves may arrive inside a collection; grab newly created curves
    curves = [o for o in bpy.context.scene.objects if o.type == 'CURVE' and o.select_get()]
    if not curves:
        # fallback: pick any curve objects
        curves = [o for o in bpy.context.scene.objects if o.type == 'CURVE']
    return curves

def setup_curve_bevel(curves, bevel_depth, bevel_res, material=None):
    for c in curves:
        c.data.bevel_depth = bevel_depth
        c.data.bevel_resolution = bevel_res
        # Make it 3D so bevel shows correctly
        c.data.dimensions = '3D'
        if material is not None:
            if c.data.materials:
                c.data.materials[0] = material
            else:
                c.data.materials.append(material)

def frame_view_on(obj):
    # optional helper: center view
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

def render_still(path):
    scene = bpy.context.scene
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

# ---------- RUN ----------
# Optional: clean default cube etc. (comment out if you don't want deletion)
# for o in list(bpy.data.objects):
#     if o.name.startswith(("Cube", "Light")):
#         bpy.data.objects.remove(o, do_unlink=True)

ensure_cycles()

metal = make_metal_material(rough=0.05)
plane = add_plane(metal)
sun = add_sun()
cam = add_camera()

curves = import_svg(SVG_PATH)
setup_curve_bevel(curves, CURVE_BEVEL_DEPTH, CURVE_BEVEL_RES, material=metal)

# Move curves a tiny bit above plane to avoid z-fighting in viewport
for c in curves:
    c.location.z += 0.0001

print(f"Imported {len(curves)} curve objects from SVG.")
# render_still(RENDER_PATH)  # uncomment to render immediately