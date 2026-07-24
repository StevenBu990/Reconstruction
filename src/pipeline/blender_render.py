"""
blender_render.py

Headless Blender render step for the fragment-repair experiment pipeline.

Invoked as:
    blender --background --python blender_render.py -- \
        --fragments_dir path/to/resolved/obj_017_pieces/ \
        --object_id obj_017 \
        --fragment_map configs/fragment_maps/obj_017.json \
        --view_presets configs/view_presets.json \
        --output_root outputs \
        --engine EEVEE \
        --resolution 1024

--fragments_dir must point to a folder containing that object's per-fragment
files, named piece_0.obj, piece_1.obj, piece_2.obj, etc. Each fragment becomes
its own Blender object, renamed to match its filename stem (e.g. "piece_3") so
fragment map JSON files can reference fragments by a stable, predictable name
regardless of internal OBJ group naming.

The Breaking Bad dataset stores each object's pieces several subfolders deep.
Resolving object_id -> fragments_dir is the manifest generator's job, not this
script's - this script only ever sees one already-resolved folder per call.

For a single object, this script renders ALL (damage_config x view) combinations
in one Blender session (to avoid repeated Blender startup cost), and writes:

    outputs/{object_id}/{damage_config}/view_{n}/input.png
    outputs/{object_id}/{damage_config}/view_{n}/ground_truth.png
    outputs/{object_id}/{damage_config}/view_{n}/metadata.json

Fragment map schema (configs/fragment_maps/{object_id}.json):
{
    "damage_configs": {
        "part1_simple": ["frag_2"],
        "part1_severe": ["frag_2", "frag_5"],
        "part2_simple": ["frag_7"],
        "part2_severe": ["frag_7", "frag_9"]
    }
}
Fragment names must match the object names Blender assigns on import
(check via the fragment ID picker helper script - see fragment_picker.py).

View presets schema (configs/view_presets.json):
{
    "views": [
        {"id": 0, "name": "front", "rotation_euler_deg": [0, 0, 0]},
        {"id": 1, "name": "right", "rotation_euler_deg": [0, 0, 90]},
        {"id": 2, "name": "back",  "rotation_euler_deg": [0, 0, 180]},
        {"id": 3, "name": "left",  "rotation_euler_deg": [0, 0, 270]}
    ]
}
"""

import bpy
import sys
import os
import glob
import json
import math
import argparse
import time
from mathutils import Vector, Euler


# --------------------------------------------------------------------------
# Argument parsing (Blender passes its own args before "--")
# --------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments_dir", required=True, help="Folder containing piece_N.obj files for this object")
    parser.add_argument("--object_id", required=True, help="Identifier used for output folder naming")
    parser.add_argument("--fragment_map", required=True, help="Path to fragment map JSON for this object")
    parser.add_argument("--view_presets", required=True, help="Path to shared view presets JSON")
    parser.add_argument("--output_root", required=True, help="Root output directory")
    parser.add_argument("--engine", default="WORKBENCH", choices=["EEVEE", "WORKBENCH", "CYCLES"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--distance_factor", type=float, default=1.5,
                         help="Camera distance as a multiple of the object's max dimension. "
                              "Lower = object fills more of the frame, less background margin.")
    parser.add_argument("--light_energy_scale", type=float, default=1000,
                         help="Point light energy = this * object's max dimension. "
                              "Lower this if renders look overexposed/too bright.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Scene setup
# --------------------------------------------------------------------------

def reset_scene():
    """Load Blender's actual factory-default startup scene: default Camera,
    default single Point Light, default flat-gray World background, same as
    what you'd see opening Blender fresh with no changes. This intentionally
    avoids introducing any new lighting/background style - the automated
    renders should look like your prior manual ones, not a custom "photo
    studio" setup. The default Cube is removed since we import real geometry
    instead."""
    bpy.ops.wm.read_factory_settings(use_empty=False)
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)


def set_render_settings(engine_choice, resolution):
    scene = bpy.context.scene

    engine_map = {
        "EEVEE": "BLENDER_EEVEE_NEXT",   # Blender 4.2+; falls back below if unavailable
        "WORKBENCH": "BLENDER_WORKBENCH",
        "CYCLES": "CYCLES",
    }
    target_engine = engine_map[engine_choice]

    # Fallback for older Blender versions where EEVEE_NEXT doesn't exist
    available_engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if target_engine not in available_engines and target_engine == "BLENDER_EEVEE_NEXT":
        target_engine = "BLENDER_EEVEE"

    scene.render.engine = target_engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False


# --------------------------------------------------------------------------
# Import + rig
# --------------------------------------------------------------------------

def import_one_obj(filepath):
    """Import a single .obj file and return the newly created mesh objects.

    IMPORTANT: forward_axis='Y', up_axis='Z' explicitly disables Blender's
    default OBJ axis-conversion (which normally assumes the standard
    Wavefront convention of forward=-Z, up=Y, and rotates the mesh to match
    Blender's native Z-up on import). If this dataset's raw files are
    already Z-up, that default conversion silently adds an unwanted extra
    rotation - which is invisible if you only inspect the raw file (e.g. via
    compute_base_rotation.py's PCA, which reads the file directly and never
    sees Blender's import-time conversion), but very visible in the actual
    render. Forcing forward_axis='Y', up_axis='Z' means "don't remap
    anything" - the imported object's local axes match the file's raw (x,y,z)
    with zero rotation applied, keeping any rotation computed on raw
    vertices valid post-import.
    """
    existing = set(bpy.data.objects)

    # Blender 4.x uses wm.obj_import; older versions use import_scene.obj
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='Y', up_axis='Z')
    else:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='Y', axis_up='Z')

    new_objects = set(bpy.data.objects) - existing
    return [obj for obj in new_objects if obj.type == "MESH"]


def import_fragments(fragments_dir):
    """Import every piece_N.obj in fragments_dir as its own fragment object.

    Each resulting object is renamed to match its file's stem (e.g. "piece_3"),
    overriding whatever internal group/object name the OBJ file itself uses.
    This guarantees fragment map JSON files can reference fragments by a name
    that's stable and predictable, independent of how each .obj was exported.

    Fragments keep Blender's default flat shading (no shade_smooth applied) -
    this matches the established methodology confirmed against prior manual
    renders. A smoother-looking reference image elsewhere is due to that
    mesh having a much higher polygon count, not a different shading mode.
    """
    piece_files = sorted(
        glob.glob(os.path.join(fragments_dir, "piece_*.obj")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
    )
    if not piece_files:
        raise RuntimeError(f"No piece_*.obj files found in {fragments_dir}")

    fragments = []
    for filepath in piece_files:
        piece_name = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "piece_3"
        imported = import_one_obj(filepath)

        if len(imported) != 1:
            print(f"[WARN] {filepath} imported as {len(imported)} objects, expected 1 - renaming all")
            for i, obj in enumerate(imported):
                obj.name = piece_name if i == 0 else f"{piece_name}_part{i}"
        else:
            imported[0].name = piece_name

        fragments.extend(imported)

    return fragments


def compute_bounding_box(fragments):
    """World-space bounding box corners across all fragments."""
    min_co = Vector((float("inf"),) * 3)
    max_co = Vector((float("-inf"),) * 3)
    for obj in fragments:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            min_co.x, min_co.y, min_co.z = min(min_co.x, world_co.x), min(min_co.y, world_co.y), min(min_co.z, world_co.z)
            max_co.x, max_co.y, max_co.z = max(max_co.x, world_co.x), max(max_co.y, world_co.y), max(max_co.z, world_co.z)
    return min_co, max_co


def create_parent_empty(fragments):
    """Create an Empty at the assembly's centroid and parent all fragments to it
    (keep_transform=True so the assembled object stays intact geometrically)."""
    min_co, max_co = compute_bounding_box(fragments)
    centroid = (min_co + max_co) / 2

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=centroid)
    empty = bpy.context.active_object
    empty.name = "AssemblyRoot"

    for obj in fragments:
        obj.select_set(True)
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    return empty, centroid, (max_co - min_co)


def setup_view_rig(assembly_root, centroid):
    """Create a ViewRig empty above AssemblyRoot. ViewRig gets the shared,
    object-agnostic 0/90/180/270 view rotation; AssemblyRoot gets the
    one-time, per-object canonicalization rotation (base_rotation_euler_deg
    from the fragment map). Keeping these on separate Empties means the same
    view_presets.json produces a consistent real-world view (front/right/
    back/left) across every object, regardless of how each one happened to
    be oriented on import."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=centroid)
    view_rig = bpy.context.active_object
    view_rig.name = "ViewRig"

    assembly_root.select_set(True)
    view_rig.select_set(True)
    bpy.context.view_layer.objects.active = view_rig
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    return view_rig


def apply_base_rotation(assembly_root, base_rotation_euler_deg):
    """Apply the one-time canonicalization rotation for this object (set once,
    determined manually per object, stored in its fragment map)."""
    assembly_root.rotation_euler = Euler(
        [math.radians(a) for a in base_rotation_euler_deg], "XYZ"
    )
    bpy.context.view_layer.update()


def recompute_bounding_box_world(fragments):
    """Recompute the world-space bounding box after base rotation has been
    applied, so camera/lighting framing reflects the canonicalized pose
    rather than the raw import orientation."""
    return compute_bounding_box(fragments)


def setup_camera(centroid, extent, distance_factor=2.2):
    """Reposition Blender's default Camera (not a new one) to frame the
    assembly, based on its bounding box. Falls back to creating one only if
    the factory-default scene didn't include it for some reason."""
    max_dim = max(extent.x, extent.y, extent.z)
    distance = max_dim * distance_factor

    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam_obj)

    # Camera on -Y axis, slightly elevated, looking at centroid
    cam_obj.location = centroid + Vector((0, -distance, distance * 0.35))

    direction = centroid - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_lighting(centroid, extent, distance_factor=2.2, light_energy_scale=1000):
    """Reposition Blender's default single Light (matches the plain point
    lamp you get opening a fresh Blender file) rather than adding a custom
    multi-light rig. Energy is scaled roughly with object size so exposure
    stays reasonable across very differently-scaled objects in the dataset.
    light_energy_scale is tunable via CLI if renders look too bright/dark."""
    max_dim = max(extent.x, extent.y, extent.z)
    distance = max_dim * distance_factor

    light_obj = bpy.data.objects.get("Light")
    if light_obj is None:
        light_data = bpy.data.lights.new("Light", type="POINT")
        light_obj = bpy.data.objects.new("Light", light_data)
        bpy.context.collection.objects.link(light_obj)

    light_obj.location = centroid + Vector((-distance, -distance, distance * 1.2))

    if light_obj.data.type == "POINT":
        light_obj.data.energy = light_energy_scale * max(max_dim, 0.1)

    return light_obj


# --------------------------------------------------------------------------
# Per-experiment operations
# --------------------------------------------------------------------------

def apply_damage_config(fragments, hidden_fragment_names):
    """Hide the fragments listed for this damage config; show all others."""
    hidden_set = set(hidden_fragment_names)
    for obj in fragments:
        is_hidden = obj.name in hidden_set
        obj.hide_render = is_hidden


def show_all_fragments(fragments):
    for obj in fragments:
        obj.hide_render = False


def apply_view_rotation(view_rig, rotation_euler_deg):
    """Rotate the ViewRig (shared preset spin), leaving AssemblyRoot's
    canonicalization rotation untouched underneath it."""
    view_rig.rotation_euler = Euler([math.radians(a) for a in rotation_euler_deg], "XYZ")
    bpy.context.view_layer.update()


def render_to(filepath):
    scene = bpy.context.scene
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def export_metadata(filepath, object_name, damage_part, view_name, assembly_root, cam_obj, image_path):
    """Writes metadata in the exact schema FullMethod.py expects:
      - object: matches a key in objects.json (e.g. "beer_bottle")
      - damage_part: matches the {object}_{damage_part} key convention in
        damages.json (e.g. "base_severe") - this is just our damage_config
        name, no translation needed as long as damages.json has a matching
        entry for every damage_config in the fragment map
      - view: view preset name (e.g. "front")
      - object_rotation_deg: the NET combined rotation actually visible in
        the render (base canonicalization + view spin together), read from
        AssemblyRoot's world matrix rather than reporting the two rig layers
        separately
      - camera_location_m / camera_rotation_deg: camera pose, rotation
        converted from radians (Blender's internal units) to degrees
      - image_path: absolute path to the damaged_input.png this metadata
        describes, since FullMethod.py opens this file directly
    """
    net_rotation = assembly_root.matrix_world.to_euler()

    metadata = {
        "object": object_name,
        "damage_part": damage_part,
        "view": view_name,
        "object_rotation_deg": {
            "x": math.degrees(net_rotation.x),
            "y": math.degrees(net_rotation.y),
            "z": math.degrees(net_rotation.z),
        },
        "camera_location_m": {
            "x": cam_obj.location.x,
            "y": cam_obj.location.y,
            "z": cam_obj.location.z,
        },
        "camera_rotation_deg": {
            "x": math.degrees(cam_obj.rotation_euler.x),
            "y": math.degrees(cam_obj.rotation_euler.y),
            "z": math.degrees(cam_obj.rotation_euler.z),
        },
        "image_path": os.path.abspath(image_path),
    }
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.fragment_map) as f:
        fragment_map = json.load(f)
    with open(args.view_presets) as f:
        view_presets = json.load(f)

    reset_scene()
    set_render_settings(args.engine, args.resolution)

    fragments = import_fragments(args.fragments_dir)
    if not fragments:
        raise RuntimeError(f"No mesh fragments found after importing {args.fragments_dir}")

    empty, centroid, _ = create_parent_empty(fragments)

    base_rotation = fragment_map.get("base_rotation_euler_deg", [0, 0, 0])
    print(f"\n[DEBUG] Loaded base_rotation_euler_deg from {args.fragment_map}: {base_rotation}\n")

    view_rig = setup_view_rig(empty, centroid)
    apply_base_rotation(empty, base_rotation)

    # Recompute bounding box AFTER canonicalization so camera/lighting frame
    # the object as it will actually appear, not its raw import orientation
    min_co, max_co = recompute_bounding_box_world(fragments)
    post_rotation_centroid = (min_co + max_co) / 2
    extent = max_co - min_co

    cam_obj = setup_camera(post_rotation_centroid, extent, distance_factor=args.distance_factor)
    setup_lighting(post_rotation_centroid, extent, distance_factor=args.distance_factor, light_energy_scale=args.light_energy_scale)

    damage_configs = fragment_map["damage_configs"]
    views = view_presets["views"]

    for damage_config_name, hidden_fragment_names in damage_configs.items():
        for view in views:
            out_dir = os.path.join(
                args.output_root, args.object_id, damage_config_name, f"view_{view['id']}"
            )
            os.makedirs(out_dir, exist_ok=True)

            apply_view_rotation(view_rig, view["rotation_euler_deg"])

            # Damaged input render
            apply_damage_config(fragments, hidden_fragment_names)
            damaged_path = os.path.join(out_dir, "input.png")
            render_to(damaged_path)

            # Ground-truth intact render (same view, nothing hidden)
            show_all_fragments(fragments)
            render_to(os.path.join(out_dir, "ground_truth.png"))

            export_metadata(
                os.path.join(out_dir, "metadata.json"),
                args.object_id,
                damage_config_name,
                view["name"],
                empty,
                cam_obj,
                damaged_path,
            )

            print(f"[OK] {args.object_id} / {damage_config_name} / view_{view['id']}")

    print(f"Finished rendering all experiments for {args.object_id}")


if __name__ == "__main__":
    main()
