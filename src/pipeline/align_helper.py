"""
align_helper.py

Interactive companion to blender_render.py, for determining
base_rotation_euler_deg WITHOUT manually reading/typing N-panel values
(which is what caused beer_bottle and wine_glass to render on their side -
easy to rotate the wrong object, misread a field, or confuse viewport orbit
with actually rotating the object).

This sets up the EXACT same rig as blender_render.py (same reset_scene,
import_fragments, create_parent_empty, setup_view_rig functions - imported
directly, not reimplemented, so there is zero chance of the two scripts
drifting apart), then adds one button in the 3D viewport sidebar: rotate
AssemblyRoot however you like using Blender's normal tools (R key, or typing
into the N-panel), get it looking upright/front-facing, then click "Save
Base Rotation to Fragment Map" - it reads AssemblyRoot's actual rotation and
writes it directly into your fragment map JSON's base_rotation_euler_deg
field. No copying numbers by hand.

Usage Option A - command line (may hit a Blender context timing issue on
some systems/versions - if you see "context is incorrect", use Option B):
    blender --python align_helper.py -- \
        --fragments_dir path/to/fractured_55 \
        --fragment_map configs/fragment_maps/wine_glass.json

Usage Option B - Blender's built-in Text Editor (recommended if Option A
fails, since the UI is already fully loaded by the time you click Run):
    1. Open Blender normally (already running with the UI loaded)
    2. Switch to the "Scripting" tab at the top of the window
    3. In the Text Editor panel, click Open, navigate to and select
       align_helper.py
    4. EDIT the two fallback path variables near the top of parse_args()
       (fragments_dir and fragment_map_path) to point at the right files
    5. Click the "Run Script" (play button) at the top of the Text Editor

Then in Blender:
    1. Open the 3D viewport sidebar (press N if not already open)
    2. Go to the "Align" tab
    3. Select AssemblyRoot in the Outliner (it's already selected on load)
    4. Rotate it (R key, or type values into the Transform panel) until the
       object looks upright and front-facing
    5. Click "Save Base Rotation to Fragment Map" in the Align tab
    6. Check the terminal - it will confirm what was written and where
"""

import sys
import os
import json
import math

import bpy

# blender_render.py must be in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render as br


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    fragments_dir = None
    fragment_map_path = None
    i = 0
    while i < len(argv):
        if argv[i] == "--fragments_dir":
            fragments_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--fragment_map":
            fragment_map_path = argv[i + 1]
            i += 2
        else:
            i += 1

    # Fallback for running via Blender's Text Editor "Run Script" button,
    # where there's no way to pass command-line arguments - just edit these
    # two lines directly before clicking Run Script.
    if not fragments_dir:
        fragments_dir = "/Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/WineGlass/2d89d2b3b6749a9d99fbba385cc0d41d/fractured_55"
    if not fragment_map_path:
        fragment_map_path = "/Users/stevenbu/Desktop/configs/fragment_maps/wine_glass.json"

    return fragments_dir, fragment_map_path


FRAGMENTS_DIR, FRAGMENT_MAP_PATH = parse_args()


class ALIGN_OT_save_base_rotation(bpy.types.Operator):
    bl_idname = "align.save_base_rotation"
    bl_label = "Save Base Rotation to Fragment Map"
    bl_description = "Write AssemblyRoot's current rotation into the fragment map JSON's base_rotation_euler_deg field"

    def execute(self, context):
        assembly_root = bpy.data.objects.get("AssemblyRoot")
        if assembly_root is None:
            self.report({'ERROR'}, "AssemblyRoot not found in scene")
            return {'CANCELLED'}

        rot_deg = [
            round(math.degrees(assembly_root.rotation_euler.x), 3),
            round(math.degrees(assembly_root.rotation_euler.y), 3),
            round(math.degrees(assembly_root.rotation_euler.z), 3),
        ]

        with open(FRAGMENT_MAP_PATH) as f:
            fragment_map = json.load(f)

        fragment_map["base_rotation_euler_deg"] = rot_deg

        with open(FRAGMENT_MAP_PATH, "w") as f:
            json.dump(fragment_map, f, indent=2)

        msg = f"Saved base_rotation_euler_deg = {rot_deg} to {FRAGMENT_MAP_PATH}"
        print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}\n")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ALIGN_PT_panel(bpy.types.Panel):
    bl_label = "Align"
    bl_idname = "ALIGN_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Align"

    def draw(self, context):
        layout = self.layout
        layout.label(text="1. Select AssemblyRoot")
        layout.label(text="2. Rotate until upright/front-facing")
        layout.label(text="3. Click below when done:")
        layout.operator("align.save_base_rotation")

        assembly_root = bpy.data.objects.get("AssemblyRoot")
        if assembly_root:
            box = layout.box()
            box.label(text="Current rotation (deg):")
            box.label(text=f"X = {math.degrees(assembly_root.rotation_euler.x):.2f}")
            box.label(text=f"Y = {math.degrees(assembly_root.rotation_euler.y):.2f}")
            box.label(text=f"Z = {math.degrees(assembly_root.rotation_euler.z):.2f}")


def setup_scene():
    br.reset_scene()

    fragments = br.import_fragments(FRAGMENTS_DIR)
    if not fragments:
        raise RuntimeError(f"No mesh fragments found in {FRAGMENTS_DIR}")

    empty, centroid, _ = br.create_parent_empty(fragments)
    br.setup_view_rig(empty, centroid)

    # Select AssemblyRoot and make it active, so it's ready to rotate immediately
    bpy.ops.object.select_all(action='DESELECT')
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty

    # Frame the object in the viewport for convenience (best-effort - not
    # critical if this fails, just a convenience so you don't have to
    # manually scroll/pan to find the object)
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                with bpy.context.temp_override(area=area):
                    bpy.ops.view3d.view_selected()
                break
    except Exception as e:
        print(f"(Could not auto-frame viewport, not critical: {e})")

    print(f"\nLoaded {len(fragments)} fragments from {FRAGMENTS_DIR}")
    print("AssemblyRoot is selected. Rotate it in the viewport, then use the "
          "'Align' tab in the sidebar (press N) to save the rotation.\n")


def register():
    bpy.utils.register_class(ALIGN_OT_save_base_rotation)
    bpy.utils.register_class(ALIGN_PT_panel)


def deferred_setup():
    """Run setup_scene() after Blender's UI/window context is fully ready.
    Calling it immediately at script load time fails with 'context is
    incorrect', since operators like obj_import need a valid 3D viewport
    context that doesn't exist yet that early in startup."""
    setup_scene()
    return None  # returning None unregisters the timer (run once)


register()
bpy.app.timers.register(deferred_setup, first_interval=0.1)
