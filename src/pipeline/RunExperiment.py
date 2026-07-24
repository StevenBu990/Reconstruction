"""
RunExperiment.py

Orchestrates the automated portion of the pipeline:
  1. For each object in the manifest, call blender_render.py ONCE (it
     internally batches all of that object's damage_configs x views in one
     Blender session - see blender_render.py's docstring).
  2. For each experiment, call FullMethod.py against that experiment's
     metadata.json to get the Gemini reconstruction.

Coregistration and evaluation are NOT automated here - both are interactive
by design (ImageCoregistration pops up click-to-align windows; you decided
to keep that manual for now rather than force an automated call around it).
Once this script finishes, you have organized input/ground_truth/generated
images per experiment ready for you to run coregistration + eval on
yourself, in whatever order you like.

Resumable: skips any experiment whose generated.png already exists, so a
partial/interrupted run can just be re-launched.

Usage:
    python RunExperiment.py \
        --manifest configs/experiment_manifest.json \
        --blender_path "/Applications/Blender.app/Contents/MacOS/Blender" \
        --blender_render_script blender_render.py \
        --view_presets configs/view_presets.json \
        --output_root outputs \
        --full_method_script FullMethod.py \
        --objects_json objects.json \
        --damages_json damage.json \
        --engine WORKBENCH --distance_factor 1.5
"""

import argparse
import json
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--blender_path", required=True)
    parser.add_argument("--blender_render_script", required=True)
    parser.add_argument("--view_presets", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--full_method_script", required=True)
    parser.add_argument("--objects_json", required=True)
    parser.add_argument("--damages_json", required=True)
    parser.add_argument("--engine", default="WORKBENCH")
    parser.add_argument("--distance_factor", type=float, default=1.5)
    parser.add_argument("--light_energy_scale", type=float, default=1000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--log", default="run_log.jsonl")
    parser.add_argument("--object_id", default=None,
                         help="Only process this one object_id - use to test the orchestrator "
                              "on a small batch before running the full manifest.")
    return parser.parse_args()


def log_event(log_path, event):
    event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def render_object(args, object_id, fragments_dir, fragment_map_path):
    """One Blender call per object - blender_render.py handles all of that
    object's damage_configs x views internally in a single session."""
    cmd = [
        args.blender_path, "--background", "--python", args.blender_render_script, "--",
        "--fragments_dir", fragments_dir,
        "--object_id", object_id,
        "--fragment_map", fragment_map_path,
        "--view_presets", args.view_presets,
        "--output_root", args.output_root,
        "--engine", args.engine,
        "--resolution", str(args.resolution),
        "--distance_factor", str(args.distance_factor),
        "--light_energy_scale", str(args.light_energy_scale),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def run_full_method(args, metadata_path):
    cmd = [
        sys.executable, args.full_method_script,
        "--metadata", metadata_path,
        "--objects", args.objects_json,
        "--damages", args.damages_json,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def main():
    args = parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.object_id:
        manifest = [exp for exp in manifest if exp["object_id"] == args.object_id]
        if not manifest:
            print(f"No experiments found for object_id={args.object_id}")
            return

    # Group experiments by object so Blender only runs once per object
    objects = {}
    for exp in manifest:
        objects.setdefault(exp["object_id"], {
            "fragments_dir": exp["fragments_dir"],
            "fragment_map_path": exp["fragment_map_path"],
            "experiments": [],
        })["experiments"].append(exp)

    print(f"Loaded {len(manifest)} experiments across {len(objects)} objects\n")

    for object_id, obj_data in objects.items():
        experiments = obj_data["experiments"]

        # Skip rendering entirely if every experiment for this object already
        # has both input.png and ground_truth.png
        needs_render = any(
            not os.path.exists(exp["input_image"]) or not os.path.exists(exp["ground_truth_image"])
            for exp in experiments
        )

        if needs_render:
            print(f"[RENDER] {object_id} ({len(experiments)} experiments)...")
            ok, stdout, stderr = render_object(
                args, object_id, obj_data["fragments_dir"], obj_data["fragment_map_path"]
            )
            if not ok:
                print(f"[FAIL] Blender render failed for {object_id}")
                log_event(args.log, {"object_id": object_id, "stage": "blender_render",
                                      "status": "failed", "stderr": stderr[-2000:]})
                continue
            log_event(args.log, {"object_id": object_id, "stage": "blender_render", "status": "ok"})
        else:
            print(f"[SKIP] {object_id} already rendered")

        for exp in experiments:
            label = f"{object_id}/{exp['damage_config']}/view_{exp['view_id']}"

            if os.path.exists(exp["generated_image"]):
                print(f"  [SKIP] {label} already has generated.png")
                continue

            if not os.path.exists(exp["metadata_path"]):
                print(f"  [FAIL] {label} missing metadata.json - render may have failed silently")
                log_event(args.log, {"experiment": label, "stage": "full_method",
                                      "status": "failed", "reason": "missing metadata.json"})
                continue

            print(f"  [GEMINI] {label}...")
            ok, stdout, stderr = run_full_method(args, exp["metadata_path"])
            if ok and os.path.exists(exp["generated_image"]):
                print(f"  [OK] {label}")
                log_event(args.log, {"experiment": label, "stage": "full_method", "status": "ok"})
            else:
                print(f"  [FAIL] {label}")
                log_event(args.log, {"experiment": label, "stage": "full_method",
                                      "status": "failed", "stderr": stderr[-2000:]})

    print("\nDone. Blender rendering + Gemini reconstruction complete for all resolvable experiments.")
    print("Coregistration and evaluation are a separate manual pass - see run_log.jsonl for any failures.")


if __name__ == "__main__":
    main()
