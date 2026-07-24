"""
CoregisterAndEvaluate.py

Second-stage orchestrator, run AFTER RunExperiment.py has produced
generated.png for each experiment. Handles:

  1. Coregistration - runs your image_register.py with target=ground_truth.png,
     source=generated.png, --sift --ransac. Since input.png and
     ground_truth.png share the exact same camera (both rendered by Blender
     at the same view for a given experiment), aligning generated.png to
     ground_truth.png's frame also aligns it to input.png's frame. This
     still pops up interactive cv2 windows (matches, target_new, source_new)
     that block on a keypress - you tap through them one at a time, per
     your plan, rather than manually clicking points.

     image_register.py's paths (data/img, result/img) are hardcoded
     relative to its working directory - rather than editing that script,
     this orchestrator runs it with cwd set to each experiment's own output
     folder, with ground_truth.png/generated.png copied into a local data/
     subfolder. Zero changes needed to image_register.py or utils.py.

  2. GT_Mask.png - extract_missing_mask.py comparing ground_truth.png vs
     input.png (no alignment needed - these already share the same camera
     frame, both rendered by Blender).

  3. GT_Fragment.png - extract_missing_mask.py comparing the ALIGNED
     reconstruction (result/source_new.png from step 1) vs input.png.

  4. evaluate.py - computes IoU/Dice between GT_Mask and GT_Fragment,
     writes metrics.json into the experiment's output folder.

Resumable: skips any experiment that already has a metrics.json.

Usage:
    python CoregisterAndEvaluate.py \
        --manifest configs/experiment_manifest.json \
        --coregistration_script image_register.py \
        --extract_mask_script extract_missing_mask.py \
        --evaluate_script evaluate.py \
        --scale 1.0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--coregistration_script", required=True)
    parser.add_argument("--extract_mask_script", required=True)
    parser.add_argument("--evaluate_script", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--object_id", default=None,
                         help="Only process this one object_id, for testing before a full run")
    parser.add_argument("--log", default="coregister_eval_log.jsonl")
    return parser.parse_args()


def log_event(log_path, event):
    event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


import cv2
import numpy as np


def estimate_foreground_fraction(image_path):
    """Rough estimate of what fraction of the image is 'object' vs background,
    using the same background-distance approach as extract_missing_mask.py.
    Used to sanity-check alignment results, not for the real mask extraction."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    corners = np.concatenate([
        img[:50, :50].reshape(-1, 3), img[:50, -50:].reshape(-1, 3),
        img[-50:, :50].reshape(-1, 3), img[-50:, -50:].reshape(-1, 3)
    ])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(img - bg, axis=2)
    foreground = dist > 15  # loose threshold, just for a rough sanity check
    return float(np.sum(foreground)) / (h * w)


def check_alignment_sanity(output_dir):
    """Flags obviously broken alignments (e.g. SIFT/RANSAC locking onto bad
    matches and producing a degenerate warp) by comparing how much of the
    frame the aligned object occupies vs the target - a real object should
    occupy a roughly similar fraction, not e.g. 5x more or less."""
    target_path = os.path.join(output_dir, "result", "target_new.jpg")
    source_path = os.path.join(output_dir, "result", "source_new.png")

    target_frac = estimate_foreground_fraction(target_path)
    source_frac = estimate_foreground_fraction(source_path)

    if target_frac is None or source_frac is None:
        return False, "could not read alignment output images"

    if target_frac < 0.01 or source_frac < 0.01:
        return False, f"near-empty foreground (target={target_frac:.3f}, source={source_frac:.3f})"

    ratio = source_frac / target_frac
    if ratio < 0.3 or ratio > 3.0:
        return False, f"foreground fraction mismatch (target={target_frac:.3f}, source={source_frac:.3f}, ratio={ratio:.2f})"

    return True, None


def run_coregistration(args, output_dir):
    """Sets up data/ + result/ subfolders inside output_dir, then runs
    image_register.py with cwd=output_dir so its hardcoded relative paths
    resolve correctly without any code changes. target=ground_truth.png,
    source=generated.png - Gemini's output gets warped onto ground_truth's
    frame. Since input.png and ground_truth.png share the exact same camera
    (both rendered by Blender at the same view), this also aligns the
    result to input.png's frame for the downstream mask comparison."""
    data_dir = os.path.join(output_dir, "data")
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    shutil.copy(os.path.join(output_dir, "ground_truth.png"), os.path.join(data_dir, "ground_truth.png"))
    shutil.copy(os.path.join(output_dir, "generated.png"), os.path.join(data_dir, "generated.png"))

    cmd = [
        sys.executable, os.path.abspath(args.coregistration_script),
        "ground_truth.png", "generated.png",
        "--sift", "--ransac", "--scale", str(args.scale),
    ]
    # This call blocks until you close/advance the cv2 windows it pops up
    result = subprocess.run(cmd, cwd=output_dir)
    aligned_path = os.path.join(result_dir, "source_new.png")
    return result.returncode == 0 and os.path.exists(aligned_path), aligned_path


def run_extract_mask(args, reference, comparison, output_name, output_dir):
    cmd = [
        sys.executable, os.path.abspath(args.extract_mask_script),
        "--reference", reference,
        "--comparison", comparison,
        "--output", output_name,
        "--output_dir", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def run_evaluate(args, gt_mask, gt_fragment, output_dir):
    cmd = [
        sys.executable, os.path.abspath(args.evaluate_script),
        "--gt_mask", gt_mask,
        "--gt_fragment", gt_fragment,
        "--output_dir", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def main():
    args = parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.object_id:
        manifest = [exp for exp in manifest if exp["object_id"] == args.object_id]

    print(f"Processing {len(manifest)} experiments\n")

    for exp in manifest:
        output_dir = exp["output_dir"]
        label = f"{exp['object_id']}/{exp['damage_config']}/view_{exp['view_id']}"

        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            print(f"[SKIP] {label} already evaluated")
            continue

        if not os.path.exists(exp["generated_image"]):
            print(f"[SKIP] {label} - no generated.png yet, run RunExperiment.py first")
            continue

        print(f"[COREG] {label} - tap through the popup windows to continue...")
        ok, aligned_path = run_coregistration(args, output_dir)
        if not ok:
            print(f"  [FAIL] coregistration failed for {label}")
            log_event(args.log, {"experiment": label, "stage": "coregistration", "status": "failed"})
            continue

        sane, reason = check_alignment_sanity(output_dir)
        if not sane:
            print(f"  [FLAG] {label} - alignment looks broken: {reason}")
            metrics = {"iou": None, "dice": None, "flag": "bad_alignment", "reason": reason}
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            log_event(args.log, {"experiment": label, "stage": "alignment_sanity", "status": "flagged", "reason": reason})
            continue

        print(f"  [MASK] computing GT_Mask...")
        gt_path = os.path.join(output_dir, "ground_truth.png")
        input_path = os.path.join(output_dir, "input.png")
        ok, err = run_extract_mask(args, gt_path, input_path, "GT_Mask.png", output_dir)
        if not ok:
            print(f"  [FAIL] GT_Mask extraction failed for {label}")
            log_event(args.log, {"experiment": label, "stage": "gt_mask", "status": "failed", "stderr": err[-1000:]})
            continue

        print(f"  [MASK] computing GT_Fragment...")
        ok, err = run_extract_mask(args, aligned_path, input_path, "GT_Fragment.png", output_dir)
        if not ok:
            print(f"  [FAIL] GT_Fragment extraction failed for {label}")
            log_event(args.log, {"experiment": label, "stage": "gt_fragment", "status": "failed", "stderr": err[-1000:]})
            continue

        print(f"  [EVAL] computing IoU/Dice...")
        gt_mask_path = os.path.join(output_dir, "GT_Mask.png")
        gt_fragment_path = os.path.join(output_dir, "GT_Fragment.png")
        ok, err = run_evaluate(args, gt_mask_path, gt_fragment_path, output_dir)
        if ok and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            if metrics.get("flag"):
                print(f"  [FLAG] {label} - {metrics['flag']}")
            else:
                print(f"  [OK] {label} - IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}")
            log_event(args.log, {"experiment": label, "stage": "evaluate", "status": "ok", "metrics": metrics})
        else:
            print(f"  [FAIL] evaluation failed for {label}")
            log_event(args.log, {"experiment": label, "stage": "evaluate", "status": "failed", "stderr": err[-1000:]})

    print("\nDone.")


if __name__ == "__main__":
    main()
