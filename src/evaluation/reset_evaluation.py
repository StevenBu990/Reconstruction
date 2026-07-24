"""
reset_evaluation.py

Deletes coregistration/mask/evaluation outputs (metrics.json, GT_Mask.png,
GT_Fragment.png, overlay images, data/ and result/ subfolders) for selected
experiments, WITHOUT touching input.png, ground_truth.png, generated.png,
or metadata.json - so a re-run of CoregisterAndEvaluate.py redoes alignment
and evaluation from scratch without needing to re-render or re-run Gemini.

Usage:
    # Reset everything except beer_bottle:
    python reset_evaluation.py \
        --manifest configs/experiment_manifest.json \
        --exclude_objects beer_bottle

    # Reset only specific objects:
    python reset_evaluation.py \
        --manifest configs/experiment_manifest.json \
        --object_ids wine_glass vase
"""

import argparse
import json
import os
import shutil


EVAL_FILES = [
    "metrics.json", "GT_Mask.png", "GT_Fragment.png",
    "intact_mask.png", "fractured_mask.png", "debug_mask.png",
    "overlayFull.png", "overlay_GT_Mask.png", "overlay_GT_Fragment.png",
]
EVAL_DIRS = ["data", "result"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--object_ids", nargs="+", default=None,
                         help="Only reset these object_ids")
    parser.add_argument("--exclude_objects", nargs="+", default=None,
                         help="Reset all objects EXCEPT these")
    parser.add_argument("--experiments", nargs="+", default=None,
                         help="Only reset these exact experiments, each as "
                              "object_id/damage_config/view_name, e.g. "
                              "beer_bottle/base_simple/right")
    parser.add_argument("--dry_run", action="store_true",
                         help="Just print what would be deleted, don't actually delete")
    args = parser.parse_args()

    if sum([bool(args.object_ids), bool(args.exclude_objects), bool(args.experiments)]) > 1:
        raise ValueError("Use only one of --object_ids, --exclude_objects, --experiments")

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.object_ids:
        manifest = [e for e in manifest if e["object_id"] in args.object_ids]
    elif args.exclude_objects:
        manifest = [e for e in manifest if e["object_id"] not in args.exclude_objects]
    elif args.experiments:
        wanted = set(args.experiments)
        manifest = [
            e for e in manifest
            if f"{e['object_id']}/{e['damage_config']}/{e['view_name']}" in wanted
        ]

    count = 0
    for exp in manifest:
        output_dir = exp["output_dir"]
        label = f"{exp['object_id']}/{exp['damage_config']}/view_{exp['view_id']}"

        removed_something = False
        for filename in EVAL_FILES:
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                removed_something = True
                if args.dry_run:
                    print(f"  would remove: {path}")
                else:
                    os.remove(path)

        for dirname in EVAL_DIRS:
            path = os.path.join(output_dir, dirname)
            if os.path.exists(path):
                removed_something = True
                if args.dry_run:
                    print(f"  would remove dir: {path}")
                else:
                    shutil.rmtree(path)

        if removed_something:
            count += 1
            if not args.dry_run:
                print(f"[RESET] {label}")

    action = "Would reset" if args.dry_run else "Reset"
    print(f"\n{action} {count} experiments out of {len(manifest)} in scope")


if __name__ == "__main__":
    main()
