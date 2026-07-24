"""
aggregate_metrics.py

Walks the manifest, reads each experiment's metrics.json (if it exists),
and writes one summary CSV with IoU/Dice for every experiment - no more
grepping through log files by hand.

Usage:
    python aggregate_metrics.py \
        --manifest configs/experiment_manifest.json \
        --output results/aggregated_metrics.csv
"""

import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows = []
    for exp in manifest:
        metrics_path = os.path.join(exp["output_dir"], "metrics.json")
        row = {
            "object_id": exp["object_id"],
            "damage_config": exp["damage_config"],
            "view_id": exp["view_id"],
            "view_name": exp["view_name"],
            "iou": None,
            "dice": None,
            "gt_pixels": None,
            "pred_pixels": None,
            "flag": "not_evaluated",
        }
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            row["iou"] = metrics.get("iou")
            row["dice"] = metrics.get("dice")
            row["gt_pixels"] = metrics.get("gt_pixels")
            row["pred_pixels"] = metrics.get("pred_pixels")
            row["flag"] = metrics.get("flag", "")
        rows.append(row)

    fieldnames = ["object_id", "damage_config", "view_id", "view_name",
                  "iou", "dice", "gt_pixels", "pred_pixels", "flag"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evaluated = [r for r in rows if r["iou"] is not None]
    flagged = [r for r in rows if r["flag"] not in ("", "not_evaluated") and r["iou"] is None]
    not_yet = [r for r in rows if r["flag"] == "not_evaluated"]

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Evaluated: {len(evaluated)}")
    print(f"  Flagged (empty GT_Mask): {len(flagged)}")
    print(f"  Not yet evaluated: {len(not_yet)}")

    if evaluated:
        print(f"\n{'Object':<20}{'Damage Config':<20}{'View':<8}{'IoU':<8}{'Dice':<8}")
        print("-" * 64)
        for r in evaluated:
            print(f"{r['object_id']:<20}{r['damage_config']:<20}{r['view_name']:<8}"
                  f"{r['iou']:<8.3f}{r['dice']:<8.3f}")

        mean_iou = sum(r["iou"] for r in evaluated) / len(evaluated)
        mean_dice = sum(r["dice"] for r in evaluated) / len(evaluated)
        print("-" * 64)
        print(f"{'MEAN':<48}{mean_iou:<8.3f}{mean_dice:<8.3f}")

    if flagged:
        print(f"\nFlagged experiments:")
        for r in flagged:
            print(f"  {r['object_id']}/{r['damage_config']}/{r['view_name']} - {r['flag']}")


if __name__ == "__main__":
    main()
