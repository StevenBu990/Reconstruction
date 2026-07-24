"""
centroid_distance.py

Complements IoU/Dice with a more forgiving "how close is the repair to the
right place and size" metric - directly addressing the gap your meeting
notes flagged: IoU can't distinguish "repaired in roughly the right spot,
slightly wrong shape" from "repaired in the wrong place entirely." Both
just show up as reduced overlap under IoU.

Computes, between GT_Mask.png (true missing region) and GT_Fragment.png
(predicted missing region):
  - centroid distance (delta_x, delta_y, euclidean distance in pixels)
  - area ratio (predicted area / true area - are they roughly the right size?)

Usage:
    python centroid_distance.py \
        --gt_mask GT_Mask.png \
        --gt_fragment GT_Fragment.png \
        --output_dir . \
        --output_name centroid_metrics.json
"""

import argparse
import json
import os

import cv2
import numpy as np


def mask_centroid(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_mask", required=True)
    parser.add_argument("--gt_fragment", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--output_name", default="centroid_metrics.json")
    args = parser.parse_args()

    gt = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt > 128).astype(np.uint8)

    pred = cv2.imread(args.gt_fragment, cv2.IMREAD_GRAYSCALE)
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask = (pred > 128).astype(np.uint8)

    gt_centroid = mask_centroid(gt_mask)
    pred_centroid = mask_centroid(pred_mask)

    gt_area = int(np.sum(gt_mask))
    pred_area = int(np.sum(pred_mask))

    metrics = {
        "gt_centroid": gt_centroid,
        "pred_centroid": pred_centroid,
        "delta_x": None,
        "delta_y": None,
        "centroid_distance_px": None,
        "gt_area_px": gt_area,
        "pred_area_px": pred_area,
        "area_ratio": (pred_area / gt_area) if gt_area > 0 else None,
    }

    if gt_centroid is not None and pred_centroid is not None:
        dx = pred_centroid[0] - gt_centroid[0]
        dy = pred_centroid[1] - gt_centroid[1]
        metrics["delta_x"] = dx
        metrics["delta_y"] = dy
        metrics["centroid_distance_px"] = float(np.sqrt(dx ** 2 + dy ** 2))
        print(f"Centroid distance: {metrics['centroid_distance_px']:.1f}px "
              f"(dx={dx:.1f}, dy={dy:.1f})")
    else:
        print("Could not compute centroid distance - one of the masks is empty.")

    if gt_area > 0:
        print(f"Area ratio (pred/gt): {metrics['area_ratio']:.2f} "
              f"(gt={gt_area}px, pred={pred_area}px)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_name), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
