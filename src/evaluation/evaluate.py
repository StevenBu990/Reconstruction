"""
evaluate.py

Same exact logic as your full_evaluation script - crops to the damaged
region (with padding), computes IoU and Dice between GT_Mask and
GT_Fragment. Only change: CLI args instead of hardcoded paths, and metrics
saved to a JSON file instead of just printed.

Usage:
    python evaluate.py \
        --gt_mask GT_Mask.png \
        --gt_fragment GT_Fragment.png \
        --output_dir .
"""

import argparse
import json
import os

import cv2
import numpy as np


def extract_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create initial object mask
    mask = (gray > 75).astype(np.uint8)

    # Remove small noise and fill gaps
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_mask", required=True, help="GT_Mask.png path")
    parser.add_argument("--gt_fragment", required=True, help="GT_Fragment.png path")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--output_name", default="metrics.json",
                         help="Filename for the metrics JSON, e.g. metrics_marked.json "
                              "to avoid overwriting a baseline metrics.json")
    parser.add_argument("--padding", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load GT handle mask
    gt = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt > 128).astype(np.uint8)

    # Obtain bounding box of damaged region
    ys, xs = np.where(gt_mask > 0)

    if len(ys) == 0:
        # GT_Mask has zero foreground pixels - the "damage" wasn't detectable
        # as a difference between ground_truth and input for this view. This
        # is a real data-quality flag worth reviewing, not a crash to hide.
        print("WARNING: GT_Mask has zero foreground pixels - damage not "
              "detectable as a ground_truth/input difference for this view.")
        metrics = {
            "iou": None,
            "dice": None,
            "gt_pixels": 0,
            "pred_pixels": None,
            "intersection_pixels": None,
            "flag": "empty_gt_mask",
        }
        with open(os.path.join(args.output_dir, args.output_name), "w") as f:
            json.dump(metrics, f, indent=2)
        return

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    ymin = max(0, ymin - args.padding)
    ymax = min(gt_mask.shape[0], ymax + args.padding)
    xmin = max(0, xmin - args.padding)
    xmax = min(gt_mask.shape[1], xmax + args.padding)

    # Load prediction
    pred = cv2.imread(args.gt_fragment)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_mask = extract_mask(pred)

    cv2.imwrite(os.path.join(args.output_dir, "debug_mask.png"), pred_mask * 255)

    print("Unique mask values:", np.unique(pred_mask))
    print("Foreground pixels:", np.sum(pred_mask))
    print("Total pixels:", pred_mask.size)

    # Crop BOTH to handle region
    gt_crop = gt_mask[ymin:ymax, xmin:xmax]
    pred_crop = pred_mask[ymin:ymax, xmin:xmax]

    if np.sum(pred_crop) == 0:
        # Predicted mask has zero pixels in the cropped region - meaningfully
        # different from "prediction exists but doesn't overlap": this means
        # extract_mask found nothing distinguishable there at all (either the
        # reconstruction didn't add visible content, or alignment shifted the
        # predicted region entirely outside the crop). Flag it rather than
        # reporting an indistinguishable IoU=0.
        print("WARNING: predicted mask has zero pixels in the cropped region.")
        metrics = {
            "iou": None,
            "dice": None,
            "gt_pixels": int(np.sum(gt_crop)),
            "pred_pixels": 0,
            "intersection_pixels": 0,
            "flag": "empty_prediction",
        }
        with open(os.path.join(args.output_dir, args.output_name), "w") as f:
            json.dump(metrics, f, indent=2)
        return

    gt_bool = gt_crop.astype(bool)
    pred_bool = pred_crop.astype(bool)

    intersection = np.logical_and(gt_bool, pred_bool)
    union = np.logical_or(gt_bool, pred_bool)

    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    dice = (2 * np.sum(intersection)) / (np.sum(gt_bool) + np.sum(pred_bool) + 1e-8)

    print("IoU:", iou)
    print("Dice:", dice)

    overlay = np.zeros((*gt_crop.shape, 3), dtype=np.uint8)
    overlay[intersection] = [0, 255, 0]
    overlay[gt_bool & ~pred_bool] = [255, 0, 0]
    overlay[pred_bool & ~gt_bool] = [0, 0, 255]
    cv2.imwrite(os.path.join(args.output_dir, "overlayFull.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("GT crop:", np.sum(gt_crop))
    print("Pred crop:", np.sum(pred_crop))
    print("Intersection:", np.sum(intersection))

    metrics = {
        "iou": float(iou),
        "dice": float(dice),
        "gt_pixels": int(np.sum(gt_crop)),
        "pred_pixels": int(np.sum(pred_crop)),
        "intersection_pixels": int(np.sum(intersection)),
    }
    with open(os.path.join(args.output_dir, args.output_name), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
