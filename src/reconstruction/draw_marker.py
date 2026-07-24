"""
draw_marker.py

Two variants of the spatial-guidance experiment, both derived from
GT_Mask.png (which already encodes exactly where the true missing region
is in image space):

  --shape circle / box : Variant 1 - location only, no shape hint
  --shape outline      : Variant 2 - traces the RAW missing region contour
                          (can be jagged/noisy from pixel-level threshold
                          artifacts - risk of being visually misread, e.g.
                          a jagged blob at a bottle's neck read as a cap)
  --shape smooth_outline : Variant 2b - convex hull + polygon simplification
                          of the same contour, testing whether a cleaner
                          shape hint avoids the misreading risk above

Usage:
    python draw_marker.py \
        --gt_mask GT_Mask.png \
        --input input.png \
        --output input_with_marker.png \
        --shape circle \
        --color red \
        --padding 20
"""

import argparse

import cv2
import numpy as np


COLOR_MAP = {
    "red": (0, 0, 255),      # BGR
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_mask", required=True, help="GT_Mask.png - defines where the marker goes")
    parser.add_argument("--input", required=True, help="input.png - the damaged image to mark up")
    parser.add_argument("--output", required=True, help="Output path for the marked-up image")
    parser.add_argument("--shape", choices=["circle", "box", "outline", "smooth_outline"], default="circle")
    parser.add_argument("--color", choices=list(COLOR_MAP.keys()), default="red")
    parser.add_argument("--padding", type=int, default=20,
                         help="Extra pixels added around the missing region's extent (circle/box only)")
    parser.add_argument("--thickness", type=int, default=4)
    parser.add_argument("--smoothing_epsilon", type=float, default=0.01,
                         help="Fraction of contour perimeter used as approxPolyDP epsilon "
                              "(smooth_outline only) - higher = simpler/smoother shape")
    args = parser.parse_args()

    gt_mask = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
    mask_binary = (gt_mask > 128).astype(np.uint8)

    ys, xs = np.where(mask_binary > 0)
    if len(ys) == 0:
        raise RuntimeError(f"{args.gt_mask} has zero foreground pixels - nothing to mark. "
                            f"This experiment shouldn't run on empty_gt_mask-flagged experiments.")

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    input_img = cv2.imread(args.input)
    color = COLOR_MAP[args.color]

    if args.shape == "circle":
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        radius = int(max(xmax - xmin, ymax - ymin) / 2) + args.padding
        cv2.circle(input_img, (cx, cy), radius, color, args.thickness)
    elif args.shape == "box":
        cv2.rectangle(
            input_img,
            (max(0, xmin - args.padding), max(0, ymin - args.padding)),
            (min(input_img.shape[1], xmax + args.padding), min(input_img.shape[0], ymax + args.padding)),
            color, args.thickness
        )
    elif args.shape == "outline":  # trace the actual missing region's raw contour
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(input_img, contours, -1, color, args.thickness)
    else:  # smooth_outline - convex hull + polygon simplification, avoids jagged pixel noise
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)
        simplified = cv2.approxPolyDP(hull, args.smoothing_epsilon * perimeter, True)
        cv2.drawContours(input_img, [simplified], -1, color, args.thickness)

    cv2.imwrite(args.output, input_img)
    print(f"Wrote {args.output} with a {args.color} {args.shape} marker around the missing region")


if __name__ == "__main__":
    main()

