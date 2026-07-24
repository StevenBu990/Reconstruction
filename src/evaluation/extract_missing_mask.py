"""
extract_missing_mask.py

Same exact logic as your obtain_missing_piece.py - computes the region
present in a "reference" (intact) image but absent in a "comparison"
(damaged) image, via background-distance thresholding + Otsu. Only change:
CLI args instead of hardcoded paths, so one script produces either:

    GT_Mask.png     (reference=ground_truth.png, comparison=input.png)
    GT_Fragment.png (reference=aligned generated.png, comparison=input.png)

instead of maintaining two near-identical copies of the same file.

Usage:
    python extract_missing_mask.py \
        --reference ground_truth.png \
        --comparison input.png \
        --output GT_Mask.png \
        --output_dir .
"""

import argparse
import os

import cv2
import numpy as np


def filter_by_compactness(mask, min_circularity=0.15):
    """Removes connected components that are thin/ring-like (e.g. an
    alignment-offset halo tracing an object's whole silhouette) while
    keeping compact blob-shaped components (a genuine missing fragment).
    Circularity = 4*pi*area / perimeter^2, which is ~1 for a circle and
    approaches 0 for a thin elongated ring - a much better distinguishing
    signal than component thickness/kernel size, since a halo can be just
    as thick as a real fragment but is always much more elongated relative
    to its enclosed area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= min_circularity:
            cv2.drawContours(filtered, [contour], -1, 1, thickness=cv2.FILLED)
    return filtered


def extract_mask(image, kernel_size=5):
    # Convert RGB image to float
    img = image.astype(np.float32)

    # Estimate background color using corners
    corners = np.concatenate([
        img[:50, :50].reshape(-1, 3),
        img[:50, -50:].reshape(-1, 3),
        img[-50:, :50].reshape(-1, 3),
        img[-50:, -50:].reshape(-1, 3)
    ])

    bg = np.median(corners, axis=0)
    print("Estimated background:", bg)

    # Compute color distance from background
    dist = np.linalg.norm(img - bg, axis=2)

    # Normalize distance image to 0-255
    dist_uint8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Automatically choose threshold
    _, mask = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    print("Otsu threshold:", _)

    # Clean mask - kernel_size controls how thick a region must be to survive.
    # A larger kernel erases thin slivers (e.g. an alignment-offset halo
    # tracing an object's whole silhouette) while preserving genuinely thick
    # blob regions (a real missing fragment).
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Intact/full image (e.g. ground_truth.png or aligned generated.png)")
    parser.add_argument("--comparison", required=True, help="Damaged image (e.g. input.png)")
    parser.add_argument("--output", required=True, help="Output filename, e.g. GT_Mask.png or GT_Fragment.png")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--kernel_size", type=int, default=5,
                         help="Morphological cleanup kernel size. Larger values erase thin "
                              "slivers (e.g. an alignment-offset halo tracing an object's "
                              "whole silhouette) while preserving genuine thick fragment blobs. "
                              "Try 15-25 if you see a thin outline instead of a solid blob.")
    parser.add_argument("--filter_compactness", action="store_true",
                         help="Remove thin ring-like components (e.g. an alignment-offset "
                              "halo tracing an object's whole silhouette) from the output, "
                              "keeping only compact blob-shaped regions. Use this if the "
                              "output looks like a thin outline instead of a solid blob near "
                              "the actual damage.")
    parser.add_argument("--min_circularity", type=float, default=0.15,
                         help="Threshold for --filter_compactness (0=any shape, 1=perfect "
                              "circle). Lower = more permissive.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    intact = cv2.imread(args.reference)
    fractured = cv2.imread(args.comparison)

    print("Top-left:", intact[0, 0])
    print("Center:", intact[intact.shape[0] // 2, intact.shape[1] // 2])
    print("Bottom-right:", intact[-1, -1])

    intact = cv2.cvtColor(intact, cv2.COLOR_BGR2RGB)
    fractured = cv2.cvtColor(fractured, cv2.COLOR_BGR2RGB)

    fractured = cv2.resize(fractured, (intact.shape[1], intact.shape[0]), interpolation=cv2.INTER_NEAREST)

    print("Images:")
    print("Intact:", intact.shape)
    print("Fractured:", fractured.shape)

    intact_mask = extract_mask(intact, kernel_size=args.kernel_size)
    fractured_mask = extract_mask(fractured, kernel_size=args.kernel_size)

    cv2.imwrite(os.path.join(args.output_dir, "intact_mask.png"), intact_mask * 255)
    cv2.imwrite(os.path.join(args.output_dir, "fractured_mask.png"), fractured_mask * 255)

    missing_mask = np.logical_and(intact_mask == 1, fractured_mask == 0).astype(np.uint8)

    if args.filter_compactness:
        before_count = int(np.sum(missing_mask))
        missing_mask = filter_by_compactness(missing_mask, min_circularity=args.min_circularity)
        after_count = int(np.sum(missing_mask))
        print(f"Compactness filter: {before_count} -> {after_count} pixels "
              f"(min_circularity={args.min_circularity})")

    cv2.imwrite(os.path.join(args.output_dir, args.output), missing_mask * 255)

    overlay = np.zeros((*missing_mask.shape, 3), dtype=np.uint8)
    overlay[missing_mask == 1] = [255, 0, 0]
    cv2.imwrite(os.path.join(args.output_dir, f"overlay_{args.output}"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Intact pixels:", np.sum(intact_mask))
    print("Fractured pixels:", np.sum(fractured_mask))
    print("Missing pixels:", np.sum(missing_mask))


if __name__ == "__main__":
    main()
