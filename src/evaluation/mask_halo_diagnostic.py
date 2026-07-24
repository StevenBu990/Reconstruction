#!/usr/bin/env python3
"""
mask_halo_diagnostic.py

Diagnoses and attempts to clean up the "edge halo" artifact seen in
GT_Fragment*.png masks (extract_missing_mask.py output), where residual
sub-pixel alignment noise along a long object silhouette edge creates a
thin false-positive trail connected to (or near) the true missing-fragment
blob.

WHAT IT DOES
1. Loads a binary mask (any of your GT_Fragment*.png / GT_Mask.png files).
2. Sweeps across a range of morphological "opening" kernel sizes
   (erosion then dilation). Opening removes thin structures (like a
   halo trailing along an edge) while leaving large blobs mostly intact,
   PROVIDED the halo is thinner than the blob's narrowest dimension.
3. At each kernel size, runs connected-component labeling and reports:
   - number of separate components
   - total foreground area
   - largest component's area, bounding box, and centroid
4. Prints a table so you can look for the size where total area drops
   sharply (halo disappearing) while largest-component area stays
   roughly stable (the real blob survives) - this is the size to use.
5. Saves a visual grid image (one panel per kernel size, largest
   component highlighted) so you can SEE what's happening, not just
   read numbers - visual confirmation matters more than any single
   metric, per your own project's established practice.
6. Optionally writes out a "cleaned" mask (largest connected component
   only, at your chosen kernel size) that you can feed into
   evaluate.py / centroid_distance.py instead of the raw mask.

IMPORTANT CAVEAT (read before trusting this blindly):
If the halo is thicker than expected, or is NOT actually a separate thin
strand but is genuinely fused with the true blob at similar thickness
across its whole length, no kernel size will cleanly separate them -
this has already been observed on the teapot case in this project.
This script cannot invent a separation that doesn't exist in the pixels;
it just makes it much easier to see whether one exists and at what
kernel size, instead of guessing once and giving up.

USAGE
    python mask_halo_diagnostic.py --mask GT_Fragment_rerun3.png \\
        --output_dir . \\
        --kernel_sizes 3,5,7,9,11,13,15,17,21

    # Once you've picked a kernel size from the table/visual grid:
    python mask_halo_diagnostic.py --mask GT_Fragment_rerun3.png \\
        --output_dir . \\
        --kernel_sizes 9 \\
        --write_cleaned --cleaned_output_name GT_Fragment_rerun3_cleaned.png

Then re-run your normal scoring commands using the cleaned mask instead
of the raw one, e.g.:
    python evaluate.py --gt_mask GT_Mask.png \\
        --gt_fragment GT_Fragment_rerun3_cleaned.png \\
        --output_dir . --output_name metrics_rerun3_cleaned.json
    python centroid_distance.py --gt_mask GT_Mask.png \\
        --gt_fragment GT_Fragment_rerun3_cleaned.png \\
        --output_dir . --output_name centroid_rerun3_cleaned.json
"""

import argparse
import os
import sys

import cv2
import numpy as np


def load_binary_mask(path):
    """Load a mask file and return a clean 0/255 uint8 binary image.
    Handles masks saved as 0/1, 0/255, or grayscale with anti-aliasing."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask file: {path}")
    # Anything above a low threshold counts as foreground - masks in this
    # project are binary by construction (0/1 or 0/255) but we threshold
    # defensively in case of compression artifacts.
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return binary


def analyze_kernel_size(binary_mask, kernel_size):
    """Apply morphological opening at the given kernel size (0 = no
    opening, i.e. the raw mask), then run connected component analysis.
    Returns a dict of stats plus the labeled image for visualization."""
    if kernel_size and kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (kernel_size, kernel_size))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    else:
        opened = binary_mask.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, connectivity=8)

    # Label 0 is background; skip it.
    total_area = int(stats[1:, cv2.CC_STAT_AREA].sum()) if num_labels > 1 else 0
    num_components = num_labels - 1

    largest_idx = None
    largest_area = 0
    if num_components > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

    largest_bbox = None
    largest_centroid = None
    if largest_idx is not None:
        x, y, w, h, _ = stats[largest_idx]
        largest_bbox = (int(x), int(y), int(w), int(h))
        largest_centroid = tuple(float(c) for c in centroids[largest_idx])

    return {
        "kernel_size": kernel_size,
        "opened_mask": opened,
        "labels": labels,
        "num_components": num_components,
        "total_area": total_area,
        "largest_area": largest_area,
        "largest_bbox": largest_bbox,
        "largest_centroid": largest_centroid,
        "largest_label_idx": largest_idx,
    }


def make_visualization_panel(binary_mask, results):
    """Build a single montage image: original mask, then one panel per
    kernel size tested, with the largest connected component highlighted
    in a distinct color so it's visually obvious what survives."""
    h, w = binary_mask.shape
    panels = []

    def to_bgr_with_highlight(mask_img, labels, largest_label_idx):
        vis = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        if largest_label_idx is not None:
            highlight = (labels == largest_label_idx)
            vis[highlight] = (0, 200, 0)  # green = largest surviving component
        return vis

    orig_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(orig_vis, "original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)
    panels.append(orig_vis)

    for r in results:
        vis = to_bgr_with_highlight(r["opened_mask"], r["labels"],
                                     r["largest_label_idx"])
        label_text = f"k={r['kernel_size']} comps={r['num_components']} area={r['total_area']}"
        cv2.putText(vis, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        panels.append(vis)

    # Arrange panels in a grid, roughly square.
    n = len(panels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r_idx = i // cols
        c_idx = i % cols
        grid[r_idx * h:(r_idx + 1) * h, c_idx * w:(c_idx + 1) * w] = p

    # Downscale if huge, just for manageable file size / viewing.
    max_dim = 2400
    if max(grid.shape[:2]) > max_dim:
        scale = max_dim / max(grid.shape[:2])
        grid = cv2.resize(grid, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and optionally clean up edge-halo artifacts "
                    "in fragment masks via a morphological-opening kernel sweep.")
    parser.add_argument("--mask", required=True,
                         help="Path to the mask file to analyze "
                              "(e.g. GT_Fragment_rerun3.png)")
    parser.add_argument("--output_dir", default=".",
                         help="Directory to write the visualization grid "
                              "(and cleaned mask, if requested)")
    parser.add_argument("--kernel_sizes", default="3,5,7,9,11,13,15,17,21",
                         help="Comma-separated odd kernel sizes to test, "
                              "e.g. '3,5,7,9,11'")
    parser.add_argument("--write_cleaned", action="store_true",
                         help="If set, writes a cleaned mask (largest "
                              "component only, at the FIRST kernel size "
                              "listed in --kernel_sizes) to --cleaned_output_name")
    parser.add_argument("--cleaned_output_name", default="cleaned_mask.png",
                         help="Filename for the cleaned mask output "
                              "(only used with --write_cleaned)")
    args = parser.parse_args()

    if not os.path.isfile(args.mask):
        print(f"ERROR: mask file not found: {args.mask}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    binary_mask = load_binary_mask(args.mask)

    try:
        kernel_sizes = [int(k.strip()) for k in args.kernel_sizes.split(",") if k.strip()]
    except ValueError:
        print("ERROR: --kernel_sizes must be a comma-separated list of integers.")
        sys.exit(1)

    # Always include 0 (no opening = raw mask) as the baseline reference row.
    baseline = analyze_kernel_size(binary_mask, 0)

    results = []
    for k in kernel_sizes:
        results.append(analyze_kernel_size(binary_mask, k))

    print(f"\nAnalyzing: {args.mask}")
    print(f"Raw mask foreground pixels: {int((binary_mask > 0).sum())}\n")

    print(f"{'kernel':>6} | {'components':>10} | {'total_area':>10} | "
          f"{'largest_area':>12} | {'largest_bbox':>20} | {'largest_centroid'}")
    print("-" * 100)
    print(f"{'raw':>6} | {baseline['num_components']:>10} | "
          f"{baseline['total_area']:>10} | {baseline['largest_area']:>12} | "
          f"{str(baseline['largest_bbox']):>20} | {baseline['largest_centroid']}")
    for r in results:
        print(f"{r['kernel_size']:>6} | {r['num_components']:>10} | "
              f"{r['total_area']:>10} | {r['largest_area']:>12} | "
              f"{str(r['largest_bbox']):>20} | {r['largest_centroid']}")

    print("\nWHAT TO LOOK FOR:")
    print("  - A kernel size where total_area drops noticeably from the raw")
    print("    baseline, but largest_area stays close to its own value at")
    print("    smaller kernel sizes -> the halo broke off and got removed,")
    print("    while the real blob survived largely intact. That's a good")
    print("    candidate kernel size to use.")
    print("  - If largest_area keeps shrinking steadily alongside total_area")
    print("    with no size where it levels off, the 'halo' may actually be")
    print("    fused with the real blob at similar thickness throughout -")
    print("    in that case no kernel size will cleanly separate them, and")
    print("    this matches the known unresolved teapot-style limitation.")
    print("  - ALWAYS check the visual grid image, not just these numbers -")
    print("    a stable largest_area number could still be centered on the")
    print("    wrong thing if the true blob and halo are similar in size.\n")

    grid = make_visualization_panel(binary_mask, results)
    grid_path = os.path.join(args.output_dir, "halo_diagnostic_grid.png")
    cv2.imwrite(grid_path, grid)
    print(f"Wrote visualization grid: {grid_path}")

    if args.write_cleaned:
        chosen = results[0]
        if chosen["largest_label_idx"] is None:
            print("WARNING: no foreground survived at the chosen kernel "
                  "size - cleaned mask not written.")
        else:
            cleaned = np.where(chosen["labels"] == chosen["largest_label_idx"],
                                255, 0).astype(np.uint8)
            cleaned_path = os.path.join(args.output_dir, args.cleaned_output_name)
            cv2.imwrite(cleaned_path, cleaned)
            print(f"Wrote cleaned mask (kernel_size={chosen['kernel_size']}, "
                  f"largest component only): {cleaned_path}")
            print("You can now feed this into evaluate.py / centroid_distance.py "
                  "in place of the raw GT_Fragment*.png mask.")


if __name__ == "__main__":
    main()
