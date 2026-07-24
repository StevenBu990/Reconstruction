"""
manual_align_matplotlib.py

Fallback for when image_register.py's cv2-window-based manual click mode is
unresponsive (a known-flaky combination on some macOS + conda opencv-python
setups). Uses matplotlib for click collection instead, which is generally
more reliable for this on macOS - but calls the EXACT SAME
calculate_transform() and warp() functions from your utils.py underneath,
so the actual math/output is identical to what image_register.py would
produce, just with a different (working) way to collect your clicks.

Usage:
    python manual_align_matplotlib.py \
        --target ground_truth.png \
        --source generated.png \
        --coregistration_dir "/Users/stevenbu/Desktop/ImageCoregistration" \
        --scale 1.0

Click 4+ corresponding points on the TARGET image window, close it, then
click the SAME points in the SAME order on the SOURCE image window, close
it. Writes result/target_new.jpg and result/source_new.png, same as
image_register.py would.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("MacOSX")  # more reliable interactive backend on macOS than default
import matplotlib.pyplot as plt


def collect_clicks(image_path, title):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(f"{title}\nClick corresponding points in order, then close this window")
    points = plt.ginput(n=-1, timeout=0)  # unlimited clicks until window closed
    plt.close(fig)
    return np.array(points)


def warp_no_gui(target, source, T):
    """Identical math to utils.py's warp() - only difference is skipping the
    cv2.imshow/waitKey calls at the end, which would hit the same broken-GUI
    issue we're routing around in the first place. Still writes the same
    result/target_new.jpg and result/source_new.png files."""
    height = target.shape[0]
    width = source.shape[1]

    corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    corners_moved = np.float32([[5, 5], [5, height + 5], [5 + width, 5], [5 + width, 5 + height]])
    T_perspective = cv2.getPerspectiveTransform(corners, corners_moved)
    target_new = cv2.warpPerspective(target, T_perspective, (width + 10, height + 10))
    cv2.imwrite("result/target_new.jpg", target_new)

    T_combined = np.dot(T_perspective, T)
    bg = tuple(int(x) for x in source[0, 0])
    source_new = cv2.warpPerspective(
        source, T_combined, (width + 10, height + 10),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=bg
    )
    cv2.imwrite("result/source_new.png", source_new)

    return source_new, target_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--coregistration_dir", required=True,
                         help="Path to the folder containing utils.py")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.coregistration_dir))
    from utils import calculate_transform, cross_corr, mutual_inf

    os.makedirs("result", exist_ok=True)

    print("\n=== Click points on the TARGET image (ground truth) ===")
    print("Click 4 or more distinctive points, then close the window when done.\n")
    target_points = collect_clicks(args.target, "TARGET (ground_truth)")
    print(f"Collected {len(target_points)} points on target")

    print("\n=== Now click the SAME points, SAME order, on the SOURCE image ===")
    print("(e.g. if point 1 was the spoon's bowl tip on target, click the bowl tip here too)\n")
    source_points = collect_clicks(args.source, "SOURCE (generated) - click matching points in the SAME order")
    print(f"Collected {len(source_points)} points on source")

    if len(target_points) != len(source_points):
        print(f"ERROR: point counts don't match ({len(target_points)} vs {len(source_points)}). "
              f"Click the same number of points on both images.")
        sys.exit(1)
    if len(target_points) < 3:
        print("ERROR: need at least 3 points for an affine transform.")
        sys.exit(1)

    target_img = cv2.imread(args.target)
    source_img = cv2.imread(args.source)

    if args.scale != 1.0:
        target_img = cv2.resize(target_img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        source_img = cv2.resize(source_img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        target_points = target_points * args.scale
        source_points = source_points * args.scale

    T = calculate_transform(source_points, target_points)
    warped, target_w = warp_no_gui(target_img, source_img, T)

    print(f"\nWrote result/target_new.jpg and result/source_new.png")


if __name__ == "__main__":
    main()
