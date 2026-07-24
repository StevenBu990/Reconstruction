"""
compute_base_rotation.py

Automatically computes base_rotation_euler_deg for an object, instead of
guessing by eye in the Blender viewport (which is unreliable for
axially-symmetric objects like bottles/glasses - they can look "upright
enough" from one viewport angle while still being tilted in a way a fixed
camera later exposes).

Method: reads all fragment vertices, combines them into one point cloud,
computes the object's principal axis via PCA (the direction of greatest
extent - for a bottle/glass/vase this is reliably the vertical body axis),
and finds the rotation that aligns that axis with world Z (vertical).

This resolves tilt (forward/backward/sideways lean) automatically and
reliably. It does NOT resolve which way the object "faces" around its own
vertical axis (there's no geometric signal for that) - after applying the
suggested rotation, you may still want to nudge the Z value alone in
Blender if you want a specific side facing the camera at view_0. Rotating
around Z alone while already upright is safe and can't make it fall over.

Usage:
    python compute_base_rotation.py --fragments_dir path/to/fractured_55

Requires: numpy, scipy (pip install numpy scipy --break-system-packages)
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation


def load_all_vertices(fragments_dir, include_pieces=None):
    piece_files = glob.glob(os.path.join(fragments_dir, "piece_*.obj"))
    if not piece_files:
        raise RuntimeError(f"No piece_*.obj files found in {fragments_dir}")

    if include_pieces:
        include_set = set(include_pieces)
        piece_files = [
            p for p in piece_files
            if os.path.splitext(os.path.basename(p))[0] in include_set
        ]
        if not piece_files:
            raise RuntimeError(f"None of {include_pieces} matched files in {fragments_dir}")

    all_verts = []
    for filepath in piece_files:
        with open(filepath) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    all_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(all_verts)


def compute_principal_axis(points):
    """Returns the unit eigenvector of greatest variance (the object's
    longest/dominant axis) via PCA on the centered point cloud."""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order - largest is last
    principal_axis = eigenvectors[:, -1]
    return principal_axis / np.linalg.norm(principal_axis)


def rotation_aligning_a_to_b(a, b):
    """Rotation matrix that rotates unit vector a onto unit vector b,
    via the standard axis-angle (Rodrigues) construction from their
    cross and dot products."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        # a and b already parallel (or anti-parallel)
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation needed - pick any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(axis * np.pi).as_matrix()

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments_dir", required=True)
    parser.add_argument("--flip", action="store_true",
                         help="Align the principal axis to -Z instead of +Z. Use this if the "
                              "un-flipped rotation renders upside-down - PCA can't tell which "
                              "end of the axis is 'up', so this gives the other valid solution "
                              "directly instead of guessing at manual Euler-angle corrections.")
    parser.add_argument("--include_pieces", nargs="+", default=None,
                         help="Restrict PCA to only these piece names (e.g. --include_pieces "
                              "piece_0 piece_3 piece_5 piece_12). Use this for asymmetric shapes "
                              "(handles, spouts) where the whole object's longest axis doesn't "
                              "match the true vertical body axis - restrict to just the body "
                              "pieces instead.")
    args = parser.parse_args()

    points = load_all_vertices(args.fragments_dir, include_pieces=args.include_pieces)
    print(f"Loaded {len(points)} vertices from {args.fragments_dir}")

    principal_axis = compute_principal_axis(points)
    print(f"Principal (longest) axis: {principal_axis}")

    target = np.array([0, 0, -1]) if args.flip else np.array([0, 0, 1])
    R = rotation_aligning_a_to_b(principal_axis, target)

    euler_deg = Rotation.from_matrix(R).as_euler("xyz", degrees=True)

    print(f"\nSuggested base_rotation_euler_deg: [{euler_deg[0]:.3f}, {euler_deg[1]:.3f}, {euler_deg[2]:.3f}]")
    if args.flip:
        print("(This is the FLIPPED alternative - aligned principal axis to -Z instead of +Z)")
    print("\nThis should bring the object's longest axis to vertical (world Z).")
    print("It does NOT determine which side faces the camera at view_0 - if you")
    print("want a specific side facing front, you can still nudge the Z value")
    print("alone afterward in Blender (safe - rotating around Z while already")
    print("upright can't make it fall over).")


if __name__ == "__main__":
    main()
