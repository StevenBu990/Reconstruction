"""
check_fragment_adjacency.py

Builds the full pairwise adjacency graph between all fragments in a
fractured object directory, then either:

  1. Validates a candidate "remove set" (--remove_set) - checks both that
     the pieces you plan to hide form one contiguous cluster, AND that the
     remaining pieces stay connected after removal.

  2. Auto-searches for valid remove-sets (--seed + --sizes) - instead of
     manually guessing a candidate and re-running validation each time,
     this enumerates every subset of the given size(s) containing your seed
     piece and reports only the ones that pass both checks. Use this to
     avoid the manual guess-then-validate loop entirely.

Usage
-----
# Just print the full adjacency graph:
python check_fragment_adjacency.py --fragments_dir /path/to/fractured_29

# Check one candidate damage config:
python check_fragment_adjacency.py --fragments_dir /path/to/fractured_29 \
    --remove_set piece_0 piece_6 piece_7

# Auto-search: find all valid 1-2 piece ("simple") and 2-3 piece ("severe")
# remove-sets containing piece_1, without manually testing candidates:
python check_fragment_adjacency.py --fragments_dir /path/to/fractured_29 \
    --seed piece_1 --sizes 1 2 3

Requires: numpy, scipy (pip install numpy scipy --break-system-packages)
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

TOUCH_THRESHOLD = 1e-4  # distance below which two fragments are considered touching


def load_obj_vertices(path):
    """Read only vertex positions (v x y z lines) from an .obj file."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64)


def min_distance_between(verts_a, verts_b):
    """Minimum point-to-point distance between two vertex clouds, via KD-tree."""
    if len(verts_a) == 0 or len(verts_b) == 0:
        return np.inf
    tree = cKDTree(verts_b)
    dists, _ = tree.query(verts_a, k=1)
    return float(dists.min())


def discover_fragments(fragments_dir):
    """Find all piece_*.obj files in the directory, sorted by piece index."""
    paths = glob.glob(os.path.join(fragments_dir, "piece_*.obj"))
    if not paths:
        paths = glob.glob(os.path.join(fragments_dir, "*.obj"))

    def piece_key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name

    paths.sort(key=piece_key)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return names, paths


def build_adjacency_graph(fragments_dir):
    names, paths = discover_fragments(fragments_dir)
    if len(names) < 2:
        print(f"Found only {len(names)} fragment(s) in {fragments_dir} - need at least 2.")
        sys.exit(1)

    print(f"Loading {len(names)} fragments...")
    verts = {}
    for name, path in zip(names, paths):
        verts[name] = load_obj_vertices(path)

    n = len(names)
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)

    print("Computing pairwise distances...")
    for i in range(n):
        for j in range(i + 1, n):
            d = min_distance_between(verts[names[i]], verts[names[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return names, dist_matrix


def print_full_graph(names, dist_matrix):
    n = len(names)
    print("\nFull pairwise adjacency (TOUCHING = shares a cut seam):\n")
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                continue
            tag = " TOUCHING" if dist_matrix[i, j] < TOUCH_THRESHOLD else ""
            row.append((dist_matrix[i, j], f"{names[i]} <-> {names[j]}: {dist_matrix[i, j]:.5f}{tag}"))
        row.sort(key=lambda x: x[0])
        for _, line in row:
            print(f"  {line}")
        print()


def connected_components(names, dist_matrix, subset=None):
    """Connected components among `subset` (or all names), using
    TOUCH_THRESHOLD adjacency, considering only edges between subset members."""
    if subset is None:
        subset = names
    idx = {name: i for i, name in enumerate(names)}
    subset = list(subset)
    unvisited = set(subset)
    components = []

    while unvisited:
        start = unvisited.pop()
        stack = [start]
        comp = {start}
        while stack:
            cur = stack.pop()
            i = idx[cur]
            for other in subset:
                if other in comp:
                    continue
                j = idx[other]
                if dist_matrix[i, j] < TOUCH_THRESHOLD:
                    comp.add(other)
                    stack.append(other)
        unvisited -= comp
        components.append(comp)

    return components


def validate_remove_set(names, dist_matrix, remove_set):
    remove_set = list(remove_set)
    missing = [p for p in remove_set if p not in names]
    if missing:
        print(f"ERROR: these pieces weren't found in the fragments dir: {missing}")
        print(f"Available pieces: {names}")
        sys.exit(1)

    remaining = [n for n in names if n not in remove_set]

    print(f"\n=== Validating remove set: {remove_set} ===\n")

    remove_components = connected_components(names, dist_matrix, subset=remove_set)
    if len(remove_components) == 1:
        print(f"[OK] Remove set is internally connected (one contiguous cluster): {remove_set}")
    else:
        print(f"[PROBLEM] Remove set splits into {len(remove_components)} disconnected clusters:")
        for k, comp in enumerate(remove_components):
            print(f"    cluster {k}: {sorted(comp)}")
        print("  -> This would hide two (or more) unrelated damage regions, not one localized break.")

    if not remaining:
        print("\n[INFO] Remove set is everything - nothing left to check.")
        return

    remaining_components = connected_components(names, dist_matrix, subset=remaining)
    if len(remaining_components) == 1:
        print(f"\n[OK] Remaining pieces stay in one connected component: {sorted(remaining)}")
    else:
        print(f"\n[PROBLEM] Removing this set splits the remaining pieces into {len(remaining_components)} disconnected components:")
        for k, comp in enumerate(remaining_components):
            print(f"    component {k}: {sorted(comp)}")
        print("  -> This remove set would orphan a floating fragment in the 'undamaged' mesh - pick a different set.")


def is_valid_remove_set(names, dist_matrix, candidate):
    """True if candidate is internally connected AND leaves the remainder connected."""
    candidate_components = connected_components(names, dist_matrix, subset=list(candidate))
    if len(candidate_components) != 1:
        return False

    remaining = [n for n in names if n not in candidate]
    if not remaining:
        return True

    remaining_components = connected_components(names, dist_matrix, subset=remaining)
    return len(remaining_components) == 1


def enumerate_valid_subsets(names, dist_matrix, seed, size):
    """All subsets of exactly `size` fragments, containing `seed`, that pass
    is_valid_remove_set. Brute-force via itertools.combinations - fine for
    the small fragment counts (~2-15 pieces) in this dataset."""
    others = [n for n in names if n != seed]
    valid = []
    for combo in itertools.combinations(others, size - 1):
        candidate = (seed,) + combo
        if is_valid_remove_set(names, dist_matrix, candidate):
            valid.append(sorted(candidate))
    return valid


def suggest_sets(names, dist_matrix, seed, sizes):
    print(f"\n=== Auto-searching for valid remove-sets containing {seed} ===\n")
    for size in sizes:
        valid = enumerate_valid_subsets(names, dist_matrix, seed, size)
        print(f"Size {size}:")
        if not valid:
            print(f"  (none found - no connected group of {size} pieces containing {seed} "
                  f"keeps the remainder connected)")
        else:
            for v in valid:
                print(f"  {v}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Check fragment adjacency and validate/search damage remove-sets.")
    parser.add_argument("--fragments_dir", required=True, help="Directory containing piece_*.obj files")
    parser.add_argument("--remove_set", nargs="+", default=None,
                         help="Candidate set of piece names to remove as one damage config, e.g. piece_0 piece_6 piece_7")
    parser.add_argument("--anchor", default=None,
                         help="(legacy) print distances from this single piece to all others, closest first")
    parser.add_argument("--seed", default=None,
                         help="Auto-search mode: find all valid remove-sets containing this piece")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3],
                         help="Sizes to search with --seed, e.g. --sizes 1 2 for simple, --sizes 2 3 for severe")
    args = parser.parse_args()

    names, dist_matrix = build_adjacency_graph(args.fragments_dir)

    if args.seed:
        if args.seed not in names:
            print(f"ERROR: seed '{args.seed}' not found. Available: {names}")
            sys.exit(1)
        suggest_sets(names, dist_matrix, args.seed, args.sizes)
        return

    if args.anchor:
        if args.anchor not in names:
            print(f"ERROR: anchor '{args.anchor}' not found. Available: {names}")
            sys.exit(1)
        i = names.index(args.anchor)
        print(f"\nDistances from {args.anchor} (closest first):")
        order = sorted(range(len(names)), key=lambda j: dist_matrix[i, j])
        for j in order:
            if j == i:
                continue
            tag = "  TOUCHING" if dist_matrix[i, j] < TOUCH_THRESHOLD else ""
            print(f"  {args.anchor} <-> {names[j]}: {dist_matrix[i, j]:.5f}{tag}")

    if args.remove_set:
        validate_remove_set(names, dist_matrix, args.remove_set)
    elif not args.anchor:
        print_full_graph(names, dist_matrix)


if __name__ == "__main__":
    main()
