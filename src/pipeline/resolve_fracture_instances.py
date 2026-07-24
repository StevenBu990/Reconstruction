"""
resolve_fracture_instances.py

Stage 1 of manifest generation: for each object in your experiment set, scan
its fractured_N subfolders (each a different pre-computed fracture of the
same object), count piece_*.obj files in each, and auto-select the instance
whose piece count is closest to a target count.

This does NOT touch fragment maps - it only decides which fractured_N folder
to use per object. After running this, review resolved_objects.json, then
write/update the fragment map JSON for each object (configs/fragment_maps/
{object_id}.json) referencing piece_N names valid for the chosen instance.

Usage:
    python resolve_fracture_instances.py \
        --dataset_root /path/to/Breaking-Bad-Dataset.github.io/everyday \
        --objects_config configs/objects_to_process.json \
        --target_piece_count 8 \
        --output configs/resolved_objects.json

objects_to_process.json schema:
{
    "objects": [
        {"object_id": "beerbottle_01", "category": "BeerBottle", "object_hash": "5566f264a6fa08cd2a68e506fbd6eecf"},
        ...
    ]
}
"""

import os
import json
import glob
import argparse


def count_pieces(fractured_dir):
    return len(glob.glob(os.path.join(fractured_dir, "piece_*.obj")))


def find_fractured_instances(object_dir):
    """Return {fractured_folder_name: piece_count} for all fractured_N subfolders."""
    instances = {}
    for entry in sorted(os.listdir(object_dir)):
        full_path = os.path.join(object_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("fractured_"):
            piece_count = count_pieces(full_path)
            if piece_count > 0:
                instances[entry] = piece_count
    return instances


def pick_closest(instances, target_piece_count):
    """Pick the fractured_N whose piece count is closest to target_piece_count."""
    return min(instances.items(), key=lambda kv: abs(kv[1] - target_piece_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to the 'everyday' category root")
    parser.add_argument("--objects_config", required=True, help="Path to objects_to_process.json")
    parser.add_argument("--target_piece_count", type=int, default=8)
    parser.add_argument("--output", required=True, help="Path to write resolved_objects.json")
    args = parser.parse_args()

    with open(args.objects_config) as f:
        objects_config = json.load(f)

    resolved = {}
    for entry in objects_config["objects"]:
        object_id = entry["object_id"]
        category = entry["category"]
        object_hash = entry["object_hash"]

        object_dir = os.path.join(args.dataset_root, category, object_hash)
        if not os.path.isdir(object_dir):
            print(f"[WARN] Skipping {object_id}: directory not found at {object_dir}")
            continue

        instances = find_fractured_instances(object_dir)
        if not instances:
            print(f"[WARN] Skipping {object_id}: no fractured_N folders with piece files found")
            continue

        chosen_name, chosen_count = pick_closest(instances, args.target_piece_count)
        fragments_dir = os.path.join(object_dir, chosen_name)

        resolved[object_id] = {
            "category": category,
            "object_hash": object_hash,
            "fracture_instance": chosen_name,
            "fragments_dir": fragments_dir,
            "piece_count": chosen_count,
            "available_instances": instances,
        }

        print(f"[OK] {object_id}: chose {chosen_name} ({chosen_count} pieces) "
              f"out of {len(instances)} candidates {instances}")

    with open(args.output, "w") as f:
        json.dump(resolved, f, indent=2)

    print(f"\nWrote {len(resolved)} resolved objects to {args.output}")
    print("Next: review the choices above, then write/update fragment map JSON "
          "for each object_id referencing piece_N names valid for its chosen instance.")


if __name__ == "__main__":
    main()
