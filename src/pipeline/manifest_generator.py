"""
manifest_generator.py

Combines:
  - configs/object_registry.json (object_id -> fragments_dir)
  - configs/fragment_maps/{object_id}.json (damage_configs + base_rotation)
  - configs/view_presets.json (shared 4 views)

into one flat experiment_manifest.json: one row per (object x damage_config x
view) experiment, with everything RunExperiment.py needs to call
blender_render.py and FullMethod.py without re-deriving any paths.

Usage:
    python manifest_generator.py \
        --object_registry configs/object_registry.json \
        --fragment_maps_dir configs/fragment_maps \
        --view_presets configs/view_presets.json \
        --output_root outputs \
        --output configs/experiment_manifest.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_registry", required=True)
    parser.add_argument("--fragment_maps_dir", required=True)
    parser.add_argument("--view_presets", required=True)
    parser.add_argument("--output_root", required=True,
                         help="Root output directory (matches blender_render.py's --output_root)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.object_registry) as f:
        registry = json.load(f)["objects"]

    with open(args.view_presets) as f:
        views = json.load(f)["views"]

    manifest = []
    skipped_objects = []

    for object_id, fragments_dir in registry.items():
        fragment_map_path = os.path.join(args.fragment_maps_dir, f"{object_id}.json")
        if not os.path.exists(fragment_map_path):
            skipped_objects.append((object_id, "no fragment map found"))
            continue

        with open(fragment_map_path) as f:
            fragment_map = json.load(f)

        damage_configs = fragment_map.get("damage_configs", {})
        if not damage_configs:
            skipped_objects.append((object_id, "fragment map has no damage_configs"))
            continue

        for damage_config_name, hidden_fragments in damage_configs.items():
            for view in views:
                out_dir = os.path.join(args.output_root, object_id, damage_config_name, f"view_{view['id']}")
                manifest.append({
                    "object_id": object_id,
                    "fragments_dir": fragments_dir,
                    "fragment_map_path": fragment_map_path,
                    "damage_config": damage_config_name,
                    "hidden_fragments": hidden_fragments,
                    "view_id": view["id"],
                    "view_name": view["name"],
                    "output_dir": out_dir,
                    "input_image": os.path.join(out_dir, "input.png"),
                    "ground_truth_image": os.path.join(out_dir, "ground_truth.png"),
                    "metadata_path": os.path.join(out_dir, "metadata.json"),
                    "generated_image": os.path.join(out_dir, "generated.png"),
                })

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} experiments to {args.output}")
    print(f"({len(registry) - len(skipped_objects)} objects included)")
    if skipped_objects:
        print("\nSkipped objects:")
        for object_id, reason in skipped_objects:
            print(f"  {object_id}: {reason}")


if __name__ == "__main__":
    main()
