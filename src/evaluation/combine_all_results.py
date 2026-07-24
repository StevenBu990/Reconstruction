"""
combine_all_results.py

Merges:
  - Your manual Google Sheet export (5 objects: Ramekin/Cup, Teapot, Mug,
    Plate, Bowl) - parses the "IoU: 0.xxx" / "Dice: 0.xxx" text format,
    uses the no-SIFT fallback IoU wherever present (since that's what you
    actually used when SIFT failed), normalizes "Small"/"Simple" ->
    "simple" for consistent damage-level naming.
  - The 10 automated objects' aggregated_metrics.csv

into one combined_results.csv covering all 15 objects, plus prints summary
stats (mean by object, by damage level, overall) for your meeting.

Note: manual data's View 1-4 numbering does NOT necessarily correspond to
the automated pipeline's front/right/back/left views - these were produced
independently. They're kept as separate view labels ("manual_view_1" etc.)
rather than assumed equivalent, to avoid silently mis-mapping them.

Usage:
    python combine_all_results.py \
        --manual_csv "Experiemnt_Tracker_-_Sheet1.csv" \
        --automated_csv results/aggregated_metrics.csv \
        --output results/combined_all_15_objects.csv
"""

import argparse
import csv
import re


MANUAL_OBJECT_ID_MAP = {
    "Ramekin/Cup": "ramekin_cup",
    "Teapot": "teapot",
    "Mug": "mug",
    "Plate": "plate",
    "Bowl": "bowl",
}


def parse_prefixed_float(text, prefix):
    """Extracts a float from strings like 'IoU: 0.607...' or 'Dice: 0.957...'"""
    if not text or not text.strip():
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else None


def parse_manual_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_raw = row.get("Object", "").strip()
            if not object_raw or object_raw not in MANUAL_OBJECT_ID_MAP:
                continue  # skip empty placeholder rows

            object_id = MANUAL_OBJECT_ID_MAP[object_raw]
            view = row.get("View (1-4)", "").strip()
            damage_part = row.get("Damage Part", "").strip().lower()
            damage_level = row.get("Damage Level", "").strip().lower()
            if damage_level == "small":
                damage_level = "simple"

            iou_sift = parse_prefixed_float(row.get("IoU SIFT Computed", ""), "IoU")
            dice_sift = parse_prefixed_float(row.get("Dice Sift", ""), "Dice")
            iou_no_sift = parse_prefixed_float(row.get("IoU No SIFT Computed (if necessary)", ""), "IoU")
            notes = row.get("Notes", "").strip()

            # Use the no-SIFT fallback value as final IoU wherever it exists
            # (that's what you actually used when SIFT failed). Dice wasn't
            # separately recomputed for fallback cases in the sheet, so it
            # stays as the SIFT-based value (may not correspond to the same
            # alignment as the final IoU in fallback cases - flagged in notes).
            final_iou = iou_no_sift if iou_no_sift is not None else iou_sift
            used_fallback = iou_no_sift is not None

            rows.append({
                "object_id": object_id,
                "damage_part": damage_part,
                "damage_level": damage_level,
                "view": f"manual_view_{view}",
                "iou": final_iou,
                "dice": dice_sift,
                "flag": "used_no_sift_fallback" if used_fallback else "",
                "notes": notes,
                "source": "manual",
            })
    return rows


def parse_automated_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            damage_config = row["damage_config"]
            # damage_config is always {part}_{simple|severe}
            if damage_config.endswith("_simple"):
                part, level = damage_config[:-len("_simple")], "simple"
            elif damage_config.endswith("_severe"):
                part, level = damage_config[:-len("_severe")], "severe"
            else:
                part, level = damage_config, ""

            iou = float(row["iou"]) if row["iou"] not in ("", "None") else None
            dice = float(row["dice"]) if row["dice"] not in ("", "None") else None

            rows.append({
                "object_id": row["object_id"],
                "damage_part": part,
                "damage_level": level,
                "view": row["view_name"],
                "iou": iou,
                "dice": dice,
                "flag": row["flag"],
                "notes": "",
                "source": "automated",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_csv", required=True)
    parser.add_argument("--automated_csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manual_rows = parse_manual_csv(args.manual_csv)
    automated_rows = parse_automated_csv(args.automated_csv)
    all_rows = manual_rows + automated_rows

    fieldnames = ["object_id", "damage_part", "damage_level", "view",
                  "iou", "dice", "flag", "notes", "source"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output}")
    print(f"  Manual (5 objects): {len(manual_rows)}")
    print(f"  Automated (10 objects): {len(automated_rows)}")

    evaluated = [r for r in all_rows if r["iou"] is not None]
    print(f"  Total with a real IoU value: {len(evaluated)}")

    # Overall mean
    mean_iou = sum(r["iou"] for r in evaluated) / len(evaluated)
    dice_vals = [r["dice"] for r in evaluated if r["dice"] is not None]
    mean_dice = sum(dice_vals) / len(dice_vals) if dice_vals else None
    print(f"\nOverall mean IoU: {mean_iou:.3f}")
    if mean_dice is not None:
        print(f"Overall mean Dice ({len(dice_vals)} with a Dice value): {mean_dice:.3f}")

    # Mean by object
    print(f"\n{'Object':<20}{'N':<6}{'Mean IoU':<12}{'Source'}")
    print("-" * 50)
    objects = sorted(set(r["object_id"] for r in all_rows))
    for obj in objects:
        obj_rows = [r for r in evaluated if r["object_id"] == obj]
        if not obj_rows:
            continue
        obj_mean = sum(r["iou"] for r in obj_rows) / len(obj_rows)
        source = obj_rows[0]["source"]
        print(f"{obj:<20}{len(obj_rows):<6}{obj_mean:<12.3f}{source}")

    # Mean by damage level (simple vs severe), across all 15 objects
    print(f"\n{'Damage Level':<15}{'N':<6}{'Mean IoU'}")
    print("-" * 35)
    for level in ["simple", "severe"]:
        level_rows = [r for r in evaluated if r["damage_level"] == level]
        if level_rows:
            level_mean = sum(r["iou"] for r in level_rows) / len(level_rows)
            print(f"{level:<15}{len(level_rows):<6}{level_mean:.3f}")


if __name__ == "__main__":
    main()
