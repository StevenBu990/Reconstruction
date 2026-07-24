#!/usr/bin/env python3
"""
ai_view_judge_verbal.py

A variant of ai_view_judge.py where a HUMAN describes, in their own words,
what makes a good view for reconstruction (e.g. "pick the view where the
fracture is shown most head-on and isn't partly hidden behind another
piece"), and the judge picks whichever of the 5 candidate views best
matches that description - rather than the judge deciding on its own what
"clearest" means with no guidance.

This tests a different mechanism than the existing judge: instead of pure
model judgment, it leverages the human's own understanding of what makes a
view useful, and checks whether the model can correctly identify which
candidate matches that description.

Sends only input.png from each candidate view - NEVER ground_truth.png -
consistent with the project's existing AI View Judge convention.

USAGE
    python ai_view_judge_verbal.py \\
        --object_id vase \\
        --damage_config rim_severe \\
        --views_root outputs_judge_test/vase/rim_severe \\
        --objects objects.json \\
        --damages damage.json \\
        --view_description "Pick the view where the rim break is shown most \\
            head-on (not at a steep side angle), and where you can see \\
            clearly down into the vase opening at the break site." \\
        --output judge_choice_vase_rim_severe_verbal.json

If objects.json / damages.json don't have the expected keys, the script
will print what it found and ask you to check the file structure rather
than silently guessing wrong.
"""

import argparse
import json
import os
import sys
import glob


def load_object_description(objects_path, object_id):
    """Try a few common schema shapes for objects.json. Returns a plain
    string description, or None with a printed diagnostic if not found."""
    with open(objects_path) as f:
        data = json.load(f)

    if object_id not in data:
        print(f"WARNING: '{object_id}' not found as a top-level key in {objects_path}.")
        print(f"Top-level keys present: {list(data.keys())[:10]}")
        return None

    entry = data[object_id]
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("function", "description", "summary", "object_function"):
            if key in entry:
                return entry[key]
        print(f"WARNING: could not find a description field in objects.json['{object_id}']. "
              f"Keys present: {list(entry.keys())}")
        return None
    return None


def load_damage_description(damages_path, object_id, damage_config):
    """Try a few common schema shapes for damage.json. Returns a plain
    string description, or None with a printed diagnostic if not found."""
    with open(damages_path) as f:
        data = json.load(f)

    candidates = [
        f"{object_id}_{damage_config}",
        damage_config,
    ]
    entry = None
    matched_key = None
    for key in candidates:
        if key in data:
            entry = data[key]
            matched_key = key
            break

    if entry is None:
        print(f"WARNING: could not find a damage entry for object='{object_id}', "
              f"damage_config='{damage_config}' in {damages_path}.")
        print(f"Tried keys: {candidates}")
        print(f"Top-level keys present (sample): {list(data.keys())[:10]}")
        return None

    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("description", "summary", "detail", "damage_description"):
            if key in entry:
                return entry[key]
        print(f"WARNING: could not find a description field in damage.json['{matched_key}']. "
              f"Keys present: {list(entry.keys())}")
        return None
    return None


def find_candidate_images(views_root):
    """Find view_N/input.png files under views_root, sorted by N."""
    pattern = os.path.join(views_root, "view_*", "input.png")
    paths = glob.glob(pattern)

    def view_num(p):
        folder = os.path.basename(os.path.dirname(p))
        try:
            return int(folder.replace("view_", ""))
        except ValueError:
            return 999

    paths.sort(key=view_num)
    return paths


def build_prompt(object_description, damage_description, view_description, n_images,
                  show_reasoning=False):
    parts = []
    parts.append(
        f"You are evaluating {n_images} candidate camera views of the same "
        f"damaged object, shown in order as Image 1 through Image {n_images}."
    )
    if object_description:
        parts.append(f"\nObject function: {object_description}")
    if damage_description:
        parts.append(f"\nDamage: {damage_description}")

    parts.append(
        "\nA human reviewer has described, in their own words, what they "
        "believe makes a good view for reconstructing this damage. Your "
        "task is to pick whichever ONE of the candidate images best matches "
        "that description - not to apply your own independent judgment of "
        "'clearest' if it conflicts with what the human described."
    )
    parts.append(f"\nHuman's description of the view they want:\n\"{view_description}\"")

    if show_reasoning:
        parts.append(
            "\nFirst, briefly (2-3 sentences) describe what you observe in EACH "
            "candidate image relevant to the human's description - specifically "
            "how much of the damaged/missing region is visible and unobstructed "
            "in each one. Then, on a final separate line, respond with EXACTLY "
            f"'ANSWER: N' where N is a single digit (1 through {n_images}) "
            "indicating which image best matches the human's description."
        )
    else:
        parts.append(
            f"\nRespond with ONLY a single digit (1 through {n_images}) indicating "
            "which image best matches the human's description. Do not include "
            "any other text."
        )
    return "\n".join(parts)


def call_judge_model(prompt, image_paths):
    """Calls the Gemini model with the prompt and the candidate images.
    Uses the google-genai SDK, consistent with this project's other
    Gemini-calling scripts (FullMethod.py, ai_view_judge.py)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai package not found. Activate the 'Reconstruction' "
              "conda environment before running this script.")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set. Run 'echo $GEMINI_API_KEY' to check.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    contents = [prompt]
    for path in image_paths:
        with open(path, "rb") as f:
            image_bytes = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Human-guided verbal-description variant of the AI view judge.")
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--damage_config", required=True)
    parser.add_argument("--views_root", required=True,
                         help="Folder containing view_0, view_1, ... subfolders, "
                              "each with an input.png")
    parser.add_argument("--objects", required=True, help="Path to objects.json")
    parser.add_argument("--damages", required=True, help="Path to damage.json")
    parser.add_argument("--view_description", required=True,
                         help="The human's free-text description of what makes "
                              "a good view for this reconstruction")
    parser.add_argument("--output", required=True,
                         help="Path to write the judge_choice_*.json result")
    parser.add_argument("--show_reasoning", action="store_true",
                         help="Ask the model to briefly explain its choice before giving "
                              "the final digit, so you can diagnose WHY it picked what it "
                              "picked. Useful when a pick looks wrong and you want to know "
                              "if it's a fixable prompting issue or a real limitation.")
    args = parser.parse_args()

    image_paths = find_candidate_images(args.views_root)
    if not image_paths:
        print(f"ERROR: no view_*/input.png files found under {args.views_root}")
        sys.exit(1)
    print(f"Found {len(image_paths)} candidate views: {image_paths}")

    object_description = load_object_description(args.objects, args.object_id)
    damage_description = load_damage_description(args.damages, args.object_id, args.damage_config)

    prompt = build_prompt(object_description, damage_description,
                           args.view_description, len(image_paths),
                           show_reasoning=args.show_reasoning)

    print("\n" + "=" * 70)
    print(prompt)
    print("=" * 70 + "\n")

    raw_response = call_judge_model(prompt, image_paths)
    print(f"Judge response:\n'{raw_response}'\n")

    if args.show_reasoning:
        # Look for the "ANSWER: N" line specifically, rather than grabbing
        # the first digit anywhere in the reasoning text (which could
        # accidentally match a digit mentioned in the reasoning itself,
        # e.g. "Image 1 shows..." before the model's real final answer).
        import re
        match = re.search(r"ANSWER:\s*(\d)", raw_response)
        if not match:
            print(f"ERROR: could not find an 'ANSWER: N' line in the response.")
            sys.exit(1)
        chosen_num = int(match.group(1))
        chosen_view_id = chosen_num - 1
    else:
        try:
            chosen_num = int("".join(c for c in raw_response if c.isdigit())[:1])
            chosen_view_id = chosen_num - 1
        except (ValueError, IndexError):
            print(f"ERROR: could not parse a valid digit from response '{raw_response}'")
            sys.exit(1)

    if chosen_view_id < 0 or chosen_view_id >= len(image_paths):
        print(f"ERROR: parsed view index {chosen_view_id} is out of range "
              f"(0-{len(image_paths)-1})")
        sys.exit(1)

    chosen_path = image_paths[chosen_view_id]
    print(f"\nJudge chose view_{chosen_view_id}: {chosen_path}")

    result = {
        "object_id": args.object_id,
        "damage_config": args.damage_config,
        "human_view_description": args.view_description,
        "chosen_view_id": chosen_view_id,
        "chosen_view_path": chosen_path,
        "raw_response": raw_response,
        "show_reasoning_used": args.show_reasoning,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
