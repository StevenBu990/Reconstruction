"""
ai_view_judge.py

Sends all candidate damaged-view images (input.png only - NEVER ground
truth, that would leak the answer) to Gemini in one call, and asks it to
pick which view gives the best chance of a successful repair.

Usage:
    python ai_view_judge.py \
        --object_id beer_bottle \
        --damage_config neck_simple \
        --views_root outputs/beer_bottle/neck_simple \
        --objects objects.json \
        --damages damage.json \
        --output judge_choice.json

Expects views_root to contain view_0/input.png, view_1/input.png, etc.
(matching whatever candidate_views.json you rendered with).
"""

import argparse
import json
import os
import re

from google import genai
from google.genai import types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--damage_config", required=True)
    parser.add_argument("--views_root", required=True,
                         help="Folder containing view_0/, view_1/, etc., each with input.png")
    parser.add_argument("--objects", required=True)
    parser.add_argument("--damages", required=True)
    parser.add_argument("--output", default="judge_choice.json")
    parser.add_argument("--num_views", type=int, default=5)
    return parser.parse_args()


def build_prompt(object_name, object_info, damage_info, num_views):
    return f"""
You are evaluating {num_views} candidate camera views of the same damaged
{object_name}, shown in order as Image 1 through Image {num_views}.

Object function: {object_info['function']}

Damage: the {damage_info['part']} is broken and must be reconstructed.
{damage_info['description']}

Your task: pick which ONE of these {num_views} views gives the best chance
of successfully reconstructing the missing {damage_info['part']} - consider
which view shows the damaged region most clearly, with the least
ambiguity about its true shape and extent.

Respond with ONLY a single digit (1 through {num_views}) indicating your
chosen image number. Do not include any other text.
"""


def main():
    args = parse_args()

    with open(args.objects) as f:
        objects = json.load(f)
    with open(args.damages) as f:
        damages = json.load(f)

    object_info = objects[args.object_id]
    damage_key = f"{args.object_id}_{args.damage_config}"
    damage_info = damages[damage_key]

    image_paths = []
    for i in range(args.num_views):
        path = os.path.join(args.views_root, f"view_{i}", "input.png")
        if not os.path.exists(path):
            raise RuntimeError(f"Missing {path} - did you render all {args.num_views} candidate views?")
        image_paths.append(path)

    prompt = build_prompt(args.object_id, object_info, damage_info, len(image_paths))
    print(prompt)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    parts = []
    for path in image_paths:
        with open(path, "rb") as f:
            parts.append(types.Part.from_bytes(mime_type="image/png", data=f.read()))
    parts.append(types.Part.from_text(text=prompt))

    contents = [types.Content(role="user", parts=parts)]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    raw_text = response.candidates[0].content.parts[0].text
    print(f"Judge response: {raw_text!r}")

    match = re.search(r"\d+", raw_text)
    if not match:
        raise RuntimeError(f"Could not parse a view number from judge response: {raw_text!r}")

    chosen_number = int(match.group())
    chosen_view_id = chosen_number - 1  # views are 1-indexed in the prompt, 0-indexed as view_N folders

    if not (0 <= chosen_view_id < len(image_paths)):
        raise RuntimeError(f"Judge chose out-of-range view {chosen_number}")

    result = {
        "object_id": args.object_id,
        "damage_config": args.damage_config,
        "chosen_view_id": chosen_view_id,
        "chosen_view_path": image_paths[chosen_view_id],
        "raw_response": raw_text,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nJudge chose view_{chosen_view_id}: {image_paths[chosen_view_id]}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
