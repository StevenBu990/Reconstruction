import os
import json
import argparse

from google import genai
from google.genai import types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.json",
                         help="Path to this experiment's metadata.json")
    parser.add_argument("--objects", default="objects.json",
                         help="Path to objects.json")
    parser.add_argument("--damages", default="damage.json",
                         help="Path to damage.json")
    parser.add_argument("--output", default=None,
                         help="Where to save the reconstructed image. "
                              "Defaults to 'generated.png' next to the metadata file.")
    parser.add_argument("--marked_image", default=None,
                         help="Optional: path to a marked-up input image (e.g. from "
                              "draw_marker.py) to use INSTEAD of metadata['image_path']. "
                              "For the red-circle/spatial-guidance experiment.")
    parser.add_argument("--use_marker_note", action="store_true",
                         help="If set (only meaningful alongside --marked_image), adds a "
                              "sentence to the prompt noting the marked region.")
    parser.add_argument("--marker_hint_type", choices=["location", "shape"], default="location",
                         help="'location' (circle/box) tells the VLM to ignore the marker's "
                              "shape. 'shape' (outline) tells it the marker traces the actual "
                              "missing region's outline.")
    return parser.parse_args()


args = parse_args()

# Default output path sits next to metadata.json, so running this from any
# directory (e.g. called programmatically per-experiment) saves the result
# in the right experiment folder without needing to cd there first.
output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.metadata)), "generated.png")


# Load files
with open(args.metadata) as f:
    metadata = json.load(f)

with open(args.objects) as f:
    objects = json.load(f)

with open(args.damages) as f:
    damages = json.load(f)



# Get information

object_name = metadata["object"]

object_info = objects[object_name]


damage_key = f"{object_name}_{metadata['damage_part']}"

damage_info = damages[damage_key]



# Create metadata section exactly like your old prompt

metadata_text = f"""
Scene Metadata:

Object Configuration:

Object Rotation (deg):
X = {metadata['object_rotation_deg']['x']}
Y = {metadata['object_rotation_deg']['y']}
Z = {metadata['object_rotation_deg']['z']}


Camera Configuration:

Location:
X = {metadata['camera_location_m']['x']}
Y = {metadata['camera_location_m']['y']}
Z = {metadata['camera_location_m']['z']}


Camera Rotation (deg):
X = {metadata['camera_rotation_deg']['x']}
Y = {metadata['camera_rotation_deg']['y']}
Z = {metadata['camera_rotation_deg']['z']}
"""



# Your original prompt style

full_prompt = f"""

Reconstruct the missing part of the object in the input image and produce the complete reconstructed object.

The task is to generate a full reconstruction of the scene with the missing region restored.


The intent of the object is a {object_name}.
{object_info['function']}


The {damage_info['part']} is broken in the input image and must be reconstructed to complete the object.


Object: {object_name} {metadata['view']}
Fragment Type: {damage_info['part']}


{metadata_text}


Output Requirement:

Generate the fully repaired object consistent with the scale and viewpoint, ensuring the missing {damage_info['part']} is correctly restored.

"""

if args.use_marker_note:
    if args.marker_hint_type == "shape":
        full_prompt += (
            "\n\nNote: a colored outline has been drawn on the input image tracing "
            "the actual shape and extent of the missing region. Use this outline as "
            "a guide for both the location AND the approximate shape/proportions of "
            "the repair.\n"
        )
    else:
        full_prompt += (
            "\n\nNote: a colored circle or box has been drawn on the input image "
            "around the approximate region where the repair is needed. Use it as "
            "a location guide only - reconstruct the actual missing geometry "
            "based on the object's real structure, not the marker's shape.\n"
        )


print(full_prompt)



# Gemini

def generate(image_path):

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )


    with open(image_path, "rb") as f:
        image_bytes = f.read()


    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=image_bytes
                ),

                types.Part.from_text(
                    text=full_prompt
                )
            ]
        )
    ]


    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents
    )


    for part in response.candidates[0].content.parts:

        if part.inline_data:
            image = part.as_image()
            image.save(output_path)
            print(f"Saved {output_path}")



if __name__ == "__main__":
    generate(args.marked_image if args.marked_image else metadata["image_path"])
