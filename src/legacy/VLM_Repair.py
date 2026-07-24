import os
import json
from google import genai
from google.genai import types

with open("prompts.json", "r") as f:
    data = json.load(f)

view = "view3"

view_data = data["glass"][view]

vlm_prompt = "\n".join(
    view_data["metadata"] +
    view_data["vlm_repair"]["prompt"]
)

def generate(image_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",  # change if needed
                    data=image_bytes,
                ),
                types.Part.from_text( text= vlm_prompt)
            ],
        ),
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
    )

    for part in response.candidates[0].content.parts:
        if part.text:
            print(part.text)

        elif part.inline_data:
            image = part.as_image()
            image.save("VLM_Repair_view3.png")
            print("✅ Reconstructed image saved as reconstructed1.png")

if __name__ == "__main__":
    generate("/Users/stevenbu/Desktop/view3Input.png")