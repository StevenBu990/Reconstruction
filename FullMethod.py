import os
from google import genai
from google.genai import types

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
                types.Part.from_text(text="The intent of this object is a vase." \
                "It is an item that can be used to hold water and is typically used to hold plants or flowers." \
                "A portion of the vase is fragmented off. I want you to reconstruct the prebroken image." \
                "This view is directly looking at the vase the side from an angle.")
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
            image.save("fullVaseView3.png")
            print("✅ Reconstructed image saved as reconstructed1.png")

if __name__ == "__main__":
    generate("/Users/stevenbu/Desktop/vase3.png")
