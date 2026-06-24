import os
from google import genai
from google.genai import types

# export GEMINI_API_KEY="AQ.Ab8RN6IIcs-Jpg_RgDJAIO_yPcSvOnkPOQqxOagzmQQVagVWfA"

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
                types.Part.from_text( text= "The image shows a broken vase with a portion of the rim missing. "
                    "Generate ONLY the geometry of the missing rim needed to repair the vase. "
                    "Do not reconstruct the full vase. "
                    "Output the portion as a standalone 3D-reconstructable image or description.")
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
            image.save("VLM_Direct_Vase2.png")
            print("✅ Reconstructed image saved as reconstructed1.png")

if __name__ == "__main__":
    generate("/Users/stevenbu/Desktop/vase2.png")