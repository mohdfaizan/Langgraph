import google.generativeai as genai
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def describe_image_with_gemini(image: Image.Image) -> str:
    # model = genai.GenerativeModel('gemini-pro-vision')
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")


    response = model.generate_content(
        [
            "Describe the contents of this image in detail.",
            image
        ]
    )
    return response.text
