import base64
from typing import Literal
from pathlib import Path
from PIL import Image as PILImage
from google import genai
from ...helpers import get_settings
from ..interface import BaseNLP, ChunksModel

SETTINGS = get_settings()


class GeminiNLP(BaseNLP):
    def __init__(self):
        self.client = None
        self.model_name = None

    def connect(self) -> None:
        self.client = genai.Client(api_key=SETTINGS.GEMINI_API_KEY)

    def disconnect(self) -> None:
        self.client.close()

    def set_model_name(self, name=None):
        self.model_name = name

    def translate_image(
        self,
        img_path: Path,
        map_words: dict,
        map_phrases: dict,
        from_lang: Literal["arabic", "english"],
        to_lang: Literal["arabic", "english"],
    ) -> str:
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            with PILImage.open(img_path) as img:
                mime_type = f"image/{img.format.lower()}" if img.format else "image/png"

            image_data = base64.b64encode(image_bytes).decode("utf-8")
            text_content = self.get_prompt(map_words, map_phrases, from_lang, to_lang)

            response = self.client.interactions.create(
                model=self.model_name,
                input=[
                    {"type": "text", "text": text_content},
                    {
                        "type": "image",
                        "data": image_data,
                        "mime_type": mime_type,
                    },
                ],
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "thinking_level": "high",
                },
            )
            return response.outputs[-1].text
        except Exception as e:
            raise RuntimeError(f"Translation Failed -- e: {e}")

    def chunk(self, img_path: str) -> ChunksModel:
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()

            img = PILImage.open(img_path)
            mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            text_content = "Chunk the following page into phrases for translation."

            response = self.client.interactions.create(
                model=self.model_name,
                input=[
                    {"type": "text", "text": text_content},
                    {
                        "type": "image",
                        "data": image_data,
                        "mime_type": mime_type,
                    },
                ],
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "thinking_level": "high",
                },
                response_format=ChunksModel.model_json_schema(),
            )
            return ChunksModel.model_validate_json(response.outputs[-1].text)
        except Exception as e:
            raise RuntimeError(f"Translation Failed -- e: {e}")

    def embed(self, chunks, dim=None, task=None): ...
