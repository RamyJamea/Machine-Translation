import base64
from openai import OpenAI
from ..interface import BaseNLP
from ...helpers import get_settings

SETTINGS = get_settings()


class OpenAINLP(BaseNLP):
    def __init__(self):
        super().__init__()
        self.client: OpenAI = None
        self.model_name: str = None

    def connect(self) -> None:
        self.client = OpenAI(
            api_key=SETTINGS.OPENAI_API_KEY,
            base_url=SETTINGS.OPENAI_BASE_URL,
        )

    def disconnect(self):
        self.client.close()

    def set_model_name(self, model_name: str) -> str:
        self.model_name = model_name

    def translate_image(self, img_path, map_words, map_phrases, from_lang, to_lang):
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            text = self.get_prompt(map_words, map_phrases, from_lang, to_lang)

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )
            return response.output_text

    def chunk(self, img_path): ...
