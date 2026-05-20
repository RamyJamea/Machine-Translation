import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from ..interface import BaseNLP
from ...helpers import get_settings

SETTINGS = get_settings()


class HuggingfaceNLP(BaseNLP):
    def __init__(self):
        self.client = None
        self.model = None

    def connect(self):
        login(SETTINGS.HF_TOKEN)

    def disconnect(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_model_name(self, name: str = "google/embeddinggemma-300M"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name).to(device=device)

    def embed(
        self, chunks: list[str], dim: int = None, task: str = "STS"
    ) -> list[list[float]]:
        embeddings = self.model.encode(chunks, prompt_name=task)
        return embeddings.tolist()

    def chunk(self, img_path): ...
    def translate_image(self, img_path, map_words, map_phrases, from_lang, to_lang): ...
