from typing import Literal
from .providers import GeminiNLP, OpenAINLP, HuggingfaceNLP


class FactoryNLP:
    def create_mtf(self, provider: Literal["gemini", "openai", "huggingface"]):
        if provider.lower() == "gemini":
            return GeminiNLP()

        elif provider.lower() == "openai":
            return OpenAINLP()

        elif provider.lower() == "huggingface":
            return HuggingfaceNLP()
