from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class ChunksModel(BaseModel):
    content: list[str] = Field(description="List of phrases.")


class BaseNLP(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_model_name(self, name: str = None) -> None: ...

    @abstractmethod
    def embed(
        self, chunks: list[str], dim: int = None, task: str = None
    ) -> list[list[float]]: ...

    @abstractmethod
    def chunk(self, img_path: Path) -> ChunksModel: ...

    @abstractmethod
    def translate_image(
        self,
        img_path: Path,
        map_words: dict,
        map_phrases: dict,
        from_lang: Literal["arabic", "english"],
        to_lang: Literal["arabic", "english"],
    ) -> str: ...

    def get_prompt(
        self,
        map_words: dict,
        map_phrases: dict,
        from_lang: Literal["arabic", "english"],
        to_lang: Literal["arabic", "english"],
    ) -> str:
        template = """<prompt_configuration>
    <metadata>
        <source_language>{{from_lang}}</source_language>
        <target_language>{{to_lang}}</target_language>
        <domain>Accounting ERP Systems</domain>
    </metadata>

    <system_instruction>
        <role>You are an expert translator specializing in accounting ERP systems.</role>
        <task>Translate the text in the image from {{from_lang}} to {{to_lang}}.</task>
        <output_format>Provide only the translation with no additional introductions.</output_format>
    </system_instruction>

    <constraints>
        <requirement>You must strictly use the following terminology mappings and few-shot examples:</requirement>
        <terminology_mappings>
            {{map_phrases}}
        </terminology_mappings>
    </constraints>
</prompt_configuration>
"""
        return template.format(
            # map_words=str(map_words),
            map_phrases=str(map_phrases),
            from_lang=from_lang,
            to_lang=to_lang,
        )
