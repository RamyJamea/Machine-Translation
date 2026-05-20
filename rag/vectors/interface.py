from pathlib import Path
from abc import ABC, abstractmethod


class BaseVDB(ABC):
    @abstractmethod
    def connect(self, path: Path) -> None: ...

    @abstractmethod
    def disconnect(self, path: Path) -> None: ...

    @abstractmethod
    def init_collection(self, name: str): ...

    @abstractmethod
    def delete_collection(self, name: str) -> None: ...

    @abstractmethod
    def insert(
        self, coll_name: str, metadata: list[dict], embeddings: list[list[float]]
    ): ...

    @abstractmethod
    def semantic_search(
        self, coll_name: str, embeddings: list[list[float]], top_k: int
    ) -> list[dict]: ...
