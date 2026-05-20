import chromadb
from uuid import uuid4
from ..interface import BaseVDB


class ChromaDB(BaseVDB):
    def __init__(self):
        self.client = None

    def connect(self, path: str):
        self.client = chromadb.PersistentClient(path=path)

    def disconnect(self, path: str):
        self.client = None

    def init_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def delete_collection(self, name: str):
        self.client.delete_collection(name=name)

    def insert(
        self,
        coll_name: str,
        metadata: list[dict],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> None:
        collection = self.init_collection(coll_name)

        if not ids:
            ids = [str(uuid4()) for _ in range(len(metadata))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
        )

    def semantic_search(
        self, coll_name: str, embeddings: list[list[float]], top_k: int
    ) -> list[dict]:
        collection = self.init_collection(coll_name)
        nearest = collection.query(query_embeddings=embeddings, n_results=top_k)

        ids = nearest.get("ids", [])
        metadatas = nearest.get("metadatas", [])

        seen = set()
        unique_metadatas = []

        for id_list, meta_list in zip(ids, metadatas):
            for _id, meta in zip(id_list, meta_list):
                if _id not in seen:
                    seen.add(_id)
                    unique_metadatas.append(meta)

        return unique_metadatas