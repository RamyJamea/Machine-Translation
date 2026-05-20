from typing import Literal
from .providers import ChromaDB


class FactoryVDB:
    def create_vdb(self, provider: Literal["chromadb"]):
        if provider.lower() == "chromadb":
            return ChromaDB()
