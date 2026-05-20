import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ..nlp import BaseNLP
from ..vectors import BaseVDB


class DataProcess:
    def __init__(self, nlp: BaseNLP, vdb: BaseVDB):
        self.nlp = nlp
        self.vdb = vdb

    def upload_batches(
        self,
        coll_name: str,
        excel_path: Path,
        from_col: str,
        batch_size: int = 100,
    ) -> None:
        df = pd.read_excel(excel_path)
        df = df.dropna(subset=[from_col])

        metadata = df.to_dict(orient="records")
        chunks = df[from_col].astype(str).tolist()

        for i in tqdm(
            range(0, len(metadata), batch_size), desc=f"Uploading to {coll_name}"
        ):
            batch_meta = metadata[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]
            embeddings = self.nlp.embed(batch_chunks)
            self.vdb.insert(coll_name, batch_meta, embeddings)
