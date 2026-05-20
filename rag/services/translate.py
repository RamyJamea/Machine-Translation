import tempfile
import pypdfium2 as pdfium
from pathlib import Path
from ..nlp import BaseNLP
from ..vectors import BaseVDB


class ImageTranslate:
    def __init__(self, vdb: BaseVDB, nlp: BaseNLP, embed: BaseNLP):
        self.vdb = vdb
        self.nlp = nlp
        self.embed = embed

    def translate_pdf(self, pdf_path: Path):
        pdf = pdfium.PdfDocument(pdf_path)
        pages = list()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for i in range(len(pdf)):
                    page = pdf[i]
                    try:
                        bitmap = page.render(scale=2)
                        image = bitmap.to_pil()
                        image_filename = temp_path / f"page_{i + 1:05d}.png"
                        image.save(image_filename)

                        # chunks
                        chunks = self.nlp.chunk(image_filename).content
                        # semantic search
                        embeddings = self.embed.embed(chunks)
                        nearest = self.vdb.semantic_search("phrases", embeddings, 3)
                        # translate
                        map_phrases = "\n".join(str(n) for n in nearest)
                        translation = self.nlp.translate_image(
                            image_filename, {}, map_phrases, "arabic", "english"
                        )
                        print(
                            f"[{i+1}/{len(pdf)}] -- Chunks: {len(chunks)}, Retrieved: {len(nearest)}"
                        )

                        image.close()
                        pages.append(translation)
                    finally:
                        page.close()

        finally:
            pdf.close()

        return "---".join(pages)
