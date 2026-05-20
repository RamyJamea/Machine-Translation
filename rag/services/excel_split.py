import docx
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ExcelSplit:
    def extract_clean_text(self, file_path: Path):
        try:
            doc = docx.Document(file_path)
            paragraphs = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            return "\n\n".join(paragraphs)

        except Exception as e:
            print(f"Error reading file: {e}")
            return ""

    def get_semantic_chunks(self, raw_text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
        )
        return text_splitter.split_text(raw_text)
