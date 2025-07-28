import os
import json
import logging
from app.processing.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)

def test_process_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    processor = DocumentProcessor()

    with open(path, "rb") as f:
        result = processor.process(f)

    print(json.dumps(result["toc"], indent=2))
    print(json.dumps(result["content"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_process_file("data/client_data/Dimah Capital_Variable Remuneration and Reward Policy & Procedures Manual.docx")