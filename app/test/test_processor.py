import os
import json
import logging
from app.pipelines.components.file_loader import LocalDocxLoader
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def test_process_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    processor = LocalDocxLoader()

    # with open(path, "rb") as f:
    #     result = processor.process(f)

    structured_data = processor.load_and_structure(path)

    print(json.dumps(structured_data, indent=2))
    # print(json.dumps(result["content"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    file_path = Path("data/client_data/Dimah Capital_Variable Remuneration and Reward Policy & Procedures Manual.docx")
    test_process_file(file_path)