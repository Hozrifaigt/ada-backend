"""
Quick offline check for DocumentProcessor extraction (tables + TOC content mapping).

No Azure / LLM needed — this exercises the deterministic parse only.

Where files live:
    host      ./ADA_DATA/client_data/your-policy.docx
    container /code/data/client_data/your-policy.docx   (that host folder is mounted here)

Run it INSIDE the backend container (that's where python-docx is installed):
    docker compose exec api python app/test_processor.py
        -> parses the first .docx found in /code/data/client_data
    docker compose exec api python app/test_processor.py /code/data/client_data/your-policy.docx
        -> parses a specific file
"""
import sys
from io import BytesIO
from pathlib import Path

from app.pipelines.document_processor import DocumentProcessor

# Search order for the data folder: the script's OWN directory first (drop a .docx next to this
# file), then the container mount / host-relative fallbacks.
DEFAULT_DIRS = [
    Path(__file__).resolve().parent,
    Path("/code/data/client_data"),
    Path("ADA_DATA/client_data"),
    Path("client_data"),
]


def find_file() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            sys.exit(f"File not found: {p}")
        return p
    for folder in DEFAULT_DIRS:
        if folder.is_dir():
            docs = [f for f in folder.glob("**/*.docx") if not f.name.startswith("~$")]
            if docs:
                return docs[0]
    sys.exit("No .docx found. Drop one into ADA_DATA/client_data (host) / "
             "/code/data/client_data (container), or pass a file path as an argument.")


def preview(text: str, limit: int = 600) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n... [+{len(text) - limit} more chars]"


def walk(nodes, depth=0):
    """Print the TOC tree."""
    for node in nodes:
        print("  " * depth + f"- {node.get('topic', '')}")
        walk(node.get("subtopics", []) or [], depth + 1)


def main():
    path = find_file()
    print(f"=== Parsing: {path} ===\n")

    with open(path, "rb") as f:
        data = f.read()

    # keep_empty_sections=True mirrors the review/upload flow.
    result = DocumentProcessor().process(BytesIO(data), keep_empty_sections=True)

    meta = result.get("metadata", {})
    print("METADATA:")
    for k in ("extraction_method", "confidence", "low_reason", "num_topics",
              "headings_total", "headings_reliable", "headings_heuristic", "weak_ratio"):
        if k in meta:
            print(f"  {k}: {meta[k]}")
    print()

    print("TOC TREE:")
    walk(result.get("toc", []))
    print()

    full_text = result.get("full_text", "") or ""
    has_table_in_full = " | " in full_text  # table rows are joined with ' | '
    print(f"FULL TEXT: {len(full_text)} chars | contains table rows: {has_table_in_full}\n")

    print("TOPIC CONTENT (first ~600 chars each; [Table] marks captured table content):")
    topics = result.get("topics", [])
    total_tables = 0
    for t in topics:
        content = t.get("topic_content", "") or ""
        n_tables = content.count("[Table]")
        total_tables += n_tables
        flag = f"  <-- {n_tables} table block(s)" if n_tables else ""
        print(f"\n## {t.get('topic', '')}{flag}")
        print(preview(content))
        for s in t.get("subtopics", []) or []:
            sc = s.get("content", "") or ""
            sn = sc.count("[Table]")
            total_tables += sn
            sflag = f"  <-- {sn} table block(s)" if sn else ""
            print(f"\n  ### {s.get('name', '')}{sflag}")
            print("  " + preview(sc).replace("\n", "\n  "))

    print(f"\n=== DONE: {len(topics)} top-level topics, "
          f"{total_tables} table block(s) captured across sections ===")


if __name__ == "__main__":
    main()
