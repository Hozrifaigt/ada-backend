import os
import json
import asyncio
import logging


from pathlib import Path
from typing import Dict, Any

from app.pipelines.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class LocalDocxLoader:
    """
    Loads .docx files from the local filesystem, processes them to extract
    structured content, and yields the data.
    """
    def __init__(self):
        self.processor = DocumentProcessor()
        logger.info("LocalDocxLoader initialized.")

    def get_file_paths(self, data_path: str) -> list:
        """Finds all .docx files in the given directory."""
        path = Path(data_path)
        if not path.is_dir():
            logger.warning(f"Data path '{data_path}' is not a valid directory.")
            return []
        return list(path.glob("**/*.docx"))

    def load_and_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Loads a single .docx file from a path, processes it, and returns
        the structured data including metadata and content chunks.

        Args:
            file_path: The Path object for the .docx file.
        
        Returns:
            A dictionary containing the structured TOC, content by topic,
            and initial metadata.
        """
        logger.debug(f"Processing local file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                processed_data = self.processor.process(f)
            
            content_tree = processed_data["content"]
            
            content_by_topic = {
                item['topic']: item['content'] 
                for item in self._flatten_content(content_tree)
            }

            return {
                "metadata": {
                    "filename": file_path.name,
                    "source": str(file_path),
                },
                "toc": processed_data["toc"],
                "content_by_topic": content_by_topic
            }

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise

    def _flatten_content(self, nodes: list) -> list:
        """Helper to flatten the nested content structure."""
        flat_list = []
        for node in nodes:
            # Create a copy and remove subtopics to avoid circular references
            node_copy = node.copy()
            subtopics = node_copy.pop("subtopics", [])
            flat_list.append(node_copy)
            if subtopics:
                flat_list.extend(self._flatten_content(subtopics))
        return flat_list
