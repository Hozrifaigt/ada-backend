import logging
from pathlib import Path
from typing import Dict, Any, List

from app.pipelines.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class LocalDocxLoader:
    """Loads and processes .docx files from local filesystem."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        logger.info("LocalDocxLoader initialized")
    
    def get_file_paths(self, data_path: str) -> List[Path]:
        """Find all .docx files in directory."""
        try:
            path = Path(data_path)
            
            if not path.exists() or not path.is_dir():
                logger.error(f"Invalid directory: {data_path}")
                return []
            
            # Find .docx files, exclude temp files
            files = [
                f for f in path.glob("**/*.docx") 
                if not f.name.startswith("~$") and f.is_file()
            ]
            
            logger.info(f"Found {len(files)} .docx files")
            return files
            
        except Exception as e:
            logger.error(f"Error finding files: {e}")
            return []
    
    def load_and_structure(self, file_path: Path) -> Dict[str, Any]:
        """Load and structure a single .docx file."""
        logger.debug(f"Processing: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.name.startswith("~$"):
            raise ValueError(f"Temporary file skipped: {file_path.name}")
        
        if file_path.suffix.lower() != '.docx':
            raise ValueError(f"Not a .docx file: {file_path}")
        
        try:
            with open(file_path, "rb") as f:
                processed_data = self.processor.process(f)
            
            content_tree = processed_data["content"]
            
            # Flatten content with hierarchy info
            flattened_content = self._flatten_content(content_tree)
            content_by_topic = {}
            topic_hierarchy = {}
            
            for item in flattened_content:
                topic = item.get('topic', 'Unknown')
                content = item.get('content', '')
                if topic and content.strip():
                    content_by_topic[topic] = content
                    topic_hierarchy[topic] = {
                        'level': item.get('level', 1),
                        'parent_topic': item.get('parent_topic'),
                        'is_subtopic': item.get('is_subtopic', False)
                    }
            
            metadata = {
                "source": str(file_path)
            }
            
            return {
                "metadata": metadata,
                "toc": processed_data["toc"],
                "content_by_topic": content_by_topic,
                "topic_hierarchy": topic_hierarchy
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise
    
    def _flatten_content(self, nodes: List[Dict[str, Any]], parent_topic: str = None, level: int = 1) -> List[Dict[str, Any]]:
        """Flatten nested content structure while preserving hierarchy."""
        flat_list = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            
            current_topic = node.get("topic", "Unknown")
            
            # Add current node if it has content
            if node.get("content") and node.get("content").strip():
                node_copy = node.copy()
                node_copy.pop("subtopics", None)
                
                # Add hierarchy information
                node_copy["level"] = level
                node_copy["parent_topic"] = parent_topic
                node_copy["is_subtopic"] = parent_topic is not None
                
                flat_list.append(node_copy)
            
            # Process subtopics recursively
            subtopics = node.get("subtopics", [])
            if subtopics:
                flat_list.extend(self._flatten_content(subtopics, current_topic, level + 1))
        
        return flat_list