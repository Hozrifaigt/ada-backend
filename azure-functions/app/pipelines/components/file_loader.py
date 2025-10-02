import logging
from pathlib import Path
from typing import Dict, Any, List

from app.pipelines.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class LocalDocxLoader:
    """Loads and processes .docx files from local filesystem."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        logger.info("LocalDocxLoader initialized with new DocumentProcessor")
    
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
            
            logger.info(f"Found {len(files)} .docx files in {data_path}")
            return files
            
        except Exception as e:
            logger.error(f"Error finding files: {e}")
            return []
    
    def load_and_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and structure a single .docx file using the new processor.
        
        Returns:
            Dictionary containing:
            - toc: Hierarchical table of contents for UI display
            - topics: List of topic documents ready for indexing
            - full_text: Complete document text for policy summary
            - metadata: Document metadata
        """
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
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise