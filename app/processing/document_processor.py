import re

from typing import Dict, List, Optional, IO, Any
from docx import Document

from app.processing.base import PolicyProcessor
import logging 

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes Word (.docx) documents to extract a structured Table of Contents (TOC)
    and map the corresponding content hierarchically. It can operate using formal
    TOC styles or fall back to a heuristic-based approach for documents without them.
    
    The processor is designed to read from a file stream, making it suitable for
    use in web applications and data processing pipelines.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the processor with an optional configuration."""
        self.config = config or {}
        self.doc: Optional[Document] = None

    def process(self, file_stream: IO[bytes]) -> Dict[str, Any]:
        """
        Main processing method. Orchestrates the parsing of the document stream.

        Args:
            file_stream: A file-like object (byte stream) of the .docx file.
        
        Returns:
            A dictionary with 'toc' and 'content' keys. The 'toc' contains only
            the nested topic structure, while 'content' contains the topics
            along with their full extracted text.
            
        Raises:
            ValueError: If the document stream cannot be processed.
        """
        try:
            self.doc = Document(file_stream)
            
            if self._has_toc_styles():
                logger.info("Strategy: Found formal TOC styles. Parsing from TOC.")
                toc_structure = self._build_structure_from_toc_styles()
                nested_content = self._map_content_to_structure(toc_structure)
            else:
                logger.info("Strategy: No formal TOC. Using heuristic parsing on document body.")
                flat_list = self._extract_sections_to_flat_list_from_body()
                nested_content = self._build_nested_content_tree(flat_list)
            
            toc_only = self._create_toc_from_content_tree(nested_content)

            return {
                "toc": toc_only,
                "content": nested_content
            }
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise ValueError(f"Document processing failed: {e}") from e


    def _has_toc_styles(self) -> bool:
        """Checks if the document uses formal 'toc' styles."""
        return any(p.style and p.style.name and p.style.name.lower().startswith('toc') for p in self.doc.paragraphs if p.text.strip())

    def _build_structure_from_toc_styles(self) -> List[Dict]:
        """Parses the document's 'toc' styles to get the exact hierarchy and titles."""
        toc_nodes: List[Dict] = []
        parent_finder = {0: {"subtopics": toc_nodes}}
        
        for para in self.doc.paragraphs:
            style_name = para.style.name.lower() if para.style else ""
            if not style_name.startswith('toc '):
                continue

            text = para.text.strip()
            if not text:
                continue
                
            try:
                level = int(style_name.split(' ')[-1])
            except (ValueError, IndexError):
                continue

            parts = text.rsplit('\t', 1)
            raw_title = parts[0]
            page_number = None

            if len(parts) == 2:
                page_num_str = ''.join(filter(str.isdigit, parts[1]))
                if page_num_str:
                    page_number = int(page_num_str)
            
            title = self._get_clean_title(raw_title)
            
            if not title:
                continue
            
            new_node = {
                "topic": title, 
                "page_number": page_number, 
                "subtopics": [], 
                "content": ""
            }
            
            parent_level = level - 1
            while parent_level not in parent_finder and parent_level >= 0:
                parent_level -= 1
            
            parent_node = parent_finder[parent_level]
            parent_node["subtopics"].append(new_node)
            parent_finder[level] = new_node

        return toc_nodes

    def _map_content_to_structure(self, toc_structure: List[Dict]) -> List[Dict]:
        """Maps content using a two-pass strategy to find and assign text between heading landmarks."""
        flat_toc = self._flatten_nodes(toc_structure)
        if not flat_toc:
            return toc_structure

        landmarks = []
        topic_lookup = list(flat_toc)

        for i, para in enumerate(self.doc.paragraphs):
            para_text = para.text.strip()
            if not para_text:
                continue
            
            for node in topic_lookup:
                if self._titles_match(para_text, node["topic"]):
                    landmarks.append({"index": i, "node": node})
                    topic_lookup.remove(node)
                    break 

        landmarks.sort(key=lambda x: x['index'])

        for i, landmark in enumerate(landmarks):
            current_node = landmark["node"]
            start_index = landmark["index"]
            is_last = (i + 1) == len(landmarks)
            end_index = len(self.doc.paragraphs) if is_last else landmarks[i + 1]["index"]
            
            content_paragraphs = self.doc.paragraphs[start_index:end_index]
            content_text = "\n".join([p.text for p in content_paragraphs])
            current_node["content"] = self._clean_content(content_text)

        return toc_structure

    #--------------------------------------------------------------------------
    # Private Methods for Heuristic (Fallback) Parsing
    #--------------------------------------------------------------------------

    def _get_heading_level_from_body(self, paragraph) -> Optional[int]:
        """
        Heuristically determines heading level with a word count constraint to
        avoid misclassifying long paragraphs as headings.
        """
        text = paragraph.text.strip()
        
        # **THE FIX**: A heading should not be a long sentence. This is the most
        # important guard against the issue you identified.
        if len(text.split()) > 25:
            return None

        style_name = paragraph.style.name.lower() if paragraph.style else ""
        if 'heading' in style_name:
            try:
                level_str = style_name.split()[-1]
                if level_str.isdigit():
                    return int(level_str)
            except (ValueError, IndexError):
                pass
        
        match = re.match(r'^(\d+(\.\d+)*)\s', text)
        if match:
            return match.group(1).count('.') + 1
        
        if re.match(r'^Appendix\s[\w\d]+', text, re.IGNORECASE):
            return 1

        # This check is now much safer due to the word count limit above.
        if any(run.font.bold for run in paragraph.runs):
            return 1
        
        return None

    def _extract_sections_to_flat_list_from_body(self) -> List[Dict]:
        """Creates a flat list of all sections based on heuristic heading detection."""
        flat_list: List[Dict] = []
        current_content: List[str] = []
        current_heading: Optional[Dict] = None
        
        for para in self.doc.paragraphs:
            level = self._get_heading_level_from_body(para)
            para_text = para.text.strip()
            
            if level is not None and para_text:
                if current_heading:
                    current_heading['content'] = self._clean_content("\n".join(current_content))
                    flat_list.append(current_heading)
                
                # **ENHANCEMENT**: Use the centralized title cleaning function.
                cleaned_title = self._get_clean_title(para_text)
                current_heading = {"topic": cleaned_title, "level": level}
                current_content = [para_text]
            elif para_text and current_heading:
                current_content.append(para_text)
        
        if current_heading:
            current_heading['content'] = self._clean_content("\n".join(current_content))
            flat_list.append(current_heading)
            
        return flat_list

    def _build_nested_content_tree(self, flat_list: List[Dict]) -> List[Dict]:
        """Constructs a nested hierarchy from a flat list of sections and their levels."""
        if not flat_list:
            return []
            
        root_nodes: List[Dict] = []
        parent_stack: List[Dict] = []
        
        for section in flat_list:
            level = section['level']
            new_node = {
                "topic": section['topic'], 
                "content": section['content'], 
                "subtopics": []
            }
            
            while parent_stack and parent_stack[-1]['level'] >= level:
                parent_stack.pop()
            
            if not parent_stack:
                root_nodes.append(new_node)
            else:
                parent_stack[-1]['node']['subtopics'].append(new_node)
            
            parent_stack.append({'level': level, 'node': new_node})
            
        return root_nodes

    #--------------------------------------------------------------------------
    # Utility and Helper Methods
    #--------------------------------------------------------------------------
    
    @staticmethod
    def _get_clean_title(text: str) -> str:
        """
        Cleans a title string and shortens it if it's excessively long
        by splitting on common separators.
        """
        title = re.sub(r'\t+', ' ', text).strip()
        title = re.sub(r'[\s.]+$', '', title).strip()
        
        # Shorten titles that look like sentences.
        if len(title.split()) > 15:
            separators = [' - ', ' – ', ': ']
            for sep in separators:
                if sep in title:
                    return title.split(sep, 1)[0].strip()
        return title

    @staticmethod
    def _titles_match(para_text: str, topic_title: str) -> bool:
        """Robustly checks if a paragraph's text matches a TOC title."""
        norm_para = re.sub(r'^[\w\d.\s]+\s', '', para_text, count=1).lower().strip()
        # The topic title from the TOC is already clean, so just lowercase it.
        norm_topic = topic_title.lower().strip()
        raw_para_norm = para_text.lower().strip()
        return norm_para == norm_topic or raw_para_norm == norm_topic

    @staticmethod
    def _flatten_nodes(nodes: List[Dict]) -> List[Dict]:
        """Recursively flattens a tree of nodes into a list."""
        flat_list: List[Dict] = []
        for node in nodes:
            flat_list.append(node)
            if node.get("subtopics"):
                flat_list.extend(DocumentProcessor._flatten_nodes(node["subtopics"]))
        return flat_list

    @staticmethod
    def _clean_content(text: str) -> str:
        """Cleans final extracted text by removing artifacts and extra whitespace."""
        if not text:
            return ""
        cleaned_text = text.replace('\u2026', '').replace('\t', ' ')
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        lines = [line.strip() for line in cleaned_text.split('\n')]
        return "\n".join(filter(None, lines))

    def _create_toc_from_content_tree(self, content_tree: List[Dict]) -> List[Dict]:
        """Recursively generates a clean TOC structure (without content)."""
        toc_tree = []
        for node in content_tree:
            toc_node = {
                "topic": node["topic"],
                "page_number": node.get("page_number"),
                "subtopics": self._create_toc_from_content_tree(node.get("subtopics", []))
            }
            toc_tree.append(toc_node)
        return toc_tree
