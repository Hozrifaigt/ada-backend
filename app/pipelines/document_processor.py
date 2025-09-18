import re
import json
import logging
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from typing import Dict, List, Optional, Tuple, Any, IO
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
import uuid

logger = logging.getLogger(__name__)

@dataclass
class TOCEntry:
    """Simplified TOC entry for document processing"""
    level: int
    title: str
    content: str = ""
    children: List['TOCEntry'] = field(default_factory=list)
    source_method: str = ""
    para_index: Optional[int] = None
    has_content: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level,
            'title': self.title,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'children': [child.to_dict() for child in self.children],
            'has_content': self.has_content,
            'source_method': self.source_method
        }

class DocumentProcessor:
    """
    Production-ready document processor optimized for RAG systems.
    Extracts hierarchical structure and prepares content for Azure Search indexing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the processor with an optional configuration."""
        self.config = config or {}
        self.doc: Optional[Document] = None

    def process(self, file_stream: IO[bytes]) -> Dict[str, Any]:
        """
        Main processing method that extracts and structures document content.
        
        Returns:
            Dictionary containing:
            - toc: Hierarchical table of contents
            - topics: List of topic objects ready for indexing
            - metadata: Document metadata
        """
        try:
            self.doc = Document(file_stream)
            
            # Step 1: Try to extract from TOC styles (if they exist)
            toc_entries = self._extract_from_toc_styles()
            
            # Step 2: If no TOC found, extract from document body
            if not toc_entries:
                logger.info("No formal TOC found, extracting from document structure")
                toc_entries = self._extract_from_document_body()
            
            # Step 3: Build hierarchy
            hierarchical_toc = self._build_hierarchy(toc_entries)
            
            # Step 4: Prune empty topics (less than 30 words with no children)
            pruned_hierarchical_toc = self._prune_empty_topics(hierarchical_toc)
            
            # Step 5: Convert to topic documents for indexing
            topic_documents = self._create_topic_documents(pruned_hierarchical_toc)
            
            # Step 6: Get full text for policy summary
            full_text = self._get_full_text()
            
            return {
                "toc": self._create_simple_toc(pruned_hierarchical_toc),
                "topics": topic_documents,
                "full_text": full_text,
                "metadata": {
                    "num_topics": len(topic_documents),
                    "extraction_method": toc_entries[0].source_method if toc_entries else "none"
                }
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise ValueError(f"Document processing failed: {e}") from e

    def _is_toc_header(self, para: Paragraph) -> bool:
        """Checks if a paragraph is a likely 'Table of Contents' header."""
        text = para.text.strip().lower()
        toc_keywords = ['table of contents', 'contents', 'index']
        return text in toc_keywords

    def _extract_from_toc_styles(self) -> List[TOCEntry]:
        """Extracts TOC entries from formal TOC styles."""
        toc_entries = []
        
        # Attempt 1: Find an explicit TOC header and extract entries after it
        toc_start_index = -1
        for i, para in enumerate(self.doc.paragraphs):
            if self._is_toc_header(para):
                toc_start_index = i
                break
        
        if toc_start_index != -1:
            logger.info("Formal TOC header found. Collecting entries after this point.")
            for para in self.doc.paragraphs[toc_start_index + 1:]:
                is_toc_style = para.style and para.style.name and 'toc' in para.style.name.lower()
                if is_toc_style:
                    text = para.text.strip()
                    if text:
                        title = re.sub(r'\.+\s*\d+\s*$', '', text).strip()
                        level_match = re.search(r'toc\s*(\d+)', para.style.name.lower())
                        level = int(level_match.group(1)) if level_match else 1
                        if title:
                            toc_entries.append(TOCEntry(level=level, title=title, source_method="formal_toc"))
                elif toc_entries and para.text.strip():
                    break
        
        # Attempt 2: If no TOC header found, find any 'toc' styled paragraph
        if not toc_entries:
            logger.info("No explicit TOC header found. Looking for any 'toc' styled paragraphs.")
            for para in self.doc.paragraphs:
                is_toc_style = para.style and para.style.name and 'toc' in para.style.name.lower()
                if is_toc_style:
                    text = para.text.strip()
                    if text:
                        title = re.sub(r'\.+\s*\d+\s*$', '', text).strip()
                        level_match = re.search(r'toc\s*(\d+)', para.style.name.lower())
                        level = int(level_match.group(1)) if level_match else 1
                        if title:
                            toc_entries.append(TOCEntry(level=level, title=title, source_method="formal_toc"))
        
        # Map content to TOC entries if found
        if toc_entries:
            self._map_content_to_toc_entries(toc_entries)
        
        return toc_entries

    def _extract_from_document_body(self) -> List[TOCEntry]:
        """Extract headings from document body using reliable methods."""
        toc_entries = []
        current_content = []
        last_entry = None
        
        for i, para in enumerate(self.doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            
            is_heading, level = self._is_heading(para)
            if is_heading:
                if last_entry and current_content:
                    last_entry.content = '\n'.join(current_content).strip()
                    last_entry.has_content = bool(last_entry.content)
                    current_content = []
                
                entry = TOCEntry(
                    level=level,
                    title=text,
                    para_index=i,
                    source_method="document_body"
                )
                toc_entries.append(entry)
                last_entry = entry
            else:
                if last_entry:
                    current_content.append(text)
        
        if last_entry and current_content:
            last_entry.content = '\n'.join(current_content).strip()
            last_entry.has_content = bool(last_entry.content)
        
        self._add_table_content_to_entries(toc_entries)
        
        return toc_entries

    def _is_heading(self, para) -> Tuple[bool, int]:
        """Determine if a paragraph is a heading using reliable methods."""
        text = para.text.strip()
        
        # Method 1: Check Word heading styles
        if para.style and para.style.name:
            style_name = para.style.name.lower()
            if 'heading' in style_name:
                level_match = re.search(r'heading\s*(\d+)', style_name)
                return (True, int(level_match.group(1))) if level_match else (True, 1)
            if 'title' in style_name:
                return True, 1
            if 'subtitle' in style_name:
                return True, 2
        
        # Method 2: Check paragraph properties (outline level)
        try:
            if para._element.pPr is not None and para._element.pPr.outlineLvl is not None:
                return True, para._element.pPr.outlineLvl.val + 1
        except:
            pass
        
        # Method 3: Simple heuristics
        if 0 < len(text) <= 150 and len(text.split()) <= 20:
            is_all_bold = para.runs and all(run.bold for run in para.runs if run.text.strip())
            is_larger_font = False
            try:
                if para.runs and para.runs[0].font.size:
                    is_larger_font = para.runs[0].font.size.pt > 12
            except:
                pass
            
            is_heading_pattern = any([
                text.startswith('Chapter '),
                text.startswith('Section '),
                text.startswith('Part '),
                text.startswith('Article '),
                text.startswith('Appendix '),
                text.isupper() and len(text.split()) <= 10
            ])
            
            heading_keywords = ['introduction', 'background', 'purpose', 'scope', 'objective',
                               'overview', 'summary', 'conclusion', 'references', 'appendix',
                               'definitions', 'glossary', 'requirements', 'methodology']
            has_heading_keyword = any(keyword in text.lower() for keyword in heading_keywords)
            
            if is_heading_pattern:
                return True, 1
            if (is_all_bold or is_larger_font) and (text.istitle() or text.isupper()):
                return True, 2
            if has_heading_keyword and (is_all_bold or is_larger_font or text.istitle()):
                return True, 2
        
        return False, 0

    def _map_content_to_toc_entries(self, toc_entries: List[TOCEntry]):
        """Maps content to TOC entries using a robust two-pass approach."""
        if not toc_entries:
            return
        
        all_paras = self.doc.paragraphs
        
        # Find the paragraph index for each TOC entry title
        for entry in toc_entries:
            found_index = -1
            
            # Pass 1: Look for a perfect match (text match + heading style)
            for i, para in enumerate(all_paras):
                if self._text_matches_title(para.text.strip(), entry.title):
                    is_heading, _ = self._is_heading(para)
                    if is_heading:
                        found_index = i
                        break
            
            # Pass 2: If no perfect match, find the first text-only match
            if found_index == -1:
                for i, para in enumerate(all_paras):
                    if self._text_matches_title(para.text.strip(), entry.title):
                        found_index = i
                        break
            
            entry.para_index = found_index if found_index != -1 else None
        
        # Filter and sort entries that were successfully located
        located_entries = sorted([e for e in toc_entries if e.para_index is not None], 
                                key=lambda x: x.para_index)
        
        # Assign content between one heading and the next
        for i, entry in enumerate(located_entries):
            start_index = entry.para_index + 1
            end_index = len(all_paras)
            
            if i + 1 < len(located_entries):
                end_index = located_entries[i+1].para_index
            
            content_paras = all_paras[start_index:end_index]
            content = '\n'.join([p.text.strip() for p in content_paras if p.text.strip()])
            
            entry.content = content.strip()
            if entry.content:
                entry.has_content = True

    def _text_matches_title(self, text: str, title: str) -> bool:
        """Check if document text matches a TOC title."""
        text_normalized = ' '.join(re.sub(r'[^\w\s]', '', text).lower().split())
        title_normalized = ' '.join(re.sub(r'[^\w\s]', '', title).lower().split())
        
        if not text_normalized or not title_normalized:
            return False
        
        return title_normalized in text_normalized or text_normalized in title_normalized

    def _add_table_content_to_entries(self, toc_entries: List[TOCEntry]):
        """Add table content to the appropriate TOC entries."""
        tables_text = []
        for table in self.doc.tables:
            table_content = self._extract_table_text(table)
            if table_content:
                tables_text.append(f"\n[Table Content]\n{table_content}\n")
        
        if toc_entries and tables_text:
            toc_entries[-1].content += '\n'.join(tables_text)

    def _extract_table_text(self, table: Table) -> str:
        """Extract text from a table."""
        rows = []
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                rows.append(row_text)
        return "\n".join(rows)

    def _build_hierarchy(self, flat_toc: List[TOCEntry]) -> List[TOCEntry]:
        """Build hierarchical structure from flat list."""
        if not flat_toc:
            return []
        
        root_entries, stack = [], []
        for entry in flat_toc:
            while stack and stack[-1].level >= entry.level:
                stack.pop()
            if stack:
                stack[-1].children.append(entry)
            else:
                root_entries.append(entry)
            stack.append(entry)
        return root_entries

    def _prune_empty_topics(self, toc_entries: List[TOCEntry]) -> List[TOCEntry]:
        """
        Removes topics that have less than 30 words and no children.
        Topics with less than 30 words but with children are kept.
        """
        pruned_list = []
        for entry in toc_entries:
            # First, recursively prune the children
            if entry.children:
                entry.children = self._prune_empty_topics(entry.children)
            
            # Check removal criteria
            word_count = len(entry.content.split())
            has_no_children = not entry.children

            # Remove if: less than 5 words AND no children
            if word_count < 5 and has_no_children:
                logger.info(f"Pruning empty topic: '{entry.title[:100]}' ({word_count} words)")
                continue
            
            pruned_list.append(entry)
        
        return pruned_list

    def _create_topic_documents(self, hierarchical_toc: List[TOCEntry]) -> List[Dict[str, Any]]:
        """
        Convert hierarchical TOC to flat list of topic documents for indexing.
        Each main topic becomes one document with its subtopics embedded.
        No embeddings needed - retrieval is by ID only.
        """
        topic_documents = []
        
        for entry in hierarchical_toc:
            # Create topic document
            topic_doc = {
                "id": str(uuid.uuid4()),
                "topic": entry.title,
                "topic_content": entry.content,
                "subtopics": []
            }
            
            # Add subtopics (children) with their own IDs
            if entry.children:
                for child in entry.children:
                    subtopic_data = {
                        "id": str(uuid.uuid4()),  # Generate UUID for each subtopic
                        "name": child.title,
                        "content": child.content
                    }
                    topic_doc["subtopics"].append(subtopic_data)
                    
                    # If subtopic has its own children, add them as nested content
                    if child.children:
                        nested_content = self._get_nested_content(child.children)
                        subtopic_data["content"] = f"{child.content}\n\n{nested_content}".strip()
            
            topic_documents.append(topic_doc)
        
        return topic_documents

    def _get_nested_content(self, children: List[TOCEntry]) -> str:
        """Recursively get content from nested children."""
        content_parts = []
        for child in children:
            if child.content:
                content_parts.append(f"{child.title}:\n{child.content}")
            if child.children:
                nested = self._get_nested_content(child.children)
                if nested:
                    content_parts.append(nested)
        return "\n\n".join(content_parts)

    def _create_simple_toc(self, hierarchical_toc: List[TOCEntry]) -> List[Dict]:
        """Create a simple TOC structure for UI display."""
        toc = []
        for entry in hierarchical_toc:
            toc_item = {
                "topic": entry.title,
                "subtopics": []
            }
            
            if entry.children:
                for child in entry.children:
                    toc_item["subtopics"].append({
                        "topic": child.title,
                        "subtopics": self._create_simple_toc(child.children) if child.children else []
                    })
            
            toc.append(toc_item)
        
        return toc

    def _get_full_text(self) -> str:
        """Get the full document text."""
        paragraphs = [para.text.strip() for para in self.doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)