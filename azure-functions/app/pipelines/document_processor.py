import re
import json
import logging
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
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
    # How the heading was detected: "toc" | "style" | "outline" | "numbering" | "heuristic".
    # Used so pruning can keep real (high-confidence) empty sections but drop heuristic noise.
    confidence: str = ""
    
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

    def process(self, file_stream: IO[bytes], keep_empty_sections: bool = False) -> Dict[str, Any]:
        """
        Main processing method that extracts and structures document content.

        Args:
            keep_empty_sections: when True (review/upload flow), keep real (style/numbering-detected)
                empty sections — they are gaps to surface, not noise. Heuristic-only empties are still
                pruned. Default False keeps the original library-ingestion behaviour.

        Returns:
            Dictionary containing:
            - toc: Hierarchical table of contents
            - topics: List of topic objects ready for indexing
            - metadata: Document metadata
        """
        try:
            self.doc = Document(file_stream)
            self._keep_empty = keep_empty_sections

            # Step 1: A Table of Contents is the author's authoritative structure — it carries the
            # exact titles AND the depth, so we build straight from it (content mapped from the body)
            # and skip body-heading guessing, dropping cover-page / stray-bold false positives. Try a
            # Word-styled TOC first, then a typed 'Table of Contents' + numbered list.
            toc_entries = self._extract_from_toc_field()
            if len(toc_entries) < 2:
                toc_entries = self._extract_from_plaintext_toc()

            if len(toc_entries) >= 2:
                logger.info(f"Using the document's Table of Contents as structure "
                            f"({len(toc_entries)} entries)")
            else:
                # Step 2: No styled TOC — extract structure from the document BODY (heading styles /
                # outline / numbering, plus heuristics). A gated LLM leveling pass may refine nesting.
                toc_entries = self._extract_from_document_body()
                if len(toc_entries) < 2:
                    logger.info("Body extraction found little structure; falling back to TOC styles")
                    fallback = self._extract_from_toc_styles()
                    if fallback:
                        toc_entries = fallback

            # Keep the flat list so an LLM leveling pass can re-level it later (no-TOC docs only).
            self._flat = toc_entries
            return self._assemble_result(toc_entries)

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise ValueError(f"Document processing failed: {e}") from e

    def _assemble_result(self, flat_entries: List[TOCEntry]) -> Dict[str, Any]:
        """Build the output dict (toc/topics/full_text/metadata) from a flat heading list.
        Idempotent: children are reset so this can be re-run after re-leveling."""
        for e in flat_entries:
            e.children = []  # reset so rebuild_with_levels is idempotent
        hierarchical = self._build_hierarchy(flat_entries)
        pruned = self._prune_empty_topics(hierarchical, keep_empty_sections=getattr(self, "_keep_empty", False))
        topic_documents = self._create_topic_documents(pruned)
        full_text = self._get_full_text()
        return {
            "toc": self._create_simple_toc(pruned),
            "topics": topic_documents,
            "full_text": full_text,
            "metadata": {
                "num_topics": len(topic_documents),
                "extraction_method": flat_entries[0].source_method if flat_entries else "none",
                **self._confidence_metrics(flat_entries),
            },
        }

    # ----- Hooks for the LLM leveling pass (low-confidence docs) -----

    def first_words(self, n: int = 300) -> str:
        """First ~n words of the document with LINE BREAKS PRESERVED, so a printed TOC or numbered
        headings keep their per-line shape (plain space-joining flattens the table into prose)."""
        lines, count = [], 0
        for para in self.doc.paragraphs:
            t = para.text.strip()
            if not t:
                continue
            lines.append(t)
            count += len(t.split())
            if count >= n:
                break
        return "\n".join(lines)

    def _iter_body_paragraphs(self):
        """Every <w:p> in the document body in document order, INCLUDING paragraphs nested inside
        content controls (w:sdt) and tables, which Document.paragraphs skips. A Word-generated TOC is
        normally wrapped in a w:sdt content control, so reading it requires this deeper traversal."""
        try:
            body = self.doc.element.body
        except Exception:
            return
        for p in body.iter(qn('w:p')):
            try:
                yield Paragraph(p, self.doc)
            except Exception:
                continue

    def toc_outline(self) -> str:
        """The document's PRINTED table of contents rendered as an INDENTED outline (indentation =
        depth), reconstructed from 'TOC 1/2/3'-styled paragraphs. Returns '' when the document has no
        styled TOC. This restores the author's intended hierarchy, which is otherwise lost when the
        TOC is flattened to plain text. Traverses content controls (w:sdt) so a Word-generated TOC —
        which Document.paragraphs cannot see — is read."""
        lines = []
        for para in self._iter_body_paragraphs():
            try:
                name = (para.style.name or "").lower() if para.style else ""
            except Exception:
                name = ""
            m = re.search(r'toc\s*(\d+)', name)  # 'toc 1'..'toc 9' (not 'TOC Heading')
            if not m:
                continue
            level = int(m.group(1))
            text = self._clean_toc_title(para.text)
            if text:
                lines.append("  " * (level - 1) + "- " + text)
        return "\n".join(lines)

    def toc_context_for_llm(self, n: int = 300) -> str:
        """The first ~n words of the document, with each paragraph on its own line and any TOC entries
        indented to their depth (reconstructed from the 'TOC 1/2/3' styles), so a printed Table of
        Contents keeps its visual shape. Sent to the LLM the same way whether or not a TOC is
        detected. Traverses content controls (w:sdt) so a Word-generated TOC is included."""
        lines, count = [], 0
        for para in self._iter_body_paragraphs():
            text = para.text.strip()
            if not text:
                continue
            try:
                name = (para.style.name or "").lower() if para.style else ""
            except Exception:
                name = ""
            m = re.search(r'toc\s*(\d+)', name)  # 'toc 1'..'toc 9' → indent by depth
            indent = "  " * (int(m.group(1)) - 1) if m else ""
            lines.append(indent + text)
            count += len(text.split())
            if count >= n:
                break
        return "\n".join(lines)

    def get_heading_hints(self) -> List[Dict[str, Any]]:
        """The detected headings as lightweight hints for the LLM (no content):
        index (position in the flat list), title, current level, and how it was detected."""
        return [
            {"index": i, "title": e.title, "level": e.level, "hint": e.confidence or "heuristic"}
            for i, e in enumerate(getattr(self, "_flat", []))
        ]

    def rebuild_with_levels(self, updates: Dict[int, int], drops: Optional[set] = None) -> Dict[str, Any]:
        """Re-level the detected headings (and drop false positives), then rebuild. Content is reused
        from the deterministic slicing — never regenerated, so it stays verbatim. `updates`/`drops`
        are keyed by the index from get_heading_hints()."""
        drops = set(drops or [])
        new_flat: List[TOCEntry] = []
        prev = None
        for i, e in enumerate(getattr(self, "_flat", [])):
            if i in drops:
                # Not really a heading → fold its title+content into the previous kept section.
                if prev is not None:
                    extra = (e.title + ("\n" + e.content if e.content else "")).strip()
                    prev.content = (prev.content + "\n" + extra).strip() if prev.content else extra
                continue
            lvl = updates.get(i)
            if isinstance(lvl, int) and lvl >= 1:
                e.level = lvl
            new_flat.append(e)
            prev = e
        self._flat = new_flat
        return self._assemble_result(new_flat)

    def _is_toc_header(self, para: Paragraph) -> bool:
        """Checks if a paragraph is a likely 'Table of Contents' header."""
        text = para.text.strip().lower()
        toc_keywords = ['table of contents', 'contents', 'index']
        return text in toc_keywords

    def _level_from_numbering(self, title: str) -> Optional[int]:
        """Nesting depth from a leading section number in the title: '1' -> 1, '1.1' -> 2,
        '2.3.1' -> 3. Returns None when the title has no leading numeric label. This is more reliable
        than the 'TOC n' style when a document's TOC is styled flat (every entry 'TOC 1') despite
        having numbered sub-sections."""
        m = re.match(r'\s*(\d+(?:\.\d+)*)', title or "")
        if not m:
            return None
        return len(m.group(1).split('.'))

    def _clean_toc_title(self, text: str) -> str:
        """Recover the real title from a TOC entry's visible text, which is usually
        '<optional number>\\tTitle\\t<page>' or 'Title......<page>'. We strip ONLY the trailing page
        number — stripping everything after the first tab leaves just the leading number on numbered
        TOCs (e.g. '1', '2', '3')."""
        t = (text or "").strip()
        t = re.sub(r'\t+\s*\d+\s*$', '', t)      # trailing tab(s) + page number
        t = re.sub(r'\.{2,}\s*\d+\s*$', '', t)   # OR dot-leader + page number
        t = re.sub(r'\s+', ' ', t.replace('\t', ' ')).strip()  # normalise remaining tabs/spaces
        return t

    def _extract_from_toc_field(self) -> List[TOCEntry]:
        """Build the structure straight from the document's styled Table of Contents ('TOC 1/2/3'
        paragraphs), which carry both the exact titles and the exact depth. Traverses content controls
        (w:sdt) so a Word-generated TOC is seen. Content is then mapped from the body by matching each
        TOC title to its heading. Returns [] when the document has no styled TOC."""
        entries: List[TOCEntry] = []
        prev_level = None
        for para in self._iter_body_paragraphs():
            try:
                name = (para.style.name or "").lower() if para.style else ""
            except Exception:
                name = ""
            m = re.search(r'toc\s*(\d+)', name)  # 'toc 1'..'toc 9' (not 'TOC Heading')
            if not m:
                continue
            title = self._clean_toc_title(para.text)
            if not title:
                continue
            # Depth priority: the dotted section number (reliable even when the TOC is styled flat) →
            # else stay within the current section as a sibling of the previous entry (so an unnumbered
            # line like "External Vendors" doesn't reset to top level and hijack the entries after it)
            # → else, for the very first entry, the 'TOC n' style level.
            num_level = self._level_from_numbering(title)
            if num_level:
                level = num_level
            elif prev_level is not None:
                level = prev_level
            else:
                level = int(m.group(1))
            entries.append(TOCEntry(level=level, title=title,
                                    source_method="toc_field", confidence="toc"))
            prev_level = level
        if entries:
            self._map_content_to_toc_entries(entries)
        return entries

    def _extract_from_plaintext_toc(self) -> List[TOCEntry]:
        """Handle a TYPED (un-styled) Table of Contents: a 'Table of Contents' / 'Contents' header
        followed by a run of NUMBERED entries ('1. Title', '2.1 Title', ...). Depth comes from the
        dotted section number; content is mapped from the body AFTER the TOC region (so the TOC lines
        are not mistaken for content). Returns [] when there is no such typed TOC. (Word-styled TOCs
        are handled by _extract_from_toc_field.)"""
        paras = self.doc.paragraphs
        start = next((i for i, p in enumerate(paras) if self._is_toc_header(p)), None)
        if start is None:
            return []

        entries: List[TOCEntry] = []
        end = start + 1
        for j in range(start + 1, len(paras)):
            text = paras[j].text.strip()
            if not text:
                end = j + 1
                continue
            level = self._level_from_numbering(text)
            if level is None:
                break  # first non-numbered line after the entries → the body has begun
            title = self._clean_toc_title(text)
            if title:
                entries.append(TOCEntry(level=level, title=title,
                                        source_method="toc_text", confidence="toc"))
            end = j + 1

        if len(entries) < 2:
            return []
        self._map_content_to_toc_entries(entries, search_from=end)
        return entries

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
        """Extract headings from document body using reliable methods.
        TOC lines are excluded entirely; when the styling is 'flat' (every heading is the same
        heading style, e.g. all 'Heading 1') the nesting depth is taken from the outline numbering."""
        paras = self.doc.paragraphs

        # Determine whether the document's heading styles express depth. If every heading resolves
        # to the same style level (e.g. all 'Heading 1'), we fall back to list numbering for depth.
        style_levels = set()
        for para in paras:
            if self._is_toc_style(para) or self._is_toc_header(para):
                continue
            is_h, lvl, conf = self._is_heading(para)
            if is_h:
                style_levels.add(lvl)
        flat_styling = len(style_levels) <= 1

        toc_entries = []
        current_content = []
        last_entry = None

        for i, para in enumerate(paras):
            text = para.text.strip()
            if not text:
                continue
            # Never treat the TOC region (its header or styled entries) as a heading or as content.
            if self._is_toc_style(para) or self._is_toc_header(para):
                continue

            is_heading, level, conf = self._is_heading(para)
            if is_heading:
                # If styles can't express nesting, derive depth from the outline numbering (ilvl).
                if flat_styling:
                    ilvl = self._numbering_xml_level(para)
                    if ilvl is not None:
                        level = ilvl
                        conf = "numbering"
                if last_entry and current_content:
                    last_entry.content = '\n'.join(current_content).strip()
                    last_entry.has_content = bool(last_entry.content)
                    current_content = []

                entry = TOCEntry(
                    level=level,
                    title=text,
                    para_index=i,
                    source_method="document_body",
                    confidence=conf,
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

    def _is_toc_style(self, para) -> bool:
        """A paragraph styled as a Table-of-Contents entry/header (e.g. 'TOC 1', 'TOC Heading')."""
        return bool(para.style and para.style.name and 'toc' in para.style.name.lower())

    def _numbering_xml_level(self, para) -> Optional[int]:
        """Heading depth from Word AUTO-numbering (w:numPr/w:ilvl) — used when styling can't express
        depth (e.g. every heading is 'Heading 1' but the outline list carries ilvl 0/1/2)."""
        try:
            pPr = para._element.pPr
            if pPr is not None and pPr.numPr is not None and pPr.numPr.ilvl is not None:
                return int(pPr.numPr.ilvl.val) + 1
        except Exception:
            pass
        return None

    def _is_heading(self, para) -> Tuple[bool, int, str]:
        """Determine if a paragraph is a heading. Returns (is_heading, level, confidence)
        where confidence is 'style' | 'outline' | 'heuristic'."""
        text = para.text.strip()

        # TOC-styled paragraphs (entries / the 'Table of Contents' header) are never body headings.
        if self._is_toc_style(para):
            return False, 0, ""

        # Method 1: Check Word heading styles
        if para.style and para.style.name:
            style_name = para.style.name.lower()
            if 'heading' in style_name:
                level_match = re.search(r'heading\s*(\d+)', style_name)
                return (True, int(level_match.group(1)), "style") if level_match else (True, 1, "style")
            if 'title' in style_name:
                return True, 1, "style"
            if 'subtitle' in style_name:
                return True, 2, "style"

        # Method 2: Check paragraph properties (outline level)
        try:
            if para._element.pPr is not None and para._element.pPr.outlineLvl is not None:
                return True, para._element.pPr.outlineLvl.val + 1, "outline"
        except:
            pass

        # Method 3: Simple heuristics. NOTE: bold is intentionally NOT a heading signal — in these
        # policy docs bold is used freely for emphasis and defined terms, so treating bold text as a
        # topic/subtopic produced many false headings. We rely on explicit patterns, a larger font,
        # or heading keywords instead.
        if 0 < len(text) <= 150 and len(text.split()) <= 20:
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
                return True, 1, "heuristic"
            if is_larger_font and (text.istitle() or text.isupper()):
                return True, 2, "heuristic"
            if has_heading_keyword and (is_larger_font or text.istitle()):
                return True, 2, "heuristic"

        return False, 0, ""

    def _map_content_to_toc_entries(self, toc_entries: List[TOCEntry], search_from: int = 0):
        """Maps content to TOC entries using a robust two-pass approach. `search_from` restricts the
        body search to paragraphs at/after that index — used by the typed-TOC path so the TOC's own
        lines (which sit in the body) are not matched as the sections themselves."""
        if not toc_entries:
            return

        all_paras = self.doc.paragraphs

        # Find the paragraph index for each TOC entry title
        for entry in toc_entries:
            found_index = -1

            # Pass 1: Look for a perfect match (text match + heading style)
            for i in range(search_from, len(all_paras)):
                para = all_paras[i]
                if self._text_matches_title(para.text.strip(), entry.title):
                    is_heading, _, _ = self._is_heading(para)
                    if is_heading:
                        found_index = i
                        break

            # Pass 2: If no perfect match, find the first text-only match
            if found_index == -1:
                for i in range(search_from, len(all_paras)):
                    para = all_paras[i]
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

    def _prune_empty_topics(self, toc_entries: List[TOCEntry], keep_empty_sections: bool = False) -> List[TOCEntry]:
        """
        Removes near-empty topics with no children. Topics with children are always kept.
        When keep_empty_sections=True, empty sections detected from a high-confidence signal
        (style/outline/numbering) are KEPT (they are real sections / gaps); only heuristic-detected
        empties are pruned. When False, the original behaviour applies (prune all near-empty leaves).
        """
        pruned_list = []
        for entry in toc_entries:
            # First, recursively prune the children
            if entry.children:
                entry.children = self._prune_empty_topics(entry.children, keep_empty_sections=keep_empty_sections)

            # Check removal criteria
            word_count = len(entry.content.split())
            has_no_children = not entry.children

            if word_count < 5 and has_no_children:
                high_confidence = entry.confidence in ("style", "outline", "numbering", "toc")
                if keep_empty_sections and high_confidence:
                    pass  # real section with no content yet → keep it (it's a gap)
                else:
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

    def _confidence_metrics(self, flat_entries: List[TOCEntry]) -> Dict[str, Any]:
        """Signals for whether the detected structure is trustworthy (deterministic) or whether the
        doc is messy enough to need an LLM nesting pass. To be CALIBRATED on a real corpus.

        - reliable headings = detected from style/outline/numbering (depth is meaningful)
        - heuristic headings = detected by guesswork (depth unknown)
        - weak_ratio = heuristic / total
        - mixed_signals = both reliable and heuristic headings in the same body (inconsistent styling)
        """
        total = len(flat_entries)
        by = {}
        for e in flat_entries:
            by[e.confidence or "unknown"] = by.get(e.confidence or "unknown", 0) + 1
        reliable = by.get("style", 0) + by.get("outline", 0) + by.get("numbering", 0) + by.get("toc", 0)
        heuristic = by.get("heuristic", 0)
        weak_ratio = round(heuristic / total, 3) if total else 0.0
        levels = sorted({e.level for e in flat_entries})
        mixed = reliable > 0 and heuristic > 0

        # Flatness: a sizable doc that produced ~no nesting probably lost its hierarchy
        # (e.g. Union Coop Finance: many headings, almost all top-level).
        min_level = levels[0] if levels else 1
        top_count = sum(1 for e in flat_entries if e.level == min_level)
        top_ratio = round(top_count / total, 3) if total else 0.0
        flat = total >= 8 and (len(levels) <= 1 or top_ratio > 0.7)

        # Gate (calibrated on the client corpus): trust the deterministic parse only when the
        # structure is almost entirely from real signals (weak_ratio <= 0.1); otherwise route to the
        # LLM leveling pass. Over-triggering the LLM is safe (it only re-levels detected headings).
        # A structure built entirely from the styled TOC is authoritative — trust it and skip the LLM,
        # even if it happens to look "flat" (a flat TOC genuinely has no subsections).
        toc_sourced = total > 0 and by.get("toc", 0) == total
        if total == 0:
            confidence, reason = "empty", "no_headings"
        elif toc_sourced:
            confidence, reason = "high", ""
        elif weak_ratio > 0.1:
            confidence, reason = "low", "heuristic_headings"
        elif flat:
            confidence, reason = "low", "flat"
        else:
            confidence, reason = "high", ""

        return {
            "headings_total": total,
            "headings_reliable": reliable,
            "headings_heuristic": heuristic,
            "weak_ratio": weak_ratio,
            "top_level_ratio": top_ratio,
            "levels_present": levels,
            "mixed_signals": mixed,
            "confidence": confidence,
            "low_reason": reason,
        }