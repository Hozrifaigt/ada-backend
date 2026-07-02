import logging


from typing import Dict, Any, List, Optional
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT, WD_COLOR_INDEX
from docx.enum.style import WD_STYLE_TYPE
from datetime import datetime

from app.services.review_diff import diff_ops, filter_review_draft, review_units, removed_sections

logger = logging.getLogger(__name__)


class WordExportService:
    """Service for exporting drafts to Word documents."""
    
    def __init__(self):
        """Initialize the Word export service."""
        pass
    
    def export_draft_to_word(self, draft_data: Dict[str, Any]) -> BytesIO:
        """
        Export a complete draft to a Word document.
        
        Args:
            draft_data: Complete draft data structure
            
        Returns:
            BytesIO: Word document as bytes
        """
        try:
            # Review drafts: export only the banked (closed) sections once any have been closed.
            # If nothing is closed yet, fall back to exporting everything.
            draft_data = filter_review_draft(draft_data)

            doc = Document()

            # Set up styles
            self._setup_styles(doc)

            # Add title page
            self._add_title_page(doc, draft_data['metadata'])

            # Add page break before table of contents
            doc.add_page_break()

            # Add table of contents
            self._add_table_of_contents(doc, draft_data['toc'])

            # Add page break before content
            doc.add_page_break()
            
            # Add content
            self._add_content(doc, draft_data)
            
            # Save to BytesIO
            doc_stream = BytesIO()
            doc.save(doc_stream)
            doc_stream.seek(0)
            
            logger.info(f"Successfully exported draft {draft_data['id']} to Word")
            return doc_stream
            
        except Exception as e:
            logger.error(f"Failed to export draft to Word: {e}")
            raise ValueError(f"Word export failed: {str(e)}")

    def _setup_styles(self, doc: Document):
        """Set up document styles."""
        try:
            # Create custom heading styles if they don't exist
            styles = doc.styles
            
            # Heading 1 style
            if 'Custom Heading 1' not in styles:
                h1_style = styles.add_style('Custom Heading 1', WD_STYLE_TYPE.PARAGRAPH)
                h1_style.font.size = Pt(16)
                h1_style.font.bold = True
                h1_style.font.color.rgb = RGBColor(0, 0, 0)
                h1_style.paragraph_format.space_before = Pt(12)
                h1_style.paragraph_format.space_after = Pt(6)
            
            # Heading 2 style
            if 'Custom Heading 2' not in styles:
                h2_style = styles.add_style('Custom Heading 2', WD_STYLE_TYPE.PARAGRAPH)
                h2_style.font.size = Pt(14)
                h2_style.font.bold = True
                h2_style.font.color.rgb = RGBColor(0, 0, 0)
                h2_style.paragraph_format.space_before = Pt(12)
                h2_style.paragraph_format.space_after = Pt(6)
                
        except Exception as e:
            logger.warning(f"Could not set up custom styles: {e}")
    
    def _add_title_page(self, doc: Document, metadata: Dict[str, Any]):
        """Add a title page to the document."""
        # Title
        title_para = doc.add_heading(metadata['title'], 0)
        title_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Add spacing
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()
        
        # Metadata
        meta_para = doc.add_paragraph()
        meta_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Format dates
        created_at = metadata.get('created_at', '')
        modified_at = metadata.get('modified_at', '')
        
        if isinstance(created_at, str) and created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = created_dt.strftime('%B %d, %Y at %I:%M %p')
            except:
                pass
        
        if isinstance(modified_at, str) and modified_at:
            try:
                modified_dt = datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
                modified_at = modified_dt.strftime('%B %d, %Y at %I:%M %p')
            except:
                pass
        
        meta_para.add_run(f"Created: {created_at}\n")
        meta_para.add_run(f"Last Modified: {modified_at}\n")
        meta_para.add_run(f"Created By: {metadata.get('created_by', 'Unknown')}")
    
    def _add_table_of_contents(self, doc: Document, toc: List[Dict[str, Any]]):
        """Add table of contents."""
        doc.add_heading('Table of Contents', level=1)
        
        for topic in toc:
            # Add topic to TOC
            toc_para = doc.add_paragraph()
            toc_para.add_run(f"{topic['order']}. {topic['topic']}")
            toc_para.paragraph_format.left_indent = Inches(0)

            # Add subtopics to TOC
            for subtopic in topic.get('subtopics', []):
                sub_para = doc.add_paragraph()
                sub_para.add_run(f"    {topic['order']}.{subtopic['order']} {subtopic['topic']}")
                sub_para.paragraph_format.left_indent = Inches(0.5)
    
    def _add_content(self, doc: Document, draft_data: Dict[str, Any]):
        """Add the actual content sections."""
        toc = draft_data.get('toc', [])
        # Review exports are client deliverables: an empty section prints as a heading only —
        # never the internal "[No content generated yet]" placeholder.
        review_mode = bool((draft_data.get('metadata') or {}).get('review_mode'))

        for topic in toc:
            # Add topic heading
            doc.add_heading(f"{topic['order']}. {topic['topic']}", level=1)

            # Add topic content
            content = topic.get('content', '').strip()
            if content:
                # Split content into paragraphs and add them
                for paragraph in content.split('\n'):
                    if paragraph.strip():
                        doc.add_paragraph(paragraph.strip())
            elif not review_mode:
                doc.add_paragraph("*[No content generated yet]*")

            # Add subtopics
            for subtopic in topic.get('subtopics', []):
                # Add subtopic heading
                doc.add_heading(
                    f"{topic['order']}.{subtopic['order']} {subtopic['topic']}",
                    level=2
                )

                # Add subtopic content
                sub_content = subtopic.get('content', '').strip()
                if sub_content:
                    for paragraph in sub_content.split('\n'):
                        if paragraph.strip():
                            doc.add_paragraph(paragraph.strip())
                elif not review_mode:
                    doc.add_paragraph("*[No content generated yet]*")

            # Add spacing between topics
            doc.add_paragraph()
    
    # ===== Review-draft export variants: redline + changes overview =====
    # (diff/flatten helpers shared with the PDF builder via app.services.review_diff)

    def _add_redline_paragraph(self, doc: Document, old: str, new: str) -> None:
        """One redline paragraph: additions highlighted green, removals red + strikethrough."""
        p = doc.add_paragraph()
        for op, text in diff_ops(old, new):
            if not text:
                continue
            run = p.add_run(text + " ")
            if op == "added":
                run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
            elif op == "removed":
                run.font.highlight_color = WD_COLOR_INDEX.RED
                run.font.strike = True

    def export_review_redline(self, draft_data: Dict[str, Any],
                              original_data: Optional[Dict[str, Any]] = None) -> BytesIO:
        """Redline .docx for a review draft: every section shows the frozen client baseline vs the
        reviewed content — additions highlighted GREEN, removals RED + strikethrough. Sections the
        review dropped are appended (fully struck)."""
        try:
            doc = Document()
            self._setup_styles(doc)
            self._add_title_page(doc, draft_data['metadata'])
            doc.add_page_break()

            doc.add_heading('Redline — changes vs the original policy', level=1)
            legend = doc.add_paragraph()
            r = legend.add_run("added text")
            r.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
            legend.add_run("   ")
            r = legend.add_run("removed text")
            r.font.highlight_color = WD_COLOR_INDEX.RED
            r.font.strike = True
            doc.add_paragraph()

            for u in review_units(draft_data):
                h = doc.add_heading("", level=u["level"])
                h.add_run(u["label"])
                if u["added"]:
                    tag = h.add_run("  (new section)")
                    tag.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
                if u["old"] == u["new"]:
                    for para in u["new"].split("\n"):
                        if para.strip():
                            doc.add_paragraph(para.strip())
                else:
                    self._add_redline_paragraph(doc, u["old"], u["new"])

            removed = removed_sections(draft_data, original_data)
            if removed:
                doc.add_page_break()
                doc.add_heading('Removed sections', level=1)
                for rm in removed:
                    h = doc.add_heading("", level=2)
                    r = h.add_run(rm["title"])
                    r.font.highlight_color = WD_COLOR_INDEX.RED
                    if rm["content"]:
                        p = doc.add_paragraph()
                        r = p.add_run(rm["content"])
                        r.font.highlight_color = WD_COLOR_INDEX.RED
                        r.font.strike = True

            stream = BytesIO()
            doc.save(stream)
            stream.seek(0)
            return stream
        except Exception as e:
            logger.error(f"Failed to export redline: {e}")
            raise ValueError(f"Redline export failed: {str(e)}")

    def export_changes_overview(self, draft_data: Dict[str, Any],
                                original_data: Optional[Dict[str, Any]] = None) -> BytesIO:
        """Changes-overview .docx for a review draft: per section Added / Modified / Unchanged with
        word deltas, plus the removed-sections list — the at-a-glance companion to the redline."""
        try:
            GREEN, AMBER, RED, SLATE = RGBColor(0x15, 0x80, 0x3D), RGBColor(0xB4, 0x53, 0x09), \
                RGBColor(0xB9, 0x1C, 0x1C), RGBColor(0x64, 0x74, 0x8B)
            doc = Document()
            self._setup_styles(doc)
            meta = draft_data.get('metadata') or {}
            title = doc.add_heading(f"{meta.get('title', 'Policy')} — Changes Overview", 0)
            title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            doc.add_paragraph()

            units = review_units(draft_data)
            removed = removed_sections(draft_data, original_data)
            rows = []
            added_n = modified_n = unchanged_n = 0
            for u in units:
                if u["added"] or (not u["old"] and u["new"]):
                    status = "Added"
                    added_n += 1
                elif u["old"] == u["new"]:
                    status = "Unchanged"
                    unchanged_n += 1
                else:
                    status = "Modified"
                    modified_n += 1
                ops = diff_ops(u["old"], u["new"]) if status == "Modified" else []
                plus = sum(len(t.split()) for op, t in ops if op == "added")
                minus = sum(len(t.split()) for op, t in ops if op == "removed")
                rows.append((u, status, plus, minus))

            summary = doc.add_paragraph()
            summary.add_run(f"{len(units)} sections reviewed — ").bold = True
            r = summary.add_run(f"{added_n} added")
            r.font.color.rgb = GREEN
            summary.add_run(", ")
            r = summary.add_run(f"{modified_n} modified")
            r.font.color.rgb = AMBER
            summary.add_run(f", {unchanged_n} unchanged, ")
            r = summary.add_run(f"{len(removed)} removed")
            r.font.color.rgb = RED
            doc.add_paragraph()

            doc.add_heading('Section changes', level=1)
            for u, status, plus, minus in rows:
                p = doc.add_paragraph(style='List Bullet')
                r = p.add_run(f"{u['label']} — ")
                r.bold = True
                r = p.add_run(status)
                r.font.color.rgb = GREEN if status == "Added" else AMBER if status == "Modified" else SLATE
                if status == "Modified":
                    p.add_run(f"  (+{plus} / −{minus} words)")
                elif status == "Added" and u["new"]:
                    p.add_run(f"  ({len(u['new'].split())} words)")
                if u.get("note"):
                    r = p.add_run(f" — {u['note']}")
                    r.italic = True

            if removed:
                doc.add_heading('Removed sections', level=1)
                for rm in removed:
                    p = doc.add_paragraph(style='List Bullet')
                    r = p.add_run(rm["title"])
                    r.font.color.rgb = RED

            stream = BytesIO()
            doc.save(stream)
            stream.seek(0)
            return stream
        except Exception as e:
            logger.error(f"Failed to export changes overview: {e}")
            raise ValueError(f"Changes overview export failed: {str(e)}")

    def export_gap_summary(self, summary: Dict[str, Any], metadata: Dict[str, Any]) -> BytesIO:
        """Build a client-facing Gap Assessment Summary .docx from a GapSummary dict."""
        try:
            doc = Document()
            self._setup_styles(doc)

            title = doc.add_heading(f"{metadata.get('title', 'Policy')} — Gap Assessment Summary", 0)
            title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            doc.add_paragraph()

            # Executive narrative
            if summary.get('narrative'):
                doc.add_heading('Executive Summary', level=1)
                for para in summary['narrative'].split('\n'):
                    if para.strip():
                        doc.add_paragraph(para.strip())

            # Rollup
            doc.add_heading('Overview', level=1)
            pct = summary.get('pct_preserved')
            rollup_lines = [
                f"Sections reviewed: {summary.get('total_units', 0)} "
                f"(applied {summary.get('applied', 0)}, assessed {summary.get('assessed', 0)}, pending {summary.get('pending', 0)})",
                f"Total findings: {summary.get('total_findings', 0)}",
                f"By severity: {summary.get('by_severity', {})}",
                f"By type: {summary.get('by_type', {})}",
                f"By source: {summary.get('by_source', {})}",
                f"Benchmark-matched sections: {summary.get('benchmark_coverage', 0)}",
                f"Baseline preserved: {pct if pct is not None else 'n/a'}%",
            ]
            for line in rollup_lines:
                doc.add_paragraph(line, style='List Bullet')

            # Per-section findings
            doc.add_heading('Findings by section', level=1)
            for t in summary.get('topics', []):
                if not t.get('findings'):
                    continue
                doc.add_heading(f"{t['title']} [{t.get('status', 'pending')}]", level=2)
                if t.get('overall_summary'):
                    doc.add_paragraph(t['overall_summary'])
                for f in t.get('findings', []):
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(f"[{f.get('severity', '')}/{f.get('type', '')}] ").bold = True
                    p.add_run(f.get('description', ''))
                    if f.get('suggested_change'):
                        sub = doc.add_paragraph(f"Change: {f['suggested_change']}")
                        sub.paragraph_format.left_indent = Inches(0.5)

            doc_stream = BytesIO()
            doc.save(doc_stream)
            doc_stream.seek(0)
            return doc_stream
        except Exception as e:
            logger.error(f"Failed to export gap summary to Word: {e}")
            raise ValueError(f"Gap summary export failed: {str(e)}")

    def get_filename(self, draft_data: Dict[str, Any]) -> str:
        """
        Generate a filename for the Word export.
        
        Args:
            draft_data: Complete draft data structure
            
        Returns:
            str: Filename for the Word document
        """
        try:
            title = draft_data['metadata']['title']
            # Clean title for filename
            safe_title = "".join(
                c for c in title 
                if c.isalnum() or c in (' ', '-', '_')
            ).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            
            return f"{safe_title}_{timestamp}.docx"
            
        except Exception as e:
            logger.warning(f"Failed to generate filename: {e}")
            return f"policy_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"