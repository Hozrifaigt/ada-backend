import logging


from typing import Dict, Any, List
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from datetime import datetime

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
            else:
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
                else:
                    doc.add_paragraph("*[No content generated yet]*")

            # Add spacing between topics
            doc.add_paragraph()
    
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