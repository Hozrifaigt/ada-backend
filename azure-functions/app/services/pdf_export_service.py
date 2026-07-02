import logging
from typing import Dict, Any, List, Optional
from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

from app.services.review_diff import diff_ops, filter_review_draft, review_units, removed_sections

logger = logging.getLogger(__name__)


class PDFExportService:
    """Service for exporting drafts to PDF documents."""

    def __init__(self):
        """Initialize the PDF export service."""
        pass

    def export_draft_to_pdf(self, draft_data: Dict[str, Any]) -> BytesIO:
        """
        Export a complete draft to a PDF document.

        Args:
            draft_data: Complete draft data structure

        Returns:
            BytesIO: PDF document as bytes
        """
        try:
            # Review drafts: export only the banked (closed) sections once any have been closed —
            # same rule as the Word export (this path previously leaked unreviewed sections).
            draft_data = filter_review_draft(draft_data)

            pdf_stream = BytesIO()
            doc = SimpleDocTemplate(
                pdf_stream,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            # Set up styles
            styles = self._setup_styles()

            # Build story (document content)
            story = []

            # Add title page
            self._add_title_page(story, draft_data['metadata'], styles)
            story.append(PageBreak())

            # Add table of contents
            self._add_table_of_contents(story, draft_data['toc'], styles)
            story.append(PageBreak())

            # Add content
            self._add_content(story, draft_data, styles)

            # Build PDF
            doc.build(story)
            pdf_stream.seek(0)

            logger.info(f"Successfully exported draft {draft_data['id']} to PDF")
            return pdf_stream

        except Exception as e:
            logger.error(f"Failed to export draft to PDF: {e}")
            raise ValueError(f"PDF export failed: {str(e)}")

    # ===== Review-draft export variants: redline + changes overview =====
    # (diff/flatten helpers shared with the Word builder via app.services.review_diff)

    @staticmethod
    def _redline_markup(old: str, new: str) -> str:
        """Inline reportlab markup for a word-level redline: additions green-highlighted, removals
        red + strikethrough. Text is XML-escaped before wrapping."""
        parts = []
        for op, text in diff_ops(old, new):
            if not text:
                continue
            t = escape(text)
            if op == "added":
                parts.append(f'<font backColor="#bbf7d0" color="#166534">{t}</font>')
            elif op == "removed":
                parts.append(f'<font backColor="#fecaca" color="#991b1b"><strike>{t}</strike></font>')
            else:
                parts.append(t)
        return " ".join(parts)

    def export_review_redline(self, draft_data: Dict[str, Any],
                              original_data: Optional[Dict[str, Any]] = None) -> BytesIO:
        """Redline PDF for a review draft: per section, the frozen client baseline vs the reviewed
        content — additions green, removals red + strikethrough; removed sections appended."""
        try:
            pdf_stream = BytesIO()
            doc = SimpleDocTemplate(pdf_stream, pagesize=letter,
                                    rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            styles = self._setup_styles()
            story: List = []

            self._add_title_page(story, draft_data['metadata'], styles)
            story.append(PageBreak())
            story.append(Paragraph('Redline — changes vs the original policy', styles['CustomHeading1']))
            story.append(Paragraph(
                '<font backColor="#bbf7d0" color="#166534">added text</font>   '
                '<font backColor="#fecaca" color="#991b1b"><strike>removed text</strike></font>',
                styles['Normal']))
            story.append(Spacer(1, 16))

            for u in review_units(draft_data):
                heading = escape(u["label"])
                if u["added"]:
                    heading += '  <font backColor="#bbf7d0" color="#166534">(new section)</font>'
                story.append(Paragraph(heading, styles['CustomHeading1' if u["level"] == 1 else 'CustomHeading2']))
                if u["old"] == u["new"]:
                    for para in u["new"].split("\n"):
                        if para.strip():
                            story.append(Paragraph(escape(para.strip()), styles['Normal']))
                            story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(self._redline_markup(u["old"], u["new"]), styles['Normal']))
                    story.append(Spacer(1, 10))

            removed = removed_sections(draft_data, original_data)
            if removed:
                story.append(PageBreak())
                story.append(Paragraph('Removed sections', styles['CustomHeading1']))
                for rm in removed:
                    story.append(Paragraph(
                        f'<font color="#991b1b">{escape(rm["title"])}</font>', styles['CustomHeading2']))
                    if rm["content"]:
                        story.append(Paragraph(
                            f'<font backColor="#fecaca" color="#991b1b"><strike>{escape(rm["content"])}</strike></font>',
                            styles['Normal']))
                        story.append(Spacer(1, 8))

            doc.build(story)
            pdf_stream.seek(0)
            return pdf_stream
        except Exception as e:
            logger.error(f"Failed to export redline PDF: {e}")
            raise ValueError(f"Redline PDF export failed: {str(e)}")

    def export_changes_overview(self, draft_data: Dict[str, Any],
                                original_data: Optional[Dict[str, Any]] = None) -> BytesIO:
        """Changes-overview PDF: per-section Added / Modified / Unchanged with word deltas + the
        removed-sections list."""
        try:
            pdf_stream = BytesIO()
            doc = SimpleDocTemplate(pdf_stream, pagesize=letter,
                                    rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            styles = self._setup_styles()
            story: List = []
            meta = draft_data.get('metadata') or {}

            story.append(Paragraph(escape(f"{meta.get('title', 'Policy')} — Changes Overview"), styles['CustomTitle']))
            story.append(Spacer(1, 12))

            units = review_units(draft_data)
            removed = removed_sections(draft_data, original_data)
            rows = []
            added_n = modified_n = unchanged_n = 0
            for u in units:
                if u["added"] or (not u["old"] and u["new"]):
                    status, added_n = "Added", added_n + 1
                elif u["old"] == u["new"]:
                    status, unchanged_n = "Unchanged", unchanged_n + 1
                else:
                    status, modified_n = "Modified", modified_n + 1
                ops = diff_ops(u["old"], u["new"]) if status == "Modified" else []
                plus = sum(len(t.split()) for op, t in ops if op == "added")
                minus = sum(len(t.split()) for op, t in ops if op == "removed")
                rows.append((u, status, plus, minus))

            story.append(Paragraph(
                f'<b>{len(units)} sections reviewed</b> — '
                f'<font color="#15803d">{added_n} added</font>, '
                f'<font color="#b45309">{modified_n} modified</font>, '
                f'{unchanged_n} unchanged, '
                f'<font color="#b91c1c">{len(removed)} removed</font>',
                styles['Normal']))
            story.append(Spacer(1, 16))

            story.append(Paragraph('Section changes', styles['CustomHeading1']))
            STATUS_COLOR = {"Added": "#15803d", "Modified": "#b45309", "Unchanged": "#64748b"}
            for u, status, plus, minus in rows:
                line = f'<b>{escape(u["label"])}</b> — <font color="{STATUS_COLOR[status]}">{status}</font>'
                if status == "Modified":
                    line += f'  (+{plus} / −{minus} words)'
                elif status == "Added" and u["new"]:
                    line += f'  ({len(u["new"].split())} words)'
                if u.get("note"):
                    line += f' — <i>{escape(u["note"])}</i>'
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 4))

            if removed:
                story.append(Spacer(1, 8))
                story.append(Paragraph('Removed sections', styles['CustomHeading1']))
                for rm in removed:
                    story.append(Paragraph(f'<font color="#b91c1c">{escape(rm["title"])}</font>', styles['Normal']))
                    story.append(Spacer(1, 4))

            doc.build(story)
            pdf_stream.seek(0)
            return pdf_stream
        except Exception as e:
            logger.error(f"Failed to export changes-overview PDF: {e}")
            raise ValueError(f"Changes-overview PDF export failed: {str(e)}")

    def _setup_styles(self):
        """Set up document styles."""
        styles = getSampleStyleSheet()

        # Custom title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.black
        ))

        # Custom heading 1 style
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.black
        ))

        # Custom heading 2 style
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.black
        ))

        # Custom description style
        styles.add(ParagraphStyle(
            name='Description',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=20
        ))

        # Custom metadata style
        styles.add(ParagraphStyle(
            name='Metadata',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=colors.grey
        ))

        return styles

    def _add_title_page(self, story: List, metadata: Dict[str, Any], styles):
        """Add a title page to the document."""
        # Title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(metadata['title'], styles['CustomTitle']))
        story.append(Spacer(1, 1*inch))

        # Client information
        client_info = f"<b>Prepared for:</b> {metadata['client_metadata']['name']}"
        story.append(Paragraph(client_info, styles['Description']))
        story.append(Spacer(1, 0.3*inch))

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

        # Metadata
        meta_text = f"""
        <b>Created:</b> {created_at}<br/>
        <b>Last Modified:</b> {modified_at}<br/>
        <b>Created By:</b> {metadata.get('created_by', 'Unknown')}
        """
        story.append(Paragraph(meta_text, styles['Metadata']))

    def _add_table_of_contents(self, story: List, toc: List[Dict[str, Any]], styles):
        """Add table of contents."""
        story.append(Paragraph('Table of Contents', styles['CustomHeading1']))
        story.append(Spacer(1, 20))

        for topic in toc:
            # Add topic to TOC
            toc_text = f"{topic.get('order', 1)}. {topic.get('topic', '')}"
            story.append(Paragraph(toc_text, styles['Normal']))

            # Add subtopics to TOC
            for subtopic in topic.get('subtopics', []):
                sub_text = f"&nbsp;&nbsp;&nbsp;&nbsp;{topic.get('order', 1)}.{subtopic.get('order', 1)} {subtopic.get('topic', '')}"
                story.append(Paragraph(sub_text, styles['Normal']))

    def _add_content(self, story: List, draft_data: Dict[str, Any], styles):
        """Add the actual content sections."""
        toc = draft_data.get('toc', [])
        # Review exports are client deliverables: empty sections print as heading only.
        review_mode = bool((draft_data.get('metadata') or {}).get('review_mode'))

        for topic_index, topic in enumerate(toc):
            # Add topic heading
            topic_title = f"{topic.get('order', topic_index + 1)}. {topic.get('topic', '')}"
            story.append(Paragraph(topic_title, styles['CustomHeading1']))

            # Add topic content
            content = topic.get('content', '').strip()
            if content:
                # Split content into paragraphs and add them
                for paragraph in content.split('\n'):
                    if paragraph.strip():
                        story.append(Paragraph(paragraph.strip(), styles['Normal']))
                        story.append(Spacer(1, 6))
            elif not review_mode:
                story.append(Paragraph("<i>[No content generated yet]</i>", styles['Normal']))
                story.append(Spacer(1, 10))

            # Add subtopics
            for subtopic in topic.get('subtopics', []):
                # Add subtopic heading
                subtopic_title = f"{topic.get('order', topic_index + 1)}.{subtopic.get('order', 1)} {subtopic.get('topic', '')}"
                story.append(Paragraph(subtopic_title, styles['CustomHeading2']))

                # Add subtopic content
                sub_content = subtopic.get('content', '').strip()
                if sub_content:
                    for paragraph in sub_content.split('\n'):
                        if paragraph.strip():
                            story.append(Paragraph(paragraph.strip(), styles['Normal']))
                            story.append(Spacer(1, 6))
                elif not review_mode:
                    story.append(Paragraph("<i>[No content generated yet]</i>", styles['Normal']))
                    story.append(Spacer(1, 10))

            # Add spacing between topics
            story.append(Spacer(1, 20))

    def get_filename(self, draft_data: Dict[str, Any]) -> str:
        """
        Generate a filename for the PDF export.

        Args:
            draft_data: Complete draft data structure

        Returns:
            str: Filename for the PDF document
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

            return f"{safe_title}_{timestamp}.pdf"

        except Exception as e:
            logger.warning(f"Failed to generate filename: {e}")
            return f"policy_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"