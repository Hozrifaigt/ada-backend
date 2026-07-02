import logging
from typing import Dict, Any, List
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

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