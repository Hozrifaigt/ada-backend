"""
Service for processing regulation PDFs and extracting structured content.
"""
import re
import uuid
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from app.models.regulation_models import RegulationChunk, RegulationMetadata

logger = logging.getLogger(__name__)


class RegulationsService:
    """Service for processing regulation PDFs into structured chunks."""

    # Regex patterns for detecting articles and sections
    ARTICLE_PATTERNS = [
        r'Article\s+(\d+[A-Za-z]?)',  # Article 15, Article 15A
        r'ARTICLE\s+(\d+[A-Za-z]?)',
        r'Article\s+\((\d+[A-Za-z]?)\)',  # Article (15)
    ]


    def __init__(self):
        """Initialize the regulations service."""
        if pdfplumber is None:
            logger.warning("pdfplumber not installed. PDF extraction will not work. Install with: pip install pdfplumber")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract English text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")

        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Filter out Arabic characters (keep only English, numbers, punctuation)
                        filtered_text = self._filter_english_only(text)
                        text_content.append(filtered_text)

            full_text = "\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    def extract_text_with_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract English text from a PDF file with page numbers.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of tuples (page_number, text_content)
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")

        try:
            pages_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        filtered_text = self._filter_english_only(text)
                        pages_content.append((page_num, filtered_text))

            total_chars = sum(len(text) for _, text in pages_content)
            logger.info(f"Extracted {total_chars} characters from {len(pages_content)} pages in {pdf_path}")
            return pages_content

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    def _filter_english_only(self, text: str) -> str:
        """Filter text to keep only English characters, numbers, and punctuation."""
        # Keep English letters, numbers, common punctuation, and whitespace
        # Remove Arabic Unicode range: \u0600-\u06FF, \u0750-\u077F, \uFB50-\uFDFF, \uFE70-\uFEFF
        filtered = re.sub(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+', '', text)
        # Clean up multiple spaces and empty lines
        filtered = re.sub(r' +', ' ', filtered)
        filtered = re.sub(r'\n\s*\n', '\n\n', filtered)
        return filtered.strip()

    def detect_articles(self, text: str) -> List[Dict[str, any]]:
        """
        Detect article boundaries in the text.

        Args:
            text: The regulation text

        Returns:
            List of dictionaries with article information
        """
        articles = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            for pattern in self.ARTICLE_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    article_num = match.group(1)
                    articles.append({
                        'article_number': f"Article {article_num}",
                        'line_number': i,
                        'line_text': line.strip()
                    })
                    break

        logger.info(f"Detected {len(articles)} articles in text")
        return articles

    def create_chunks(
        self,
        text: str,
        regulation_metadata: RegulationMetadata,
        chunk_size: int = 800,
        overlap: int = 100,
        pages_content: Optional[List[Tuple[int, str]]] = None
    ) -> List[RegulationChunk]:
        """
        Create intelligent chunks from regulation text.

        Chunks are article-aware: they don't split mid-article.

        Args:
            text: The regulation text
            regulation_metadata: Metadata about the regulation
            chunk_size: Target chunk size in tokens (approximate by words)
            overlap: Overlap between chunks in tokens
            pages_content: Optional list of (page_number, text) tuples for page tracking

        Returns:
            List of RegulationChunk objects
        """
        articles = self.detect_articles(text)
        lines = text.split('\n')

        # Build a mapping of line numbers to page numbers
        line_to_page = self._build_line_to_page_map(pages_content) if pages_content else {}

        chunks = []

        # If no articles detected, use simple chunking
        if not articles:
            logger.warning("No articles detected, using simple chunking")
            return self._simple_chunk(text, regulation_metadata, chunk_size, overlap)

        # Process each article
        for i, article in enumerate(articles):
            start_line = article['line_number']

            # Determine end line (next article or end of document)
            if i + 1 < len(articles):
                end_line = articles[i + 1]['line_number']
            else:
                end_line = len(lines)

            # Extract article content
            article_lines = lines[start_line:end_line]
            article_text = '\n'.join(article_lines).strip()

            # Skip if too short (less than 20 words)
            word_count = len(article_text.split())
            if word_count < 20:
                logger.debug(f"Skipping short chunk ({word_count} words): {article_text[:50]}...")
                continue

            # Get page number for this article
            page_number = line_to_page.get(start_line)

            # If article fits in one chunk, create single chunk
            if word_count <= chunk_size:
                chunk = self._create_chunk(
                    content=article_text,
                    regulation_metadata=regulation_metadata,
                    article_number=article['article_number'],
                    page_number=page_number
                )
                chunks.append(chunk)
            else:
                # Split large article into multiple chunks with overlap
                sub_chunks = self._split_large_article(
                    article_text,
                    article['article_number'],
                    regulation_metadata,
                    chunk_size,
                    overlap,
                    page_number=page_number
                )
                chunks.extend(sub_chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(articles)} articles")
        return chunks

    def _build_line_to_page_map(self, pages_content: List[Tuple[int, str]]) -> Dict[int, int]:
        """
        Build a mapping from line numbers to page numbers.

        Args:
            pages_content: List of (page_number, text) tuples

        Returns:
            Dictionary mapping line number to page number
        """
        line_to_page = {}
        current_line = 0

        for page_num, page_text in pages_content:
            page_lines = page_text.split('\n')
            for _ in page_lines:
                line_to_page[current_line] = page_num
                current_line += 1

        return line_to_page

    def _create_chunk(
        self,
        content: str,
        regulation_metadata: RegulationMetadata,
        article_number: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> RegulationChunk:
        """Create a single regulation chunk."""
        chunk_id = str(uuid.uuid4())

        # Auto-tag topics based on keywords
        topics = self._extract_topics(content)

        return RegulationChunk(
            id=chunk_id,
            content=content,
            regulation_category=regulation_metadata.regulation_category,
            decree_name=regulation_metadata.decree_name,
            article_number=article_number,
            regulation_type=regulation_metadata.regulation_type,
            effective_date=regulation_metadata.effective_date,
            topics=topics,
            page_number=page_number
        )

    def _split_large_article(
        self,
        article_text: str,
        article_number: str,
        regulation_metadata: RegulationMetadata,
        chunk_size: int,
        overlap: int,
        page_number: Optional[int] = None
    ) -> List[RegulationChunk]:
        """Split a large article into multiple overlapping chunks."""
        words = article_text.split()
        chunks = []

        start = 0
        part_num = 1

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Add article header to each part
            if part_num > 1:
                chunk_text = f"{article_number} (continued - Part {part_num}):\n{chunk_text}"

            chunk = self._create_chunk(
                content=chunk_text,
                regulation_metadata=regulation_metadata,
                article_number=f"{article_number} (Part {part_num})",
                page_number=page_number
            )
            chunks.append(chunk)

            start = end - overlap
            part_num += 1

        return chunks

    def _simple_chunk(
        self,
        text: str,
        regulation_metadata: RegulationMetadata,
        chunk_size: int,
        overlap: int
    ) -> List[RegulationChunk]:
        """Simple word-based chunking when article detection fails."""
        words = text.split()
        chunks = []

        start = 0
        chunk_num = 1

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            chunk = self._create_chunk(
                content=chunk_text,
                regulation_metadata=regulation_metadata,
                article_number=f"Section {chunk_num}"
            )
            chunks.append(chunk)

            start = end - overlap
            chunk_num += 1

        return chunks

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        topics = []

        # Topic keywords to look for
        topic_keywords = {
            'employment contracts': ['employment contract', 'hiring', 'appointment'],
            'termination': ['termination', 'dismissal', 'end of service'],
            'leave': ['leave', 'vacation', 'absence', 'holiday'],
            'probation': ['probation', 'probationary period'],
            'working hours': ['working hours', 'work time', 'shift'],
            'wages': ['wage', 'salary', 'remuneration', 'compensation'],
            'overtime': ['overtime', 'additional hours'],
            'benefits': ['benefit', 'allowance', 'entitlement'],
            'health and safety': ['health', 'safety', 'occupational'],
            'discrimination': ['discrimination', 'equal treatment'],
            'disciplinary': ['disciplinary', 'penalty', 'sanction'],
            'grievance': ['grievance', 'complaint'],
            'training': ['training', 'development'],
            'remote work': ['remote', 'work from home', 'flexible work'],
        }

        text_lower = text.lower()

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics
