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

    CHAPTER_PATTERNS = [
        r'Chapter\s+(\d+[:\s]+.*?)(?:\n|$)',  # Chapter 3: Employment Contracts
        r'CHAPTER\s+(\d+[:\s]+.*?)(?:\n|$)',
        r'Section\s+(\d+[:\s]+.*?)(?:\n|$)',  # Section 5: Leave Entitlements
    ]

    # Patterns for detecting cross-references
    REFERENCE_PATTERNS = [
        r'subject to Article\s+(\d+)',
        r'as per provisions of Article\s+(\d+)',
        r'in accordance with Article\s+(\d+)',
        r'notwithstanding Article\s+(\d+)',
        r'pursuant to Article\s+(\d+)',
        r'referred to in Article\s+(\d+)',
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

    def detect_chapters(self, text: str) -> List[Dict[str, any]]:
        """
        Detect chapter/section boundaries in the text.

        Args:
            text: The regulation text

        Returns:
            List of dictionaries with chapter information
        """
        chapters = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            for pattern in self.CHAPTER_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    chapters.append({
                        'chapter_title': match.group(0).strip(),
                        'line_number': i
                    })
                    break

        logger.info(f"Detected {len(chapters)} chapters in text")
        return chapters

    def extract_cross_references(self, text: str) -> List[str]:
        """
        Extract article cross-references from text.

        Args:
            text: The article text

        Returns:
            List of referenced article numbers
        """
        references = set()

        for pattern in self.REFERENCE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                references.add(f"Article {match}")

        return list(references)

    def create_chunks(
        self,
        text: str,
        regulation_metadata: RegulationMetadata,
        chunk_size: int = 800,
        overlap: int = 100
    ) -> List[RegulationChunk]:
        """
        Create intelligent chunks from regulation text.

        Chunks are article-aware: they don't split mid-article.

        Args:
            text: The regulation text
            regulation_metadata: Metadata about the regulation
            chunk_size: Target chunk size in tokens (approximate by words)
            overlap: Overlap between chunks in tokens

        Returns:
            List of RegulationChunk objects
        """
        articles = self.detect_articles(text)
        chapters = self.detect_chapters(text)
        lines = text.split('\n')

        chunks = []

        # If no articles detected, use simple chunking
        if not articles:
            logger.warning("No articles detected, using simple chunking")
            return self._simple_chunk(text, regulation_metadata, chunk_size, overlap)

        # Find the current chapter for each article
        article_chapters = self._map_articles_to_chapters(articles, chapters)

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

            # Skip if too short
            if len(article_text) < 50:
                continue

            # Approximate token count (1 token ≈ 0.75 words)
            word_count = len(article_text.split())

            # If article fits in one chunk, create single chunk
            if word_count <= chunk_size:
                chunk = self._create_chunk(
                    content=article_text,
                    regulation_metadata=regulation_metadata,
                    article_number=article['article_number'],
                    chapter=article_chapters.get(i)
                )
                chunks.append(chunk)
            else:
                # Split large article into multiple chunks with overlap
                sub_chunks = self._split_large_article(
                    article_text,
                    article['article_number'],
                    regulation_metadata,
                    article_chapters.get(i),
                    chunk_size,
                    overlap
                )
                chunks.extend(sub_chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(articles)} articles")
        return chunks

    def _map_articles_to_chapters(
        self,
        articles: List[Dict],
        chapters: List[Dict]
    ) -> Dict[int, str]:
        """Map each article to its chapter."""
        article_to_chapter = {}

        for i, article in enumerate(articles):
            article_line = article['line_number']

            # Find the most recent chapter before this article
            current_chapter = None
            for chapter in reversed(chapters):
                if chapter['line_number'] < article_line:
                    current_chapter = chapter['chapter_title']
                    break

            if current_chapter:
                article_to_chapter[i] = current_chapter

        return article_to_chapter

    def _create_chunk(
        self,
        content: str,
        regulation_metadata: RegulationMetadata,
        article_number: Optional[str] = None,
        chapter: Optional[str] = None
    ) -> RegulationChunk:
        """Create a single regulation chunk."""
        chunk_id = str(uuid.uuid4())
        references = self.extract_cross_references(content)

        # Auto-tag topics based on keywords
        topics = self._extract_topics(content)

        return RegulationChunk(
            id=chunk_id,
            content=content,
            regulation_category=regulation_metadata.regulation_category,
            decree_name=regulation_metadata.decree_name,
            article_number=article_number,
            chapter=chapter,
            regulation_type=regulation_metadata.regulation_type,
            effective_date=regulation_metadata.effective_date,
            superseded_by=regulation_metadata.superseded_by,
            topics=topics,
            references=references
        )

    def _split_large_article(
        self,
        article_text: str,
        article_number: str,
        regulation_metadata: RegulationMetadata,
        chapter: Optional[str],
        chunk_size: int,
        overlap: int
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
                chapter=chapter
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
                article_number=f"Section {chunk_num}",
                chapter=None
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
