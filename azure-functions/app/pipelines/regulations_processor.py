"""
Pipeline for processing regulation PDFs and indexing them for retrieval.
"""
import asyncio
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from azure.storage.blob import BlobServiceClient

from app.services.regulations_service import RegulationsService
from app.config.settings import settings
from app.models.regulation_models import (
    RegulationMetadata,
    RegulationChunk,
    RegulationProcessingResult,
    RegulationsIndexStatus
)
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)


class RegulationsProcessor:
    """
    Pipeline for processing regulation PDFs and indexing them.
    """

    # Mapping of regulation files to metadata
    # This is manually curated to handle amendment relationships
    REGULATION_METADATA_MAP = {
        "Federal Decree-Law No. 33 of 2021 Regarding the Regulation of Employment Relationship and its amendments (2).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 33 of 2021",
            "regulation_type": "primary",
            "effective_date": "2022-02-02",
            "decree_number": "33 of 2021"
        },
        "Federal Decree Law No. 20 of 2023 Amending Certain Provisions of Federal Decree Law No. 33 of 2021 Regarding the Regulation of Employment Relationships (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 20 of 2023",
            "regulation_type": "amendment",
            "effective_date": "2023-02-01",
            "decree_number": "20 of 2023",
            "amends": "Federal Decree-Law No. 33 of 2021"
        },
        "Federal Decree-Law No. 9 of 2022 Concerning Domestic Workers (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 9 of 2022",
            "regulation_type": "primary",
            "effective_date": "2022-06-15",
            "decree_number": "9 of 2022"
        },
        "Federal Decree Law No. 21 of 2023 Amending Certain Provisions of Federal Decree Law No. 9 of 2022 Concerning Domestic Workers.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 21 of 2023",
            "regulation_type": "amendment",
            "effective_date": "2023-06-01",
            "decree_number": "21 of 2023",
            "amends": "Federal Decree-Law No. 9 of 2022"
        },
        "Federal Decree-Law No. 13 of 2022 Concerning Unemployment Insurance Scheme.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 13 of 2022",
            "regulation_type": "primary",
            "effective_date": "2023-01-01",
            "decree_number": "13 of 2022"
        },
        "Cabinet Decision No. 106 of 2022 pertaining to the executive regulations of Federal Decree Law No. 9 of 2022.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Cabinet Decision No. 106 of 2022",
            "regulation_type": "executive",
            "effective_date": "2022-09-01",
            "decree_number": "106 of 2022",
            "implements": "Federal Decree-Law No. 9 of 2022"
        },
        "Cabinet Resolution _Executive Regulations Decree-Law No. 33 (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Cabinet Resolution - Executive Regulations for Decree-Law No. 33",
            "regulation_type": "executive",
            "effective_date": "2022-02-02",
            "decree_number": "Executive Regulations",
            "implements": "Federal Decree-Law No. 33 of 2021"
        },
        "Federal Decree-Law No. 47 of 2021 (1).docx": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 47 of 2021",
            "regulation_type": "primary",
            "effective_date": "2021-12-01",
            "decree_number": "47 of 2021"
        }
    }

    def __init__(
        self,
        embedder: Optional[AzureEmbedder] = None,
        vector_store: Optional[AzureVectorStore] = None,
        llm_service = None
    ):
        """
        Initialize the regulations processor.

        Args:
            embedder: Embedder instance (will create if not provided)
            vector_store: Vector store instance (will create if not provided)
            llm_service: LLM service for topic extraction (will create if not provided)
        """
        self.regulations_service = RegulationsService()
        self.embedder = embedder or AzureEmbedder()
        self.vector_store = vector_store or AzureVectorStore(embedder=self.embedder)

        # Azure Blob Storage client for regulations container
        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self.blob_service_client.get_container_client(
            settings.AZURE_STORAGE_REGULATIONS_CONTAINER_NAME
        )

        # LLM service for enhanced topic extraction
        if llm_service is None:
            try:
                from app.pipelines.components.llm import AzureLLM
                from app.prompts.manager import PromptManager
                self.llm_service = AzureLLM(prompt_manager=PromptManager(settings.PROMPT_MANAGER_PATH))
            except Exception as e:
                logger.warning(f"Could not initialize LLM service for topic extraction: {e}. Using keyword-based extraction.")
                self.llm_service = None
        else:
            self.llm_service = llm_service

    def _download_blobs_to_temp(self) -> tuple[Path, List[Path]]:
        """
        Download all PDF files from the Azure Blob Storage regulations container to a temp directory.

        Returns:
            Tuple of (temp_dir path, list of downloaded file paths)
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="regulations_"))
        downloaded_files = []

        blobs = self.container_client.list_blobs()
        for blob in blobs:
            if blob.name.lower().endswith(".pdf"):
                local_path = temp_dir / blob.name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blob_client = self.container_client.get_blob_client(blob.name)
                with open(local_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                downloaded_files.append(local_path)
                logger.info(f"Downloaded blob: {blob.name}")

        return temp_dir, downloaded_files

    async def process_all_regulations(self, max_concurrent_pdfs: int = 3) -> RegulationsIndexStatus:
        """
        Process all regulation PDFs from Azure Blob Storage in parallel.

        Args:
            max_concurrent_pdfs: Maximum number of PDFs to process concurrently (default 3)

        Returns:
            RegulationsIndexStatus with processing results
        """
        logger.info(f"Starting regulation processing from Azure Blob Storage container: {settings.AZURE_STORAGE_REGULATIONS_CONTAINER_NAME}")

        # Download blobs to temp directory
        temp_dir, pdf_files = self._download_blobs_to_temp()
        logger.info(f"Downloaded {len(pdf_files)} PDF files to {temp_dir} (max {max_concurrent_pdfs} concurrent)")

        try:
            # Use semaphore to limit concurrent PDF processing
            semaphore = asyncio.Semaphore(max_concurrent_pdfs)

            async def process_with_semaphore(pdf_file: Path) -> RegulationProcessingResult:
                async with semaphore:
                    try:
                        return await self.process_regulation(str(pdf_file))
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_file.name}: {e}")
                        return RegulationProcessingResult(
                            regulation_name=pdf_file.name,
                            file_path=str(pdf_file),
                            chunks_created=0,
                            articles_found=0,
                            success=False,
                            error_message=str(e),
                            processing_time_seconds=0.0
                        )

            # Process all PDFs in parallel (limited by semaphore)
            results = await asyncio.gather(*[
                process_with_semaphore(pdf_file) for pdf_file in pdf_files
            ])

            # Calculate totals
            total_chunks = sum(r.chunks_created for r in results if r.success)

            # Create index status
            status = RegulationsIndexStatus(
                total_regulations=len(results),
                total_chunks=total_chunks,
                last_updated=datetime.utcnow(),
                regulations=list(results)
            )

            logger.info(f"Processing complete: {len(results)} regulations, {total_chunks} chunks")
            return status

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {temp_dir}")

    async def process_regulation(self, pdf_path: str) -> RegulationProcessingResult:
        """
        Process a single regulation PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            RegulationProcessingResult with processing details
        """
        start_time = time.time()
        pdf_file = Path(pdf_path)

        logger.info(f"Processing regulation: {pdf_file.name}")

        try:
            # Get metadata for this regulation
            metadata_dict = self.REGULATION_METADATA_MAP.get(pdf_file.name)
            if not metadata_dict:
                logger.warning(f"No metadata mapping for {pdf_file.name}, using defaults")
                metadata_dict = {
                    "regulation_category": "Unknown",
                    "decree_name": pdf_file.stem,
                    "regulation_type": "primary",
                    "effective_date": None,
                    "decree_number": None
                }

            metadata = RegulationMetadata(
                file_path=str(pdf_path),
                **metadata_dict
            )

            # Extract text from PDF with page numbers
            pages_content = self.regulations_service.extract_text_with_pages(str(pdf_path))
            text = "\n".join(page_text for _, page_text in pages_content)

            if not text or len(text) < 100:
                raise ValueError("Insufficient text extracted from PDF")

            # Detect articles
            articles = self.regulations_service.detect_articles(text)
            articles_count = len(articles)

            # Create chunks with page tracking
            chunks = self.regulations_service.create_chunks(
                text=text,
                regulation_metadata=metadata,
                chunk_size=800,
                overlap=100,
                pages_content=pages_content
            )

            if not chunks:
                raise ValueError("No chunks created from regulation")

            # Enhance chunks with LLM-based topic extraction
            enhanced_chunks = await self._enhance_chunks_with_topics(chunks, metadata.decree_name)

            # Index enhanced chunks in vector store
            await self._index_chunks(enhanced_chunks)

            processing_time = time.time() - start_time

            result = RegulationProcessingResult(
                regulation_name=metadata.decree_name,
                file_path=str(pdf_path),
                chunks_created=len(chunks),
                articles_found=articles_count,
                success=True,
                error_message=None,
                processing_time_seconds=round(processing_time, 2)
            )

            logger.info(
                f"Successfully processed {metadata.decree_name}: "
                f"{len(chunks)} chunks, {articles_count} articles, "
                f"{processing_time:.2f}s"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {pdf_file.name}: {e}")

            return RegulationProcessingResult(
                regulation_name=pdf_file.name,
                file_path=str(pdf_path),
                chunks_created=0,
                articles_found=0,
                success=False,
                error_message=str(e),
                processing_time_seconds=round(processing_time, 2)
            )

    async def _enhance_chunks_with_topics(
        self,
        chunks: List[RegulationChunk],
        decree_name: str,
        batch_size: int = 25
    ) -> List[RegulationChunk]:
        """
        Enhance chunks with LLM-based topic extraction in parallel batches.

        Uses LLM to extract 5+ main topics from each chunk,
        improving retrieval accuracy for policy generation.

        Args:
            chunks: List of regulation chunks
            decree_name: Name of the decree for context
            batch_size: Number of chunks to process in parallel per batch (default 25)

        Returns:
            Enhanced chunks with improved topics
        """
        if not self.llm_service:
            logger.info("LLM service not available, using keyword-based topics")
            return chunks

        logger.info(f"Enhancing {len(chunks)} chunks with LLM topics (batch_size={batch_size})")
        enhanced_chunks = []

        # Process chunks in parallel batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            # Create tasks for parallel processing within the batch
            async def enhance_single_chunk(chunk: RegulationChunk) -> RegulationChunk:
                try:
                    topics = await self._extract_topics_with_llm(
                        content=chunk.content,
                        article_number=chunk.article_number,
                        decree_name=decree_name
                    )
                    chunk.topics = topics
                except Exception as e:
                    logger.warning(f"Failed to enhance topics for chunk {chunk.id}: {e}. Using original topics.")
                return chunk

            # Process batch in parallel
            batch_results = await asyncio.gather(*[
                enhance_single_chunk(chunk) for chunk in batch
            ])
            enhanced_chunks.extend(batch_results)

            logger.info(f"Enhanced topics for {len(enhanced_chunks)}/{len(chunks)} chunks")

        logger.info(f"Enhanced {len(enhanced_chunks)} chunks with LLM-extracted topics")
        return enhanced_chunks

    async def _extract_topics_with_llm(
        self,
        content: str,
        article_number: Optional[str],
        decree_name: str
    ) -> List[str]:
        """
        Extract 5+ main topics from regulation content using LLM.
        
        Args:
            content: The regulation content
            article_number: Article number (if available)
            decree_name: Name of the decree
            
        Returns:
            List of 5-10 topic keywords
        """
        # Limit content size for LLM processing
        content_preview = content[:1500] if len(content) > 1500 else content
        
        prompt = f"""You are analyzing a regulation article for topic extraction to enable policy search.

Regulation: {decree_name}
{f'Article: {article_number}' if article_number else ''}

Content preview:
{content_preview}

Extract 8-12 topics from this regulation article using BOTH perspectives:

1. LEGAL/REGULATORY TERMS: How this is referenced in official legal documents
   (formal terminology, article references, legal concepts)

2. BUSINESS/POLICY TERMS: How professionals would search for this when writing policies
   (common business terms, policy names, practical concepts)

Include BOTH legal terminology AND business terminology for each concept covered.
This helps match policy searches to regulatory content.

Return ONLY a JSON array of strings, no other text.
Example format: ["legal term 1", "business term 1", "legal term 2", "business term 2", ...]"""
        
        try:
            response = await self.llm_service.generate(prompt)
            
            # Parse JSON response
            import json
            topics = json.loads(response)
            
            # Ensure we have at least 5 topics
            if not isinstance(topics, list):
                topics = [topics] if topics else []
                
            # Limit to top 12 topics
            topics = topics[:12] if len(topics) > 12 else topics
            
            # Fallback to keyword-based if LLM returns too few
            if len(topics) < 5:
                keyword_topics = self.regulations_service._extract_topics(content)
                topics.extend(keyword_topics)
                topics = list(set(topics))[:12]  # Deduplicate and limit
            
            logger.debug(f"Extracted {len(topics)} topics: {topics}")
            return topics
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM topic response: {e}. Using keyword extraction.")
            return self.regulations_service._extract_topics(content)
        except Exception as e:
            logger.warning(f"LLM topic extraction failed: {e}. Falling back to keyword extraction.")
            return self.regulations_service._extract_topics(content)

    async def _index_chunks(self, chunks: List[RegulationChunk]) -> None:
        """
        Index regulation chunks using hybrid vectors.
        
        Creates topic-enriched embeddings by concatenating:
        - topics (LLM-extracted 5-10 main concepts)
        - decree_name (regulation source)
        - article_number (section reference)
        - content (actual text)
        
        This enables both BM25 keyword search (with topic field boosting ^10)
        and semantic vector search for improved retrieval accuracy.

        Args:
            chunks: List of enhanced regulation chunks to index
        """
        try:
            logger.info(f"Indexing {len(chunks)} chunks in vector store")

            # Index the chunks
            success = await self.vector_store.index_regulations(chunks)

            if not success:
                raise Exception("Failed to index regulation chunks in vector store")

            logger.info(f"Successfully indexed {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise

    def get_amendment_chain(self, regulation_name: str) -> List[str]:
        """
        Get the amendment chain for a regulation.

        Args:
            regulation_name: Name of the regulation

        Returns:
            List of regulation names in the amendment chain (newest to oldest)
        """
        chain = [regulation_name]

        # Find all files that amend or are amended by this regulation
        for filename, metadata in self.REGULATION_METADATA_MAP.items():
            if metadata.get("amends") == regulation_name:
                chain.insert(0, metadata["regulation_name"])
            elif metadata["regulation_name"] == regulation_name:
                if "amends" in metadata:
                    parent = metadata["amends"]
                    chain.append(parent)

        return chain

    def get_active_regulations(self) -> List[str]:
        """
        Get list of all regulation names.

        Returns:
            List of regulation names
        """
        return [metadata["decree_name"] for metadata in self.REGULATION_METADATA_MAP.values()]
