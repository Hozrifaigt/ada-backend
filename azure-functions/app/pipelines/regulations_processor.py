"""
Pipeline for processing regulation PDFs and indexing them for retrieval.
"""
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from app.services.regulations_service import RegulationsService
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
            "decree_number": "33 of 2021",
            "supersedes": ["Federal Decree-Law No. 8 of 1980"],
            "superseded_by": None  # Partially superseded by amendments
        },
        "Federal Decree Law No. 20 of 2023 Amending Certain Provisions of Federal Decree Law No. 33 of 2021 Regarding the Regulation of Employment Relationships (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 20 of 2023",
            "regulation_type": "amendment",
            "effective_date": "2023-02-01",
            "decree_number": "20 of 2023",
            "supersedes": [],
            "superseded_by": None,
            "amends": "Federal Decree-Law No. 33 of 2021"
        },
        "Federal Decree-Law No. 9 of 2022 Concerning Domestic Workers (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 9 of 2022",
            "regulation_type": "primary",
            "effective_date": "2022-06-15",
            "decree_number": "9 of 2022",
            "supersedes": [],
            "superseded_by": None
        },
        "Federal Decree Law No. 21 of 2023 Amending Certain Provisions of Federal Decree Law No. 9 of 2022 Concerning Domestic Workers.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 21 of 2023",
            "regulation_type": "amendment",
            "effective_date": "2023-06-01",
            "decree_number": "21 of 2023",
            "supersedes": [],
            "superseded_by": None,
            "amends": "Federal Decree-Law No. 9 of 2022"
        },
        "Federal Decree-Law No. 13 of 2022 Concerning Unemployment Insurance Scheme.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 13 of 2022",
            "regulation_type": "primary",
            "effective_date": "2023-01-01",
            "decree_number": "13 of 2022",
            "supersedes": [],
            "superseded_by": None
        },
        "Cabinet Decision No. 106 of 2022 pertaining to the executive regulations of Federal Decree Law No. 9 of 2022.pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Cabinet Decision No. 106 of 2022",
            "regulation_type": "executive",
            "effective_date": "2022-09-01",
            "decree_number": "106 of 2022",
            "supersedes": [],
            "superseded_by": None,
            "implements": "Federal Decree-Law No. 9 of 2022"
        },
        "Cabinet Resolution _Executive Regulations Decree-Law No. 33 (1).pdf": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Cabinet Resolution - Executive Regulations for Decree-Law No. 33",
            "regulation_type": "executive",
            "effective_date": "2022-02-02",
            "decree_number": "Executive Regulations",
            "supersedes": [],
            "superseded_by": None,
            "implements": "Federal Decree-Law No. 33 of 2021"
        },
        "Federal Decree-Law No. 47 of 2021 (1).docx": {
            "regulation_category": "UAE Labor Law",
            "decree_name": "Federal Decree-Law No. 47 of 2021",
            "regulation_type": "primary",
            "effective_date": "2021-12-01",
            "decree_number": "47 of 2021",
            "supersedes": [],
            "superseded_by": None
        }
    }

    def __init__(
        self,
        regulations_dir: str = "/code/app/regulations",
        embedder: Optional[AzureEmbedder] = None,
        vector_store: Optional[AzureVectorStore] = None
    ):
        """
        Initialize the regulations processor.

        Args:
            regulations_dir: Directory containing regulation PDFs
            embedder: Embedder instance (will create if not provided)
            vector_store: Vector store instance (will create if not provided)
        """
        self.regulations_dir = Path(regulations_dir)
        self.regulations_service = RegulationsService()
        self.embedder = embedder or AzureEmbedder()
        self.vector_store = vector_store or AzureVectorStore(embedder=self.embedder)

        if not self.regulations_dir.exists():
            raise ValueError(f"Regulations directory not found: {regulations_dir}")

    async def process_all_regulations(self) -> RegulationsIndexStatus:
        """
        Process all regulation PDFs in the regulations directory.

        Returns:
            RegulationsIndexStatus with processing results
        """
        logger.info(f"Starting regulation processing from: {self.regulations_dir}")

        results = []
        total_chunks = 0

        # Get all PDF files
        pdf_files = list(self.regulations_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                result = await self.process_regulation(str(pdf_file))
                results.append(result)
                if result.success:
                    total_chunks += result.chunks_created
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results.append(RegulationProcessingResult(
                    regulation_name=pdf_file.name,
                    file_path=str(pdf_file),
                    chunks_created=0,
                    articles_found=0,
                    success=False,
                    error_message=str(e),
                    processing_time_seconds=0.0
                ))

        # Create index status
        status = RegulationsIndexStatus(
            total_regulations=len(results),
            total_chunks=total_chunks,
            last_updated=datetime.utcnow(),
            regulations=results
        )

        logger.info(f"Processing complete: {len(results)} regulations, {total_chunks} chunks")
        return status

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
                    "decree_number": None,
                    "supersedes": [],
                    "superseded_by": None
                }

            metadata = RegulationMetadata(
                file_path=str(pdf_path),
                **metadata_dict
            )

            # Extract text from PDF
            text = self.regulations_service.extract_text_from_pdf(str(pdf_path))

            if not text or len(text) < 100:
                raise ValueError("Insufficient text extracted from PDF")

            # Detect articles
            articles = self.regulations_service.detect_articles(text)
            articles_count = len(articles)

            # Create chunks
            chunks = self.regulations_service.create_chunks(
                text=text,
                regulation_metadata=metadata,
                chunk_size=800,
                overlap=100
            )

            if not chunks:
                raise ValueError("No chunks created from regulation")

            # Index chunks in vector store
            await self._index_chunks(chunks)

            processing_time = time.time() - start_time

            result = RegulationProcessingResult(
                regulation_name=metadata.regulation_name,
                file_path=str(pdf_path),
                chunks_created=len(chunks),
                articles_found=articles_count,
                success=True,
                error_message=None,
                processing_time_seconds=round(processing_time, 2)
            )

            logger.info(
                f"Successfully processed {metadata.regulation_name}: "
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

    async def _index_chunks(self, chunks: List[RegulationChunk]) -> None:
        """
        Index regulation chunks in the vector store.

        Args:
            chunks: List of regulation chunks to index
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
        Get list of active (non-superseded) regulation names.

        Returns:
            List of active regulation names
        """
        active = []

        for metadata in self.REGULATION_METADATA_MAP.values():
            # Include if not fully superseded
            if not metadata.get("superseded_by"):
                active.append(metadata["regulation_name"])

        return active
