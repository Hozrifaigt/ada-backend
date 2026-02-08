"""
Script to index regulation PDFs into Azure AI Search.

Usage:
    python -m app.scripts.index_regulations
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipelines.regulations_processor import RegulationsProcessor
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def index_regulations():
    """Main function to process and index regulations."""
    logger.info("=" * 80)
    logger.info("REGULATION INDEXING SCRIPT")
    logger.info("=" * 80)

    try:
        # Initialize components
        logger.info("Initializing embedder and vector store...")
        embedder = AzureEmbedder()
        vector_store = AzureVectorStore(embedder=embedder)

        # Initialize regulations processor
        regulations_dir = settings.REGULATIONS_PATH
        logger.info(f"Regulations directory: {regulations_dir}")

        processor = RegulationsProcessor(
            regulations_dir=regulations_dir,
            embedder=embedder,
            vector_store=vector_store
        )

        # Process all regulations
        logger.info("\nStarting regulation processing...")
        logger.info("-" * 80)

        status = await processor.process_all_regulations()

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total regulations processed: {status.total_regulations}")
        logger.info(f"Total chunks created: {status.total_chunks}")
        logger.info(f"Last updated: {status.last_updated}")
        logger.info("")

        # Display individual regulation results
        logger.info("Individual Results:")
        logger.info("-" * 80)

        successful = 0
        failed = 0

        for result in status.regulations:
            status_symbol = "✓" if result.success else "✗"
            logger.info(f"\n{status_symbol} {result.regulation_name}")
            logger.info(f"  File: {Path(result.file_path).name}")

            if result.success:
                logger.info(f"  Chunks: {result.chunks_created}")
                logger.info(f"  Articles: {result.articles_found}")
                logger.info(f"  Time: {result.processing_time_seconds}s")
                successful += 1
            else:
                logger.error(f"  Error: {result.error_message}")
                failed += 1

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Successful: {successful}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"Total chunks indexed: {status.total_chunks}")

        if failed > 0:
            logger.warning(f"\n⚠️  {failed} regulation(s) failed to process")
            logger.warning("Please review the errors above and fix any issues")
            return False
        else:
            logger.info("\n✅ All regulations processed successfully!")
            return True

    except Exception as e:
        logger.error(f"\n❌ Fatal error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Entry point for the script."""
    logger.info("\nStarting regulation indexing...")
    logger.info(f"Azure Search Endpoint: {settings.AZURE_SEARCH_ENDPOINT}")
    logger.info(f"Azure Search Index: {settings.AZURE_SEARCH_INDEX_NAME}\n")

    # Check if pdfplumber is installed
    try:
        import pdfplumber
        logger.info("✓ pdfplumber is installed")
    except ImportError:
        logger.error("✗ pdfplumber is not installed")
        logger.error("Please install with: pip install pdfplumber")
        sys.exit(1)

    # Run the async function
    success = asyncio.run(index_regulations())

    if success:
        logger.info("\n🎉 Regulation indexing completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Regulation indexing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
