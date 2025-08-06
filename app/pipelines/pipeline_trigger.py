import logging


from app.config.settings import settings
from app.prompts.manager import PromptManager

from app.pipelines.components.llm import AzureLLM
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.indexing_pipeline import IndexingPipeline
from app.pipelines.components.file_loader import LocalDocxLoader
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.index_manager import AzureSearchIndexManager


logger = logging.getLogger(__name__)

def get_pipeline() -> IndexingPipeline:
    """Constructs Azure indexing pipeline."""
    logger.info("Constructing Azure indexing pipeline")
    
    prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
    loader = LocalDocxLoader() 
    embedder = AzureEmbedder()
    vector_store = AzureVectorStore(embedder=embedder)
    llm = AzureLLM(prompt_manager=prompt_manager)
    
    return IndexingPipeline(
        loader=loader,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        batch_size=settings.BATCH_SIZE,
    )


def run_indexing_pipeline():
    """Run the indexing pipeline."""
    logger.info("Starting indexing pipeline")

    index_manager = AzureSearchIndexManager()
    if not index_manager.ensure_index_exists():
        logger.error("Failed to create/verify index")
        raise Exception("Index setup failed")

    try:
        pipeline = get_pipeline()
        pipeline.run(
            data_path=settings.DATA_PATH,
            collection_name=settings.AZURE_SEARCH_INDEX_NAME
        )
        logger.info("Indexing pipeline completed successfully")
    except Exception as e:
        logger.error(f"Indexing pipeline failed: {e}")
        raise