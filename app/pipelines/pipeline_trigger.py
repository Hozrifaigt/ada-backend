import logging

from app.prompts.manager import PromptManager
from app.config.settings import settings, PROMPT_FILE_PATH
from app.pipelines.indexing_pipeline import IndexingPipeline

from app.pipelines.components.file_loader import LocalDocxLoader
from app.pipelines.components.azure_components import (
        AzureEmbedder, AzureVectorStore, AzureLLM
    )

logger = logging.getLogger(__name__)


def get_pipeline() -> IndexingPipeline:

    logger.info("Constructing Azure-native indexing pipeline.")

    prompt_manager = PromptManager(PROMPT_FILE_PATH)

    loader = LocalDocxLoader() 
    embedder = AzureEmbedder()
    vector_store = AzureVectorStore(embedder_client=embedder.client)
    llm = AzureLLM(prompt_manager=prompt_manager)

    return IndexingPipeline(
        loader=loader,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        batch_size=settings.BATCH_SIZE,
    )


def run_indexing_pipeline():
    logger.info("Background indexing task started") 
    try:
        pipeline = get_pipeline()
        pipeline.run(
            data_path=settings.DATA_PATH,
            collection_name=settings.AZURE_SEARCH_INDEX_NAME
        )
    except Exception as e:
        logger.error(f"Background indexing task FAILED: {e}")
