import logging

from app.config.settings import settings
from app.prompts.manager import PromptManager
from app.pipelines.indexing_pipeline import IndexingPipeline

logger = logging.getLogger(__name__)

PROMPT_FILE_PATH = "app/prompts/prompts.v1.yml"

def get_pipeline() -> IndexingPipeline:
    prompt_manager = PromptManager(PROMPT_FILE_PATH)

    if settings.HOST_MODE == "local":
        from app.pipelines.components.local_components import (
            LocalDocxLoader, LocalEmbedder, LocalVectorStore, LocalLLM
        )
        return IndexingPipeline(
            loader=LocalDocxLoader(),
            embedder=LocalEmbedder(),
            vector_store=LocalVectorStore(),
            llm=LocalLLM(prompt_manager=prompt_manager, max_workers=settings.MAX_WORKERS),
            batch_size=settings.BATCH_SIZE,
        )
    elif settings.HOST_MODE == "azure":
        from app.pipelines.components.azure_components import (
            AzureBlobLoader, AzureEmbedder, AzureVectorStore, AzureLLM
        )
        return IndexingPipeline(
            loader=AzureBlobLoader(),
            embedder=AzureEmbedder(),
            vector_store=AzureVectorStore(),
            llm=AzureLLM(prompt_manager=prompt_manager),
            batch_size=settings.BATCH_SIZE,
        )
    else:
        raise ValueError(f"Invalid HOST_MODE: {settings.HOST_MODE}")

def run_indexing_pipeline():
    logger.info("Background indexing task started") 
    try:
        pipeline = get_pipeline()
        pipeline.run(
            data_path=settings.DATA_PATH,
            collection_name=settings.POLICY_CONTENT_COLLECTION_NAME
        )
    except Exception as e:
        logger.error(f"Background indexing task FAILED: {e}") 
