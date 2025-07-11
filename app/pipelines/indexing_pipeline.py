import logging
import asyncio
from tqdm.asyncio import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.pipelines.base import BaseLoader, BaseEmbedder, BaseVectorStore, BaseLLM


logger = logging.getLogger(__name__)

class IndexingPipeline:
    """
    Orchestrates the end-to-end process of loading documents, processing them,
    and storing them in a vector database for retrieval.
    
    This pipeline is asynchronous and relies on component-based abstractions
    to remain agnostic to the underlying technologies (e.g., local vs. cloud).
    """
    def __init__(
        self,
        loader: BaseLoader,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        batch_size: int = 64
    ):
        """
        Initializes the pipeline with all its necessary components and configuration.
        
        Args:
            loader: A component that implements the BaseLoader interface for loading documents.
            embedder: A component that implements the BaseEmbedder interface for creating vectors.
            vector_store: A component that implements the BaseVectorStore interface for storage.
            llm: A component that implements the BaseLLM interface for metadata generation.
            batch_size: The number of documents to process in each batch.
        """
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.batch_size = batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        logger.info(
            f"IndexingPipeline initialized with batch_size={self.batch_size} "
            f"using components: {loader.__class__.__name__}, {embedder.__class__.__name__}, "
            f"{vector_store.__class__.__name__}, {llm.__class__.__name__}"
        )

    def run(self, data_path: str, collection_name: str):
        """
        Synchronous entrypoint for the pipeline.
        This wraps the main asynchronous `arun` method.
        """
        try:
            # Use asyncio.run() to execute the main async logic
            asyncio.run(self.arun(data_path, collection_name))
        except Exception as e:
            logger.critical("The indexing pipeline failed to run to completion.", exc_info=True)


    async def arun(self, data_path: str, collection_name: str):
        """
        The main asynchronous execution logic of the pipeline.
        """
        logger.info("Starting Indexing Pipeline Run")

        # Step 1: Setup Vector Store
        logger.info(f"Step 1: Setting up vector store collection: '{collection_name}'")
        self.vector_store.setup(collection_name, self.embedder.vector_size)

        # Step 2: Load Documents
        logger.info(f"Step 2: Loading documents from source: '{data_path}'")
        docs = list(self.loader.load(data_path))
        if not docs:
            logger.warning("No documents found at the source. Pipeline will exit.")
            return


        logger.info("Step 3: Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(docs)
        total_chunks = len(chunks)
        logger.info(f"Created {total_chunks} text chunks.")
        if total_chunks == 0:
            logger.warning("No text chunks were created after splitting. Pipeline will exit.")
            return

        logger.info(f"Step 4 & 5: Processing chunks in batches of {self.batch_size}...")
        
        for i in tqdm(range(0, total_chunks, self.batch_size), desc="Processing Batches"):
            # A. Get the current batch of chunks
            batch_chunks = chunks[i : i + self.batch_size]
            page_contents = [chunk.page_content for chunk in batch_chunks]

            # B. Generate Metadata using the LLM component
            try:
                metadatas = await self.llm.ainvoke_batch(page_contents)
            except Exception as e:
                logger.error(f"Metadata generation failed for batch starting at index {i}. Skipping batch.", exc_info=True)
                continue # Move to the next batch

            # C. Combine chunks with their new metadata
            for j, chunk in enumerate(batch_chunks):
                # .update() safely adds the new keys ('summary', 'keywords')
                chunk.metadata.update(metadatas[j])

            # D. Generate embeddings for the batch
            try:
                embeddings = self.embedder.embed_documents(page_contents)
            except Exception as e:
                logger.error(f"Embedding generation failed for batch starting at index {i}. Skipping batch.", exc_info=True)
                continue

            # E. Add embeddings to metadata and upsert to vector store
            for j, chunk in enumerate(batch_chunks):
                chunk.metadata['embedding'] = embeddings[j]
            
            try:
                self.vector_store.upsert(batch_chunks, collection_name)
                logger.debug(f"Successfully upserted batch starting at index {i}.")
            except Exception as e:
                logger.error(f"Upsert failed for batch starting at index {i}.", exc_info=True)
                # Continue to the next batch

        logger.info("--- Indexing Pipeline Finished Successfully ---")