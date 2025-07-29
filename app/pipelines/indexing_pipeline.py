import logging
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from langchain_core.documents import Document

# Note: We now import the concrete classes, not a base class
from app.pipelines.components.file_loader import LocalDocxLoader
from app.pipelines.components.azure_components import AzureEmbedder, AzureVectorStore, AzureLLM

logger = logging.getLogger(__name__)

class IndexingPipeline:
    def __init__(
        self,
        loader: LocalDocxLoader,
        embedder: AzureEmbedder,
        vector_store: AzureVectorStore,
        llm: AzureLLM,
        batch_size: int = 32
    ):
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.batch_size = batch_size
        logger.info("IndexingPipeline initialized with concrete Azure components.")

    def run(self, data_path: str, collection_name: str):
        try:
            asyncio.run(self.arun(data_path, collection_name))
        except Exception as e:
            logger.critical("Indexing pipeline failed to run.", exc_info=True)

    async def arun(self, data_path: str, collection_name: str):
        logger.info("--- Starting Azure Indexing Pipeline Run ---")

        # Step 1: Setup Vector Store
        self.vector_store.setup(collection_name, self.embedder.vector_size)

        # Step 2: Discover and structure documents
        file_paths = self.loader.get_file_paths(data_path)
        if not file_paths:
            logger.warning("No document files found. Exiting.")
            return

        all_policy_records = []
        all_content_chunks = []

        for file_path in tqdm(file_paths, desc="Structuring Documents"):
            try:
                structured_data = self.loader.load_and_structure(file_path)
                
                # A. Prepare Policy Record (with full text for now)
                content_by_topic = structured_data.get("content_by_topic", {})
                full_document_text = "\n\n".join(content_by_topic.values())
                
                if not full_document_text:
                    logger.warning(f"Skipping empty document: {file_path.name}")
                    continue

                policy_metadata = structured_data.get("metadata", {})
                policy_metadata.update({"record_type": "policy", "toc": structured_data.get("toc", [])})
                all_policy_records.append(Document(page_content=full_document_text, metadata=policy_metadata))

                # B. Prepare Content Chunks
                for topic, text in content_by_topic.items():
                    chunk_metadata = {"source_filename": policy_metadata.get("filename"), "topic": topic, "record_type": "chunk"}
                    all_content_chunks.append(Document(page_content=text, metadata=chunk_metadata))

            except Exception as e:
                logger.error(f"Failed to structure file {file_path}. Skipping.", exc_info=True)
        
        logger.info(f"Created {len(all_policy_records)} policy records and {len(all_content_chunks)} content chunks.")
        
        processed_docs = []

        # Step 3A: Process Policy Records individually using Map-Reduce
        logger.info("Step 3A: Generating summaries for full policy documents...")
        for policy_record in tqdm(all_policy_records, desc="Summarizing Policies"):
            try:
                summary_text = await self.llm.asummarize_document(policy_record)
                
                policy_record.page_content = summary_text
                policy_record.metadata["summary"] = summary_text

                embedding = self.embedder.embed_documents([summary_text])[0]
                policy_record.metadata['embedding'] = embedding
                processed_docs.append(policy_record)
            except Exception as e:
                logger.error(f"Failed to process policy record {policy_record.metadata['filename']}.", exc_info=True)

        # Step 3B: Process Content Chunks (Sync)
        logger.info("Step 3B: Generating embeddings for content chunks...")
        
        for i in tqdm(range(0, len(all_content_chunks), self.batch_size), desc="Embedding Chunks"):
            batch_chunks = all_content_chunks[i : i + self.batch_size]
            
            # Embed the raw text of the chunks
            page_contents = [chunk.page_content for chunk in batch_chunks]
            embeddings = self.embedder.embed_documents(page_contents)

            for j, chunk in enumerate(batch_chunks):
                chunk.metadata['embedding'] = embeddings[j]
                processed_docs.append(chunk)

        # Step 4: Upsert All Processed Records
        logger.info(f"Step 4: Upserting a total of {len(processed_docs)} records.")
        if processed_docs:
            self.vector_store.upsert(processed_docs, collection_name)
        
        logger.info("--- Indexing Pipeline Finished Successfully ---")
