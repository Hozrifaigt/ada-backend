import logging
import asyncio

from tqdm.asyncio import tqdm
from app.pipelines.components.llm import AzureLLM
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.file_loader import LocalDocxLoader
from app.pipelines.components.vector_store import AzureVectorStore


logger = logging.getLogger(__name__)

class IndexingPipeline:
    """Production-ready indexing pipeline optimized for RAG systems."""
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
        logger.info("Production IndexingPipeline initialized")

    def run(self, data_path: str, collection_name: str):
        """Run pipeline synchronously."""
        try:
            asyncio.run(self.arun(data_path, collection_name))
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    async def arun(self, data_path: str, collection_name: str):
        """Main production pipeline execution."""
        logger.info("=== Starting Production RAG Indexing Pipeline ===")
        
        # Step 1: Load and structure documents
        logger.info("Step 1: Loading and structuring documents")
        file_paths = self.loader.get_file_paths(data_path)
        if not file_paths:
            logger.warning("No files found")
            return
        
        all_policy_docs = []
        all_chunk_docs = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                structured_data = self.loader.load_and_structure(file_path)
                
                # Extract basic info
                content_by_topic = structured_data.get("content_by_topic", {})
                full_text = "\n\n".join(content_by_topic.values())
                
                if not full_text.strip():
                    logger.warning(f"Empty document: {file_path.name}")
                    continue
                
                metadata = structured_data.get("metadata", {})
                topic_hierarchy = structured_data.get("topic_hierarchy", {})
                
                # FIX: Create proper policy document structure
                policy_doc = {
                    "filename": file_path.name,  # Add filename field
                    "source": str(file_path),    # Keep full path as source
                    "full_text": full_text,      # Will be replaced with summary
                    "toc": structured_data.get("toc", []),
                }
                all_policy_docs.append(policy_doc)
                
                # Create chunk documents
                for topic, text in content_by_topic.items():
                    if not text.strip():
                        continue
                    
                    hierarchy_info = topic_hierarchy.get(topic, {})
                    
                    chunk_doc = {
                        "content": text,
                        "source": str(file_path),
                        "topic": topic,
                        "level": hierarchy_info.get("level", 1),
                        "parent_topic": hierarchy_info.get("parent_topic"),
                        "is_subtopic": hierarchy_info.get("is_subtopic", False)
                    }
                    all_chunk_docs.append(chunk_doc)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_policy_docs)} policies and {len(all_chunk_docs)} chunks")
        
        # Step 2: Generate summaries for policy documents
        logger.info("Step 2: Generating LLM summaries for policy documents")
        
        for policy_doc in tqdm(all_policy_docs, desc="Summarizing policies"):
            try:
                file_path = policy_doc["source"]
                full_text = policy_doc["full_text"]
                
                logger.info(f"Direct summarization for {file_path} ({len(text.split())} words)")
                summary = await self.llm.asummarize_document(full_text)
                
                policy_doc["description"] = summary
                policy_doc.pop("full_text")
                
                logger.info(f"Summarized {policy_doc['filename']}: {len(summary)} chars")
                
            except Exception as e:
                logger.error(f"Failed to summarize {policy_doc.get('filename', 'unknown')}: {e}")
                policy_doc["description"] = policy_doc.get("full_text", "")[:1000] + "..."
                policy_doc.pop("full_text", None)
        
        # Step 3: Store policy documents in Azure Search
        logger.info("Step 3: Storing policy documents in Azure Search")
        policy_success = self.vector_store.upsert_policy_documents(all_policy_docs)
        
        if not policy_success:
            logger.error("Failed to store some policy documents")
        
        # Step 4: Store chunk documents in Azure Search
        logger.info("Step 4: Storing chunk documents in Azure Search")
        chunk_success = self.vector_store.upsert_chunk_documents(all_chunk_docs)
        
        if not chunk_success:
            logger.error("Failed to store some chunk documents")
        

        if policy_success and chunk_success:
            logger.info("=== Pre-Processing Pipeline Completed Successfully ===")
            logger.info(f"Stored {len(all_policy_docs)} policies and {len(all_chunk_docs)} chunks")
        else:
            logger.warning("=== Pipeline completed with some errors ===")
        

        logger.info("RAG System Ready! Available capabilities:")