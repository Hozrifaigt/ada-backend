import logging
import asyncio
import json
import uuid
from typing import List, Dict, Any

from tqdm.asyncio import tqdm
from app.pipelines.components.llm import AzureLLM
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.file_loader import LocalDocxLoader
from app.pipelines.components.vector_store import AzureVectorStore


logger = logging.getLogger(__name__)

class IndexingPipeline:
    """Simplified indexing pipeline - only embeds policy descriptions."""
    
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
        logger.info("Simplified IndexingPipeline initialized (no topic embeddings)")

    def run(self, data_path: str, collection_name: str):
        """Run pipeline synchronously."""
        try:
            asyncio.run(self.arun(data_path, collection_name))
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    async def arun(self, data_path: str, collection_name: str):
        """Main pipeline execution - simplified without topic embeddings."""
        logger.info("=== Starting Simplified RAG Indexing Pipeline ===")
        
        # Step 1: Load and structure documents
        logger.info("Step 1: Loading and structuring documents")
        file_paths = self.loader.get_file_paths(data_path)
        if not file_paths:
            logger.warning("No files found")
            return
        
        all_policy_docs = []
        all_topic_docs = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                # Process document using new structure
                structured_data = self.loader.load_and_structure(file_path)
                
                # Extract components
                toc = structured_data.get("toc", [])
                topics = structured_data.get("topics", [])
                full_text = structured_data.get("full_text", "")
                metadata = structured_data.get("metadata", {})
                
                if not full_text.strip():
                    logger.warning(f"Empty document: {file_path.name}")
                    continue
                
                # Process topic documents (no embeddings needed)
                policy_id = str(uuid.uuid4())
                processed_topic_docs = []
                
                for topic_data in topics:
                    # Simply create topic document without embeddings
                    topic_doc = {
                        "id": topic_data["id"],
                        "policy_id": policy_id,
                        "source": str(file_path),
                        "topic": topic_data["topic"],
                        "topic_content": topic_data.get("topic_content", ""),
                        "subtopics": json.dumps(topic_data.get("subtopics", []))  # Store as JSON string
                    }
                    processed_topic_docs.append(topic_doc)
                
                # Store topic IDs for policy reference
                topic_ids = [doc["id"] for doc in processed_topic_docs]
                
                # Create policy document (will add embedding later)
                policy_doc = {
                    "id": policy_id,
                    "filename": file_path.name,
                    "source": str(file_path),
                    "full_text": full_text,  # Temporary, for summary generation
                    "toc": toc,
                    "topic_ids": topic_ids,
                    "num_topics": metadata.get("num_topics", 0)
                }
                
                all_policy_docs.append(policy_doc)
                all_topic_docs.extend(processed_topic_docs)
                
                logger.info(f"Processed {file_path.name}: {len(processed_topic_docs)} topics")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_policy_docs)} policies with {len(all_topic_docs)} total topics")
        
        # Step 2: Generate summaries and embeddings for policy documents ONLY
        logger.info("Step 2: Generating policy summaries and embeddings")
        
        for policy_doc in tqdm(all_policy_docs, desc="Summarizing policies"):
            try:
                # Generate summary
                full_text = policy_doc.pop("full_text")  # Remove full text after using
                logger.info(f"Generating summary for {policy_doc['filename']}")
                summary = await self.llm.asummarize_document(full_text)
                
                policy_doc["description"] = summary
                
                # Generate embedding for policy summary ONLY
                policy_doc["description_vector"] = self.embedder.embed_query(summary)
                
                logger.info(f"Summarized {policy_doc['filename']}: {len(summary)} chars")
                
            except Exception as e:
                logger.error(f"Failed to summarize {policy_doc.get('filename', 'unknown')}: {e}")
                policy_doc["description"] = "Summary generation failed"
                policy_doc["description_vector"] = self.embedder.embed_query("Summary generation failed")
        
        # Step 3: Store policy documents in Azure Search
        logger.info("Step 3: Storing policy documents in Azure Search")
        policy_success = self.vector_store.upsert_policy_documents(all_policy_docs)
        
        if not policy_success:
            logger.error("Failed to store some policy documents")
        
        # Step 4: Store topic documents in Azure Search (no embeddings)
        logger.info("Step 4: Storing topic documents in Azure Search")
        topic_success = self.vector_store.upsert_topic_documents(all_topic_docs)
        
        if not topic_success:
            logger.error("Failed to store some topic documents")
        
        if policy_success and topic_success:
            logger.info("=== Simplified Pipeline Completed Successfully ===")
            logger.info(f"Stored {len(all_policy_docs)} policies and {len(all_topic_docs)} topics")
            logger.info("Note: Only policy descriptions have embeddings for similarity search")
        else:
            logger.warning("=== Pipeline completed with some errors ===")