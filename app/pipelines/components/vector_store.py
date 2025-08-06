import uuid
import json
import logging


from typing import List, Dict, Any
from app.config.settings import settings
from app.pipelines.components.embedder import AzureEmbedder

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class AzureVectorStore:
    """Production-ready Azure AI Search store with proper JSON structure."""
    
    def __init__(self, embedder: AzureEmbedder):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.embedder = embedder
        
        logger.info(f"Initializing Production AzureVectorStore:")
        logger.info(f"Endpoint: {settings.AZURE_SEARCH_ENDPOINT}")
        logger.info(f"Index: {self.index_name}")
        
        try:
            self.search_client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=self.index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
            )
            logger.info("Production AzureVectorStore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AzureVectorStore: {e}")
            raise
    
    def upsert_policy_documents(self, policy_docs: List[Dict[str, Any]]) -> bool:
        """Store policy documents with proper structure for RAG."""
        if not policy_docs:
            logger.warning("No policy documents to store")
            return True
        
        logger.info(f"Storing {len(policy_docs)} policy documents")
        
        try:
            documents_to_upload = []
            
            for policy_doc in policy_docs:
                # Generate embedding for the description
                description = policy_doc['description']
                embedding = self.embedder.embed_documents([description])[0]
                
                # Convert TOC to JSON string to store complex object
                toc_json = json.dumps(policy_doc.get('toc', []))
                
                # Create properly structured document for Azure Search
                search_doc = {
                    "id": str(uuid.uuid4()),
                    "record_type": "policy",
                    "description": description,
                    "description_vector": embedding,
                    "source": policy_doc['source'],
                    "toc": toc_json,  # Store as JSON string
                    # Required fields that aren't used for policies but can't be null
                    "content": "",  # Empty string instead of None
                    "topic": "",    # Empty string instead of None
                    "level": 0,     # Default value instead of None
                    "parent_topic": "",  # Empty string instead of None
                    "is_subtopic": False,  # Default value instead of None
                    "content_vector": embedding  # Use same vector to avoid null
                }
                
                documents_to_upload.append(search_doc)
                logger.debug(f"Prepared policy: {policy_doc.get('filename', 'unknown')}")
            
            # Upload to Azure Search
            result = self.search_client.upload_documents(documents_to_upload)
            
            successful = sum(1 for item in result if item.succeeded)
            failed = len(result) - successful
            
            if failed > 0:
                logger.error(f"Failed to upload {failed} policy documents")
                for item in result:
                    if not item.succeeded:
                        logger.error(f"Policy upload error: {item.error_message}")
            
            logger.info(f"Successfully stored {successful} policy documents")
            return failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store policy documents: {e}")
            return False
    
    def upsert_chunk_documents(self, chunk_docs: List[Dict[str, Any]]) -> bool:
        """Store chunk documents with proper structure for RAG."""
        if not chunk_docs:
            logger.warning("No chunk documents to store")
            return True
        
        logger.info(f"Storing {len(chunk_docs)} chunk documents")
        
        try:
            # Use batch size from settings - same as main pipeline batch size
            batch_size = settings.BATCH_SIZE
            total_successful = 0
            total_failed = 0
            
            for i in range(0, len(chunk_docs), batch_size):
                batch = chunk_docs[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunk_docs)-1)//batch_size + 1}")
                
                # Generate embeddings for batch
                texts = [chunk['content'] for chunk in batch]
                embeddings = self.embedder.embed_documents(texts)
                
                documents_to_upload = []
                
                for j, chunk_doc in enumerate(batch):
                    # Create a dummy vector for description_vector (same as content_vector)
                    dummy_vector = embeddings[j]  # Use the same embedding as content
                    
                    search_doc = {
                        "id": str(uuid.uuid4()),
                        "record_type": "chunk",
                        "content": chunk_doc['content'],
                        "content_vector": embeddings[j],
                        "source": chunk_doc['source'],
                        "topic": chunk_doc['topic'],
                        "level": chunk_doc.get('level', 1),
                        "parent_topic": chunk_doc.get('parent_topic'),
                        "is_subtopic": chunk_doc.get('is_subtopic', False),
                        # Required fields that aren't used for chunks but can't be null
                        "description": "",  # Empty string instead of None
                        "description_vector": dummy_vector,  # Use dummy vector instead of None
                        "toc": "[]"  # Empty JSON array as string
                    }
                    
                    documents_to_upload.append(search_doc)
                
                # Upload batch to Azure Search
                result = self.search_client.upload_documents(documents_to_upload)
                
                batch_successful = sum(1 for item in result if item.succeeded)
                batch_failed = len(result) - batch_successful
                
                total_successful += batch_successful
                total_failed += batch_failed
                
                if batch_failed > 0:
                    logger.error(f"Failed to upload {batch_failed} chunks in batch")
                    for item in result:
                        if not item.succeeded:
                            logger.error(f"Chunk upload error: {item.error_message}")
            
            logger.info(f"Successfully stored {total_successful} chunk documents, {total_failed} failed")
            return total_failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store chunk documents: {e}")
            return False