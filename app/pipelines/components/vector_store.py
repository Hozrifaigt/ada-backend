import uuid
import json
import logging

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config.settings import settings
from app.pipelines.components.embedder import AzureEmbedder

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of document upload operation."""
    successful: int
    failed: int
    total: int
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0


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
    
    def _create_base_document(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Create base document structure with required fields."""
        return {
            "id": doc_id or str(uuid.uuid4()),
            "record_type": "",
            "content": "",
            "content_vector": [],
            "source": "",
            "topic": "",
            "level": 0,
            "parent_topic": "",
            "is_subtopic": False,
            "description": "",
            "description_vector": [],
            "toc": "[]"
        }
    
    def _create_policy_document(self, policy_doc: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """Create properly structured policy document for Azure Search."""
        search_doc = self._create_base_document()
        
        description = policy_doc['description']
        toc_json = json.dumps(policy_doc.get('toc', []))
        
        search_doc.update({
            "record_type": "policy",
            "description": description,
            "description_vector": embedding,
            "source": policy_doc['source'],
            "toc": toc_json,
            "content_vector": embedding
        })
        
        return search_doc
    
    def _create_chunk_document(self, chunk_doc: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """Create properly structured chunk document for Azure Search."""
        search_doc = self._create_base_document()
        
        search_doc.update({
            "record_type": "chunk",
            "content": chunk_doc['content'],
            "content_vector": embedding,
            "source": chunk_doc['source'],
            "topic": chunk_doc['topic'],
            "level": chunk_doc.get('level', 1),
            "parent_topic": chunk_doc.get('parent_topic', ""),
            "is_subtopic": chunk_doc.get('is_subtopic', False),
            "description_vector": embedding
        })
        
        return search_doc
    
    def _upload_batch(self, documents: List[Dict[str, Any]]) -> UploadResult:
        """Upload a batch of documents and return results."""
        try:
            result = self.search_client.upload_documents(documents)
            
            successful = sum(1 for item in result if item.succeeded)
            failed = len(result) - successful
            
            if failed > 0:
                logger.error(f"Failed to upload {failed} documents in batch")
                for item in result:
                    if not item.succeeded:
                        logger.error(f"Upload error for doc {getattr(item, 'key', 'unknown')}: {item.error_message}")
            
            return UploadResult(successful=successful, failed=failed, total=len(documents))
            
        except AzureError as e:
            logger.error(f"Azure error during batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))
        except Exception as e:
            logger.error(f"Unexpected error during batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))
    
    def upsert_policy_documents(self, policy_docs: List[Dict[str, Any]]) -> bool:
        """Store policy documents with proper structure for RAG."""
        if not policy_docs:
            logger.warning("No policy documents to store")
            return True
        
        logger.info(f"Storing {len(policy_docs)} policy documents")
        
        try:
            # Extract descriptions for embedding - single API call for all policies
            descriptions = [policy_doc['description'] for policy_doc in policy_docs]
            logger.info(f"Generating embeddings for {len(descriptions)} policy descriptions in one API call")
            
            embeddings = self.embedder.embed_documents(descriptions)
            
            if len(embeddings) != len(policy_docs):
                raise ValueError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(policy_docs)}")
            
            # Create documents with their corresponding embeddings
            documents_to_upload = []
            for policy_doc, embedding in zip(policy_docs, embeddings):
                search_doc = self._create_policy_document(policy_doc, embedding)
                documents_to_upload.append(search_doc)
                logger.debug(f"Prepared policy: {policy_doc.get('filename', policy_doc.get('source', 'unknown'))}")
            
            # Upload documents
            result = self._upload_batch(documents_to_upload)
            
            logger.info(f"Policy upload complete: {result.successful} successful, {result.failed} failed "
                       f"(success rate: {result.success_rate:.2%})")
            return result.failed == 0
            
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
            batch_size = getattr(settings, 'BATCH_SIZE', 24)
            total_result = UploadResult(successful=0, failed=0, total=len(chunk_docs))
            
            for i in range(0, len(chunk_docs), batch_size):
                batch = chunk_docs[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunk_docs) - 1) // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Generate embeddings for batch
                texts = [chunk['content'] for chunk in batch]
                embeddings = self.embedder.embed_documents(texts)
                
                # Create documents
                documents_to_upload = []
                for chunk_doc, embedding in zip(batch, embeddings):
                    search_doc = self._create_chunk_document(chunk_doc, embedding)
                    documents_to_upload.append(search_doc)
                
                # Upload batch
                batch_result = self._upload_batch(documents_to_upload)
                
                # Accumulate results
                total_result.successful += batch_result.successful
                total_result.failed += batch_result.failed
                
                logger.info(f"Batch {batch_num} complete: {batch_result.successful} successful, "
                           f"{batch_result.failed} failed")
            
            logger.info(f"Chunk upload complete: {total_result.successful} successful, "
                       f"{total_result.failed} failed (success rate: {total_result.success_rate:.2%})")
            return total_result.failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store chunk documents: {e}")
            return False