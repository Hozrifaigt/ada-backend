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
    """Simplified Azure AI Search store - vectors only for policy descriptions."""
    
    def __init__(self, embedder: AzureEmbedder):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.embedder = embedder
        
        logger.info(f"Initializing Simplified AzureVectorStore:")
        logger.info(f"Endpoint: {settings.AZURE_SEARCH_ENDPOINT}")
        logger.info(f"Index: {self.index_name}")
        
        try:
            self.search_client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=self.index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
            )
            logger.info("Simplified AzureVectorStore initialized (vectors for policies only)")
        except Exception as e:
            logger.error(f"Failed to initialize AzureVectorStore: {e}")
            raise
    
    def _create_base_document(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Create base document structure with required fields."""
        return {
            "id": doc_id or str(uuid.uuid4()),
            "record_type": "",
            # Policy fields
            "filename": "",
            "source": "",
            "description": "",
            "description_vector": [],
            "toc": "[]",
            "topic_ids": [],
            # Topic fields (no vectors)
            "policy_id": "",
            "topic": "",
            "topic_content": "",
            "subtopics": "[]"
        }
    
    def _create_policy_document(self, policy_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Create properly structured policy document for Azure Search."""
        search_doc = self._create_base_document(policy_doc.get("id"))
        
        # Serialize TOC to JSON string
        toc_json = json.dumps(policy_doc.get('toc', []))
        topic_ids = policy_doc.get('topic_ids', [])
        
        search_doc.update({
            "record_type": "policy",
            "filename": policy_doc.get('filename', ''),
            "source": policy_doc.get('source', ''),
            "description": policy_doc.get('description', ''),
            "description_vector": policy_doc.get('description_vector', []),
            "toc": toc_json,
            "topic_ids": topic_ids
        })
        
        return search_doc
    
    def _create_topic_document(self, topic_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Create topic document without vectors - retrieval by ID only."""
        search_doc = self._create_base_document(topic_doc.get("id"))
        
        # Subtopics already come as JSON string from pipeline
        subtopics_json = topic_doc.get('subtopics', '[]')
        if not isinstance(subtopics_json, str):
            subtopics_json = json.dumps(subtopics_json)
        
        search_doc.update({
            "record_type": "topic",
            "policy_id": topic_doc.get('policy_id', ''),
            "source": topic_doc.get('source', ''),
            "topic": topic_doc.get('topic', ''),
            "topic_content": topic_doc.get('topic_content', ''),
            "subtopics": subtopics_json
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
        """Store policy documents with embeddings."""
        if not policy_docs:
            logger.warning("No policy documents to store")
            return True
        
        logger.info(f"Storing {len(policy_docs)} policy documents (with embeddings)")
        
        try:
            documents_to_upload = []
            for policy_doc in policy_docs:
                search_doc = self._create_policy_document(policy_doc)
                documents_to_upload.append(search_doc)
                logger.debug(f"Prepared policy: {policy_doc.get('filename', 'unknown')}")
            
            result = self._upload_batch(documents_to_upload)
            
            logger.info(f"Policy upload complete: {result.successful} successful, {result.failed} failed "
                       f"(success rate: {result.success_rate:.2%})")
            return result.failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store policy documents: {e}")
            return False
    
    def upsert_topic_documents(self, topic_docs: List[Dict[str, Any]]) -> bool:
        """Store topic documents WITHOUT embeddings - retrieval by ID only."""
        if not topic_docs:
            logger.warning("No topic documents to store")
            return True
        
        logger.info(f"Storing {len(topic_docs)} topic documents (no embeddings)")
        
        try:
            batch_size = getattr(settings, 'BATCH_SIZE', 24)
            total_result = UploadResult(successful=0, failed=0, total=len(topic_docs))
            
            for i in range(0, len(topic_docs), batch_size):
                batch = topic_docs[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(topic_docs) - 1) // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                documents_to_upload = []
                for topic_doc in batch:
                    search_doc = self._create_topic_document(topic_doc)
                    documents_to_upload.append(search_doc)
                
                batch_result = self._upload_batch(documents_to_upload)
                
                total_result.successful += batch_result.successful
                total_result.failed += batch_result.failed
                
                logger.info(f"Batch {batch_num} complete: {batch_result.successful} successful, "
                           f"{batch_result.failed} failed")
            
            logger.info(f"Topic upload complete: {total_result.successful} successful, "
                       f"{total_result.failed} failed (success rate: {total_result.success_rate:.2%})")
            return total_result.failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store topic documents: {e}")
            return False
    
    def search_similar_policies(self, query_vector: List[float], threshold: float = 0.6, top_k: int = 5) -> List[Dict]:
        """
        Search for similar policies using vector similarity on descriptions.
        This is the ONLY vector search in the system.
        """
        try:
            from azure.search.documents.models import VectorizedQuery
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="description_vector"
            )
            
            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                filter="record_type eq 'policy'",
                select="id,filename,description,source,toc,topic_ids",
                top=top_k
            )
            
            processed_results = []
            for i, result in enumerate(results):
                # Parse TOC JSON
                toc = json.loads(result.get("toc", "[]"))
                
                # Calculate similarity score (rank-based for simplicity)
                similarity_score = max(95 - (i * 10), 10) / 100
                
                if similarity_score >= threshold:
                    processed_results.append({
                        "id": result["id"],
                        "filename": result["filename"],
                        "description": result["description"],
                        "source": result["source"],
                        "toc": toc,
                        "topic_ids": result.get("topic_ids", []),
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search similar policies: {e}")
            return []
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """
        Get a topic document by its ID - direct lookup, no vector search.
        """
        try:
            topic = self.search_client.get_document(key=topic_id)
            
            if topic and topic.get("record_type") == "topic":
                # Parse subtopics JSON
                topic["subtopics"] = json.loads(topic.get("subtopics", "[]"))
                return topic
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get topic {topic_id}: {e}")
            
            # Fallback to search
            try:
                results = list(self.search_client.search(
                    search_text="*",
                    filter=f"record_type eq 'topic' and id eq '{topic_id}'",
                    select="id,policy_id,topic,topic_content,subtopics,source",
                    top=1
                ))
                
                if results:
                    topic = results[0]
                    topic["subtopics"] = json.loads(topic.get("subtopics", "[]"))
                    return topic
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve topic by search: {e}")
                return None
    
    def get_topics_by_policy(self, policy_id: str) -> List[Dict]:
        """
        Get all topics for a specific policy using the stored topic_ids.
        Direct ID lookups only - no vector search.
        """
        try:
            # First get the policy document to retrieve topic IDs
            policy = self.search_client.get_document(key=policy_id)
            
            if not policy or policy.get("record_type") != "policy":
                logger.warning(f"Policy {policy_id} not found")
                return []
            
            topic_ids = policy.get("topic_ids", [])
            
            if not topic_ids:
                logger.warning(f"No topics found for policy {policy_id}")
                return []
            
            # Retrieve all topics by their IDs
            topics = []
            for topic_id in topic_ids:
                topic = self.get_topic_by_id(topic_id)
                if topic:
                    topics.append(topic)
                else:
                    logger.warning(f"Topic {topic_id} not found for policy {policy_id}")
            
            return topics
            
        except Exception as e:
            logger.error(f"Failed to get topics for policy {policy_id}: {e}")
            return []
    
    def get_policy_by_id(self, policy_id: str) -> Optional[Dict]:
        """
        Get a policy document by its ID.
        """
        try:
            policy = self.search_client.get_document(key=policy_id)
            
            if policy and policy.get("record_type") == "policy":
                # Parse TOC JSON
                policy["toc"] = json.loads(policy.get("toc", "[]"))
                return policy
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get policy {policy_id}: {e}")
            return None
    
    def get_all_policies(self, max_results: int = 50) -> List[Dict]:
        """
        Get all policy documents (no vector search, just listing).
        """
        try:
            results = self.search_client.search(
                search_text="*",
                filter="record_type eq 'policy'",
                select="id,filename,description,source,toc,topic_ids",
                top=max_results
            )
            
            policies = []
            for result in results:
                # Parse TOC JSON
                result["toc"] = json.loads(result.get("toc", "[]"))
                policies.append(result)
            
            return policies
            
        except Exception as e:
            logger.error(f"Failed to retrieve all policies: {e}")
            return []