import logging
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from app.config.settings import settings
from app.pipelines.components.azure_components import AzureEmbedder

logger = logging.getLogger(__name__)

class ProductionRAGQueryService:
    """Production-ready RAG query service with semantic, hybrid, and metadata search."""
    
    def __init__(self, embedder: AzureEmbedder):
        self.embedder = embedder
        self.search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        )
        logger.info("Production RAG Query Service initialized")
    
    # ===== POLICY QUERIES =====
    
    def get_all_policies(self, top: int = 100) -> List[Dict[str, Any]]:
        """Get all policy documents with fast server-side filtering."""
        try:
            results = self.search_client.search(
                search_text="*",
                filter="record_type eq 'policy'",  # Server-side filtering!
                select=["id", "description", "filename", "source", "toc"],
                top=top
            )
            
            policies = [dict(result) for result in results]
            logger.info(f"Retrieved {len(policies)} policies")
            return policies
            
        except Exception as e:
            logger.error(f"Failed to get all policies: {e}")
            return []
    
    def semantic_search_policies(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        """Semantic search on policy descriptions using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="description_vector"
            )
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter="record_type eq 'policy'",
                select=["id", "description", "filename", "@search.score"],
                top=top
            )
            
            policies = []
            for result in results:
                policy = dict(result)
                policy["search_score"] = result.get("@search.score", 0)
                policies.append(policy)
            
            logger.info(f"Semantic search found {len(policies)} policies for query: '{query}'")
            return policies
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def hybrid_search_policies(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword search on policies."""
        try:
            query_embedding = self.embedder.embed_query(query)
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="description_vector"
            )
            
            results = self.search_client.search(
                search_text=query,  # Keyword search
                vector_queries=[vector_query],  # Semantic search
                filter="record_type eq 'policy'",
                select=["id", "description", "filename", "@search.score"],
                top=top
            )
            
            policies = []
            for result in results:
                policy = dict(result)
                policy["search_score"] = result.get("@search.score", 0)
                policies.append(policy)
            
            logger.info(f"Hybrid search found {len(policies)} policies for query: '{query}'")
            return policies
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    # ===== CHUNK QUERIES =====
    
    def get_all_chunks_for_policy(self, filename: str, top: int = 1000) -> List[Dict[str, Any]]:
        """Get all chunks for a specific policy with fast server-side filtering."""
        try:
            results = self.search_client.search(
                search_text="*",
                filter=f"record_type eq 'chunk' and source eq '{filename}'",  # Server-side filtering!
                select=["id", "content", "topic", "level", "parent_topic", "is_subtopic"],
                top=top,
                order_by=["level asc", "topic asc"]  # Order by hierarchy
            )
            
            chunks = [dict(result) for result in results]
            logger.info(f"Retrieved {len(chunks)} chunks for policy '{filename}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for policy '{filename}': {e}")
            return []
    
    def get_chunk_by_topic_and_policy(self, topic: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get specific chunk by exact topic and policy match."""
        try:
            results = self.search_client.search(
                search_text="*",
                filter=f"record_type eq 'chunk' and topic eq '{topic}' and source eq '{filename}'",
                select=["id", "content", "topic", "level", "parent_topic", "is_subtopic"],
                top=1
            )
            
            for result in results:
                logger.info(f"Found chunk for topic '{topic}' in policy '{filename}'")
                return dict(result)
            
            logger.warning(f"No chunk found for topic '{topic}' in policy '{filename}'")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk by topic and policy: {e}")
            return None
    
    def semantic_search_chunks(self, query: str, filename: Optional[str] = None, top: int = 10) -> List[Dict[str, Any]]:
        """Semantic search on chunk content with optional policy filtering."""
        try:
            query_embedding = self.embedder.embed_query(query)
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
            
            # Build filter
            filter_clause = "record_type eq 'chunk'"
            if filename:
                filter_clause += f" and source eq '{filename}'"
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filter_clause,
                select=["id", "content", "topic", "source", "level", "parent_topic", "is_subtopic", "@search.score"],
                top=top
            )
            
            chunks = []
            for result in results:
                chunk = dict(result)
                chunk["search_score"] = result.get("@search.score", 0)
                chunks.append(chunk)
            
            logger.info(f"Semantic search found {len(chunks)} chunks for query: '{query}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunk search failed: {e}")
            return []
    
    def get_chunks_by_hierarchy(self, parent_topic: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks under a specific parent topic (hierarchical search)."""
        try:
            filter_clause = f"record_type eq 'chunk' and parent_topic eq '{parent_topic}'"
            if filename:
                filter_clause += f" and source eq '{filename}'"
            
            results = self.search_client.search(
                search_text="*",
                filter=filter_clause,
                select=["id", "content", "topic", "source", "level", "parent_topic", "is_subtopic"],
                order_by=["level asc", "topic asc"]
            )
            
            chunks = [dict(result) for result in results]
            logger.info(f"Found {len(chunks)} chunks under parent topic '{parent_topic}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks by hierarchy: {e}")
            return []
    
    def get_chunks_by_level(self, level: int, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks at a specific hierarchy level."""
        try:
            filter_clause = f"record_type eq 'chunk' and level eq {level}"
            if filename:
                filter_clause += f" and source eq '{filename}'"
            
            results = self.search_client.search(
                search_text="*",
                filter=filter_clause,
                select=["id", "content", "topic", "source", "level", "parent_topic", "is_subtopic"],
                order_by=["topic asc"]
            )
            
            chunks = [dict(result) for result in results]
            logger.info(f"Found {len(chunks)} chunks at level {level}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks by level: {e}")
            return []