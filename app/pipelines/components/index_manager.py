import logging

from app.config.settings import settings
from typing import Dict, Any, Optional, List

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticField,
    SemanticPrioritizedFields
)

logger = logging.getLogger(__name__)

class AzureSearchIndexManager:
    """Manages Azure Search index - simplified with vectors only for policy descriptions."""

    def __init__(self, index_name: Optional[str] = None):        
        self.index_name = index_name or settings.AZURE_SEARCH_INDEX_NAME
        self.index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        )
        logger.info(f"Azure Search Index Manager initialized for index: {self.index_name}")
    
    def check_index_exists(self, index_name: Optional[str] = None) -> bool:
        """Check if the specified index exists."""
        target_index = index_name or self.index_name
        try:
            self.index_client.get_index(target_index)
            logger.info(f"Index '{target_index}' exists")
            return True
        except ResourceNotFoundError:
            logger.info(f"Index '{target_index}' does not exist")
            return False
        except Exception as e:
            logger.error(f"Error checking if index '{target_index}' exists: {e}")
            return False
    
    def create_rag_index(self, index_name: Optional[str] = None) -> bool:
        """Create simplified RAG index - vectors only for policy descriptions."""
        target_index = index_name or self.index_name
        try:
            logger.info(f"Creating simplified index: {target_index}")
            
            fields = [
                # Common fields
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="record_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                
                # Policy-specific fields (WITH vector for similarity search)
                SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="description", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.microsoft"),
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=settings.AZURE_OPENAI_EMBEDDING_DIMENSIONS,
                    vector_search_profile_name="default-vector-profile"
                ),
                SimpleField(name="toc", type=SearchFieldDataType.String, searchable=False),
                SimpleField(name="topic_ids", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=False),
                
                # Topic-specific fields (NO vectors - retrieval by ID only)
                SimpleField(name="policy_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="topic", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True),
                SimpleField(name="topic_content", type=SearchFieldDataType.String, searchable=False),  # Not searchable, just storage
                SimpleField(name="subtopics", type=SearchFieldDataType.String, searchable=False)  # JSON string, no vectors
            ]
            
            # Vector search configuration (only for policy descriptions)
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-algorithm"
                    )
                ]
            )
            
            # Semantic search configuration (simplified - only for policy search)
            semantic_config = SemanticConfiguration(
                name="policy-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="filename"),
                    content_fields=[
                        SemanticField(field_name="description")
                    ],
                    keywords_fields=[
                        SemanticField(field_name="source")
                    ]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            # Create the index
            index = SearchIndex(
                name=target_index,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            result = self.index_client.create_index(index)
            logger.info(f"Successfully created simplified RAG index: {result.name}")
            logger.info("Note: Vector search enabled only for policy descriptions")
                       
            return True
            
        except Exception as e:
            logger.error(f"Failed to create RAG index '{target_index}': {e}")
            return False
    
    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Delete the specified index."""
        target_index = index_name or self.index_name
        try:
            if self.check_index_exists(target_index):
                self.index_client.delete_index(target_index)
                logger.info(f"Successfully deleted index: {target_index}")
                return True
            else:
                logger.info(f"Index '{target_index}' doesn't exist, nothing to delete")
                return True
        except Exception as e:
            logger.error(f"Failed to delete index '{target_index}': {e}")
            return False
    
    def get_index_stats(self, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get index statistics for the specified index."""
        target_index = index_name or self.index_name
        try:
            if not self.check_index_exists(target_index):
                return None
            
            index = self.index_client.get_index(target_index)
            
            # Count vector fields
            vector_fields = [f for f in index.fields 
                           if hasattr(f, 'vector_search_dimensions') 
                           and f.vector_search_dimensions is not None]
            
            stats = {
                "name": index.name,
                "field_count": len(index.fields),
                "vector_field_count": len(vector_fields),
                "vector_search_enabled": index.vector_search is not None,
                "semantic_search_enabled": index.semantic_search is not None,
                "fields": [
                    {
                        "name": f.name, 
                        "type": str(f.type), 
                        "searchable": getattr(f, 'searchable', False),
                        "filterable": getattr(f, 'filterable', False),
                        "has_vector": hasattr(f, 'vector_search_dimensions') and f.vector_search_dimensions is not None
                    } 
                    for f in index.fields
                ]
            }
            
            logger.info(f"Index stats for '{target_index}': {stats['field_count']} fields, "
                       f"{stats['vector_field_count']} vector fields, "
                       f"semantic search: {stats['semantic_search_enabled']}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats for '{target_index}': {e}")
            return None
    
    def ensure_index_exists(self, index_name: Optional[str] = None) -> bool:
        """Ensure the specified RAG index exists, create if it doesn't."""
        target_index = index_name or self.index_name
        if self.check_index_exists(target_index):
            logger.info(f"RAG index '{target_index}' already exists")
            return True
        else:
            logger.info(f"RAG index '{target_index}' doesn't exist, creating...")
            return self.create_rag_index(target_index)
    
    def recreate_index(self, index_name: Optional[str] = None) -> bool:
        """Delete and recreate the specified index."""
        target_index = index_name or self.index_name
        logger.warning(f"Recreating index '{target_index}' - ALL DATA WILL BE LOST")
        
        if self.delete_index(target_index):
            logger.info("Index deleted, creating new simplified index...")
            return self.create_rag_index(target_index)
        else:
            logger.error("Failed to delete index, cannot recreate")
            return False