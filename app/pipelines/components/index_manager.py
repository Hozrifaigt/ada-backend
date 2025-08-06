import logging
from typing import Dict, Any, Optional, List
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
    SemanticPrioritizedFields,
    ComplexField
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

from app.config.settings import settings

logger = logging.getLogger(__name__)

class AzureSearchIndexManager:
    """Manages Azure Search index creation and schema for RAG system."""
    
    def __init__(self, index_name: Optional[str] = None):
        # Use provided index name or fall back to settings default
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
        """Create production-ready RAG index with proper schema."""
        target_index = index_name or self.index_name
        try:
            logger.info(f"Creating index: {target_index}")
            
            fields = [
                # Common fields
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="record_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                
                # Policy-specific fields
                SearchableField(name="description", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.microsoft"),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
                
                # Chunk-specific fields  
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.microsoft"),
                SearchableField(name="topic", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True),
                SimpleField(name="level", type=SearchFieldDataType.Int32, filterable=True, sortable=True, facetable=True),
                SearchableField(name="parent_topic", type=SearchFieldDataType.String, searchable=True, filterable=True),
                SimpleField(name="is_subtopic", type=SearchFieldDataType.Boolean, filterable=True, facetable=True),
                
                # Vector fields for semantic search
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="default-vector-profile"
                ),
                SearchField(
                    name="content_vector", 
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="default-vector-profile"
                ),
                
                # TOC field as complex object - FIX: Change to store structured data
                SimpleField(name="toc", type=SearchFieldDataType.String, searchable=False, filterable=False)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="default-hnsw-algorithm")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-algorithm"
                    )
                ]
            )
            
            # Configure semantic search for better hybrid search
            semantic_config = SemanticConfiguration(
                name="rag-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="topic"),
                    content_fields=[
                        SemanticField(field_name="description"),
                        SemanticField(field_name="content"),
                    ],
                    keywords_fields=[
                        SemanticField(field_name="source"),
                        SemanticField(field_name="parent_topic")
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
            
            # Create the index
            result = self.index_client.create_index(index)
            logger.info(f"Successfully created RAG index: {result.name}")
                       
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
            
            stats = {
                "name": index.name,
                "field_count": len(index.fields),
                "vector_search_enabled": index.vector_search is not None,
                "semantic_search_enabled": index.semantic_search is not None,
                "fields": [{"name": f.name, "type": str(f.type), "searchable": getattr(f, 'searchable', False)} for f in index.fields]
            }
            
            logger.info(f"Index stats for '{target_index}': {stats}")
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
            logger.info("Index deleted, creating new one...")
            return self.create_rag_index(target_index)
        else:
            logger.error("Failed to delete index, cannot recreate")
            return False
    
    def list_all_indexes(self) -> List[str]:
        """List all indexes in the Azure Search service."""
        try:
            indexes = self.index_client.list_indexes()
            index_names = [index.name for index in indexes]
            logger.info(f"Found {len(index_names)} indexes: {index_names}")
            return index_names
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

def main():
    """Main function to run index management."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    manager = AzureSearchIndexManager()
    
    # Check if index exists and create if needed
    if manager.ensure_index_exists():
        print("Index is ready!")
        
        # Show index stats
        stats = manager.get_index_stats()
        if stats:
            print(f"📊 Index has {stats['field_count']} fields")
            print(f"🔍 Vector search: {'✅' if stats['vector_search_enabled'] else '❌'}")
            print(f"🔍 Semantic search: {'✅' if stats['semantic_search_enabled'] else '❌'}")
    else:
        print("❌ Failed to create RAG index")

if __name__ == "__main__":
    main()