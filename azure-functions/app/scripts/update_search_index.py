import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_index_schema():
    """Update the search index to include regulation fields."""
    try:
        # Create index client
        credential = AzureKeyCredential(settings.AZURE_SEARCH_KEY)
        index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=credential
        )

        logger.info(f"Connecting to Azure Search: {settings.AZURE_SEARCH_ENDPOINT}")
        logger.info(f"Index: {settings.AZURE_SEARCH_INDEX_NAME}")

        # Get existing index
        try:
            existing_index = index_client.get_index(settings.AZURE_SEARCH_INDEX_NAME)
            logger.info(f"Found existing index with {len(existing_index.fields)} fields")
        except Exception as e:
            logger.error(f"Could not retrieve index: {e}")
            logger.error("Please ensure the index exists first")
            return False

        # Define new regulation fields to add
        new_fields = [
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                retrievable=True,
                sortable=False,
                facetable=False
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                filterable=False,
                retrievable=False,
                sortable=False,
                facetable=False,
                vector_search_dimensions=1536,
                vector_search_profile_name="default-vector-profile"
            ),
            SearchField(
                name="regulation_name",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                retrievable=True,
                sortable=False,
                facetable=True
            ),
            SearchField(
                name="article_number",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                retrievable=True,
                sortable=False,
                facetable=False
            ),
            SearchField(
                name="chapter",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                retrievable=True,
                sortable=False,
                facetable=False
            ),
            SearchField(
                name="regulation_type",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                retrievable=True,
                sortable=False,
                facetable=True
            ),
            SearchField(
                name="effective_date",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                retrievable=True,
                sortable=True,
                facetable=False
            ),
            SearchField(
                name="superseded_by",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                retrievable=True,
                sortable=False,
                facetable=False
            ),
            SearchField(
                name="topics",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                retrievable=True,
                sortable=False,
                facetable=True
            ),
            SearchField(
                name="references",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=False,
                retrievable=True,
                sortable=False,
                facetable=False
            ),
        ]

        # Check which fields already exist
        existing_field_names = {field.name for field in existing_index.fields}
        fields_to_add = []

        for field in new_fields:
            if field.name in existing_field_names:
                logger.info(f"✓ Field '{field.name}' already exists")
            else:
                logger.info(f"+ Adding field '{field.name}'")
                fields_to_add.append(field)

        if not fields_to_add:
            logger.info("\n✅ All regulation fields already exist in the index!")
            return True

        # Add new fields to existing index
        logger.info(f"\nAdding {len(fields_to_add)} new fields to the index...")

        existing_index.fields.extend(fields_to_add)

        # Update the index
        index_client.create_or_update_index(existing_index)

        logger.info("\n✅ Index schema updated successfully!")
        logger.info(f"Total fields in index: {len(existing_index.fields)}")

        return True

    except Exception as e:
        logger.error(f"\n❌ Failed to update index schema: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    logger.info("=" * 60)
    logger.info("Azure AI Search Index Schema Update")
    logger.info("=" * 60)
    logger.info("\nThis script will add regulation fields to your index.\n")

    success = update_index_schema()

    if success:
        logger.info("\n🎉 Schema update complete!")
        logger.info("\nYou can now run the regulation indexing endpoint:")
        logger.info("POST /api/v1/regulations/index")
        sys.exit(0)
    else:
        logger.error("\n❌ Schema update failed")
        logger.error("Please check the errors above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
