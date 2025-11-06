import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
from app.models.draft_models import ClientMetadata, DraftMetadata, CompleteDraft

logger = logging.getLogger(__name__)


class DraftStorageService:
    """Azure Table Storage service for temporary draft management."""
    
    def __init__(self, connection_string: str, table_name: str = "Drafts"):
        """Initialize Azure Table Storage client."""
        self.table_name = table_name
        self.connection_string = connection_string
        
        try:
            self.table_service = TableServiceClient.from_connection_string(
                conn_str=connection_string
            )
            self.table_client = TableClient.from_connection_string(
                conn_str=connection_string,
                table_name=table_name
            )
            
            # Create table if it doesn't exist
            self._ensure_table_exists()
            
            logger.info(f"DraftStorageService initialized with table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DraftStorageService: {e}")
            raise
    
    def _ensure_table_exists(self):
        """Ensure the drafts table exists, create if not."""
        try:
            self.table_client.create_table()
            logger.info(f"Created table: {self.table_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Table {self.table_name} already exists")
            else:
                logger.warning(f"Error ensuring table exists: {e}")
    
    def _serialize_draft(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize draft data for Azure Table Storage.
        Azure Table Storage has limitations on property sizes and types.
        Split large draft_data across multiple properties if needed.
        """
        try:
            draft_id = draft["id"]
            draft_json = json.dumps(draft, ensure_ascii=False)

            # Check if draft_data exceeds 60KB (leave some buffer)
            max_size = 60 * 1024  # 60KB buffer

            # Create entity with required keys
            entity = {
                'PartitionKey': 'draft',
                'RowKey': draft_id,
                'draft_id': draft_id,
                'title': draft["metadata"]["title"],
                'description': draft["metadata"]["description"],
                'created_by': draft["metadata"]["created_by"],
                'created_at': draft["metadata"]["created_at"],
                'modified_at': draft["metadata"]["modified_at"],

                'client_metadata': json.dumps(draft["metadata"]['client_metadata']),
                'most_similar_policy_id': draft["metadata"]['most_similar_policy_id'],
                'function': draft["metadata"].get('function', ''),
                'policy_type': draft["metadata"].get('policy_type', ''),
            }

            # Handle large draft data by splitting across multiple properties
            # Use character-based splitting to avoid UTF-8 issues
            if len(draft_json) <= (max_size // 2):  # Divide by 2 for UTF-16 safety
                # Small enough to fit in one property
                entity['draft_data'] = draft_json
                entity['draft_chunks'] = 1
            else:
                # Split into character-based chunks to avoid UTF-8 corruption
                chunk_size = max_size // 2  # Safe for UTF-16 encoding
                chunks = []

                for i in range(0, len(draft_json), chunk_size):
                    chunk = draft_json[i:i+chunk_size]
                    chunks.append(chunk)

                # Store chunks as separate properties
                entity['draft_chunks'] = len(chunks)
                for i, chunk in enumerate(chunks):
                    entity[f'draft_data_{i}'] = chunk

                logger.info(f"Split large draft into {len(chunks)} chunks (total chars: {len(draft_json)})")

            return entity

        except Exception as e:
            logger.error(f"Failed to serialize draft: {e}")
            raise ValueError(f"Draft serialization failed: {str(e)}")

    
    def _deserialize_draft(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize draft data from Azure Table Storage."""
        try:
            # Check if data is chunked
            chunk_count = entity.get('draft_chunks', 1)

            if chunk_count == 1:
                # Single chunk - use draft_data directly
                draft_json = entity.get('draft_data', '')
            else:
                # Multiple chunks - reassemble
                chunks = []
                for i in range(chunk_count):
                    chunk_key = f'draft_data_{i}'
                    if chunk_key in entity:
                        chunks.append(entity[chunk_key])
                    else:
                        logger.warning(f"Missing chunk {i} for draft")

                draft_json = ''.join(chunks)
                logger.info(f"Reassembled draft from {chunk_count} chunks")

            # Parse the JSON draft data
            draft = json.loads(draft_json)

            # Ensure TOC items have required fields (handle legacy data)
            if 'toc' in draft:
                for idx, topic in enumerate(draft['toc']):
                    # Add order field if missing
                    if 'order' not in topic:
                        topic['order'] = idx + 1

                    # Ensure subtopics have order field
                    if 'subtopics' in topic:
                        for sub_idx, subtopic in enumerate(topic['subtopics']):
                            if 'order' not in subtopic:
                                subtopic['order'] = sub_idx + 1

            return draft

        except Exception as e:
            logger.error(f"Failed to deserialize draft: {e}")
            raise ValueError(f"Draft deserialization failed: {str(e)}")
    
    def create_draft(self, draft: CompleteDraft) -> bool:

        """
        Create a new draft in Table Storage.
        Returns True if successful, False otherwise.
        """
        try:
            draft_dict = draft.model_dump(mode='json')

            entity = self._serialize_draft(draft_dict)
            self.table_client.create_entity(entity=entity)

            logger.info(f"Created draft: {draft_dict['id']}")
            return True
            
        except AzureError as e:
            logger.error(f"Azure error creating draft: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to create draft: {e}")
            return False
    
    def get_draft(self, draft_id: str) -> Optional[CompleteDraft]:
        """
        Retrieve a draft by ID.
        Returns the CompleteDraft model or None if not found.
        """
        try:
            entity = self.table_client.get_entity(
                partition_key='draft',
                row_key=draft_id
            )

            draft_dict = self._deserialize_draft(entity)
            # Convert dictionary to CompleteDraft model
            draft_model = CompleteDraft(**draft_dict)
            logger.debug(f"Retrieved draft: {draft_id}")
            return draft_model
            
        except ResourceNotFoundError:
            logger.warning(f"Draft not found: {draft_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get draft {draft_id}: {e}")
            return None
    
    def update_draft(self, draft_id: str, draft_model: CompleteDraft) -> bool:
        """
        Update an existing draft.
        Returns True if successful, False otherwise.
        """
        try:
            # Ensure draft ID matches
            draft_model.id = draft_id

            # DEBUG: Log conversation history before serialization
            total_conversations = 0
            for topic in draft_model.toc:
                total_conversations += len(topic.conversation_history)
                for subtopic in topic.subtopics:
                    total_conversations += len(subtopic.conversation_history)
            logger.info(f"DEBUG: Before serialization - draft has {total_conversations} total conversation entries")

            # Convert the Pydantic model to a dictionary
            draft_dict = draft_model.model_dump(mode='json')

            # DEBUG: Log conversation history after model_dump
            total_conversations_dict = 0
            for topic in draft_dict.get('toc', []):
                total_conversations_dict += len(topic.get('conversation_history', []))
                for subtopic in topic.get('subtopics', []):
                    total_conversations_dict += len(subtopic.get('conversation_history', []))
            logger.info(f"DEBUG: After model_dump - dict has {total_conversations_dict} total conversation entries")

            entity = self._serialize_draft(draft_dict)
            self.table_client.upsert_entity(entity=entity)


            logger.info(f"Updated draft: {draft_id}")
            return True
            
        except AzureError as e:
            logger.error(f"Azure error updating draft {draft_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to update draft {draft_id}: {e}")
            return False
    
    def delete_draft(self, draft_id: str) -> bool:
        """
        Delete a draft from storage.
        Returns True if successful, False if not found or error.
        """
        try:
            self.table_client.delete_entity(
                partition_key='draft',
                row_key=draft_id
            )
            
            logger.info(f"Deleted draft: {draft_id}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Draft not found for deletion: {draft_id}")
            return False
        except AzureError as e:
            logger.error(f"Azure error deleting draft {draft_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete draft {draft_id}: {e}")
            return False
    
    def list_all_drafts(
        self,
        title: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        created_by: Optional[str] = None,
        industry: Optional[str] = None,
        function: Optional[str] = None,
        policy_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all drafts in storage with optional filtering.
        Returns a list of draft summary information (not full content).

        Note: Azure Table Storage has limited query capabilities. Some filters
        are applied at different levels:
        - Database-level (exact match): function, policy_type
        - Post-fetch (partial/case-insensitive): title, created_by, country, city, industry
        """
        try:
            logger.info(f"📋 list_all_drafts called with filters: title={title}, country={country}, city={city}, created_by={created_by}, industry={industry}, function={function}, policy_type={policy_type}")

            # Build query filter for database-level filtering
            # Azure Table Storage only supports exact matches in queries
            # Note: created_by is now post-fetch filtered for partial matching
            query_filters = ["PartitionKey eq 'draft'"]

            if function:
                query_filters.append(f"function eq '{function}'")
            if policy_type:
                query_filters.append(f"policy_type eq '{policy_type}'")

            query_filter = " and ".join(query_filters)
            logger.info(f"🔍 Query filter: {query_filter}")

            entities = self.table_client.query_entities(
                query_filter=query_filter,
                select=['RowKey', 'draft_id', 'title', 'description',
                       'created_by', 'created_at', 'modified_at', 'client_metadata',
                       'most_similar_policy_id', 'function', 'policy_type']
            )

            drafts = []
            entity_count = 0
            for entity in entities:
                try:
                    entity_count += 1
                    draft_id = entity.get('draft_id')
                    logger.info(f"Processing entity {entity_count}: draft_id={draft_id}, title={entity.get('title')}")

                    client_metadata = entity.get('client_metadata')
                    if client_metadata:
                        if isinstance(client_metadata, str):
                            try:
                                # Parse JSON string to dict
                                client_metadata = json.loads(client_metadata)
                                logger.info(f"  Parsed client_metadata: {client_metadata}")
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse client_metadata for draft {draft_id}")
                                client_metadata = None
                        # If it's already a dict, keep it as is
                        elif not isinstance(client_metadata, dict):
                            client_metadata = None

                    # Apply post-fetch filters for fields in client_metadata and title
                    # Title filter - case-insensitive partial match
                    if title and title.lower() not in entity['title'].lower():
                        logger.info(f"  ❌ Filtered out by title: '{title}' not in '{entity['title']}'")
                        continue

                    # Created by filter - case-insensitive partial match
                    if created_by and created_by.lower() not in entity['created_by'].lower():
                        logger.info(f"  ❌ Filtered out by created_by: '{created_by}' not in '{entity['created_by']}'")
                        continue

                    # Client metadata filters
                    if client_metadata:
                        if country and client_metadata.get('country', '').lower() != country.lower():
                            logger.info(f"  ❌ Filtered out by country: '{country}' != '{client_metadata.get('country', '')}'")
                            continue
                        if city and client_metadata.get('city', '').lower() != city.lower():
                            logger.info(f"  ❌ Filtered out by city: '{city}' != '{client_metadata.get('city', '')}'")
                            continue
                        if industry and client_metadata.get('industry', '').lower() != industry.lower():
                            logger.info(f"  ❌ Filtered out by industry: '{industry}' != '{client_metadata.get('industry', '')}'")
                            continue
                    else:
                        # If we're filtering on client_metadata fields but entity has no metadata, filter it out
                        if country or city or industry:
                            logger.info(f"  ❌ Filtered out: no client_metadata but filters require it")
                            continue

                    # Get just the summary info without deserializing full draft
                    draft_summary = {
                        'draft_id': entity['draft_id'],
                        'title': entity['title'],
                        'description': entity.get('description', ''),
                        'created_by': entity['created_by'],
                        'created_at': entity['created_at'],
                        'modified_at': entity['modified_at'],
                        'client_metadata': client_metadata,
                        'most_similar_policy_id': entity.get('most_similar_policy_id'),
                        'function': entity.get('function', ''),
                        'policy_type': entity.get('policy_type', '')
                    }
                    logger.info(f"  ✅ Draft passed all filters: {draft_id}")
                    drafts.append(draft_summary)
                except Exception as e:
                    logger.warning(f"Skipping malformed draft entity: {e}")
                    continue

            # Sort by modified_at descending (most recent first)
            drafts.sort(
                key=lambda x: x['modified_at'],
                reverse=True
            )

            logger.info(f"✅ Retrieved {len(drafts)} drafts after applying filters (processed {entity_count} total entities)")
            return drafts

        except AzureError as e:
            logger.error(f"Azure error listing drafts: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list drafts: {e}")
            return []
    
    def get_drafts_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all drafts created by a specific user.
        Note: In collaborative mode, all drafts are visible to all users,
        but this method is kept for potential filtering needs.
        """
        try:
            query_filter = f"PartitionKey eq 'draft' and created_by eq '{user_id}'"
            
            entities = self.table_client.query_entities(
                query_filter=query_filter,
                select=['RowKey', 'draft_id', 'title', 'description', 
                       'created_at', 'modified_at']
            )
            
            drafts = []
            for entity in entities:
                try:
                    draft_summary = {
                        'draft_id': entity['draft_id'],
                        'title': entity['title'],
                        'description': entity['description'],
                        'created_at': entity['created_at'],
                        'modified_at': entity['modified_at']
                    }
                    drafts.append(draft_summary)
                except Exception as e:
                    logger.warning(f"Skipping malformed draft entity: {e}")
                    continue
            
            drafts.sort(
                key=lambda x: x['modified_at'],
                reverse=True
            )
            
            logger.info(f"Retrieved {len(drafts)} drafts for user {user_id}")
            return drafts
            
        except Exception as e:
            logger.error(f"Failed to get drafts for user {user_id}: {e}")
            return []
    
    def cleanup_old_drafts(self, days_old: int = 30) -> int:
        """
        Clean up drafts older than specified days.
        Returns the number of drafts deleted.
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            # Query old drafts
            query_filter = f"PartitionKey eq 'draft' and modified_at lt '{cutoff_str}'"
            
            entities = self.table_client.query_entities(
                query_filter=query_filter,
                select=['RowKey', 'draft_id']
            )
            
            deleted_count = 0
            for entity in entities:
                try:
                    self.table_client.delete_entity(
                        partition_key='draft',
                        row_key=entity['RowKey']
                    )
                    deleted_count += 1
                    logger.info(f"Deleted old draft: {entity['draft_id']}")
                except Exception as e:
                    logger.warning(f"Failed to delete old draft {entity['draft_id']}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old drafts")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old drafts: {e}")
            return 0