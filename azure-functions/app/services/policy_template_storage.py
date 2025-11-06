"""Azure Table Storage service for policy templates."""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

from app.models.policy_template_models import PolicyTemplate

logger = logging.getLogger(__name__)


class PolicyTemplateStorageService:
    """Azure Table Storage service for policy template management."""

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

            logger.info(f"PolicyTemplateStorageService initialized with table: {table_name}")

        except Exception as e:
            logger.error(f"Failed to initialize PolicyTemplateStorageService: {e}")
            raise

    def _ensure_table_exists(self):
        """Ensure the table exists, create if not."""
        try:
            self.table_client.create_table()
            logger.info(f"Created table: {self.table_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Table {self.table_name} already exists")
            else:
                logger.warning(f"Error ensuring table exists: {e}")

    def _serialize_template(self, template: PolicyTemplate) -> Dict[str, Any]:
        """Serialize template data for Azure Table Storage."""
        try:
            template_dict = template.model_dump()
            template_json = json.dumps(template_dict, ensure_ascii=False)

            max_size = 60 * 1024  # 60KB buffer

            # Create entity with partition key based on function
            entity = {
                'PartitionKey': f'template_{template.function.lower().replace(" ", "_")}',
                'RowKey': template.template_id,
                'template_id': template.template_id,
                'function': template.function,
                'policy_type': template.policy_type,
                'is_template': True,
                'created_at': template.created_at,
                'modified_at': template.modified_at,
                'source': template.source,
            }

            # Handle large template data by splitting
            if len(template_json) <= (max_size // 2):
                entity['template_data'] = template_json
                entity['template_chunks'] = 1
            else:
                chunk_size = max_size // 2
                chunks = []

                for i in range(0, len(template_json), chunk_size):
                    chunk = template_json[i:i+chunk_size]
                    chunks.append(chunk)

                entity['template_chunks'] = len(chunks)
                for i, chunk in enumerate(chunks):
                    entity[f'template_data_{i}'] = chunk

                logger.info(f"Split large template into {len(chunks)} chunks")

            return entity

        except Exception as e:
            logger.error(f"Failed to serialize template: {e}")
            raise

    def _deserialize_template(self, entity: Dict[str, Any]) -> PolicyTemplate:
        """Deserialize template data from Azure Table Storage."""
        try:
            # Reconstruct template JSON from chunks
            chunks_count = entity.get('template_chunks', 1)

            if chunks_count == 1:
                template_json = entity['template_data']
            else:
                # Combine chunks
                chunks = []
                for i in range(chunks_count):
                    chunk = entity.get(f'template_data_{i}', '')
                    chunks.append(chunk)
                template_json = ''.join(chunks)

            # Parse JSON and create PolicyTemplate
            template_dict = json.loads(template_json)
            return PolicyTemplate(**template_dict)

        except Exception as e:
            logger.error(f"Failed to deserialize template: {e}")
            raise

    def create_template(self, template: PolicyTemplate) -> bool:
        """Store a policy template."""
        try:
            entity = self._serialize_template(template)
            self.table_client.upsert_entity(entity=entity)
            logger.info(f"Stored template: {template.template_id} ({template.function} - {template.policy_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to store template {template.template_id}: {e}")
            return False

    def get_template(self, template_id: str, function: str) -> Optional[PolicyTemplate]:
        """Retrieve a policy template by ID."""
        try:
            partition_key = f'template_{function.lower().replace(" ", "_")}'
            entity = self.table_client.get_entity(
                partition_key=partition_key,
                row_key=template_id
            )
            return self._deserialize_template(entity)
        except ResourceNotFoundError:
            logger.warning(f"Template {template_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve template {template_id}: {e}")
            return None

    def list_templates_by_function(self, function: str) -> List[PolicyTemplate]:
        """List all templates for a specific function."""
        try:
            partition_key = f'template_{function.lower().replace(" ", "_")}'
            filter_query = f"PartitionKey eq '{partition_key}'"

            entities = self.table_client.query_entities(query_filter=filter_query)

            templates = []
            for entity in entities:
                try:
                    template = self._deserialize_template(entity)
                    templates.append(template)
                except Exception as e:
                    logger.error(f"Failed to deserialize template: {e}")
                    continue

            logger.info(f"Retrieved {len(templates)} templates for function '{function}'")
            return templates

        except Exception as e:
            logger.error(f"Failed to list templates for function '{function}': {e}")
            return []

    def list_all_templates(self) -> List[PolicyTemplate]:
        """List all policy templates."""
        try:
            filter_query = "is_template eq true"
            entities = self.table_client.query_entities(query_filter=filter_query)

            templates = []
            for entity in entities:
                try:
                    template = self._deserialize_template(entity)
                    templates.append(template)
                except Exception as e:
                    logger.error(f"Failed to deserialize template: {e}")
                    continue

            logger.info(f"Retrieved {len(templates)} total templates")
            return templates

        except Exception as e:
            logger.error(f"Failed to list all templates: {e}")
            return []

    def get_policy_types_by_function(self, function: str) -> List[str]:
        """Get list of policy types for a specific function."""
        try:
            templates = self.list_templates_by_function(function)
            policy_types = list(set([t.policy_type for t in templates]))
            policy_types.sort()
            return policy_types
        except Exception as e:
            logger.error(f"Failed to get policy types for function '{function}': {e}")
            return []

    def get_all_functions(self) -> List[str]:
        """Get list of all available functions."""
        try:
            templates = self.list_all_templates()
            functions = list(set([t.function for t in templates]))
            functions.sort()
            return functions
        except Exception as e:
            logger.error(f"Failed to get all functions: {e}")
            return []

    def delete_template(self, template_id: str, function: str) -> bool:
        """Delete a policy template."""
        try:
            partition_key = f'template_{function.lower().replace(" ", "_")}'
            self.table_client.delete_entity(
                partition_key=partition_key,
                row_key=template_id
            )
            logger.info(f"Deleted template: {template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    def delete_all_templates(self) -> int:
        """Delete all policy templates. Returns count of deleted templates."""
        try:
            templates = self.list_all_templates()
            deleted_count = 0

            for template in templates:
                if self.delete_template(template.template_id, template.function):
                    deleted_count += 1

            logger.info(f"Deleted {deleted_count} templates")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete all templates: {e}")
            return 0
