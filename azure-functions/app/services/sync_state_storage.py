import logging
from datetime import datetime, timezone
from typing import Optional

from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceNotFoundError

from app.config.settings import settings

logger = logging.getLogger(__name__)

PARTITION_KEY = "sharepoint"
ROW_KEY = "library"
TABLE_NAME = "SyncState"


class SyncStateStorage:
    """Persists the Microsoft Graph delta token for the SharePoint CDC pipeline."""

    def __init__(self):
        self._table_client = None

    def _get_client(self):
        if self._table_client is None:
            service = TableServiceClient.from_connection_string(
                settings.AZURE_TABLE_STORAGE_CONNECTION_STRING
            )
            service.create_table_if_not_exists(TABLE_NAME)
            self._table_client = service.get_table_client(TABLE_NAME)
        return self._table_client

    def get_delta_url(self) -> Optional[str]:
        """Return the saved delta URL, or None if this is the first sync."""
        try:
            entity = self._get_client().get_entity(
                partition_key=PARTITION_KEY, row_key=ROW_KEY
            )
            return entity.get("delta_url") or None
        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Could not read delta URL from SyncState table: {e}")
            return None

    def save_delta_url(self, delta_url: str) -> None:
        """Persist the new delta URL after a successful sync pass."""
        try:
            entity = {
                "PartitionKey": PARTITION_KEY,
                "RowKey": ROW_KEY,
                "delta_url": delta_url,
                "last_sync_at": datetime.now(timezone.utc).isoformat(),
            }
            self._get_client().upsert_entity(entity)
        except Exception as e:
            logger.error(f"Failed to save delta URL to SyncState table: {e}")

    def reset_delta(self) -> None:
        """Delete the saved delta token so the next sync does a full re-scan."""
        try:
            self._get_client().delete_entity(partition_key=PARTITION_KEY, row_key=ROW_KEY)
            logger.info("Delta token cleared — next sync will do a full re-scan")
        except ResourceNotFoundError:
            pass  # nothing to clear
        except Exception as e:
            logger.error(f"Failed to clear delta token: {e}")

    def get_status(self) -> dict:
        """Return sync state for the status endpoint."""
        try:
            entity = self._get_client().get_entity(
                partition_key=PARTITION_KEY, row_key=ROW_KEY
            )
            return {
                "synced_before": True,
                "last_sync_at": entity.get("last_sync_at"),
                "delta_token_present": bool(entity.get("delta_url")),
            }
        except ResourceNotFoundError:
            return {
                "synced_before": False,
                "last_sync_at": None,
                "delta_token_present": False,
            }
        except Exception as e:
            logger.warning(f"Could not read sync status: {e}")
            return {
                "synced_before": None,
                "last_sync_at": None,
                "delta_token_present": False,
            }
