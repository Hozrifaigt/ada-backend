import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import msal

from app.config.settings import settings

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]


class SharePointLoader:
    """
    Reads .docx files from a SharePoint document library using the
    Microsoft Graph API delta query for incremental (CDC) syncing.
    Authenticates as a service principal using AZURE_CLIENT_ID/SECRET/TENANT_ID.
    """

    def __init__(self):
        self._msal_app = msal.ConfidentialClientApplication(
            client_id=settings.AZURE_CLIENT_ID,
            client_credential=settings.AZURE_CLIENT_SECRET,
            authority=f"https://login.microsoftonline.com/{settings.AZURE_TENANT_ID}",
        )
        self._token: Optional[str] = None
        self._drive_id: Optional[str] = None
        self._site_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        result = self._msal_app.acquire_token_silent(GRAPH_SCOPE, account=None)
        if not result:
            result = self._msal_app.acquire_token_for_client(scopes=GRAPH_SCOPE)
        if "access_token" not in result:
            raise RuntimeError(
                f"Failed to acquire Graph API token: {result.get('error_description')}"
            )
        return result["access_token"]

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}

    # ------------------------------------------------------------------
    # Site / Drive discovery (cached per instance lifetime)
    # ------------------------------------------------------------------

    def _get_site_id(self) -> str:
        if self._site_id:
            return self._site_id
        parsed = urlparse(settings.SHAREPOINT_SITE_URL)
        hostname = parsed.hostname
        site_path = parsed.path.lstrip("/")
        url = f"{GRAPH_BASE}/sites/{hostname}:/{site_path}"
        resp = httpx.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        self._site_id = resp.json()["id"]
        logger.info(f"Resolved site ID: {self._site_id}")
        return self._site_id

    def _get_drive_id(self) -> str:
        if self._drive_id:
            return self._drive_id
        site_id = self._get_site_id()
        url = f"{GRAPH_BASE}/sites/{site_id}/drives"
        resp = httpx.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        drives = resp.json().get("value", [])
        target = settings.SHAREPOINT_DRIVE_NAME.lower()
        for drive in drives:
            if drive.get("name", "").lower() == target:
                self._drive_id = drive["id"]
                logger.info(f"Resolved drive ID for '{settings.SHAREPOINT_DRIVE_NAME}': {self._drive_id}")
                return self._drive_id
        raise RuntimeError(
            f"Drive '{settings.SHAREPOINT_DRIVE_NAME}' not found in site. "
            f"Available: {[d.get('name') for d in drives]}"
        )

    # ------------------------------------------------------------------
    # Delta query — returns only changed items since last sync
    # ------------------------------------------------------------------

    def list_changes(
        self, delta_url: Optional[str] = None
    ) -> Tuple[List[Dict], List[str], str]:
        """
        Fetch changes from SharePoint using the Graph delta API.

        Args:
            delta_url: The delta URL from the previous sync. None triggers a full scan.

        Returns:
            (added_or_modified, deleted_ids, new_delta_url)
            - added_or_modified: list of DriveItem dicts for new/changed .docx files
            - deleted_ids: list of item IDs that were deleted
            - new_delta_url: URL to pass on the next call
        """
        drive_id = self._get_drive_id()
        folder_path = settings.SHAREPOINT_FOLDER_PATH.strip("/")

        if delta_url:
            url = delta_url
        elif folder_path:
            url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{folder_path}:/delta"
        else:
            url = f"{GRAPH_BASE}/drives/{drive_id}/root/delta"

        added_or_modified: List[Dict] = []
        deleted_ids: List[str] = []
        new_delta_url = ""

        while url:
            resp = httpx.get(url, headers=self._headers(), timeout=60)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("value", []):
                # Deleted items carry a 'deleted' facet
                if "deleted" in item:
                    deleted_ids.append(item["id"])
                    continue
                name = item.get("name", "")
                if name.endswith(".docx") and not name.startswith("~$"):
                    added_or_modified.append(item)

            next_link = data.get("@odata.nextLink")
            delta_link = data.get("@odata.deltaLink")

            if delta_link:
                new_delta_url = delta_link
                url = None
            elif next_link:
                url = next_link
            else:
                url = None

        logger.info(
            f"Delta query complete: {len(added_or_modified)} added/modified, "
            f"{len(deleted_ids)} deleted"
        )
        return added_or_modified, deleted_ids, new_delta_url

    # ------------------------------------------------------------------
    # File download
    # ------------------------------------------------------------------

    def download_file(self, item_id: str) -> bytes:
        """Download a DriveItem's content and return as bytes."""
        drive_id = self._get_drive_id()
        url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/content"
        resp = httpx.get(url, headers=self._headers(), timeout=120, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
