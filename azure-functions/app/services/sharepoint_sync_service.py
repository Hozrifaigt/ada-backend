import asyncio
import json
import logging
import uuid
from io import BytesIO
from typing import Dict, List, Any

from app.config.settings import settings
from app.pipelines.components.sharepoint_loader import SharePointLoader
from app.pipelines.document_processor import DocumentProcessor
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.services.sync_state_storage import SyncStateStorage

logger = logging.getLogger(__name__)


def _function_from_item(item: Dict[str, Any]) -> str:
    """
    Derive the policy's function from its SharePoint folder path.
    The library is organised as <Function>/<...>/file.docx, so the top-level
    folder name is the function (e.g. 'HR', 'IT', 'Finance', 'Governance',
    'Procurement'). Returned lowercase so matching is case-insensitive.
    """
    path = item.get("parentReference", {}).get("path", "")
    # path looks like "/drives/{id}/root:/HR/Policies" or "/drive/root:/HR"
    if "root:" in path:
        sub = path.split("root:", 1)[1].lstrip("/")
        if sub:
            return sub.split("/")[0].strip().lower()
    return ""


class SharePointSyncService:
    """
    CDC pipeline that syncs .docx files from SharePoint into Azure Search.
    Uses the Microsoft Graph delta API so only changed files are processed
    on incremental runs (not a full re-index every time).
    """

    def run_sync(self) -> Dict[str, Any]:
        """Entry point called by the admin endpoint (synchronous wrapper)."""
        return asyncio.run(self._async_sync())

    async def _async_sync(self) -> Dict[str, Any]:
        loader = SharePointLoader()
        sync_state = SyncStateStorage()
        embedder = AzureEmbedder()
        vector_store = AzureVectorStore(embedder=embedder)
        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        llm = AzureLLM(prompt_manager=prompt_manager)
        processor = DocumentProcessor()

        stats = {"added": 0, "updated": 0, "deleted": 0, "errors": 0}

        try:
            saved_delta_url = sync_state.get_delta_url()
            is_first_run = saved_delta_url is None
            logger.info(
                f"SharePoint sync started ({'full scan' if is_first_run else 'delta scan'})"
            )

            added_or_modified, deleted_ids, new_delta_url = loader.list_changes(
                delta_url=saved_delta_url
            )

            # Process deletions first
            for item_id in deleted_ids:
                try:
                    if vector_store.delete_policy_by_source(item_id):
                        stats["deleted"] += 1
                        logger.info(f"Removed deleted SharePoint item {item_id} from index")
                except Exception as e:
                    logger.error(f"Failed to delete item {item_id}: {e}")
                    stats["errors"] += 1

            # Process new and modified files
            for item in added_or_modified:
                item_id = item["id"]
                filename = item.get("name", item_id)
                web_url = item.get("webUrl", "")
                function = _function_from_item(item)
                try:
                    file_bytes = loader.download_file(item_id)
                    structured = processor.process(BytesIO(file_bytes))

                    topics = structured.get("topics", [])
                    full_text = structured.get("full_text", "")
                    toc = structured.get("toc", [])

                    if not full_text.strip():
                        logger.warning(f"Skipping empty document: {filename}")
                        continue

                    # Remove existing version of this file from the index (upsert behaviour)
                    vector_store.delete_policy_by_source(item_id)

                    policy_id = str(uuid.uuid4())

                    # Build topic documents with embeddings
                    topic_docs = []
                    topic_contents = [t.get("topic_content", "") for t in topics]

                    if topic_contents:
                        topic_embeddings = embedder.embed_documents(topic_contents)
                        for i, topic_data in enumerate(topics):
                            topic_docs.append({
                                "id": topic_data["id"],
                                "policy_id": policy_id,
                                "source": item_id,
                                "topic": topic_data["topic"],
                                "topic_content": topic_data.get("topic_content", ""),
                                "topic_vector": topic_embeddings[i],
                                "subtopics": json.dumps(topic_data.get("subtopics", [])),
                            })

                    # Generate policy summary (used for display in the UI)
                    summary = await llm.asummarize_document(full_text)

                    # Build the MATCHING vector from the policy's title + section
                    # headings rather than the generic summary. Headings carry the
                    # real topical signal (e.g. "Tendering, Contract Award" vs
                    # "Endpoint Security, Release Deployment"), which discriminates
                    # far better than boilerplate governance prose.
                    clean_title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
                    topic_names = [t.get("topic", "") for t in topics if t.get("topic")]
                    match_text = f"{clean_title}. " + ", ".join(topic_names)
                    description_vector = embedder.embed_query(match_text)
                    logger.info(f"Match vector built from title+headings for '{filename}': {match_text[:120]}")

                    policy_doc = {
                        "id": policy_id,
                        "filename": filename,
                        "source": item_id,
                        "web_url": web_url,
                        "function": function,
                        "description": summary,
                        "description_vector": description_vector,
                        "toc": toc,
                        "topic_ids": [d["id"] for d in topic_docs],
                        "num_topics": len(topic_docs),
                    }

                    vector_store.upsert_policy_documents([policy_doc])
                    if topic_docs:
                        vector_store.upsert_topic_documents(topic_docs)

                    if is_first_run:
                        stats["added"] += 1
                    else:
                        stats["updated"] += 1

                    logger.info(f"Indexed '{filename}' ({len(topic_docs)} topics)")

                except Exception as e:
                    logger.error(f"Failed to process SharePoint item '{filename}': {e}")
                    stats["errors"] += 1

            # Persist new delta token only on full success (or partial — still useful)
            if new_delta_url:
                sync_state.save_delta_url(new_delta_url)
                logger.info("Delta token saved for next sync")

            logger.info(f"SharePoint sync complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"SharePoint sync failed: {e}")
            raise
