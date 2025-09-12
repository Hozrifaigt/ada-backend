"""
Draft Service - Core business logic for draft management
"""

import uuid
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


from app.models.policy_models import SimilarPolicyRequest, PolicyMatch
from app.models.draft_models import ClientMetadata, DraftMetadata, CompleteDraft

logger = logging.getLogger(__name__)


class DraftService:
    """Business logic service for draft management."""
    
    def __init__(self, storage_service):
        """Initialize draft service with storage dependency."""
        self.storage = storage_service
    
    async def initialize_draft_from_description(
        self,
        description: str,
        title: str,
        user_id: str,
        similarity_service,
        store,
        client_metadata: ClientMetadata
    ) -> Dict[str, Any]:
        """
        Initialize a new draft from user description.
        Returns draft with suggested TOC from similar policies.
        """
        try:
            # Search for similar policies
            request = SimilarPolicyRequest(
                description=description,
                threshold=0.7
            )
            similar_policies = await similarity_service.find_similar_policies(request, store)
            most_similar_policy_id = similar_policies[0].policy_id if similar_policies else None
            
            # Create draft structure
            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            # Get the most similar policy ID (if any)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata, # Pass the Pydantic object directly
                description=description,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                most_similar_policy_id=most_similar_policy_id
            )
            
            draft_to_save = CompleteDraft( 
                id=str(uuid.uuid4()),
                metadata=draft_metadata,
                toc=[]
            )

            # if similar_policies:
            #     merged_toc = self._merge_tocs_from_policies(similar_policies)
            #     draft_to_save.toc = merged_toc
            if similar_policies:
                draft_to_save.toc = similar_policies[0].toc 

            success = self.storage.create_draft(draft_to_save)

            if success:
                logger.info(f"Initialized draft {draft_to_save.id} with {len(draft_to_save.toc)} suggested topics")
                
                # The API response still needs to be a dictionary, so we convert it here.

                draft_dict = draft_to_save.model_dump()

                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_dict,
                    "similar_policies_found": len(similar_policies),
                    "most_similar_policy_id": most_similar_policy_id,
                    "client_metadata": client_metadata.model_dump()
                }
            else:
                logger.error(f"Failed to save initialized draft {draft_to_save.id}")
                return None

                
        except Exception as e:
            logger.error(f"Failed to initialize draft: {e}")
            return None
 
    def _merge_tocs_from_policies(self, policies: List[PolicyMatch]) -> List[Dict[str, Any]]:
        """
        Extract and merge TOCs from similar policies.
        Prioritize topics that appear in multiple policies.
        """
        topic_frequency = {}
        topic_details = {}
        
        print(f"Number of policies: {len(policies)}")

        for i, policy in enumerate(policies):

            # if i != 0: break  # let's just keep only the first policy
        
            if not policy.toc:
                continue
                
            for topic in policy.toc:
                topic_key = topic.get('topic', '').lower().strip()
                if not topic_key:
                    continue
                
                if topic_key not in topic_frequency:
                    topic_frequency[topic_key] = 0
                    topic_details[topic_key] = {
                        "id": str(uuid.uuid4()),
                        "topic": topic.get('topic', ''),
                        "order": len(topic_details) + 1,
                        "source_topic_id": topic.get('source_topic_id', ''),
                        "source_policy_id": policy.policy_id,  # PolicyMatch is a Pydantic model, use attribute access
                        "subtopics": []
                    }
                topic_frequency[topic_key] += 1
                
                # Merge subtopics if they exist
                if 'subtopics' in topic and topic['subtopics']:
                    for subtopic in topic['subtopics']:
                        subsection = {
                            "id": str(uuid.uuid4()),
                            "title": subtopic.get('title', ''),
                            "order": len(topic_details[topic_key]["subtopics"]) + 1,
                            "source_subtopic_id": subtopic.get('subtopic_id')
                        }
                        # Avoid duplicate subtopics
                        existing_titles = [s['title'].lower() for s in topic_details[topic_key]["subtopics"]]
                        if subsection['title'].lower() not in existing_titles:
                            topic_details[topic_key]["subtopics"].append(subsection)
        
        # Sort by frequency and return as list
        sorted_topics = sorted(
            topic_details.items(),
            key=lambda x: topic_frequency[x[0]],
            reverse=True
        )
        
        # Reorder and return
        toc = []
        for idx, (_, topic) in enumerate(sorted_topics, 1):
            topic["order"] = idx
            toc.append(topic)
        
        return toc
    
    def _format_toc_for_model(self, raw_toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform raw TOC data to match Pydantic model structure.
        Handles different input formats and ensures required fields are present.
        """
        formatted_toc = []
        
        for idx, item in enumerate(raw_toc, 1):
            # Handle different possible input formats
            if isinstance(item, dict):
                # Extract title from 'topic' field or 'title' field
                title = item.get('title') or item.get('topic', f'Topic {idx}')
                
                # Clean up title if it contains tab-separated data or IDs
                if '\t' in title:
                    title = title.split('\t')[1] if len(title.split('\t')) > 1 else title.split('\t')[0]
                
                formatted_topic = {
                    'id': item.get('id') or str(uuid.uuid4()),
                    'title': title.strip(),
                    'order': item.get('order', idx),
                    'source_topic_id': item.get('source_topic_id'),
                    'subtopics': []
                }
                
                # Handle subtopics if they exist
                subtopics = item.get('subtopics', [])
                if subtopics:
                    for sub_idx, subtopic in enumerate(subtopics, 1):
                        if isinstance(subtopic, dict):
                            sub_title = subtopic.get('title') or subtopic.get('topic', f'Subtopic {sub_idx}')
                            
                            # Clean up subtopic title
                            if '\t' in sub_title:
                                sub_title = sub_title.split('\t')[1] if len(sub_title.split('\t')) > 1 else sub_title.split('\t')[0]
                            
                            formatted_subtopic = {
                                'id': subtopic.get('id') or str(uuid.uuid4()),
                                'title': sub_title.strip(),
                                'order': subtopic.get('order', sub_idx),
                                'source_subtopic_id': subtopic.get('source_subtopic_id')
                            }
                            formatted_topic['subtopics'].append(formatted_subtopic)
                
                formatted_toc.append(formatted_topic)
        
        return formatted_toc
    
    def update_toc(self, draft_id: str, updated_toc: List[Dict[str, Any]]) -> bool:
        """
        Update the TOC structure of a draft from frontend.
        """
        try:
            logger.info(f"Starting TOC update for draft {draft_id}")
            
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found for TOC update")
                return False
            
            logger.info(f"Successfully retrieved draft {draft_id}")
            
            # Transform TOC data to match Pydantic model structure
            formatted_toc = self._format_toc_for_model(updated_toc)
            
            # Update TOC with formatted content
            draft["toc"] = formatted_toc
            draft["metadata"]["modified_at"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Updated TOC with {len(formatted_toc)} items")
            
            # Initialize topics array to match TOC structure
            draft["topics"] = self._initialize_topics_from_toc(formatted_toc)
            
            logger.info(f"Initialized {len(draft['topics'])} topics")
            
            # Convert to CompleteDraft model for storage
            try:
                draft_model = CompleteDraft(**draft)
                logger.info("Successfully converted draft to CompleteDraft model")
            except Exception as model_error:
                logger.error(f"Failed to convert to CompleteDraft model: {model_error}")
                return False
            
            success = self.storage.update_draft(draft_id, draft_model)
            
            if success:
                logger.info(f"Successfully updated TOC for draft {draft_id} with {len(formatted_toc)} topics")
            else:
                logger.error(f"Storage update failed for draft {draft_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update TOC for draft {draft_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_topics_from_toc(self, toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize empty topics structure matching the TOC."""
        topics = []
        
        for topic in toc:
            # Handle different possible ID field names
            topic_id = topic.get("id") or topic.get("topic_id") or str(uuid.uuid4())
            
            section = {
                "id": topic_id,
                "content": "",
                "summary": "",
                "subtopics": []
            }

            for subtopic in topic.get("subtopics", []):
                # Handle different possible ID field names for subtopics too
                subtopic_id = subtopic.get("id") or subtopic.get("subtopic_id") or str(uuid.uuid4())
                section["subtopics"].append({
                    "id": subtopic_id,
                    "content": "",
                    "summary": ""
                })
            
            topics.append(section)
        
        return topics
    
    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete draft by ID."""
        return self.storage.get_draft(draft_id)
    
    def list_drafts(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all drafts. 
        user_id parameter kept for compatibility but all drafts are visible to all users.
        """
        return self.storage.list_all_drafts()
    
    def update_topic_content(
        self,
        draft_id: str,
        topic_id: str,
        content: str,
        subtopic_id: Optional[str] = None
    ) -> bool:
        """
        Update content for a topic or subtopic.
        Automatically creates summary if content > 150 words.
        """
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.warning(f"Draft {draft_id} not found")
                return False

            # Find and update the topic
            for topic in draft["topics"]:
                if topic["id"] == topic_id:
                    if subtopic_id:
                        # Update subtopic
                        for subtopic in topic["subtopics"]:
                            if subtopic["id"] == subtopic_id:
                                subtopic["content"] = content
                                subtopic["summary"] = self._create_summary_if_needed(content)
                                break
                    else:
                        # Update main topic
                        topic["content"] = content
                        topic["summary"] = self._create_summary_if_needed(content)
                    
                    # Update modified timestamp
                    draft["metadata"]["modified_at"] = datetime.now(timezone.utc).isoformat()
                    
                    # Save updated draft
                    return self.storage.update_draft(draft_id, draft)

            logger.warning(f"Topic {topic_id} not found in draft {draft_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update topic content: {e}")
            return False
    
    def _create_summary_if_needed(self, content: str) -> str:
        """
        Create summary placeholder if content > 150 words.
        Actual summary generation will be done by LLM during context building.
        """
        if not content:
            return ""
        
        word_count = len(content.split())
        if word_count > 150:
            # Return a marker that indicates summary should be generated
            # Actual summary will be created by LLM when needed
            return "NEEDS_SUMMARY"
        
        return ""

    def get_topic_by_id(self, draft: Dict[str, Any], topic_id: str) -> Optional[Dict[str, Any]]:
        """Get a topic from draft by its ID."""
        for topic in draft.get("topics", []):
            if topic["id"] == topic_id:
                return topic
        return None
    
    def get_toc_item_by_id(self, draft: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
        """Get a TOC item (topic) by its ID."""
        for topic in draft.get("toc", []):
            if topic["id"] == item_id:
                return topic
        return None
    
    def build_sequential_context(
        self,
        draft: Dict[str, Any],
        target_topic_id: str,
        target_subtopic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context from all previous topics up to the target.
        Returns structured context for LLM consumption.
        """
        context_parts = {
            "document_context": {
                "title": draft["metadata"]["title"],
                "description": draft["metadata"]["description"],
                "client_metadata": draft["metadata"].get("client_metadata", {})
            },
            "previous_topics": [],
            "current_topic_toc": None
        }
        
        # Find target position in TOC
        target_found = False
        
        for toc_item in draft["toc"]:
            if target_found:
                break

            topic = self.get_topic_by_id(draft, toc_item["id"])
            if not topic:
                continue

            if toc_item["id"] == target_topic_id:
                context_parts["current_topic_toc"] = toc_item

                if target_subtopic_id:
                    # Include main topic content if generating subtopic
                    if topic["content"]:
                        context_parts["current_topic_content"] = {
                            "title": toc_item["title"],
                            "content": topic["content"] if topic["summary"] == "" else topic["summary"]
                        }

                    # Include previous subtopics of current topic
                    for sub_toc in toc_item.get("subtopics", []):
                        if sub_toc["id"] == target_subtopic_id:
                            break

                        for subtopic in topic.get("subtopics", []):
                            if subtopic["id"] == sub_toc["id"] and subtopic["content"]:
                                context_parts["previous_topics"].append({
                                    "title": f"{toc_item['title']} - {sub_toc['title']}",
                                    "content": subtopic["content"] if subtopic["summary"] == "" else subtopic["summary"],
                                    "is_subtopic": True
                                })
                
                target_found = True
            else:
                # Add all content from topics before target
                if topic["content"]:
                    context_parts["previous_topics"].append({
                        "title": toc_item["title"],
                        "content": topic["content"] if topic["summary"] == "" else topic["summary"],
                        "is_subtopic": False
                    })
                
                # Add subtopics
                for sub_toc in toc_item.get("subtopics", []):
                    for subtopic in topic.get("subtopics", []):
                        if subtopic["id"] == sub_toc["id"] and subtopic["content"]:
                            context_parts["previous_topics"].append({
                                "title": f"{toc_item['title']} - {sub_toc['title']}",
                                "content": subtopic["content"] if subtopic["summary"] == "" else subtopic["summary"],
                                "is_subtopic": True
                            })
        
        return context_parts
    
    def delete_draft(self, draft_id: str) -> bool:
        """Delete a draft from storage."""
        try:
            success = self.storage.delete_draft(draft_id)
            if success:
                logger.info(f"Deleted draft {draft_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete draft {draft_id}: {e}")
            return False
    
    def delete_all_drafts(self) -> Dict[str, Any]:
        """Delete all drafts from storage."""
        try:
            # Get all drafts first
            all_drafts = self.storage.list_all_drafts()
            total_count = len(all_drafts)
            
            if total_count == 0:
                logger.info("No drafts found to delete")
                return {
                    "deleted_count": 0,
                    "total_count": 0,
                    "message": "No drafts found to delete"
                }
            
            # Delete each draft
            deleted_count = 0
            failed_deletions = []
            
            for draft in all_drafts:
                # The list_all_drafts returns 'draft_id' not 'id'
                draft_id = draft.get("draft_id")
                if draft_id:
                    try:
                        if self.storage.delete_draft(draft_id):
                            deleted_count += 1
                        else:
                            failed_deletions.append(draft_id)
                    except Exception as e:
                        logger.error(f"Failed to delete draft {draft_id}: {e}")
                        failed_deletions.append(draft_id)
            
            logger.info(f"Deleted {deleted_count}/{total_count} drafts")
            
            return {
                "deleted_count": deleted_count,
                "total_count": total_count,
                "failed_count": len(failed_deletions),
                "failed_draft_ids": failed_deletions,
                "message": f"Successfully deleted {deleted_count} out of {total_count} drafts"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete all drafts: {e}")
            return {
                "deleted_count": 0,
                "total_count": 0,
                "failed_count": 0,
                "failed_draft_ids": [],
                "message": f"Failed to delete all drafts: {str(e)}"
            }
    
    async def generate_content_for_topic(
        self,
        draft_id: str,
        topic_id: str,
        user_prompt: str,
        subtopic_id: Optional[str],
        vector_store,
        llm_service
    ) -> Dict[str, Any]:
        """
        Generate content for a topic or subtopic using the generation service.
        
        Returns:
            Dict with content and sources used
        """
        try:
            # Get the draft
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found")
                return None
            
            # Generate content using the generation service
            content, source_ids = await self.generation_service.generate_topic_content(
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                vector_store=vector_store,
                llm_service=llm_service
            )
            
            # Update the draft with generated content
            success = self.update_topic_content(
                draft_id=draft_id,
                topic_id=topic_id,
                content=content,
                subtopic_id=subtopic_id
            )
            
            if success:
                # Check if we need to generate a summary
                word_count = len(content.split())
                summary = ""
                if word_count > 150:
                    # Get the topic title for summary generation
                    draft = self.storage.get_draft(draft_id)  # Refresh draft
                    toc_item = self.get_toc_item_by_id(draft, topic_id)
                    
                    if subtopic_id:
                        # Find subtopic title
                        for sub in toc_item.get("subtopics", []):
                            if sub["id"] == subtopic_id:
                                title = sub["title"]
                                break
                    else:
                        title = toc_item["title"]
                    
                    summary = await self.generation_service.generate_summary(
                        content=content,
                        title=title,
                        llm_service=llm_service
                    )
                    
                    # Update summary in draft
                    self._update_summary(draft_id, topic_id, subtopic_id, summary)
                
                return {
                    "content": content,
                    "sources_used": source_ids,
                    "summary": summary,
                    "word_count": word_count
                }
            else:
                logger.error(f"Failed to save generated content for topic {topic_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    def _update_summary(
        self,
        draft_id: str,
        topic_id: str,
        subtopic_id: Optional[str],
        summary: str
    ) -> bool:
        """Update the summary for a topic or subtopic."""
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                return False
            
            for topic in draft["topics"]:
                if topic["id"] == topic_id:
                    if subtopic_id:
                        for subtopic in topic["subtopics"]:
                            if subtopic["id"] == subtopic_id:
                                subtopic["summary"] = summary
                                break
                    else:
                        topic["summary"] = summary
                    
                    return self.storage.update_draft(draft_id, draft)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
            return False
    
    def get_draft_progress(self, draft_id: str) -> Dict[str, Any]:
        """Calculate draft completion progress."""
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                return {"error": "Draft not found"}
            
            total_items = 0
            completed_items = 0
            
            for topic in draft.get("topics", []):
                total_items += 1
                if topic.get("content"):
                    completed_items += 1
                
                for subtopic in topic.get("subtopics", []):
                    total_items += 1
                    if subtopic.get("content"):
                        completed_items += 1
            
            return {
                "total": total_items,
                "completed": completed_items,
                "percentage": (completed_items / total_items * 100) if total_items > 0 else 0,
                "remaining": total_items - completed_items
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate progress for draft {draft_id}: {e}")
            return {"error": "Failed to calculate progress"}
