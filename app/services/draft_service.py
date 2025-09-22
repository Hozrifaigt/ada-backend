import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


from app.models.policy_models import SimilarPolicyRequest, PolicyMatch
from app.models.draft_models import ClientMetadata, DraftMetadata, CompleteDraft
from app.services.generation_service import ContentGenerationService

logger = logging.getLogger(__name__)


class DraftService:
    """Business logic service for draft management."""

    def __init__(self, storage_service):
        """Initialize draft service with storage dependency."""
        self.storage = storage_service
        self.generation_service = ContentGenerationService()
    
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

            if similar_policies:
                # Transform the TOC to properly set source_topic_id and source_subtopic_id
                draft_to_save.toc = self._transform_toc_for_draft(similar_policies[0].toc)

            success = self.storage.create_draft(draft_to_save)

            if success:
                logger.info(f"Initialized draft {draft_to_save.id} with {len(draft_to_save.toc)} suggested topics")
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

    def _transform_toc_for_draft(self, toc_from_policy: List[Dict[str, Any]]) -> List:
        """
        Transform TOC from similar policy to draft format.
        Creates new IDs for draft while preserving source IDs from vector DB.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for topic_item in toc_from_policy:
            # Create subtopics with new IDs but preserve source IDs
            subtopics = []
            for sub in topic_item.get('subtopics', []):
                subtopic = TOCSubtopic(
                    subtopic_id=str(uuid.uuid4()),  # New ID for draft
                    topic=sub.get('topic', ''),
                    order=sub.get('order', 0),
                    source_subtopic_id=sub.get('subtopic_id'),  # Preserve vector DB ID
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            # Create topic with new ID but preserve source ID
            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),  # New ID for draft
                topic=topic_item.get('topic', ''),
                order=topic_item.get('order', 0),
                source_topic_id=topic_item.get('topic_id'),  # Preserve vector DB ID
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

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
                        "source_topic_id": topic.get('topic_id', ''),  # Use topic_id from vector DB as source
                        "source_policy_id": policy.policy_id,  # PolicyMatch is a Pydantic model, use attribute access
                        "subtopics": []
                    }
                topic_frequency[topic_key] += 1
                
                # Merge subtopics if they exist
                if 'subtopics' in topic and topic['subtopics']:
                    for subtopic in topic['subtopics']:
                        subsection = {
                            "id": str(uuid.uuid4()),
                            "topic": subtopic.get('topic', ''),
                            "order": len(topic_details[topic_key]["subtopics"]) + 1,
                            "source_subtopic_id": subtopic.get('subtopic_id', '')  # Use subtopic_id from vector DB as source
                        }
                        # Avoid duplicate subtopics
                        existing_titles = [s['topic'].lower() for s in topic_details[topic_key]["subtopics"]]
                        if subsection['topic'].lower() not in existing_titles:
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
                title = item.get('topic', f'Topic {idx}')
                
                formatted_topic = {
                    'topic_id': item.get('id') or str(uuid.uuid4()),  # Use 'id' for draft topic ID
                    'title': title.strip(),
                    'order': item.get('order', idx),
                    'source_topic_id': item.get('source_topic_id'),  # Preserve source_topic_id if it exists
                    'subtopics': []
                }
                
                # Handle subtopics if they exist
                subtopics = item.get('subtopics', [])
                if subtopics:
                    for sub_idx, subtopic in enumerate(subtopics, 1):
                        if isinstance(subtopic, dict):
                            sub_title = subtopic.get('title') or subtopic.get('topic', f'Subtopic {sub_idx}')
                            
                            formatted_subtopic = {
                                'id': subtopic.get('id') or str(uuid.uuid4()),
                                'title': sub_title.strip(),
                                'order': subtopic.get('order', sub_idx),
                                'source_subtopic_id': subtopic.get('source_subtopic_id')  # Preserve source_subtopic_id if it exists
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

            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found for TOC update")
                return False

            logger.info(f"Successfully retrieved draft {draft_id}")

            # Transform TOC data to match Pydantic model structure
            formatted_toc = self._format_toc_for_model(updated_toc)

            # Convert formatted TOC to TOCTopic models
            from app.models.draft_models import TOCTopic, TOCSubtopic

            toc_topics = []
            for toc_item in formatted_toc:
                # Convert subtopics
                subtopics = []
                for sub_item in toc_item.get('subtopics', []):
                    subtopic = TOCSubtopic(
                        subtopic_id=sub_item['id'],
                        topic=sub_item['title'],
                        order=sub_item['order'],
                        source_subtopic_id=sub_item.get('source_subtopic_id'),
                        content="",
                        summary="",
                        conversation_history=[]
                    )
                    subtopics.append(subtopic)

                # Convert main topic
                topic = TOCTopic(
                    topic_id=toc_item['topic_id'],  # Use topic_id, not id
                    topic=toc_item['title'],
                    order=toc_item['order'],
                    source_topic_id=toc_item.get('source_topic_id'),
                    content="",
                    summary="",
                    conversation_history=[],
                    subtopics=subtopics
                )
                toc_topics.append(topic)

            # Update TOC in the model
            draft_model.toc = toc_topics
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            logger.info(f"Updated TOC with {len(toc_topics)} topics")

            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully updated TOC for draft {draft_id} with {len(toc_topics)} topics")
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update TOC for draft {draft_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    
    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete draft by ID."""
        draft_model = self.storage.get_draft(draft_id)
        if draft_model:
            return draft_model.model_dump()
        return None
    
    def list_drafts(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all drafts. 
        user_id parameter kept for compatibility but all drafts are visible to all users.
        """
        return self.storage.list_all_drafts()
    
    async def update_topic_content(
        self,
        draft_id: str,
        topic_id: str,
        content: str,
        subtopic_id: Optional[str] = None,
        llm_service = None
    ) -> bool:
        """
        Update content for a topic or subtopic using typed models.
        """
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.warning(f"Draft {draft_id} not found")
                return False

            # Find and update in the TOC structure
            topic_found = False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    topic_found = True

                    if subtopic_id:
                        # Update subtopic content
                        subtopic_found = False
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.content = content
                                # Generate LLM summary if content > 150 words and LLM service available
                                if llm_service:
                                    toc_subtopic.summary = await self.generation_service.generate_summary(
                                        content=content,
                                        title=toc_subtopic.topic,
                                        llm_service=llm_service
                                    )
                                else:
                                    toc_subtopic.summary = self._create_simple_summary_for_context(content)
                                subtopic_found = True
                                logger.info(f"Updated subtopic {subtopic_id} content")
                                break

                        if not subtopic_found:
                            logger.warning(f"Subtopic {subtopic_id} not found in topic {topic_id}")
                            return False
                    else:
                        # Update main topic content
                        toc_topic.content = content
                        # Generate LLM summary if content > 150 words and LLM service available
                        if llm_service:
                            toc_topic.summary = await self.generation_service.generate_summary(
                                content=content,
                                title=toc_topic.topic,
                                llm_service=llm_service
                            )
                        else:
                            toc_topic.summary = self._create_simple_summary_for_context(content)
                        logger.info(f"Updated topic {topic_id} content")
                    break

            if not topic_found:
                logger.warning(f"Topic {topic_id} not found in draft {draft_id}")
                return False

            # Update modified timestamp
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            # Save the updated model
            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully saved content for topic {topic_id}" +
                           (f", subtopic {subtopic_id}" if subtopic_id else ""))
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update topic content: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _create_simple_summary_for_context(self, content: str) -> str:
        """
        Create a simple summary for context building by truncating content > 150 words.
        This is used for context building when full content would be too long.
        """
        if not content:
            return ""

        word_count = len(content.split())
        if word_count > 150:
            # Take first 150 words as summary for context
            words = content.split()
            return " ".join(words[:150]) + "..."

        return ""  # No summary needed for short content

    def get_toc_item_by_id(self, draft: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
        """Get a TOC item (topic) by its ID."""
        # Convert to model if needed for compatibility
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        for toc_topic in draft_model.toc:
            if toc_topic.topic_id == item_id:
                return {
                    "id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "title": toc_topic.topic,  # For backward compatibility
                    "order": toc_topic.order,
                    "subtopics": [
                        {
                            "id": sub.subtopic_id,
                            "topic": sub.topic,
                            "title": sub.topic  # For backward compatibility
                        }
                        for sub in toc_topic.subtopics
                    ]
                }
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
        Note: Still accepts dict for API compatibility but works with typed structure internally.
        """
        # Convert to model if needed
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        context_parts = {
            "document_context": {
                "title": draft_model.metadata.title,
                "description": draft_model.metadata.description,
                "client_metadata": draft_model.metadata.client_metadata.model_dump() if draft_model.metadata.client_metadata else {}
            },
            "previous_topics": [],
            "current_topic_toc": None
        }

        # Find target position in TOC
        target_found = False

        for toc_topic in draft_model.toc:
            if target_found:
                break

            if toc_topic.topic_id == target_topic_id:
                context_parts["current_topic_toc"] = {
                    "topic_id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "order": toc_topic.order
                }

                if target_subtopic_id:
                    # Include main topic content if generating subtopic
                    if toc_topic.content:
                        context_parts["current_topic_content"] = {
                            "title": toc_topic.topic,
                            "content": toc_topic.content if not toc_topic.summary else toc_topic.summary
                        }

                    # Include previous subtopics of current topic
                    for toc_subtopic in toc_topic.subtopics:
                        if toc_subtopic.subtopic_id == target_subtopic_id:
                            break

                        if toc_subtopic.content:
                            context_parts["previous_topics"].append({
                                "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                                "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
                                "is_subtopic": True
                            })

                target_found = True
            else:
                # Add all content from topics before target
                if toc_topic.content:
                    context_parts["previous_topics"].append({
                        "title": toc_topic.topic,
                        "content": toc_topic.content if not toc_topic.summary else toc_topic.summary,
                        "is_subtopic": False
                    })

                # Add subtopics
                for toc_subtopic in toc_topic.subtopics:
                    if toc_subtopic.content:
                        context_parts["previous_topics"].append({
                            "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                            "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
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
            

            # Log conversation history BEFORE generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"BEFORE generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"BEFORE generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Generate content using the generation service
            content, source_ids = await self.generation_service.generate_topic_content(
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                vector_store=vector_store,
                llm_service=llm_service
            )

            # Log conversation history AFTER generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"AFTER generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"AFTER generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Save the generated content to the topic/subtopic
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        # Save content to subtopic
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.content = content
                                logger.info(f"Saved content to subtopic {subtopic_id}")
                                break
                    else:
                        # Save content to main topic
                        toc_topic.content = content
                        logger.info(f"Saved content to topic {topic_id}")
                    break

            # Check if we need to generate a summary before saving
            word_count = len(content.split())
            summary = ""
            if word_count > 150:
                # Get the topic title for summary generation
                toc_item = self.get_toc_item_by_id(draft.model_dump(), topic_id)

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

                # Update summary directly in the draft object
                for toc_topic in draft.toc:
                    if toc_topic.topic_id == topic_id:
                        if subtopic_id:
                            for toc_subtopic in toc_topic.subtopics:
                                if toc_subtopic.subtopic_id == subtopic_id:
                                    toc_subtopic.summary = summary
                                    break
                        else:
                            toc_topic.summary = summary
                        break

            # Log conversation history state before saving (for debugging)
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"Saving subtopic {subtopic_id} with {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"Saving topic {topic_id} with {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Save the entire modified draft (includes conversation history, content, and summary)
            success = self.storage.update_draft(draft_id, draft)

            if success:
                
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
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.summary = summary
                                break
                    else:
                        toc_topic.summary = summary

                    return self.storage.update_draft(draft_id, draft_model)

            return False

        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
            return False
    
    def get_draft_progress(self, draft_id: str) -> Dict[str, Any]:
        """Calculate draft completion progress."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return {"error": "Draft not found"}

            total_items = 0
            completed_items = 0

            for toc_topic in draft_model.toc:
                total_items += 1
                if toc_topic.content:
                    completed_items += 1

                for toc_subtopic in toc_topic.subtopics:
                    total_items += 1
                    if toc_subtopic.content:
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
