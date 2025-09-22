import logging
from datetime import datetime, timezone
from pathlib import Path

from typing import Dict, Any, List, Optional, Tuple
from app.prompts.manager import PromptManager
from app.services.query_classification_service import QueryClassificationService, create_classification_service
from app.models.classification_models import QueryCategory
from app.models.draft_models import CompleteDraft, DraftMetadata
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ContentGenerationService:
    """Service for generating content with RAG and sequential context."""

    def __init__(self):
        """Initialize the generation service."""
        # Initialize query classification service
        self.classification_service = create_classification_service()
        self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)

    async def generate_topic_content(
        self,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        vector_store,
        llm_service
    ) -> Tuple[str, List[str]]:
        """
        Generate content for a topic or subtopic using RAG and context.

        Returns:
            Tuple of (generated_content, source_ids_used)
        """
        try:
            # Step 1: Classify the user query
            toc_item = self._find_toc_item(draft.toc, topic_id)
            if not toc_item:
                raise ValueError(f"Topic {topic_id} not found in TOC")

            # Prepare context for classification
            topic_title = toc_item.topic
            classification_context = {
                "topic_title": topic_title,
                "document_type": "policy",
                "document_title": draft.metadata.title
            }

            # Get conversation history for classification
            conversation_history = self._get_conversation_history(
                draft, topic_id, subtopic_id, limit=3
            )
            formatted_history = self._format_conversation_history_for_prompt(conversation_history)

            # Classify the query with conversation context
            classification_result = await self.classification_service.classify_query(
                user_prompt, classification_context, formatted_history
            )

            # Handle non-legitimate queries
            if not classification_result.classification.should_proceed:
                logger.info(
                    f"Query blocked - Category: {classification_result.classification.category}, "
                    f"Query: {user_prompt[:100]}..."
                )
                suggested_response = self.classification_service.get_suggested_response(
                    classification_result.classification
                )
                # Return the suggested response instead of generated content
                return suggested_response, []

            # Step 2: Determine what we're generating
            if subtopic_id:
                subtopic_item = self._find_subtopic(toc_item, subtopic_id)
                if not subtopic_item:
                    raise ValueError(f"Subtopic {subtopic_id} not found")
                title = subtopic_item.topic
                source_id = subtopic_item.source_subtopic_id
            else:
                title = toc_item.topic
                source_id = toc_item.source_topic_id

            # Note: We already have formatted_history from classification step above

            # Step 4: Try to retrieve existing content if source_id exists
            if source_id:
                existing_content = await self._retrieve_existing_content(source_id, vector_store)
                if existing_content:
                    adapted_content = await self._adapt_existing_content(
                        existing_content=existing_content,
                        user_prompt=user_prompt,
                        draft=draft,
                        topic_id=topic_id,
                        subtopic_id=subtopic_id,
                        conversation_history=formatted_history,
                        llm_service=llm_service
                    )

                    # Add conversation to draft history before returning
                    logger.info(f"DEBUG: About to add conversation for topic {topic_id}, subtopic {subtopic_id} (adapted content path)")
                    self._add_conversation_to_draft(
                        draft, topic_id, user_prompt, adapted_content, subtopic_id
                    )

                    return adapted_content, [source_id]

            # Step 5: Search for similar topics (for custom topics)
            similar_topics = await self._search_similar_topics(title, vector_store, threshold=0.7)

            # Step 6: Build complete context
            sequential_context = self._build_sequential_context(
                draft,
                topic_id,
                subtopic_id
            )

            # Step 7: Generate content based on what we found
            if similar_topics:
                # Generate with similar topics as reference
                content = await self._generate_with_references(
                    title=title,
                    user_prompt=user_prompt,
                    similar_topics=similar_topics,
                    sequential_context=sequential_context,
                    draft_metadata=draft.metadata,
                    conversation_history=formatted_history,
                    llm_service=llm_service
                )
                source_ids = [topic.get("id", topic.get("topic_id", "unknown")) for topic in similar_topics[:3]]
            else:
                # Generate from scratch
                content = await self._generate_from_scratch(
                    title=title,
                    user_prompt=user_prompt,
                    sequential_context=sequential_context,
                    draft_metadata=draft.metadata,
                    conversation_history=formatted_history,
                    llm_service=llm_service
                )
                source_ids = []

            # Step 8: Add conversation to draft history
            logger.info(f"DEBUG: About to add conversation for topic {topic_id}, subtopic {subtopic_id}")
            self._add_conversation_to_draft(
                draft, topic_id, user_prompt, content, subtopic_id
            )

            # Verify conversation was added
            topic = self._find_toc_item(draft.toc, topic_id)
            if topic:
                if subtopic_id:
                    subtopic = self._find_subtopic(topic, subtopic_id)
                    if subtopic:
                        logger.info(f"DEBUG: After adding conversation - subtopic {subtopic_id} now has {len(subtopic.conversation_history)} entries")
                else:
                    logger.info(f"DEBUG: After adding conversation - topic {topic_id} now has {len(topic.conversation_history)} entries")

            return content, source_ids

        except Exception as e:
            import traceback
            logger.error(f"Failed to generate content: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _retrieve_existing_content(
        self,
        source_id: str,
        vector_store
    ) -> Optional[str]:
        """Retrieve content for an existing topic/subtopic by ID."""
        try:
            # Use the vector store's get_topic_by_id method
            topic = vector_store.get_topic_by_id(source_id)

            if topic:
                return topic.get("topic_content", "")

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve content for {source_id}: {e}")
            return None

    async def _search_similar_topics(
        self,
        topic_title: str,
        vector_store,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar topics in the knowledge base."""
        try:
            # Generate embedding for the topic title
            embedding = vector_store.embedder.embed_query(topic_title)

            # Perform vector search using the new search_similar_topics method
            results = vector_store.search_similar_topics(
                query_vector=embedding,
                threshold=threshold,
                top_k=5
            )

            return [
                {
                    "id": r.get("id"),
                    "title": r.get("topic"),
                    "content": r.get("topic_content", "")[:500]  # First 500 chars for prompt efficiency
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Failed to search similar topics: {e}")
            return []

    async def _adapt_existing_content(
        self,
        existing_content: str,
        user_prompt: str,
        draft: Dict[str, Any],
        topic_id: str,
        subtopic_id: Optional[str],
        conversation_history: str,
        llm_service
    ) -> str:
        """Adapt existing content based on user requirements and context."""

        sequential_context = self._build_sequential_context(
            draft, topic_id, subtopic_id
        )

        # Use prompt manager
        prompt = self.prompt_manager.get_prompt(
            "adapt_existing_content",
            document_title=draft.metadata.title,
            document_description=draft.metadata.description,
            previous_topics=self._format_context_for_prompt(sequential_context),
            existing_content=existing_content,
            user_requirements=user_prompt,
            conversation_history=conversation_history
        )

        response = await llm_service.generate(prompt)
        return response.strip()


    async def _generate_with_references(
        self,
        title: str,
        user_prompt: str,
        similar_topics: List[Dict[str, Any]],
        sequential_context: Dict[str, Any],
        draft_metadata: DraftMetadata,
        conversation_history: str,
        llm_service
    ) -> str:
        """Generate content using similar topics as reference."""

        # Format similar topics for prompt
        references = "\n\n".join([
            f"Reference {i+1} - {topic['title']}:\n{topic['content']}"
            for i, topic in enumerate(similar_topics[:3])
        ])

        # Use prompt manager
        prompt = self.prompt_manager.get_prompt(
            "generate_with_references",
            document_title=draft_metadata.title,
            document_description=draft_metadata.description,
            topic_title=title,
            user_requirements=user_prompt,
            previous_topics=self._format_context_for_prompt(sequential_context),
            similar_topics=references,
            conversation_history=conversation_history
        )

        response = await llm_service.generate(prompt)
        return response.strip()


    async def _generate_from_scratch(
        self,
        title: str,
        user_prompt: str,
        sequential_context: Dict[str, Any],
        draft_metadata: DraftMetadata,
        conversation_history: str,
        llm_service
    ) -> str:
        """Generate content from scratch when no references are available."""

        prompt = self.prompt_manager.get_prompt(
            "generate_from_scratch",
            document_title=draft_metadata.title,
            document_description=draft_metadata.description,
            topic_title=title,
            user_requirements=user_prompt,
            previous_topics=self._format_context_for_prompt(sequential_context),
            conversation_history=conversation_history
        )

        response = await llm_service.generate(prompt)
        return response.strip()


    def _build_sequential_context(
        self,
        draft: CompleteDraft,
        target_topic_id: str,
        target_subtopic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build context from previous sections."""
        context = {
            "previous_topics": [],
            "current_topic": None
        }

        for topic_toc in draft.toc:
            # Find corresponding content
            topic_id = topic_toc.topic_id

            if topic_id == target_topic_id:
                context["current_topic"] = topic_toc.topic

                # If generating subtopic, include main topic content
                if target_subtopic_id and topic_toc.content:
                    content_to_use = topic_toc.content
                    if len(content_to_use.split()) <= 150:
                        context["current_topic_content"] = content_to_use
                    else:
                        # Use summary or first 150 words
                        context["current_topic_content"] = " ".join(content_to_use.split()[:150]) + "..."

                # Include previous subtopics if generating subtopic
                if target_subtopic_id:
                    for subtopic_toc in topic_toc.subtopics:
                        if subtopic_toc.subtopic_id == target_subtopic_id:
                            break

                        if subtopic_toc.content:
                            self._add_to_context(
                                context["previous_topics"],
                                f"{topic_toc.topic} - {subtopic_toc.topic}",
                                subtopic_toc.content
                            )
                break
            else:
                # Add all content from topics before target
                if topic_toc.content:
                    self._add_to_context(
                        context["previous_topics"],
                        topic_toc.topic,
                        topic_toc.content
                    )

                # Add subtopics
                for subtopic_toc in topic_toc.subtopics:
                    if subtopic_toc.content:
                        self._add_to_context(
                            context["previous_topics"],
                            f"{topic_toc.topic} - {subtopic_toc.topic}",
                            subtopic_toc.content
                        )

        return context

    def _add_to_context(
        self,
        context_list: List[Dict[str, str]],
        title: str,
        content: str
    ):
        """Add content to context list with appropriate summary."""
        word_count = len(content.split())
        if word_count <= 150:
            context_list.append({
                "title": title,
                "content": content
            })
        else:
            # Take first 150 words as summary
            summary = " ".join(content.split()[:150]) + "..."
            context_list.append({
                "title": title,
                "content": summary
            })

    def _format_context_for_prompt(self, sequential_context: Dict[str, Any]) -> str:
        """Format context for inclusion in LLM prompt."""
        if not sequential_context["previous_topics"]:
            return "No previous topics yet."

        formatted = []
        for item in sequential_context["previous_topics"]:
            formatted.append(f"- {item['title']}:\n  {item['content']}")

        return "\n\n".join(formatted)

    def _find_toc_item(
        self,
        toc: List,
        topic_id: str
    ):
        """Find a topic in the TOC by ID."""
        for topic in toc:
            if topic.topic_id == topic_id:
                return topic
        return None

    def _find_subtopic(
        self,
        topic,
        subtopic_id: str
    ):
        """Find a subtopic within a topic."""
        for subtopic in topic.subtopics:
            if subtopic.subtopic_id == subtopic_id:
                return subtopic
        return None


    def _get_conversation_history(
        self,
        draft,
        topic_id: str,
        subtopic_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a topic or subtopic."""
        try:
            # Find topic in TOC
            topic = self._find_toc_item(draft.toc, topic_id)
            if not topic:
                return []

            if subtopic_id:
                # Get subtopic conversation history
                subtopic = self._find_subtopic(topic, subtopic_id)
                if subtopic and subtopic.conversation_history:
                    history = [entry.model_dump() for entry in subtopic.conversation_history]
                else:
                    return []
            else:
                # Get topic conversation history
                if topic.conversation_history:
                    history = [entry.model_dump() for entry in topic.conversation_history]
                else:
                    return []

            # Sort by timestamp (newest first) and limit to last N messages
            sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
            return sorted_history[:limit]

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def _format_conversation_history_for_prompt(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Format conversation history for inclusion in LLM prompts."""
        if not conversation_history:
            return "No previous conversation history."

        # Reverse to show chronological order (oldest first)
        chronological_history = list(reversed(conversation_history))

        formatted_history = []
        for i, conv in enumerate(chronological_history, 1):
            user_msg = conv['user_message']
            ai_response = conv['ai_response']

            # Truncate long messages for prompt efficiency
            if len(user_msg) > 200:
                user_msg = user_msg[:197] + "..."
            if len(ai_response) > 300:
                ai_response = ai_response[:297] + "..."

            formatted_history.append(
                f"Previous Interaction {i}:\n"
                f"User: {user_msg}\n"
                f"Assistant: {ai_response}"
            )

        return "\n\n".join(formatted_history)

    def _add_conversation_to_draft(
        self,
        draft: CompleteDraft,
        topic_id: str,
        user_message: str,
        ai_response: str,
        subtopic_id: Optional[str] = None
    ) -> None:
        """Add a conversation entry to the draft's conversation history."""
        try:
            from app.models.draft_models import ConversationEntry

            timestamp = datetime.now(timezone.utc).isoformat()

            conversation_entry = ConversationEntry(
                timestamp=timestamp,
                user_message=user_message,
                ai_response=ai_response
            )

            # Find topic in TOC
            topic = self._find_toc_item(draft.toc, topic_id)
            if not topic:
                logger.warning(f"Topic {topic_id} not found in draft")
                return

            if subtopic_id:
                # Add to subtopic conversation history
                subtopic = self._find_subtopic(topic, subtopic_id)
                if subtopic:
                    subtopic.conversation_history.append(conversation_entry)
                    logger.info(f"Added conversation entry for subtopic {subtopic_id}")
                else:
                    logger.warning(f"Subtopic {subtopic_id} not found in topic {topic_id}")
            else:
                # Add to topic conversation history
                topic.conversation_history.append(conversation_entry)
                logger.info(f"Added conversation entry for topic {topic_id}")

        except Exception as e:
            logger.error(f"Failed to add conversation to draft: {e}")

    async def generate_summary(
        self,
        content: str,
        title: str,
        llm_service
    ) -> str:
        """Generate a summary for content > 150 words."""
        word_count = len(content.split())
        if word_count <= 150:
            return ""

        prompt = self.prompt_manager.get_prompt(
            "generate_summary",
            topic_title=title,
            content=content
        )

        response = await llm_service.generate(prompt)
        return response.strip()