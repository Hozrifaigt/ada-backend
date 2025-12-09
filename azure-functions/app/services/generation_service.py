import logging
from datetime import datetime, timezone
from pathlib import Path

from typing import Dict, Any, List, Optional, Tuple
from app.prompts.manager import PromptManager
from app.services.query_classification_service import QueryClassificationService, create_classification_service
from app.services.reflection_agent import ReflectionAgent, ReflectionCritique
from app.models.classification_models import QueryCategory
from app.models.draft_models import CompleteDraft, DraftMetadata
from app.models.regulation_models import RegulationChunk
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ContentGenerationService:
    """Service for generating content with RAG and sequential context."""

    def __init__(self, reflection_agent: Optional[ReflectionAgent] = None):
        """
        Initialize the generation service.
        
        Args:
            reflection_agent: Optional reflection agent for response critique
        """
        # Initialize query classification service
        self.classification_service = create_classification_service()
        self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        self.reflection_agent = reflection_agent

    def _map_detail_level(self, detail_level: int) -> str:
        """
        Map detail level int (1-5) to descriptive string for prompts.
        Based on the Light (Principle-Based) vs Heavy (Prescriptive) policy framework.

        Args:
            detail_level: Integer from 1 to 5

        Returns:
            Descriptive string for prompt
        """
        mapping = {
            1: "Light (Principle-Based) - Focus on high-level principles and outcomes",
            2: "Light-to-Moderate - Balanced approach with some flexibility",
            3: "Moderate - Balance between principles and procedures",
            4: "Moderate-to-Heavy - More detailed with clear procedures",
            5: "Heavy (Prescriptive) - Detailed, step-by-step, compliance focused"
        }
        return mapping.get(detail_level, mapping[3])  # Default to medium

    async def generate_topic_content(
        self,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        vector_store,
        llm_service
    ) -> Tuple[str, List[str], bool, Optional[ReflectionCritique]]:
        """
        Generate content for a topic or subtopic using RAG and context.

        Returns:
            Tuple of (generated_content, source_ids_used, is_chat_response, critique)
            where critique is None for chat responses or if reflection agent unavailable
        """
        try:
            # Step 1: Find the topic in TOC
            toc_item = self._find_toc_item(draft.toc, topic_id)
            if not toc_item:
                raise ValueError(f"Topic {topic_id} not found in TOC")

            # Step 2: Get conversation history for classification (before classification)
            classification_history = self._get_conversation_history(
                draft, topic_id, subtopic_id, limit=3, for_llm_context=False
            )
            formatted_classification_history = self._format_conversation_history_for_prompt(classification_history)

            # Step 3: Classify the user query
            # Prepare context for classification
            topic_title = toc_item.topic
            classification_context = {
                "topic_title": topic_title,
                "document_type": "policy",
                "document_title": draft.metadata.title
            }

            # Classify the query with conversation context (using classification history)
            classification_result = await self.classification_service.classify_query(
                user_prompt, classification_context, formatted_classification_history
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
                # Return the suggested response as a chat response, not content (no critique for blocked queries)
                return suggested_response, [], True, None

            # Step 4: Determine what we're generating
            subtopic_item = None  # Initialize for later reuse
            if subtopic_id:
                subtopic_item = self._find_subtopic(toc_item, subtopic_id)
                if not subtopic_item:
                    raise ValueError(f"Subtopic {subtopic_id} not found")
                topic_title = subtopic_item.topic
                source_id = subtopic_item.source_subtopic_id
            else:
                topic_title = toc_item.topic
                source_id = toc_item.source_topic_id

            # # Step 5: Get conversation history for generation - last 3 legitimate only
            # generation_history = self._get_conversation_history(
            #     draft, topic_id, subtopic_id, limit=3, for_llm_context=True
            # )
            # formatted_generation_history = self._format_conversation_history_for_prompt(generation_history)

            # # Step 6: Try to retrieve existing content if source_id exists
            # if source_id:
            #     existing_content = await self._retrieve_existing_content(source_id, vector_store)
            #     if existing_content:
            #         adapted_content = await self._adapt_existing_content(
            #             existing_content=existing_content,
            #             user_prompt=user_prompt,
            #             draft=draft,
            #             topic_id=topic_id,
            #             subtopic_id=subtopic_id,
            #             conversation_history=formatted_generation_history,
            #             llm_service=llm_service
            #         )

            #         # Add conversation to draft history before returning
            #         logger.info(f"DEBUG: About to add conversation for topic {topic_id}, subtopic {subtopic_id} (adapted content path)")
            #         self._add_conversation_to_draft(
            #             draft, topic_id, user_prompt, adapted_content, subtopic_id
            #         )

            #         return adapted_content, [source_id], False

            # # Step 7: Search for similar topics (for custom topics without source_id)
            # similar_topics = await self._search_similar_topics(topic_title, vector_store, threshold=0.7)

            # Step 7.5: Search for relevant regulations
            regulation_chunks = await self._search_relevant_regulations(
                topic_title=topic_title,
                user_prompt=user_prompt,
                draft_metadata=draft.metadata,
                vector_store=vector_store
            )
            
            # Step 8: Build complete context
            sequential_context = self._build_sequential_context(
                draft,
                topic_id,
                subtopic_id
            )

            # Format the full TOC for LLM context
            full_toc = self._format_full_toc(draft.toc)

            # Step 9: Build topic structure guidance
            # If generating a main topic with subtopics, guide LLM to be introductory
            topic_structure_guidance = self._build_topic_structure_guidance(
                toc_item=toc_item,
                subtopic_id=subtopic_id
            )

            # Step 10: Generate content (with regulations context if available)
            content = await self._generate_from_scratch(
                title=topic_title,
                user_prompt=user_prompt,
                sequential_context=sequential_context,
                draft_metadata=draft.metadata,
                conversation_history="",
                llm_service=llm_service,
                full_toc=full_toc,
                regulation_chunks=regulation_chunks,
                topic_structure_guidance=topic_structure_guidance
            )
            source_ids = []

            # Step 11: Reflection & Critique (if agent available)
            # Flow: critique_and_fix() does one critique + regenerate if needed (3 LLM calls max)
            # Note: We reuse toc_item and subtopic_item from Step 4 above
            critique = None
            if self.reflection_agent:
                try:
                    # Get conversation history (reuse toc_item from Step 1, subtopic_item from Step 4)
                    if subtopic_id and subtopic_item:
                        conversation_history = subtopic_item.conversation_history
                    else:
                        conversation_history = toc_item.conversation_history

                    # Perform critique and auto-fix if needed (single call handles everything)
                    critique = await self.reflection_agent.critique_and_fix(
                        generated_content=content,
                        draft=draft,
                        topic_id=topic_id,
                        subtopic_id=subtopic_id,
                        user_prompt=user_prompt,
                        conversation_history=conversation_history,
                        regulation_chunks=regulation_chunks,
                        source_ids=source_ids,
                        detail_level=draft.metadata.detail_level
                    )

                    # Use fixed content if available (score was below threshold)
                    if critique.fixed_content:
                        logger.info("Using auto-fixed content from reflection agent")
                        content = critique.fixed_content

                    # Log critique results
                    await self.reflection_agent.log_critique_results(
                        critique=critique,
                        draft_id=draft.id,
                        topic_id=topic_id,
                        subtopic_id=subtopic_id
                    )

                    logger.info(
                        f"Reflection critique completed - Score: {critique.score:.2f}, "
                        f"Alignment: {critique.alignment_score:.2f}, "
                        f"Regulation Compliance: {critique.regulation_compliance_score:.2f}"
                    )

                except Exception as e:
                    logger.error(f"Error in reflection critique: {e}", exc_info=True)
                    # Continue without critique on error

            # Step 12: Add final content to conversation history
            # Only add after critique so we store the best version (original or fixed)
            self._add_conversation_to_draft(
                draft, topic_id, user_prompt, content, subtopic_id
            )

            return content, source_ids, False, critique

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

    async def _search_relevant_regulations(
        self,
        topic_title: str,
        user_prompt: str,
        draft_metadata: DraftMetadata,
        vector_store
    ) -> List[RegulationChunk]:
        """
        Search for relevant regulation chunks based on topic and context.

        Args:
            topic_title: Title of the current topic
            user_prompt: User's request/query (not used in search - often generic like "generate content")
            draft_metadata: Draft metadata with regulations field
            vector_store: Vector store instance

        Returns:
            List of relevant RegulationChunk objects
        """
        # Only search if regulations field is set
        if not draft_metadata.regulations or draft_metadata.regulations.strip() == "":
            logger.info("No regulations specified, skipping regulation search")
            return []

        try:
            # Build search query from topic title and policy type
            # Note: user_prompt is excluded as it's often generic (e.g., "generate content for this topic")
            # policy_type is more descriptive (e.g., "Leave Policy", "Termination Policy")
            query_text = f"{topic_title} {draft_metadata.policy_type}"

            # Build context for better search
            context = {
                "industry": draft_metadata.client_metadata.industry,
                "policy_type": draft_metadata.policy_type,
                "function": draft_metadata.function,
                "client_requests": draft_metadata.client_specific_requests or "",
                "sector_comments": draft_metadata.sector_specific_comments or ""
            }

            logger.info(f"Searching regulations for: {draft_metadata.regulations} with query: {query_text}")

            # Search regulations using settings constants
            regulation_chunks = await vector_store.search_relevant_regulations(
                query_text=query_text,
                regulation_name=draft_metadata.regulations,
                top_k=settings.REGULATION_SEARCH_TOP_K,
                min_score=settings.REGULATION_SEARCH_MIN_SCORE,
                context=context
            )

            # Log retrieved regulations for debugging
            if regulation_chunks:
                logger.info(f"Retrieved {len(regulation_chunks)} regulations for '{topic_title}':")
                for chunk in regulation_chunks[:3]:
                    topics_preview = chunk.topics[:3] if chunk.topics else []
                    logger.info(f"  - {chunk.article_number} (score: {chunk.score:.2f}): {topics_preview}")
            else:
                logger.warning(f"No regulations found for '{topic_title}' with query: {query_text}")

            return regulation_chunks

        except Exception as e:
            logger.error(f"Failed to search regulations: {e}")
            return []

    def _format_regulations_for_prompt(self, regulation_chunks: List[RegulationChunk]) -> str:
        """
        Format regulation chunks for inclusion in the prompt.

        Args:
            regulation_chunks: List of RegulationChunk objects

        Returns:
            Formatted string for prompt
        """
        if not regulation_chunks:
            return ""

        formatted_parts = []
        for i, chunk in enumerate(regulation_chunks):
            article_info = f"{chunk.article_number}" if chunk.article_number else "Section"
            decree_name = chunk.decree_name
            effective_date = chunk.effective_date or "Unknown"
            content = chunk.content

            formatted_parts.append(
                f"[Regulation {i+1}] {article_info} - {decree_name} (Effective: {effective_date})\n{content}"
            )

        return "\n\n".join(formatted_parts)

    def _build_topic_structure_guidance(
        self,
        toc_item,
        subtopic_id: Optional[str]
    ) -> str:
        """
        Build guidance for content depth based on topic structure.

        If generating a main topic that has subtopics, instruct LLM to be
        introductory/high-level since details will be covered in subtopics.

        Args:
            toc_item: The topic being generated
            subtopic_id: If provided, we're generating a subtopic (not main topic)

        Returns:
            Guidance string for the prompt
        """
        # If generating a subtopic, no special guidance needed
        if subtopic_id:
            return ""

        # Check if main topic has subtopics
        if not toc_item.subtopics or len(toc_item.subtopics) == 0:
            return ""

        # Main topic has subtopics - provide guidance to be introductory
        subtopic_names = [sub.topic for sub in toc_item.subtopics]
        subtopics_list = ", ".join(f'"{name}"' for name in subtopic_names)

        guidance = f"""TOPIC STRUCTURE GUIDANCE:
This topic "{toc_item.topic}" has the following subtopics that will be created separately:
{subtopics_list}

IMPORTANT: Since detailed content will be provided in each subtopic, this main topic should:
1. Provide a HIGH-LEVEL OVERVIEW and introduction to the subject
2. Explain the PURPOSE and SCOPE of this policy area
3. List the key principles or general guidelines
4. DO NOT go into specific details that belong in the subtopics listed above
5. Keep content introductory - the subtopics will cover specifics like procedures, eligibility, and detailed requirements

For example, if this is "Leaves" with subtopics "Annual Leave", "Sick Leave", "Maternity Leave":
- This main topic should introduce the company's leave philosophy and general entitlements
- Specific details about annual leave duration, sick leave procedures, or maternity leave eligibility should be deferred to their respective subtopics"""

        return guidance

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

        # Format the full TOC for context
        full_toc = self._format_full_toc(draft.toc)

        # Use prompt manager
        prompt = self.prompt_manager.get_prompt(
            "adapt_existing_content",
            document_title=draft.metadata.title,
            document_description=draft.metadata.description or "",
            function=draft.metadata.function,
            policy_type=draft.metadata.policy_type,
            industry=draft.metadata.client_metadata.industry,
            client_specific_requests=draft.metadata.client_specific_requests or "",
            sector_specific_comments=draft.metadata.sector_specific_comments or "",
            regulations=draft.metadata.regulations or "",
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            full_table_of_contents=full_toc,
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
        llm_service,
        full_toc: str,
        regulation_chunks: List[RegulationChunk] = None
    ) -> str:
        """Generate content using similar topics as reference."""

        # Format similar topics for prompt
        references = "\n\n".join([
            f"Reference {i+1} - {topic['title']}:\n{topic['content']}"
            for i, topic in enumerate(similar_topics[:3])
        ])

        # Format regulations for prompt
        regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])

        # Use prompt manager
        prompt = self.prompt_manager.get_prompt(
            "generate_with_references",
            document_title=draft_metadata.title,
            document_description=draft_metadata.description or "",
            function=draft_metadata.function,
            policy_type=draft_metadata.policy_type,
            industry=draft_metadata.client_metadata.industry,
            client_specific_requests=draft_metadata.client_specific_requests or "",
            sector_specific_comments=draft_metadata.sector_specific_comments or "",
            regulations=draft_metadata.regulations or "",
            regulations_context=regulations_context,
            detail_level=self._map_detail_level(draft_metadata.detail_level),
            full_table_of_contents=full_toc,
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
        llm_service,
        full_toc: str,
        regulation_chunks: List[RegulationChunk] = None,
        topic_structure_guidance: str = ""
    ) -> str:
        """Generate content from scratch when no references are available."""

        # Format regulations for prompt
        regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])

        prompt = self.prompt_manager.get_prompt(
            "generate_from_scratch",
            document_title=draft_metadata.title,
            document_description=draft_metadata.description or "",
            function=draft_metadata.function,
            policy_type=draft_metadata.policy_type,
            industry=draft_metadata.client_metadata.industry,
            client_specific_requests=draft_metadata.client_specific_requests or "",
            sector_specific_comments=draft_metadata.sector_specific_comments or "",
            regulations=draft_metadata.regulations or "",
            regulations_context=regulations_context,
            detail_level=self._map_detail_level(draft_metadata.detail_level),
            full_table_of_contents=full_toc,
            topic_title=title,
            user_requirements=user_prompt,
            previous_topics=self._format_context_for_prompt(sequential_context),
            conversation_history=conversation_history,
            topic_structure_guidance=topic_structure_guidance
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

    def _format_full_toc(self, toc: List) -> str:
        """Format the full table of contents with summaries for LLM context."""
        if not toc:
            return "No table of contents available."

        formatted = []
        for topic in toc:
            # Add main topic with summary if available
            topic_line = f"{topic.order}. {topic.topic}"

            # Add status indicator and summary
            if topic.content:
                if topic.summary:
                    topic_line += f" [COMPLETED - Summary: {topic.summary}]"
                else:
                    # If content exists but no summary (content < 150 words), show brief content
                    word_count = len(topic.content.split())
                    if word_count <= 50:
                        topic_line += f" [COMPLETED - Content: {topic.content}]"
                    else:
                        # Take first 50 words as brief summary
                        brief = " ".join(topic.content.split()[:50]) + "..."
                        topic_line += f" [COMPLETED - Brief: {brief}]"
            else:
                topic_line += " [PENDING]"

            formatted.append(topic_line)

            # Add subtopics with summaries if they exist
            if topic.subtopics:
                for subtopic in topic.subtopics:
                    subtopic_line = f"   {topic.order}.{subtopic.order} {subtopic.topic}"

                    # Add status indicator and summary
                    if subtopic.content:
                        if subtopic.summary:
                            subtopic_line += f" [COMPLETED - Summary: {subtopic.summary}]"
                        else:
                            # If content exists but no summary (content < 150 words)
                            word_count = len(subtopic.content.split())
                            if word_count <= 50:
                                subtopic_line += f" [COMPLETED - Content: {subtopic.content}]"
                            else:
                                # Take first 50 words as brief summary
                                brief = " ".join(subtopic.content.split()[:50]) + "..."
                                subtopic_line += f" [COMPLETED - Brief: {brief}]"
                    else:
                        subtopic_line += " [PENDING]"

                    formatted.append(subtopic_line)

        return "\n".join(formatted)

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
        limit: int = 3,
        for_llm_context: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a topic or subtopic.

        Args:
            draft: The draft object
            topic_id: Topic ID
            subtopic_id: Optional subtopic ID
            limit: Maximum number of entries to return
            for_llm_context: If True, filter to only legitimate content responses
        """
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

            # Sort by timestamp (newest first)
            sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)

            if for_llm_context:
                # Filter to get only legitimate conversations for LLM
                # (ones that resulted in actual content, not chat responses)
                chat_response_indicators = [
                    "I'm here to help with policy",
                    "I can't help with that request",
                    "Let's focus on creating",
                    "How can I assist you with creating",
                    "I need more specific information",
                    "Let's focus on policy content",
                    "I can't assist with that"
                ]

                legitimate_history = []
                for entry in sorted_history:
                    ai_response = entry.get('ai_response', '')
                    # Check if it's a meaningful content response
                    # Check first 200 chars for efficiency
                    is_chat_response = any(
                        indicator in ai_response[:200]
                        for indicator in chat_response_indicators
                    )

                    if not is_chat_response and ai_response.strip() and len(ai_response) > 100:
                        # This is legitimate content (not a short chat response)
                        legitimate_history.append(entry)
                        if len(legitimate_history) >= limit:
                            break

                logger.info(f"Filtered history: {len(legitimate_history)} legitimate out of {len(sorted_history)} total")
                return legitimate_history

            # For non-LLM context, return all recent entries
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