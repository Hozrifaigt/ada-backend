import json
import logging
import math
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from typing import Dict, Any, List, Optional, Tuple
from app.prompts.manager import PromptManager
from app.services.query_classification_service import QueryClassificationService, create_classification_service
from app.services.reflection_agent import ReflectionAgent, ReflectionCritique
from app.models.classification_models import QueryCategory
from app.models.draft_models import CompleteDraft, DraftMetadata, GapReport, GapFinding
from app.models.regulation_models import RegulationChunk
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Process-level cache of uploaded-regulation chunk embeddings, keyed by a hash of the chunk set, so the
# per-section "Check regulations" step doesn't re-embed the same regulation text on every click. Bounded.
_UPLOADED_REG_EMBED_CACHE: Dict[str, List[List[float]]] = {}
_UPLOADED_REG_EMBED_CACHE_MAX = 8


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


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
            # If this draft was started from an uploaded policy, adapt the existing content
            # instead of generating from scratch
            existing_uploaded_content = None
            if draft.metadata.toc_source == "uploaded_policy":
                if subtopic_id and subtopic_item:
                    existing_uploaded_content = subtopic_item.content or None
                elif not subtopic_id:
                    existing_uploaded_content = toc_item.content or None

            if existing_uploaded_content:
                content = await self._adapt_existing_content(
                    existing_content=existing_uploaded_content,
                    user_prompt=user_prompt,
                    draft=draft,
                    topic_id=topic_id,
                    subtopic_id=subtopic_id,
                    conversation_history="",
                    llm_service=llm_service,
                    regulation_chunks=regulation_chunks
                )
            else:
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
            query_text = f"{topic_title} {draft_metadata.effective_policy_type}"

            # Build context for better search
            context = {
                "industry": draft_metadata.client_metadata.industry,
                "policy_type": draft_metadata.effective_policy_type,
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
        llm_service,
        regulation_chunks: List[RegulationChunk] = None
    ) -> str:
        """Adapt existing content based on user requirements and context."""

        sequential_context = self._build_sequential_context(
            draft, topic_id, subtopic_id
        )

        # Format the full TOC for context
        full_toc = self._format_full_toc(draft.toc)

        # Regulation context comes ONLY from retrieved chunks — the whole uploaded_regulations_text
        # used to be appended here and overflowed the context window as files accumulated.
        regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])

        # Use prompt manager
        prompt = self.prompt_manager.get_prompt(
            "adapt_existing_content",
            document_title=draft.metadata.title,
            document_description=draft.metadata.description or "",
            function=draft.metadata.function,
            policy_type=draft.metadata.effective_policy_type,
            industry=draft.metadata.client_metadata.industry,
            client_specific_requests=draft.metadata.client_specific_requests or "",
            sector_specific_comments=draft.metadata.sector_specific_comments or "",
            regulations=draft.metadata.regulations or "",
            regulations_context=regulations_context,
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            full_table_of_contents=full_toc,
            previous_topics=self._format_context_for_prompt(sequential_context),
            existing_content=existing_content,
            user_requirements=user_prompt,
            conversation_history=conversation_history
        )

        response = await llm_service.generate(prompt)
        return response.strip()


    # ============================================================
    # GAP ASSESSMENT (Review Policy / review_mode)
    # ============================================================

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        """Parse a JSON object from an LLM response, tolerating markdown fences / extra prose."""
        text = (response or "").strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    async def infer_heading_levels(self, document_start: str, headings: List[Dict[str, Any]], llm_service):
        """LLM re-leveling for low-confidence docs. Given the doc's first words (its TOC, if any) and
        the detected heading list, return (updates {index: level}, drops {index}). Content is never
        sent and never returned — only the nesting decisions."""
        rendered = "\n".join(
            f"{h['index']}: [{h.get('hint', '')}] {h['title']}  (current_level {h['level']})"
            for h in headings
        )
        prompt = self.prompt_manager.get_prompt(
            "infer_heading_levels",
            document_start=document_start or "(none)",
            headings=rendered or "(none)",
        )
        response = await llm_service.generate(prompt)
        data = self._parse_json_object(response)

        # The TOC is the source of truth → apply the LLM's levels/drops directly. (The only safety
        # net is upstream: a result that produces zero topics is not adopted.)
        updates: Dict[int, int] = {}
        drops = set()
        for item in data.get("levels", []):
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if item.get("is_heading") is False:
                drops.add(idx)
                continue
            lvl = item.get("level")
            if isinstance(lvl, int) and lvl >= 1:
                updates[idx] = lvl
        return updates, drops

    def _format_findings_for_prompt(self, findings: List[Dict[str, Any]]) -> str:
        """Format confirmed gap findings as an instruction list for the apply prompt."""
        if not findings:
            return "(No changes confirmed.)"
        lines = []
        for i, f in enumerate(findings, 1):
            lines.append(
                f"{i}. [{f.get('type', '')}/{f.get('source', '')}] {f.get('description', '')}\n"
                f"   Apply: {f.get('suggested_change', '')}"
            )
        return "\n".join(lines)

    async def generate_gap_report(
        self,
        topic_id: str,
        topic_title: str,
        existing_content: str,
        benchmark_content: Optional[str],
        draft,
        previous_policy_excerpt: str,
        llm_service,
        regulation_chunks: List[RegulationChunk] = None,
        benchmark_topic_name: Optional[str] = None,
        surrounding_context: str = "",
        benchmark_only: bool = False,
    ) -> GapReport:
        """Compare a client topic's content against the benchmark + regulations and return a structured gap report.
        benchmark_only=True scopes the assessment to the benchmark (no regulation context) for the Review tab's
        Benchmark step."""
        if benchmark_only:
            regulations_context = ""
        else:
            # Retrieved chunks only — callers pass uploaded-regulation passages via regulation_chunks
            # (see _regulation_chunks_for_unit); the whole-text dump overflowed the context window.
            regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])

        prompt = self.prompt_manager.get_prompt(
            "gap_assessment",
            topic_title=topic_title,
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            client_specific_requests=draft.metadata.client_specific_requests or "",
            sector_specific_comments=draft.metadata.sector_specific_comments or "",
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            existing_content=existing_content or "(This topic has no client content yet.)",
            benchmark_content=benchmark_content or "None available",
            regulations_context=regulations_context or "None available",
            previous_policy_excerpt=previous_policy_excerpt or "None available",
            surrounding_context=surrounding_context or "None",
        )

        response = await llm_service.generate(prompt)
        try:
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse gap report JSON for topic '{topic_title}': {e}")
            logger.error(f"LLM response: {response[:500]}")
            data = {"findings": [], "overall_summary": "Gap assessment could not be parsed."}

        findings: List[GapFinding] = []
        for f in data.get("findings", []):
            try:
                ftype = str(f.get("type", "enhancement"))
                evidence = (str(f.get("evidence", "")).strip() or None)
                # Evidence gate (Stage 3d): missing/non_compliant findings must be backed by a quote
                # from the benchmark/regulation. Drop "made up" gaps with no evidence.
                if ftype in ("missing", "non_compliant") and not evidence:
                    logger.info(f"Dropping unevidenced {ftype} finding for '{topic_title}'")
                    continue
                findings.append(GapFinding(
                    type=ftype,
                    source=str(f.get("source", "benchmark")),
                    severity=str(f.get("severity", "medium")),
                    description=str(f.get("description", "")).strip(),
                    suggested_change=str(f.get("suggested_change", "")).strip(),
                    evidence=evidence,
                    evidence_source=(str(f.get("evidence_source", "")).strip() or None),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed gap finding: {e}")

        return GapReport(
            topic_id=topic_id,
            findings=findings,
            benchmark_topic_matched=benchmark_topic_name,
            overall_summary=str(data.get("overall_summary", "")).strip(),
        )

    async def apply_gap_to_content(
        self,
        existing_content: str,
        benchmark_content: Optional[str],
        confirmed_findings: List[Dict[str, Any]],
        draft,
        topic_id: str,
        subtopic_id: Optional[str],
        conversation_history: str,
        llm_service,
        regulation_chunks: List[RegulationChunk] = None,
    ) -> str:
        """Apply confirmed gap findings to a topic's existing content while preserving the baseline."""
        sequential_context = self._build_sequential_context(draft, topic_id, subtopic_id)
        full_toc = self._format_full_toc(draft.toc)

        # Retrieved chunks only (uploaded-regulation passages arrive via regulation_chunks).
        regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])

        prompt = self.prompt_manager.get_prompt(
            "adapt_with_gap",
            document_title=draft.metadata.title,
            document_description=draft.metadata.description or "",
            function=draft.metadata.function,
            industry=draft.metadata.client_metadata.industry,
            client_specific_requests=draft.metadata.client_specific_requests or "",
            sector_specific_comments=draft.metadata.sector_specific_comments or "",
            regulations=draft.metadata.regulations or "",
            regulations_context=regulations_context,
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            full_table_of_contents=full_toc,
            previous_topics=self._format_context_for_prompt(sequential_context),
            benchmark_content=benchmark_content or "None available",
            existing_content=existing_content,
            confirmed_findings=self._format_findings_for_prompt(confirmed_findings),
            review_intensity=getattr(draft.metadata, "review_intensity", "preserve") or "preserve",
        )

        response = await llm_service.generate(prompt)
        return response.strip()

    def _format_topics_for_reconcile(self, topics: List[Dict[str, Any]], snippet_chars: int = 400) -> str:
        """Format client/benchmark topics (title + short content snippet + subtopic titles) for the
        reconcile prompt. Subtopics are shown so the LLM can propose missing ones via add_subtopics."""
        if not topics:
            return "(none)"
        lines = []
        for t in topics:
            title = (t.get("topic") or t.get("name") or t.get("title") or "").strip()
            if not title:
                continue
            content = (t.get("topic_content") or t.get("content") or "").strip()
            snippet = content[:snippet_chars].replace("\n", " ")
            lines.append(f"- {title}" + (f": {snippet}…" if snippet else ""))
            subs = t.get("subtopics") or []
            if isinstance(subs, str):
                try:
                    subs = json.loads(subs)
                except Exception:
                    subs = []
            for s in subs:
                st = (s if isinstance(s, str) else (s.get("name") or s.get("topic") or s.get("title") or "")).strip()
                if st:
                    lines.append(f"    - (subtopic) {st}")
        return "\n".join(lines) if lines else "(none)"

    def _assemble_full_policy_text(self, draft) -> str:
        """Ordered title+content over all topics/subtopics, for a whole-policy pass."""
        parts = []
        for i, t in enumerate(draft.toc, 1):
            parts.append(f"## {i}. {t.topic}\n{(t.content or '').strip()}")
            for j, s in enumerate(t.subtopics, 1):
                parts.append(f"### {i}.{j} {s.topic}\n{(s.content or '').strip()}")
        return "\n\n".join(parts).strip()

    async def run_consistency_check(self, draft, llm_service) -> Dict[str, Any]:
        """Whole-policy consistency pass → parsed issues dict."""
        full = self._assemble_full_policy_text(draft)
        cap = 48000  # ~12k tokens; map-reduce over sections is a future enhancement for larger policies
        text = full if len(full) <= cap else full[:cap]
        if len(full) > cap:
            logger.warning(f"Consistency check truncated policy text from {len(full)} to {cap} chars")
        prompt = self.prompt_manager.get_prompt(
            "policy_consistency",
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            policy_text=text,
        )
        response = await llm_service.generate(prompt)
        try:
            return self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse consistency JSON: {e}")
            return {"issues": [], "summary": "Consistency check could not be parsed."}

    async def generate_gap_summary_narrative(self, draft, summary, llm_service) -> str:
        """Write the executive-summary narrative for a gap-assessment review."""
        rollup = (
            f"Units reviewed: {summary.total_units} (applied {summary.applied}, assessed {summary.assessed}, "
            f"pending {summary.pending}). Findings: {summary.total_findings} "
            f"(severity {summary.by_severity}, type {summary.by_type}, source {summary.by_source}). "
            f"Benchmark-matched units: {summary.benchmark_coverage}. "
            f"Baseline preserved: {summary.pct_preserved if summary.pct_preserved is not None else 'n/a'}%."
        )
        highlights = "\n".join(
            f"- {t.title} ({t.status}): {t.overall_summary}" for t in summary.topics if t.overall_summary
        )[:4000]
        prompt = self.prompt_manager.get_prompt(
            "gap_summary_narrative",
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            rollup=rollup,
            highlights=highlights or "None",
        )
        response = await llm_service.generate(prompt)
        return response.strip()

    async def reconcile_toc(
        self,
        client_topics: List[Dict[str, Any]],
        benchmark_topics: List[Dict[str, Any]],
        draft_metadata,
        llm_service,
        similarity_hints: str = "",
    ) -> Dict[str, Any]:
        """Ask the LLM to decide the final TOC and the client<->benchmark mapping. `similarity_hints`
        are content-based client<->benchmark pairs (computed upstream) that the LLM should merge rather
        than duplicate. Returns parsed dict."""
        prompt = self.prompt_manager.get_prompt(
            "reconcile_toc",
            title=draft_metadata.title,
            function=draft_metadata.function,
            industry=draft_metadata.client_metadata.industry,
            regulations=draft_metadata.regulations or "",
            client_specific_requests=draft_metadata.client_specific_requests or "",
            sector_specific_comments=draft_metadata.sector_specific_comments or "",
            detail_level=self._map_detail_level(draft_metadata.detail_level),
            review_intensity=getattr(draft_metadata, "review_intensity", "preserve") or "preserve",
            client_topics=self._format_topics_for_reconcile(client_topics),
            benchmark_topics=self._format_topics_for_reconcile(benchmark_topics),
            similarity_hints=(similarity_hints or "None"),
        )

        response = await llm_service.generate(prompt)
        try:
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse reconcile_toc JSON: {e}")
            logger.error(f"LLM response: {response[:500]}")
            raise
        return data

    async def generate_toc_from_text(self, pasted_text: str, current_toc_text: str, llm_service) -> List[Dict[str, Any]]:
        """Parse a pasted/described TOC into a structured [{title, subtopics:[...]}] list (Stage 1c)."""
        prompt = self.prompt_manager.get_prompt(
            "toc_from_text",
            pasted_text=pasted_text or "",
            current_toc=current_toc_text or "(none)",
        )
        response = await llm_service.generate(prompt)
        try:
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse toc_from_text JSON: {e}")
            return []
        return data.get("toc", []) or []

    async def consolidate_theme_content(
        self, topic_title: str, existing_content: str, whole_policy_text: str,
        benchmark_content: Optional[str], draft, llm_service,
        regulation_chunks: List[RegulationChunk] = None,
    ) -> str:
        """Consolidate everything about a theme from across the whole policy into one section (Stage 2)."""
        # Retrieved chunks only (uploaded-regulation passages arrive via regulation_chunks).
        regulations_context = self._format_regulations_for_prompt(regulation_chunks or [])
        prompt = self.prompt_manager.get_prompt(
            "consolidate_theme",
            topic_title=topic_title,
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            existing_content=existing_content or "(none yet)",
            whole_policy_text=whole_policy_text or "(none)",
            benchmark_content=benchmark_content or "None available",
            regulations_context=regulations_context or "None available",
        )
        response = await llm_service.generate(prompt)
        return response.strip()

    # ============================================================
    # REVIEW PIPELINE (review_mode drafts — per-section benchmark → law → close)
    # ============================================================

    def _chunk_regulation_text(self, text: str, target_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
        """Split a (potentially long) regulation document into retrievable passages. Splits on blank-line /
        article boundaries, greedily packs paragraphs to ~target_chars, carries a small overlap for
        continuity, and hard-splits any single oversized paragraph."""
        text = (text or "").strip()
        if not text:
            return []
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) + 2 > target_chars:
                chunks.append(buf)
                buf = (buf[-overlap_chars:] + "\n\n" + p) if overlap_chars else p
            else:
                buf = (buf + "\n\n" + p) if buf else p
        if buf:
            chunks.append(buf)
        out: List[str] = []
        for c in chunks:
            if len(c) <= target_chars * 2:
                out.append(c)
            else:
                for i in range(0, len(c), target_chars):
                    out.append(c[i:i + target_chars])
        return out

    def _embed_reg_chunks(self, chunks: List[str], vector_store) -> List[List[float]]:
        """Embed regulation chunks with a process-level cache (keyed by the chunk set) to avoid re-embedding
        the same uploaded regulation on every per-section check."""
        key = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
        cached = _UPLOADED_REG_EMBED_CACHE.get(key)
        if cached is not None:
            return cached
        vecs = vector_store.embedder.embed_documents(chunks)
        if len(_UPLOADED_REG_EMBED_CACHE) >= _UPLOADED_REG_EMBED_CACHE_MAX:
            _UPLOADED_REG_EMBED_CACHE.pop(next(iter(_UPLOADED_REG_EMBED_CACHE)))
        _UPLOADED_REG_EMBED_CACHE[key] = vecs
        return vecs

    async def search_uploaded_regulations(
        self, query_text: str, uploaded_text: str, vector_store, top_k: int = 5, min_score: float = 0.0,
    ) -> List[RegulationChunk]:
        """Draft-scoped RAG over the reviewer's UPLOADED regulation text: chunk → embed (cached) → cosine
        top-k against the section query. Returns only the most relevant passages as RegulationChunks, so the
        prompt gets the right articles instead of the whole document dumped in. Non-fatal on any failure."""
        if not (uploaded_text or "").strip() or vector_store is None:
            return []
        chunks = self._chunk_regulation_text(uploaded_text)
        if not chunks:
            return []
        try:
            vecs = self._embed_reg_chunks(chunks, vector_store)
            # Cap the query length so an unusually long section can't exceed the embedding token limit.
            qv = vector_store.embedder.embed_query((query_text or "")[:16000])
        except Exception as e:
            logger.warning(f"Uploaded-regulation embedding failed (non-fatal): {e}")
            return []
        scored = sorted(
            ((_cosine(qv, v), i) for i, v in enumerate(vecs) if v),
            key=lambda x: x[0], reverse=True,
        )
        out: List[RegulationChunk] = []
        for rank, (score, i) in enumerate(scored[:top_k], 1):
            if score < min_score:
                continue
            out.append(RegulationChunk(
                id=f"uploaded-{i}",
                content=chunks[i],
                regulation_category="Uploaded",
                decree_name="Uploaded Regulation",
                article_number=f"Excerpt {rank}",
                regulation_type="uploaded",
                score=score,
            ))
        return out

    def _regulations_context(self, draft, regulation_chunks) -> str:
        """Formatted regulation chunks for a prompt. Covers both the library index AND the draft's uploaded
        regulation — the uploaded text is now RETRIEVED into the most-relevant chunks upstream
        (search_uploaded_regulations) rather than dumped whole, so this no longer appends the full document."""
        return self._format_regulations_for_prompt(regulation_chunks or [])

    async def generate_benchmark_as_is(
        self, topic_title: str, baseline_content: str, benchmark_content: Optional[str],
        draft, llm_service,
    ) -> str:
        """The 'use benchmark as-is' rewrite: the benchmark section adapted to this client (prose)."""
        prompt = self.prompt_manager.get_prompt(
            "benchmark_as_is",
            topic_title=topic_title,
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            baseline_content=baseline_content or "(none yet)",
            benchmark_content=benchmark_content or "None available",
        )
        response = await llm_service.generate(prompt)
        return response.strip()

    async def review_regulation_check(
        self, topic_title: str, existing_content: str, draft, llm_service,
        regulation_chunks: List[RegulationChunk] = None,
    ) -> Dict[str, Any]:
        """Law/regulation check of one section → {applies, extract, suggestion}."""
        regulations_context = self._regulations_context(draft, regulation_chunks)
        if not (regulations_context or "").strip():
            return {"applies": False, "extract": "", "suggestion": "", "check_failed": False}
        prompt = self.prompt_manager.get_prompt(
            "review_regulation",
            topic_title=topic_title,
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            existing_content=existing_content or "(none yet)",
            regulations_context=regulations_context,
        )
        response = await llm_service.generate(prompt)
        try:
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse review_regulation JSON for '{topic_title}': {e}")
            # A failed CHECK must not read as "no regulation applies" — that's a compliance
            # false-negative. check_failed lets the UI say "try again" instead.
            return {"applies": False, "extract": "", "suggestion": "", "check_failed": True}
        return {
            "applies": bool(data.get("applies", False)),
            "extract": str(data.get("extract", "")).strip(),
            "suggestion": str(data.get("suggestion", "")).strip(),
            "check_failed": False,
        }

    async def redistribute_assignments(
        self, good: List[Dict[str, Any]], sources: List[Dict[str, Any]],
        leftover_indices: List[int], suggestions: Dict[int, int], filled_targets: set,
        draft, llm_service,
    ) -> Dict[int, int]:
        """Ask the LLM to assign the LEFTOVER client content units (the ones exact-title matching didn't
        place) to the best NEW section, respecting topic/subtopic level and the embedding `suggestions`.
        Returns {source_index: good_index} (good_index -1 = unassigned). Falls back to the suggestions."""
        good_block = "\n".join(
            f"{i}: [{g.get('level', 'topic')}]{' (already filled)' if i in filled_targets else ''} {g['title']}"
            for i, g in enumerate(good)
        )
        src_block = "\n".join(
            f"{i}: [{sources[i].get('level', 'topic')}] [suggested {suggestions.get(i, -1)}] {sources[i]['title']} — "
            f"{(sources[i]['content'] or '').strip()[:300].replace(chr(10), ' ')}"
            for i in leftover_indices
        )
        prompt = self.prompt_manager.get_prompt(
            "redistribute_content",
            document_title=draft.metadata.title,
            good_sections=good_block,
            source_units=src_block,
        )
        try:
            response = await llm_service.generate(prompt)
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse redistribute_content JSON: {e}")
            return {i: suggestions.get(i, -1) for i in leftover_indices}
        out: Dict[int, int] = {}
        for a in data.get("assignments", []):
            try:
                out[int(a["unit"])] = int(a["section"])
            except (TypeError, ValueError, KeyError):
                continue
        # Any leftover the LLM omitted falls back to its embedding suggestion.
        for i in leftover_indices:
            if i not in out:
                out[i] = suggestions.get(i, -1)
        return out

    async def review_chat_turn(
        self, topic_title: str, baseline_content: str, existing_content: str,
        benchmark_content: Optional[str], whole_policy_text: str, user_message: str,
        conversation_history: str, draft, llm_service, regulation_chunks: List[RegulationChunk] = None,
    ) -> Dict[str, Any]:
        """One turn of the per-section review chat → {reply, proposed_content}."""
        prompt = self.prompt_manager.get_prompt(
            "review_chat",
            topic_title=topic_title,
            document_title=draft.metadata.title,
            industry=draft.metadata.client_metadata.industry,
            detail_level=self._map_detail_level(draft.metadata.detail_level),
            baseline_content=baseline_content or "(none yet)",
            existing_content=existing_content or "(none yet)",
            benchmark_content=benchmark_content or "None available",
            whole_policy_text=whole_policy_text or "(none)",
            regulations_context=self._regulations_context(draft, regulation_chunks) or "None available",
            conversation_history=conversation_history or "(none)",
            user_message=user_message,
        )
        response = await llm_service.generate(prompt)
        try:
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse review_chat JSON for '{topic_title}': {e}")
            return {"reply": response.strip(), "proposed_content": None}
        proposed = data.get("proposed_content")
        if proposed is not None:
            proposed = str(proposed).strip() or None
        return {"reply": str(data.get("reply", "")).strip(), "proposed_content": proposed}

    async def extract_toc_from_content(self, document_excerpt: str, llm_service) -> List[Dict[str, Any]]:
        """Recover the document's Table of Contents by giving the LLM the raw first pages and letting
        it find the printed TOC (or infer structure from headings) - instead of feeding it a
        possibly-wrong list of deterministically detected headings. Returns a flat, document-ordered
        list of {"title", "level"} (titles verbatim so they map back to the body). [] on failure."""
        if not document_excerpt or not document_excerpt.strip():
            return []
        prompt = self.prompt_manager.get_prompt("extract_toc", document_excerpt=document_excerpt)
        try:
            response = await llm_service.generate(prompt)
            data = self._parse_json_object(response)
        except Exception as e:
            logger.error(f"Failed to parse extract_toc JSON: {e}")
            return []
        # No printed TOC detected → return nothing so the caller keeps the deterministic structure.
        if str(data.get("source", "")).strip().lower() == "none":
            return []
        out: List[Dict[str, Any]] = []
        for it in (data.get("toc") or []):
            title = str(it.get("title", "")).strip()
            if not title:
                continue
            try:
                level = int(it.get("level", 1))
            except (TypeError, ValueError):
                level = 1
            out.append({"title": title, "level": max(1, level)})
        return out


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
            policy_type=draft_metadata.effective_policy_type,
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
            policy_type=draft_metadata.effective_policy_type,
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