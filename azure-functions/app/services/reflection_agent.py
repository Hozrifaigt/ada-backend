import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from app.config.settings import settings
from app.models.draft_models import CompleteDraft, DraftMetadata, ClientMetadata
from app.models.regulation_models import RegulationChunk

logger = logging.getLogger(__name__)


@dataclass
class ReflectionCritique:
    """Result of reflection/critique analysis."""
    score: float  # 0-1 score for overall quality
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]
    compliance_gaps: List[str]
    alignment_score: float  # How well it aligns with client/sector needs
    regulation_compliance_score: float  # 0-1 score for regulation compliance (highest priority)
    recommendations: str
    requires_revision: bool
    fixed_content: Optional[str] = None  # Auto-fixed content if revision was performed
    critique_failed: bool = False  # True when the critique itself errored — scores are meaningless


class ReflectionAgent:
    """
    Reflection agent that critiques generated policy content.
    
    Ensures generated content:
    1. Aligns with client metadata and requirements
    2. Adheres to sector-specific comments
    3. Follows client-specific requests
    4. Complies with applicable regulations
    5. Maintains appropriate detail level
    6. Is coherent and well-structured
    """

    def __init__(self, llm_service):
        """
        Initialize the reflection agent.
        
        Args:
            llm_service: LLM service for critique analysis
        """
        self.llm_service = llm_service

    async def critique_response(
        self,
        generated_content: str,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        conversation_history: List[Dict[str, str]],
        regulation_chunks: Optional[List[RegulationChunk]] = None,
        source_ids: Optional[List[str]] = None
    ) -> ReflectionCritique:
        """
        Critique the generated policy content comprehensively.
        
        Analyzes:
        - Alignment with client metadata (industry, requirements, comments)
        - Compliance with sector-specific guidelines
        - Adherence to applicable regulations
        - Detail level appropriateness
        - Content coherence and structure
        - Policy consistency with rest of draft
        
        Args:
            generated_content: The content that was generated
            draft: Complete draft metadata and structure
            topic_id: Topic being generated
            subtopic_id: Subtopic if applicable
            user_prompt: Original user request
            conversation_history: Previous interactions on this topic
            regulation_chunks: Relevant regulations used
            source_ids: Source materials used
            
        Returns:
            ReflectionCritique with detailed analysis
        """
        try:
            # Build comprehensive context for critique
            critique_context = self._build_critique_context(
                generated_content=generated_content,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
                regulation_chunks=regulation_chunks,
                source_ids=source_ids
            )

            # Call LLM for detailed critique
            critique_result = await self._perform_critique(critique_context)

            return critique_result

        except Exception as e:
            logger.error(f"Error in reflection agent critique: {e}")
            # A FAILED critique must not trigger an auto-rewrite: the zero scores and
            # "Critique failed" weakness are not real feedback, and regenerating from them
            # degrades the content. Keep the original and flag the failure.
            return ReflectionCritique(
                score=0.0,
                strengths=[],
                weaknesses=[f"Critique failed: {str(e)}"],
                improvements=[],
                compliance_gaps=[],
                alignment_score=0.0,
                regulation_compliance_score=0.0,
                recommendations="Unable to perform critique. Please review manually.",
                requires_revision=False,
                critique_failed=True
            )

    def _build_critique_context(
        self,
        generated_content: str,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        conversation_history: List[Dict[str, str]],
        regulation_chunks: Optional[List[RegulationChunk]] = None,
        source_ids: Optional[List[str]] = None
    ) -> str:
        """Build comprehensive context document for LLM critique."""
        
        # Find the topic in TOC
        toc_item = None
        for topic in draft.toc:
            if topic.topic_id == topic_id:
                toc_item = topic
                break
        
        topic_title = toc_item.topic if toc_item else topic_id
        
        # Build context sections
        sections = []
        
        # Section 1: Policy Metadata
        sections.append("=== POLICY METADATA ===")
        sections.append(f"Title: {draft.metadata.title}")
        sections.append(f"Description: {draft.metadata.description or 'N/A'}")
        sections.append(f"Function: {draft.metadata.function}")
        sections.append(f"Policy Type: {draft.metadata.effective_policy_type}")
        sections.append(f"Detail Level: {draft.metadata.detail_level}/5")
        sections.append(f"Applicable Regulations: {draft.metadata.regulations}")
        sections.append("")
        
        # Section 2: Client Metadata
        sections.append("=== CLIENT METADATA ===")
        client = draft.metadata.client_metadata
        sections.append(f"Client Name: {client.name}")
        sections.append(f"Country: {client.country}")
        sections.append(f"City: {client.city}")
        sections.append(f"Industry: {client.industry}")
        sections.append("")
        
        # Section 3: Client Requirements & Comments
        sections.append("=== CLIENT REQUIREMENTS & COMMENTS ===")
        sections.append(f"Client-Specific Requests: {draft.metadata.client_specific_requests or 'None'}")
        sections.append(f"Sector-Specific Comments: {draft.metadata.sector_specific_comments or 'None'}")
        sections.append("")
        
        # Section 4: Topic Information & Structure
        sections.append("=== TOPIC INFORMATION ===")
        sections.append(f"Topic: {topic_title}")

        if subtopic_id and toc_item:
            # Generating a subtopic - find and display it
            for subtopic in toc_item.subtopics:
                if subtopic.subtopic_id == subtopic_id:
                    sections.append(f"Subtopic: {subtopic.topic}")
                    sections.append("Content Type: SUBTOPIC (should contain detailed, specific content)")
                    break
        elif toc_item and toc_item.subtopics and len(toc_item.subtopics) > 0:
            # Generating a main topic that HAS subtopics
            subtopic_names = [sub.topic for sub in toc_item.subtopics]
            sections.append(f"Content Type: MAIN TOPIC WITH SUBTOPICS")
            sections.append(f"Subtopics: {', '.join(subtopic_names)}")
            sections.append("IMPORTANT: This main topic should be HIGH-LEVEL/INTRODUCTORY only.")
            sections.append("Detailed content should be deferred to the subtopics listed above.")
        else:
            # Generating a main topic without subtopics
            sections.append("Content Type: STANDALONE TOPIC (no subtopics - can contain full detail)")

        sections.append(f"User Prompt: {user_prompt}")
        sections.append("")
        
        # Section 5: Conversation History
        # Entries arrive as ConversationEntry models (from the draft TOC) or dicts; the keys are
        # user_message/ai_response. The old dict-only .get('user_prompt') access crashed on models
        # (AttributeError), which made EVERY critique on a topic with history fail → forced a
        # regeneration driven by a meaningless "Critique failed" critique.
        sections.append("=== CONVERSATION HISTORY ===")
        if conversation_history:
            for i, msg in enumerate(conversation_history[-5:], 1):  # Last 5 messages
                entry = msg.model_dump() if hasattr(msg, 'model_dump') else (msg or {})
                user_msg = entry.get('user_message') or 'N/A'
                ai_response = entry.get('ai_response') or 'N/A'
                sections.append(f"[{i}] User: {user_msg[:200]}")
                sections.append(f"    Response: {ai_response[:200]}...")
        else:
            sections.append("No previous conversation")
        sections.append("")
        
        # Section 6: Regulation Context
        sections.append("=== APPLICABLE REGULATIONS ===")
        if regulation_chunks:
            sections.append(f"Found {len(regulation_chunks)} relevant regulation(s)")
            for i, chunk in enumerate(regulation_chunks[:3], 1):  # Top 3
                effective_date = chunk.effective_date or "Unknown"
                sections.append(f"[Regulation {i}] {chunk.decree_name} - Article {chunk.article_number} (Effective: {effective_date})")
                sections.append(f"  Topics: {', '.join(chunk.topics)}")
                sections.append(f"  Content: {chunk.content[:500]}..." if len(chunk.content) > 500 else f"  Content: {chunk.content}")
        else:
            sections.append("No regulations provided")
        sections.append("")
        
        # Section 7: Generated Content — the critic must see what it's judging. The old
        # 1000-char preview meant compliance/quality scores were based on ~the first two
        # paragraphs only. Cap generously; note the truncation when it happens.
        content_cap = 12000
        sections.append("=== GENERATED CONTENT ===")
        sections.append(f"Length: {len(generated_content)} characters")
        if len(generated_content) <= content_cap:
            sections.append(f"Content:\n{generated_content}")
        else:
            sections.append(f"Content (truncated at {content_cap} of {len(generated_content)} chars):\n"
                            f"{generated_content[:content_cap]}...")
        sections.append("")
        
        # Section 8: Sources Used
        sections.append("=== SOURCES USED ===")
        if source_ids:
            sections.append(f"Source materials: {', '.join(source_ids)}")
        else:
            sections.append("Generated from scratch (no pre-existing content)")
        sections.append("")
        
        return "\n".join(sections)

    async def _perform_critique(self, context: str) -> ReflectionCritique:
        """
        Call LLM to perform detailed critique analysis.
        
        Args:
            context: Complete context for critique
            
        Returns:
            ReflectionCritique object
        """
        
        prompt = f"""You are an expert policy reviewer critiquing generated HR/employment policy content.

Your task is to analyze the generated policy content with **REGULATION COMPLIANCE AS THE HIGHEST PRIORITY**.

PRIORITY ORDER FOR EVALUATION:
1. **REGULATION COMPLIANCE (CRITICAL)** - Content MUST align with applicable regulations (UAE Labor Law, etc.)
   - Does the content correctly reference and follow the provided regulations?
   - Are there any statements that contradict the regulations?
   - Are regulatory requirements properly incorporated?
2. **CONTENT DEPTH APPROPRIATENESS** - Check the "Content Type" in TOPIC INFORMATION section:
   - MAIN TOPIC WITH SUBTOPICS: Content should be HIGH-LEVEL/INTRODUCTORY only. Penalize if too detailed.
   - SUBTOPIC: Content should be DETAILED and SPECIFIC. Penalize if too vague.
   - STANDALONE TOPIC: Content can have full detail as appropriate.
3. Client metadata (industry, location, specific requirements)
4. Sector-specific comments and guidelines
5. Detail level appropriateness (1=light principles to 5=detailed procedures)
6. Content quality, clarity, and structure

CONTEXT FOR CRITIQUE:
{context}

CRITIQUE REQUIREMENTS:
Analyze and provide JSON response with:
{{
  "score": <0-1 overall quality score>,
  "alignment_score": <0-1 how well aligned with client requirements>,
  "regulation_compliance_score": <0-1 how well content complies with provided regulations - THIS IS CRITICAL>,
  "strengths": [<list of positive aspects>],
  "weaknesses": [<list of problematic areas>],
  "improvements": [<list of specific improvements needed>],
  "compliance_gaps": [<list of regulatory gaps - be specific about which regulations are not addressed>],
  "recommendations": "<comprehensive recommendations paragraph>",
  "requires_revision": <boolean - TRUE if regulation_compliance_score < 0.85 OR content depth is inappropriate>
}}

REGULATION COMPLIANCE SCORING GUIDE:
- 1.0: Perfect compliance - all regulations properly referenced and followed
- 0.85-0.99: Good compliance - minor gaps but no contradictions
- 0.70-0.84: Needs improvement - some regulations missing or unclear
- 0.50-0.69: Poor compliance - significant regulatory gaps
- Below 0.50: Unacceptable - contradicts regulations or ignores them entirely

CONTENT DEPTH SCORING (affects overall score):
- If "MAIN TOPIC WITH SUBTOPICS": Penalize if content goes into details that belong in subtopics
- If "SUBTOPIC": Penalize if content is too vague or introductory
- Content depth issues should be listed in "weaknesses" and "improvements"

Focus especially on:
- Does the content align with the APPLICABLE REGULATIONS section?
- Are there any statements that could violate labor law?
- Are employee rights and employer obligations correctly stated?
- Is the content legally defensible?
- Is the content depth appropriate for the Content Type?

Return ONLY valid JSON, no additional text."""

        try:
            response = await self.llm_service.generate(prompt)

            # Parse JSON response, tolerating markdown fences / surrounding prose
            # (strict json.loads failed on ```json fences, turning good critiques into errors).
            text = (response or "").strip()
            try:
                critique_json = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start == -1 or end <= start:
                    raise
                critique_json = json.loads(text[start:end + 1])

            return ReflectionCritique(
                score=float(critique_json.get("score", 0.0)),
                alignment_score=float(critique_json.get("alignment_score", 0.0)),
                regulation_compliance_score=float(critique_json.get("regulation_compliance_score", 0.0)),
                strengths=critique_json.get("strengths", []),
                weaknesses=critique_json.get("weaknesses", []),
                improvements=critique_json.get("improvements", []),
                compliance_gaps=critique_json.get("compliance_gaps", []),
                recommendations=critique_json.get("recommendations", ""),
                requires_revision=critique_json.get("requires_revision", False)
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse critique JSON: {e}")
            logger.error(f"LLM response: {response[:500]}")
            raise Exception(f"Invalid JSON from critique LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Error performing critique: {e}")
            raise

    async def log_critique_results(
        self,
        critique: ReflectionCritique,
        draft_id: str,
        topic_id: str,
        subtopic_id: Optional[str] = None
    ) -> None:
        """
        Log critique results for monitoring and improvement.
        
        Args:
            critique: The critique results
            draft_id: ID of the draft
            topic_id: ID of the topic
            subtopic_id: ID of subtopic if applicable
        """
        
        location = f"draft:{draft_id}/topic:{topic_id}"
        if subtopic_id:
            location += f"/subtopic:{subtopic_id}"
        
        logger.info(f"REFLECTION CRITIQUE [{location}]")
        logger.info(f"  Quality Score: {critique.score:.2f}")
        logger.info(f"  Alignment Score: {critique.alignment_score:.2f}")
        logger.info(f"  Requires Revision: {critique.requires_revision}")
        
        if critique.strengths:
            logger.info(f"  Strengths: {', '.join(critique.strengths)}")
        
        if critique.weaknesses:
            logger.warning(f"  Weaknesses: {', '.join(critique.weaknesses)}")
        
        if critique.compliance_gaps:
            logger.warning(f"  Compliance Gaps: {', '.join(critique.compliance_gaps)}")
        
        logger.info(f"  Recommendations: {critique.recommendations[:200]}...")

    async def critique_and_fix(
        self,
        generated_content: str,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        conversation_history: List[Dict[str, str]],
        regulation_chunks: Optional[List[RegulationChunk]] = None,
        source_ids: Optional[List[str]] = None,
        detail_level: int = 3
    ) -> ReflectionCritique:
        """
        Critique generated content and automatically fix if needed.

        Workflow (3 LLM calls max):
        1. Perform initial critique
        2. If regulation_compliance_score >= 0.85 AND score >= 0.80: Return as-is
        3. Otherwise: Auto-regenerate improved version and return

        Args:
            generated_content: Initial generated content
            draft: Complete draft with metadata
            topic_id: Topic being generated
            subtopic_id: Subtopic if applicable
            user_prompt: Original user request
            conversation_history: Previous interactions
            regulation_chunks: Relevant regulations
            source_ids: Source materials used
            detail_level: Policy detail level (1-5)

        Returns:
            ReflectionCritique with fixed_content if revision was needed
        """
        try:
            # Step 1: Perform critique (LLM Call #2)
            critique = await self.critique_response(
                generated_content=generated_content,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
                regulation_chunks=regulation_chunks,
                source_ids=source_ids
            )

            logger.info(
                f"Critique - Score: {critique.score:.2f}, "
                f"Alignment: {critique.alignment_score:.2f}, "
                f"Regulation Compliance: {critique.regulation_compliance_score:.2f}"
            )

            # A failed critique carries no usable feedback — keep the original content
            # rather than rewriting it from "Critique failed" noise.
            if critique.critique_failed:
                logger.warning("Critique failed; skipping auto-fix and keeping original content")
                return critique

            # Step 2: Check if revision is needed
            # Regulation compliance is the highest priority
            regulation_acceptable = critique.regulation_compliance_score >= 0.85
            score_acceptable = critique.score >= 0.80

            if regulation_acceptable and score_acceptable and not critique.requires_revision:
                logger.info("Content meets quality standards, no revision needed")
                return critique

            # Step 3: Auto-fix the content (LLM Call #3)
            logger.info(
                f"Content needs revision (regulation_compliance={critique.regulation_compliance_score:.2f}, "
                f"score={critique.score:.2f}), generating improved version..."
            )

            fixed_content = await self._generate_improved_version(
                original_content=generated_content,
                critique=critique,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                regulation_chunks=regulation_chunks,
                detail_level=detail_level
            )

            if not fixed_content:
                logger.warning("Failed to auto-fix content, returning original critique")
                return critique

            # Return critique with fixed content (no second critique)
            critique.fixed_content = fixed_content
            logger.info("Generated improved version, returning with fixed content")
            return critique

        except Exception as e:
            logger.error(f"Error in critique_and_fix: {e}", exc_info=True)
            # Return original critique without fix on error
            return await self.critique_response(
                generated_content=generated_content,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
                regulation_chunks=regulation_chunks,
                source_ids=source_ids
            )

    async def _generate_improved_version(
        self,
        original_content: str,
        critique: ReflectionCritique,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        regulation_chunks: Optional[List[RegulationChunk]] = None,
        detail_level: int = 3
    ) -> Optional[str]:
        """
        Generate improved version based on critique feedback.
        
        Args:
            original_content: The content that needs improvement
            critique: The critique analysis
            draft: Complete draft with metadata
            topic_id: Topic ID
            subtopic_id: Subtopic ID if applicable
            user_prompt: Original user request
            regulation_chunks: Relevant regulations
            detail_level: Policy detail level
            
        Returns:
            Improved content or None if generation fails
        """
        try:
            # Build improvement prompt
            prompt = self._build_improvement_prompt(
                original_content=original_content,
                critique=critique,
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                regulation_chunks=regulation_chunks,
                detail_level=detail_level
            )
            
            # Generate improved version
            improved_content = await self.llm_service.generate(prompt)
            
            if not improved_content or len(improved_content.strip()) < 100:
                logger.warning("Generated improvement is too short")
                return None
            
            logger.debug(f"Generated improved version ({len(improved_content)} chars)")
            return improved_content
            
        except Exception as e:
            logger.error(f"Error generating improved version: {e}")
            return None

    def _build_improvement_prompt(
        self,
        original_content: str,
        critique: ReflectionCritique,
        draft: CompleteDraft,
        topic_id: str,
        subtopic_id: Optional[str],
        user_prompt: str,
        regulation_chunks: Optional[List[RegulationChunk]] = None,
        detail_level: int = 3
    ) -> str:
        """Build prompt for generating improved content."""
        
        detail_mapping = {
            1: "Light (Principle-Based) - Focus on high-level principles only",
            2: "Light-to-Moderate - Balanced with some flexibility",
            3: "Moderate - Balance between principles and procedures",
            4: "Moderate-to-Heavy - Detailed with clear procedures",
            5: "Heavy (Prescriptive) - Detailed, step-by-step, compliance focused"
        }
        
        # Build sections
        sections = []
        
        sections.append("You are an expert policy writer tasked with improving policy content based on critique feedback.")
        sections.append("")
        
        sections.append("ORIGINAL CONTENT:")
        sections.append(f"{original_content}")
        sections.append("")
        
        sections.append("CRITIQUE FEEDBACK:")
        sections.append(f"Quality Score: {critique.score:.2f}/1.0")
        sections.append(f"Alignment Score: {critique.alignment_score:.2f}/1.0")
        sections.append("")
        
        if critique.strengths:
            sections.append("Strengths to maintain:")
            for strength in critique.strengths:
                sections.append(f"  ✓ {strength}")
            sections.append("")
        
        if critique.weaknesses:
            sections.append("Issues to address:")
            for weakness in critique.weaknesses:
                sections.append(f"  ✗ {weakness}")
            sections.append("")
        
        if critique.improvements:
            sections.append("Specific improvements needed:")
            for improvement in critique.improvements:
                sections.append(f"  → {improvement}")
            sections.append("")
        
        if critique.compliance_gaps:
            sections.append("Compliance gaps to fill:")
            for gap in critique.compliance_gaps:
                sections.append(f"  ⚠ {gap}")
            sections.append("")
        
        sections.append("CONTEXT:")
        sections.append(f"Policy Title: {draft.metadata.title}")
        sections.append(f"Client: {draft.metadata.client_metadata.name} ({draft.metadata.client_metadata.industry})")
        sections.append(f"Detail Level: {detail_mapping.get(detail_level, detail_mapping[3])}")
        sections.append(f"Client Requests: {draft.metadata.client_specific_requests or 'None'}")
        sections.append(f"Sector Comments: {draft.metadata.sector_specific_comments or 'None'}")
        sections.append("")
        
        if regulation_chunks:
            sections.append("APPLICABLE REGULATIONS (use these to fix compliance gaps):")
            for i, chunk in enumerate(regulation_chunks[:3], 1):
                effective_date = chunk.effective_date or "Unknown"
                sections.append(f"[Regulation {i}] {chunk.decree_name} - Article {chunk.article_number} (Effective: {effective_date})")
                sections.append(f"  Content: {chunk.content}")
            sections.append("")

        sections.append("INSTRUCTIONS:")
        sections.append("1. Maintain the original structure and overall message")
        sections.append("2. Address ALL identified issues and improvements")
        sections.append("3. Fill compliance gaps with relevant regulation references")
        sections.append("4. Ensure alignment with client requirements and sector comments")
        sections.append("5. Keep detail level consistent with specified level")
        sections.append("6. Make content more specific and actionable")
        sections.append("7. Add examples where critique mentioned they're missing")
        sections.append("8. Improve clarity and remove ambiguous language")
        sections.append("")
        
        sections.append("Generate an improved version that addresses all feedback while maintaining professional quality.")
        
        return "\n".join(sections)

    async def log_improvement_results(
        self,
        initial_score: float,
        initial_alignment: float,
        final_critique: ReflectionCritique,
        draft_id: str,
        topic_id: str,
        subtopic_id: Optional[str] = None
    ) -> None:
        """
        Log results of critique-and-fix process for monitoring.
        
        Args:
            initial_score: Score before improvement
            initial_alignment: Alignment before improvement
            final_critique: Final critique after auto-fix
            draft_id: ID of the draft
            topic_id: ID of the topic
            subtopic_id: ID of subtopic if applicable
        """
        
        location = f"draft:{draft_id}/topic:{topic_id}"
        if subtopic_id:
            location += f"/subtopic:{subtopic_id}"
        
        if final_critique.fixed_content:
            logger.info(f"CONTENT AUTO-IMPROVED [{location}]")
            logger.info(f"  Before: Score {initial_score:.2f}, Alignment {initial_alignment:.2f}")
            logger.info(f"  After:  Score {final_critique.score:.2f}, Alignment {final_critique.alignment_score:.2f}")
            score_improvement = final_critique.score - initial_score
            alignment_improvement = final_critique.alignment_score - initial_alignment
            logger.info(f"  Improvement: Score {score_improvement:+.2f}, Alignment {alignment_improvement:+.2f}")
        else:
            logger.info(f"CONTENT ACCEPTED AS-IS [{location}]")
            logger.info(f"  Score: {final_critique.score:.2f}, Alignment: {final_critique.alignment_score:.2f}")

