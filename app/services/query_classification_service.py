"""
Query Classification Service
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from app.models.classification_models import (
    QueryClassification,
    ClassificationResult,
    QueryCategory
)
from app.prompts.manager import PromptManager
from app.config.settings import settings

logger = logging.getLogger(__name__)


class QueryClassificationService:
    """Service for classifying user queries using LLM before content generation."""

    def __init__(self):
        """Initialize the classification service."""

        # Initialize prompt manager
        try:
            self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        except (AttributeError, FileNotFoundError):
            # Fallback to default path if settings not configured
            prompts_file = Path(__file__).parent.parent / "prompts" / "prompts.v1.yml"
            self.prompt_manager = PromptManager(str(prompts_file))

        logger.info("QueryClassificationService initialized")

    async def classify_query(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: str = ""
    ) -> ClassificationResult:
        """
        Classify a user query using LLM to determine if it should proceed to content generation.

        Args:
            user_query: The user's input query
            context: Optional context (topic_title, document_type, etc.)
            conversation_history: Previous conversation for this topic/subtopic

        Returns:
            ClassificationResult with classification and processing details
        """
        start_time = time.time()

        try:
            # Use LLM classification with conversation history
            llm_result = await self._classify_with_llm(user_query, context, conversation_history)
            if llm_result:
                processing_time = int((time.time() - start_time) * 1000)
                return ClassificationResult(
                    classification=llm_result,
                    processing_time_ms=processing_time,
                    method_used="llm_based"
                )

            # If LLM fails for any reason, log it but allow the request to proceed
            logger.warning("LLM classification failed, allowing request to proceed")

            # Default fallback - assume legitimate request
            classification = QueryClassification(
                category=QueryCategory.LEGITIMATE_REQUEST,
                confidence=0.5,
                reason="LLM classification failed, defaulting to legitimate",
                should_proceed=True
            )
            processing_time = int((time.time() - start_time) * 1000)
            return ClassificationResult(
                classification=classification,
                processing_time_ms=processing_time,
                method_used="default_fallback"
            )

        except Exception as e:
            logger.error(f"Unexpected error in classification: {e}")
            # On any error, default to allowing the request
            classification = QueryClassification(
                category=QueryCategory.LEGITIMATE_REQUEST,
                confidence=0.5,
                reason=f"Classification error: {str(e)}",
                should_proceed=True
            )
            processing_time = int((time.time() - start_time) * 1000)
            return ClassificationResult(
                classification=classification,
                processing_time_ms=processing_time,
                method_used="error_fallback"
            )

    async def _classify_with_llm(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: str = ""
    ) -> Optional[QueryClassification]:
        """Classify query using LLM-based approach with conversation history."""
        try:
            # Import here to avoid circular imports
            from app.api.policy.dependencies import get_llm_service

            llm_service = get_llm_service()

            # Create classification prompt
            prompt = self._create_classification_prompt(user_query, context, conversation_history)

            # Get LLM response
            response = await llm_service.generate(prompt)

            # Parse the response
            classification = self._parse_llm_response(response)

            return classification

        except Exception as e:
            logger.error(f"Error in LLM-based classification: {e}")
            return None

    def _create_classification_prompt(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: str = ""
    ) -> str:
        """Create prompt for LLM classification using prompt manager."""

        topic_title = context.get("topic_title", "Unknown") if context else "Unknown"
        document_title = context.get("document_title", "Policy Document") if context else "Policy Document"

        # Use prompt manager to get the classification prompt
        prompt = self.prompt_manager.get_prompt(
            "classify_query",
            document_title=document_title,
            topic_title=topic_title,
            conversation_history=conversation_history if conversation_history else "No previous conversation.",
            user_query=user_query
        )

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[QueryClassification]:
        """Parse LLM response into QueryClassification object."""
        try:
            lines = response.strip().split('\n')
            result = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()

            # Extract values
            category_str = result.get('category', 'LEGITIMATE_REQUEST')
            confidence = float(result.get('confidence', '0.5'))
            reason = result.get('reason', 'LLM classification')
            should_proceed = result.get('should_proceed', 'true').lower() == 'true'

            # Map category string to enum
            category_mapping = {
                'LEGITIMATE_REQUEST': QueryCategory.LEGITIMATE_REQUEST,
                'SMALL_TALK': QueryCategory.SMALL_TALK,
                'INAPPROPRIATE': QueryCategory.INAPPROPRIATE,
            }

            category = category_mapping.get(category_str, QueryCategory.LEGITIMATE_REQUEST)

            return QueryClassification(
                category=category,
                confidence=confidence,
                reason=reason,
                should_proceed=should_proceed,
                rule_name="llm_classification"
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return default legitimate classification on parse error
            return QueryClassification(
                category=QueryCategory.LEGITIMATE_REQUEST,
                confidence=0.5,
                reason="Parse error, defaulting to legitimate",
                should_proceed=True,
                rule_name="parse_error_fallback"
            )

    def get_suggested_response(self, classification: QueryClassification) -> str:
        """Get a helpful response for non-legitimate queries."""
        if classification.suggested_response:
            return classification.suggested_response

        # Default responses based on category
        responses = {
            QueryCategory.SMALL_TALK: "I'm here to help with policy document generation. How can I assist you with creating or improving policy content?",
            QueryCategory.INAPPROPRIATE: "I can't help with that request. Let's focus on creating professional policy documentation.",
        }

        return responses.get(classification.category, "Let's focus on creating policy content. What would you like to work on?")


def create_classification_service() -> QueryClassificationService:
    """Factory function to create classification service."""
    return QueryClassificationService()