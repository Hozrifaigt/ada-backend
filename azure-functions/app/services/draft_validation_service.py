import json
import logging
from typing import Dict, List, Optional
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class DraftValidationService:
    """Service for validating draft creation parameters using LLM."""

    def __init__(self, llm: AzureLLM, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def validate_description_with_llm(self, description: str) -> Dict:
        """
        Use LLM to validate the quality of a draft description.

        Args:
            description: The description to validate

        Returns:
            Dictionary with validation results from LLM
        """
        try:
            # Get the validation prompt
            prompt = self.prompt_manager.get_prompt(
                "validate_draft_description",
                description=description
            )

            # Get LLM response
            response = await self.llm.generate(prompt)

            # Parse JSON response
            try:
                result = json.loads(response)

                # Ensure all required fields are present
                return {
                    "is_valid": result.get("is_valid", False),
                    "issues": result.get("issues", []),
                    "suggestions": result.get("suggestions", []),
                    "description_quality_score": result.get("quality_score", 0),
                    "improved_description": result.get("improved_description")
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM validation response as JSON: {response}")
                return {
                    "is_valid": False,
                    "issues": ["Failed to validate description properly"],
                    "suggestions": ["Please try again with a more detailed description"],
                    "description_quality_score": 0,
                    "improved_description": None
                }

        except Exception as e:
            logger.error(f"Error validating description with LLM: {e}")
            return {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Please try again"],
                "description_quality_score": 0,
                "improved_description": None
            }

    async def validate_draft_request(self, title: str, description: str,
                                    client_metadata: Dict) -> Dict:
        """
        Complete validation of a draft request using LLM.

        Args:
            title: Draft title (not validated)
            description: Draft description to validate
            client_metadata: Client metadata (not used in LLM validation)

        Returns:
            Dictionary with validation results
        """
        # Only validate description with LLM, no title validation
        validation_result = await self.validate_description_with_llm(description)

        return validation_result