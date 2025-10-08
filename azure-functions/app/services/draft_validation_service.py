import re
import logging
from typing import Dict, List, Tuple, Optional
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class DraftValidationService:
    """Service for validating draft creation parameters."""

    # Common low-quality patterns
    LOW_QUALITY_PATTERNS = [
        r'^n/a$',
        r'^na$',
        r'^none$',
        r'^test$',
        r'^asdf',
        r'^xxx',
        r'^todo',
        r'^tbd',
        r'^to be determined',
        r'^not applicable',
        r'^general',
        r'^overall',
        r'^policy$',
        r'^hr policy$',
        r'^it policy$',
        r'^company policy$'
    ]

    # Minimum requirements for a good description
    MIN_WORD_COUNT = 10
    MIN_SENTENCE_COUNT = 2
    MIN_SPECIFIC_TERMS = 2

    # Keywords that indicate specific content
    SPECIFIC_KEYWORDS = [
        'remote', 'hybrid', 'vacation', 'leave', 'sick', 'maternity', 'paternity',
        'code of conduct', 'ethics', 'compliance', 'security', 'data', 'privacy',
        'harassment', 'discrimination', 'diversity', 'inclusion', 'performance',
        'review', 'promotion', 'compensation', 'benefits', 'health', 'safety',
        'training', 'development', 'onboarding', 'termination', 'resignation',
        'confidentiality', 'intellectual property', 'expense', 'reimbursement',
        'travel', 'equipment', 'byod', 'work from home', 'flexible', 'overtime',
        'attendance', 'punctuality', 'dress code', 'social media', 'conflict',
        'grievance', 'whistleblower', 'investigation', 'disciplinary', 'probation'
    ]

    def __init__(self, llm: Optional[AzureLLM] = None, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.prompt_manager = prompt_manager

    def validate_description(self, description: str, title: str, client_metadata: Dict) -> Tuple[bool, List[str], List[str], int]:
        """
        Validate the quality of a draft description.

        Returns:
            Tuple of (is_valid, issues, suggestions, quality_score)
        """
        issues = []
        suggestions = []

        # Clean the description
        description_clean = description.strip().lower()

        # Check for low-quality patterns
        for pattern in self.LOW_QUALITY_PATTERNS:
            if re.match(pattern, description_clean, re.IGNORECASE):
                issues.append(f"Description '{description}' is too generic or meaningless")
                suggestions.append("Provide a detailed description of what this policy should cover")
                return False, issues, suggestions, 0

        # Check word count
        words = description.split()
        word_count = len(words)
        if word_count < self.MIN_WORD_COUNT:
            issues.append(f"Description is too short ({word_count} words, minimum {self.MIN_WORD_COUNT})")
            suggestions.append("Add more details about the specific aspects this policy should address")

        # Check for sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', description) if s.strip()]
        sentence_count = len(sentences)
        if sentence_count < self.MIN_SENTENCE_COUNT:
            issues.append(f"Description lacks detail (only {sentence_count} sentence(s))")
            suggestions.append("Explain multiple aspects or requirements of this policy")

        # Check for specific keywords
        specific_terms_found = sum(1 for keyword in self.SPECIFIC_KEYWORDS if keyword in description_clean)
        if specific_terms_found < self.MIN_SPECIFIC_TERMS:
            issues.append("Description lacks specific policy-related terms")
            suggestions.append("Include specific areas like: remote work, leave policies, code of conduct, data security, etc.")

        # Check if description is just repeating the title
        title_words = set(title.lower().split())
        description_words = set(description_clean.split())
        if len(description_words - title_words) < 5:
            issues.append("Description is too similar to the title")
            suggestions.append("Expand on the title with specific requirements and scope")

        # Check for company/industry context
        has_context = any([
            client_metadata.get('industry', '').lower() in description_clean,
            'our company' in description_clean,
            'our organization' in description_clean,
            'we ' in description_clean,
            client_metadata.get('name', '').lower() in description_clean if client_metadata.get('name') else False
        ])

        if not has_context and word_count > 5:
            suggestions.append(f"Consider mentioning your {client_metadata.get('industry', 'industry')} context for more relevant content")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            description,
            word_count,
            sentence_count,
            specific_terms_found,
            has_context
        )

        # Determine if valid
        is_valid = len(issues) == 0 or (quality_score >= 50 and not any('too generic' in issue for issue in issues))

        # Add suggestions based on score
        if quality_score < 30:
            suggestions.append("This description needs significant improvement. Please provide specific details about what the policy should cover.")
        elif quality_score < 60:
            suggestions.append("Consider adding more specific requirements or scenarios this policy should address.")

        return is_valid, issues, suggestions, quality_score

    def _calculate_quality_score(self, description: str, word_count: int,
                                sentence_count: int, specific_terms: int,
                                has_context: bool) -> int:
        """Calculate a quality score from 0-100."""
        score = 0

        # Word count scoring (max 30 points)
        if word_count >= 50:
            score += 30
        elif word_count >= 30:
            score += 25
        elif word_count >= 20:
            score += 20
        elif word_count >= 10:
            score += 10
        else:
            score += max(0, word_count * 2)

        # Sentence structure (max 20 points)
        if sentence_count >= 3:
            score += 20
        elif sentence_count >= 2:
            score += 15
        elif sentence_count >= 1:
            score += 10

        # Specific terms (max 30 points)
        score += min(30, specific_terms * 10)

        # Context bonus (max 10 points)
        if has_context:
            score += 10

        # Grammar and structure (max 10 points)
        if description[0].isupper() and description.rstrip().endswith(('.', '!', '?')):
            score += 5
        if not re.search(r'(.)\1{3,}', description):  # No excessive repetition
            score += 5

        return min(100, score)

    async def suggest_improved_description(self, description: str, title: str,
                                          client_metadata: Dict) -> Optional[str]:
        """
        Use LLM to suggest an improved description.

        Returns:
            Improved description or None if LLM is not available
        """
        if not self.llm or not self.prompt_manager:
            return None

        try:
            prompt = self.prompt_manager.get_prompt(
                "improve_draft_description",
                original_description=description,
                title=title,
                industry=client_metadata.get('industry', 'General'),
                company_name=client_metadata.get('name', 'the organization'),
                city=client_metadata.get('city', ''),
                country=client_metadata.get('country', '')
            )

            improved = await self.llm.generate(prompt, max_tokens=500)
            return improved.strip()

        except Exception as e:
            logger.error(f"Error generating improved description: {e}")
            return None

    def validate_title(self, title: str) -> Tuple[bool, List[str]]:
        """
        Validate the title of a draft.

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        # Check if title is too short
        if len(title.strip()) < 5:
            issues.append("Title is too short (minimum 5 characters)")

        # Check if title is too generic
        generic_titles = ['policy', 'hr', 'it', 'document', 'draft', 'test']
        if title.strip().lower() in generic_titles:
            issues.append("Title is too generic")

        # Check for special characters that might cause issues
        if re.search(r'[<>:"/\\|?*]', title):
            issues.append("Title contains invalid characters")

        is_valid = len(issues) == 0
        return is_valid, issues

    async def validate_draft_request(self, title: str, description: str,
                                    client_metadata: Dict) -> Dict:
        """
        Complete validation of a draft request.

        Returns:
            Dictionary with validation results
        """
        all_issues = []
        all_suggestions = []

        # Validate title
        title_valid, title_issues = self.validate_title(title)
        all_issues.extend(title_issues)

        # Validate description
        desc_valid, desc_issues, desc_suggestions, quality_score = self.validate_description(
            description, title, client_metadata
        )
        all_issues.extend(desc_issues)
        all_suggestions.extend(desc_suggestions)

        # Get improved description if needed
        improved_description = None
        if not desc_valid and self.llm and self.prompt_manager:
            improved_description = await self.suggest_improved_description(
                description, title, client_metadata
            )
            if improved_description:
                all_suggestions.insert(0, "Click 'Use Suggested Description' to use our improved version")

        is_valid = title_valid and desc_valid

        return {
            "is_valid": is_valid,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "description_quality_score": quality_score,
            "improved_description": improved_description
        }