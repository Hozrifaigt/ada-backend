import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings
from app.models.draft_models import ClientMetadata

logger = logging.getLogger(__name__)


class TOCGenerationService:
    """Service for generating Table of Contents using LLM when no similar policies are found."""

    def __init__(self, llm_service: AzureLLM):
        """Initialize TOC generation service with LLM dependency."""
        self.llm = llm_service
        self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        logger.info("TOCGenerationService initialized")

    async def generate_toc(
        self,
        title: str,
        client_metadata: ClientMetadata,
        function: str = "",
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3,
        max_topics: int = 8,
        max_subtopics_per_topic: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate a structured TOC based on policy title and client context.
        No description required - uses title and metadata to generate appropriate structure.

        Args:
            title: Policy title (e.g., "Leave Policy", "Code of Conduct")
            client_metadata: Client information (industry, location, etc.)
            function: Function category (e.g., 'HR')
            client_specific_requests: Client-specific requirements (optional)
            sector_specific_comments: Sector-specific considerations (optional)
            regulations: Applicable regulations
            detail_level: Level of detail (1-5)
            max_topics: Maximum number of main topics to generate
            max_subtopics_per_topic: Maximum subtopics per main topic

        Returns:
            List of topic dictionaries with same structure as similarity search results
        """
        try:
            logger.info(f"Generating TOC for: {title}")

            # Get the prompt template
            prompt_template = self.prompt_manager.get_prompt_template("generate_toc")

            # Map detail level to descriptive string
            detail_level_map = {
                1: "Light (Principle-Based) - Focus on high-level principles and outcomes",
                2: "Light-to-Moderate - Balanced approach with some flexibility",
                3: "Moderate - Balance between principles and procedures",
                4: "Moderate-to-Heavy - More detailed with clear procedures",
                5: "Heavy (Prescriptive) - Detailed, step-by-step, compliance focused"
            }
            detail_level_str = detail_level_map.get(detail_level, detail_level_map[3])

            # Format the prompt with context
            formatted_prompt = prompt_template.format(
                title=title,
                function=function or "",
                industry=client_metadata.industry,
                country=client_metadata.country,
                city=client_metadata.city,
                company_name=client_metadata.name,
                client_specific_requests=client_specific_requests or "",
                sector_specific_comments=sector_specific_comments or "",
                regulations=regulations or "",
                detail_level=detail_level_str,
                max_topics=max_topics,
                max_subtopics=max_subtopics_per_topic
            )

            # Generate TOC structure using LLM
            logger.info("Calling LLM to generate TOC structure")
            llm_response = await self.llm.generate(formatted_prompt)

            # Parse the LLM response to extract TOC structure
            toc_structure = self._parse_llm_toc_response(llm_response)

            # Transform to match expected format
            formatted_toc = self._format_toc_for_draft(toc_structure)

            logger.info(f"Generated TOC with {len(formatted_toc)} topics")
            return formatted_toc

        except Exception as e:
            logger.error(f"Failed to generate TOC: {e}", exc_info=True)
            return self._get_fallback_toc(title, client_metadata.industry)

    async def generate_toc_from_description(
        self,
        description: Optional[str],
        title: str,
        client_metadata: ClientMetadata,
        function: str = "",
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3,
        max_topics: int = 8,
        max_subtopics_per_topic: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate a structured TOC based on policy description and client context.

        DEPRECATED: Use generate_toc() instead for new implementations.

        Args:
            description: Policy description provided by user (optional)
            title: Policy title
            client_metadata: Client information (industry, location, etc.)
            function: Function category (e.g., 'HR')
            client_specific_requests: Client-specific requirements (optional)
            sector_specific_comments: Sector-specific considerations (optional)
            regulations: Applicable regulations
            detail_level: Level of detail (1-5)
            max_topics: Maximum number of main topics to generate
            max_subtopics_per_topic: Maximum subtopics per main topic

        Returns:
            List of topic dictionaries with same structure as similarity search results
        """
        try:
            logger.info(f"Generating TOC for: {title}")

            # Get the prompt template
            prompt_template = self.prompt_manager.get_prompt_template("generate_toc_from_description")

            # Map detail level to descriptive string
            detail_level_map = {
                1: "Light (Principle-Based) - Focus on high-level principles and outcomes",
                2: "Light-to-Moderate - Balanced approach with some flexibility",
                3: "Moderate - Balance between principles and procedures",
                4: "Moderate-to-Heavy - More detailed with clear procedures",
                5: "Heavy (Prescriptive) - Detailed, step-by-step, compliance focused"
            }
            detail_level_str = detail_level_map.get(detail_level, detail_level_map[3])

            # Format the prompt with context
            formatted_prompt = prompt_template.format(
                title=title,
                description=description or "",
                function=function or "",
                industry=client_metadata.industry,
                country=client_metadata.country,
                city=client_metadata.city,
                company_name=client_metadata.name,
                client_specific_requests=client_specific_requests or "",
                sector_specific_comments=sector_specific_comments or "",
                regulations=regulations or "",
                detail_level=detail_level_str,
                max_topics=max_topics,
                max_subtopics=max_subtopics_per_topic
            )

            # Generate TOC structure using LLM
            logger.info("Calling LLM to generate TOC structure")
            llm_response = await self.llm.generate(formatted_prompt)

            # Parse the LLM response to extract TOC structure
            toc_structure = self._parse_llm_toc_response(llm_response)

            # Transform to match expected format (similar to similarity search results)
            formatted_toc = self._format_toc_for_draft(toc_structure)

            logger.info(f"Generated TOC with {len(formatted_toc)} topics")
            return formatted_toc

        except Exception as e:
            logger.error(f"Failed to generate TOC: {e}", exc_info=True)
            # Return a basic fallback TOC structure
            return self._get_fallback_toc(title, client_metadata.industry)

    def _parse_llm_toc_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract structured TOC.
        Expects JSON format but handles plain text fallback.
        """
        try:
            # First try to extract JSON from the response
            # LLM might return JSON wrapped in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # Try to find JSON structure in the response
                json_str = response.strip()
                # Look for array brackets
                if "[" in json_str and "]" in json_str:
                    start = json_str.find("[")
                    end = json_str.rfind("]") + 1
                    json_str = json_str[start:end]

            # Parse JSON
            toc_data = json.loads(json_str)

            # Validate structure
            if isinstance(toc_data, list) and all(isinstance(item, dict) for item in toc_data):
                return toc_data
            else:
                logger.warning("Invalid TOC structure from LLM, using text parsing")
                return self._parse_text_toc(response)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return self._parse_text_toc(response)
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return []

    def _parse_text_toc(self, response: str) -> List[Dict[str, Any]]:
        """
        Fallback parser for plain text TOC format.
        Looks for numbered or bulleted lists.
        """
        toc = []
        current_topic = None

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for main topic patterns (1., 1), -, *)
            if any(line.startswith(p) for p in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']):
                # Save previous topic if exists
                if current_topic:
                    toc.append(current_topic)
                # Start new topic
                topic_text = line.split('.', 1)[1].strip() if '.' in line else line
                current_topic = {
                    "topic": topic_text,
                    "subtopics": []
                }
            elif current_topic and any(line.startswith(p) for p in ['  -', '  •', '  *', '    -', 'a.', 'b.', 'c.', 'i.', 'ii.']):
                # This is a subtopic
                subtopic_text = line.lstrip('- •*').strip()
                if '.' in subtopic_text and subtopic_text[0].isalpha():
                    subtopic_text = subtopic_text.split('.', 1)[1].strip()
                current_topic["subtopics"].append({"topic": subtopic_text})

        # Don't forget the last topic
        if current_topic:
            toc.append(current_topic)

        return toc

    def _format_toc_for_draft(self, toc_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform parsed TOC structure to match expected draft format.
        Generates UUIDs and adds required fields.
        """
        formatted_toc = []

        for idx, topic_data in enumerate(toc_structure, 1):
            # Generate unique IDs (these will be draft IDs, not source IDs)
            topic_id = str(uuid.uuid4())

            # Format subtopics
            subtopics = []
            for sub_idx, subtopic_data in enumerate(topic_data.get("subtopics", []), 1):
                subtopic = {
                    "subtopic_id": str(uuid.uuid4()),
                    "topic": subtopic_data.get("topic", f"Subtopic {sub_idx}"),
                    "order": sub_idx,
                    # No source_subtopic_id since this is AI-generated
                }
                subtopics.append(subtopic)

            # Create formatted topic
            formatted_topic = {
                "topic_id": topic_id,
                "topic": topic_data.get("topic", f"Topic {idx}"),
                "order": idx,
                "subtopics": subtopics,
                # Mark as AI-generated by not including source_topic_id
                "is_ai_generated": True
            }

            formatted_toc.append(formatted_topic)

        return formatted_toc

    def _get_fallback_toc(self, title: str, industry: str) -> List[Dict[str, Any]]:
        """
        Generate a basic fallback TOC structure based on common policy sections.
        Used when LLM generation fails.
        """
        logger.info(f"Using fallback TOC for {industry} industry")

        # Industry-specific fallback templates
        industry_templates = {
            "healthcare": [
                {"topic": "Purpose and Scope", "subtopics": [
                    {"topic": "Policy Objectives"},
                    {"topic": "Applicability"},
                    {"topic": "Regulatory Compliance"}
                ]},
                {"topic": "Definitions and Terms", "subtopics": [
                    {"topic": "Key Terminology"},
                    {"topic": "Acronyms and Abbreviations"}
                ]},
                {"topic": "Roles and Responsibilities", "subtopics": [
                    {"topic": "Management Responsibilities"},
                    {"topic": "Staff Responsibilities"},
                    {"topic": "Third-Party Obligations"}
                ]},
                {"topic": "Policy Requirements", "subtopics": [
                    {"topic": "Mandatory Requirements"},
                    {"topic": "Best Practices"},
                    {"topic": "Documentation Standards"}
                ]},
                {"topic": "Compliance and Monitoring", "subtopics": [
                    {"topic": "Audit Procedures"},
                    {"topic": "Performance Metrics"},
                    {"topic": "Reporting Requirements"}
                ]}
            ],
            "finance": [
                {"topic": "Executive Summary", "subtopics": [
                    {"topic": "Policy Overview"},
                    {"topic": "Key Objectives"}
                ]},
                {"topic": "Regulatory Framework", "subtopics": [
                    {"topic": "Applicable Regulations"},
                    {"topic": "Compliance Standards"}
                ]},
                {"topic": "Risk Management", "subtopics": [
                    {"topic": "Risk Assessment"},
                    {"topic": "Mitigation Strategies"}
                ]},
                {"topic": "Operational Procedures", "subtopics": [
                    {"topic": "Standard Processes"},
                    {"topic": "Exception Handling"}
                ]},
                {"topic": "Governance and Oversight", "subtopics": [
                    {"topic": "Board Oversight"},
                    {"topic": "Management Controls"}
                ]}
            ],
            "technology": [
                {"topic": "Introduction", "subtopics": [
                    {"topic": "Background"},
                    {"topic": "Objectives"}
                ]},
                {"topic": "Technical Standards", "subtopics": [
                    {"topic": "Architecture Guidelines"},
                    {"topic": "Security Requirements"}
                ]},
                {"topic": "Implementation Guidelines", "subtopics": [
                    {"topic": "Development Standards"},
                    {"topic": "Deployment Procedures"}
                ]},
                {"topic": "Maintenance and Support", "subtopics": [
                    {"topic": "Monitoring Requirements"},
                    {"topic": "Update Procedures"}
                ]},
                {"topic": "Documentation Requirements", "subtopics": [
                    {"topic": "Technical Documentation"},
                    {"topic": "User Documentation"}
                ]}
            ]
        }

        # Get industry-specific template or use default
        industry_lower = industry.lower()
        if industry_lower in industry_templates:
            template = industry_templates[industry_lower]
        else:
            # Default template for any industry
            template = [
                {"topic": "Purpose and Scope", "subtopics": [
                    {"topic": "Objectives"},
                    {"topic": "Applicability"}
                ]},
                {"topic": "Definitions", "subtopics": [
                    {"topic": "Key Terms"},
                    {"topic": "Abbreviations"}
                ]},
                {"topic": "Policy Statement", "subtopics": [
                    {"topic": "Core Principles"},
                    {"topic": "Requirements"}
                ]},
                {"topic": "Procedures", "subtopics": [
                    {"topic": "Implementation Steps"},
                    {"topic": "Workflows"}
                ]},
                {"topic": "Compliance", "subtopics": [
                    {"topic": "Monitoring"},
                    {"topic": "Enforcement"}
                ]}
            ]

        # Format the template
        return self._format_toc_for_draft(template)