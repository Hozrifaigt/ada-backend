"""Service for parsing policy templates from Excel files."""

import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import pandas as pd

from app.models.policy_template_models import PolicyTemplate
from app.models.draft_models import TOCTopic, TOCSubtopic

logger = logging.getLogger(__name__)


class ExcelParserService:
    """Parse policy templates from Excel files."""

    def __init__(self):
        """Initialize the Excel parser service."""
        pass

    async def parse_excel(self, file_content: bytes) -> Tuple[List[PolicyTemplate], List[str]]:
        """
        Parse Excel file and return policy templates.

        Args:
            file_content: Bytes content of the Excel file

        Returns:
            Tuple of (templates, errors)
        """
        try:
            # Read Excel file
            df = pd.read_excel(BytesIO(file_content))

            # Validate columns
            required_columns = ['Function', 'Policies', 'Topics', 'Topic Content', 'Subtopics', 'Subtopic Content']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                return [], [f"Missing required columns: {', '.join(missing_columns)}"]

            # Forward fill empty cells (Excel merged cell pattern)
            df['Function'] = df['Function'].ffill()
            df['Policies'] = df['Policies'].ffill()
            df['Topics'] = df['Topics'].ffill()
            df['Topic Content'] = df['Topic Content'].fillna('')
            df['Subtopics'] = df['Subtopics'].fillna('')
            df['Subtopic Content'] = df['Subtopic Content'].fillna('')

            # Build templates
            templates = self._build_templates(df)

            logger.info(f"Successfully parsed {len(templates)} policy templates from Excel")
            return templates, []

        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            return [], [f"Failed to parse Excel file: {str(e)}"]

    def _build_templates(self, df: pd.DataFrame) -> List[PolicyTemplate]:
        """
        Build PolicyTemplate objects from DataFrame.

        The DataFrame has a flat structure where empty cells mean "same as above".
        We need to build a hierarchical structure grouped by Function → Policy → Topic → Subtopic.
        """
        templates = []

        # Group by Function and Policy
        grouped = df.groupby(['Function', 'Policies'], dropna=False)

        for (function, policy_type), policy_df in grouped:
            # Skip if policy name is NaN or empty
            if pd.isna(policy_type) or str(policy_type).strip() == '':
                continue

            # Build TOC for this policy
            toc = self._build_toc(policy_df)

            # Create template
            template = PolicyTemplate(
                template_id=str(uuid.uuid4()),
                function=str(function).strip(),
                policy_type=str(policy_type).strip(),
                toc=toc
            )

            templates.append(template)

        return templates

    def _build_toc(self, policy_df: pd.DataFrame) -> List[TOCTopic]:
        """
        Build table of contents for a single policy.

        Args:
            policy_df: DataFrame rows for a single policy

        Returns:
            List of TOCTopic objects
        """
        toc = []
        current_topic = None
        topic_order = 0

        # Group by Topic
        grouped = policy_df.groupby('Topics', dropna=False)

        for topic_name, topic_df in grouped:
            # Skip if topic name is NaN or empty
            if pd.isna(topic_name) or str(topic_name).strip() == '':
                continue

            topic_order += 1
            topic_name = str(topic_name).strip()

            # Get topic content (first row's Topic Content for this topic)
            topic_content = str(topic_df.iloc[0]['Topic Content']).strip() if not pd.isna(topic_df.iloc[0]['Topic Content']) else ''

            # Build subtopics
            subtopics = self._build_subtopics(topic_df)

            # Create TOCTopic
            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),
                topic=topic_name,
                order=topic_order,
                content=topic_content,
                summary='',
                conversation_history=[],
                subtopics=subtopics
            )

            toc.append(topic)

        return toc

    def _build_subtopics(self, topic_df: pd.DataFrame) -> List[TOCSubtopic]:
        """
        Build subtopics for a single topic.

        Args:
            topic_df: DataFrame rows for a single topic

        Returns:
            List of TOCSubtopic objects
        """
        subtopics = []
        subtopic_order = 0

        for _, row in topic_df.iterrows():
            subtopic_name = str(row['Subtopics']).strip() if not pd.isna(row['Subtopics']) else ''

            # Skip if no subtopic name
            if not subtopic_name or subtopic_name == '' or subtopic_name == 'nan':
                continue

            subtopic_order += 1

            # Get subtopic content
            subtopic_content = str(row['Subtopic Content']).strip() if not pd.isna(row['Subtopic Content']) else ''

            # Create TOCSubtopic
            subtopic = TOCSubtopic(
                subtopic_id=str(uuid.uuid4()),
                topic=subtopic_name,
                order=subtopic_order,
                content=subtopic_content,
                summary='',
                conversation_history=[]
            )

            subtopics.append(subtopic)

        return subtopics

    def get_template_summary(self, template: PolicyTemplate) -> Dict[str, Any]:
        """Get summary information about a template."""
        topic_count = len(template.toc)
        subtopic_count = sum(len(topic.subtopics) for topic in template.toc)

        return {
            'template_id': template.template_id,
            'function': template.function,
            'policy_type': template.policy_type,
            'topic_count': topic_count,
            'subtopic_count': subtopic_count,
            'created_at': template.created_at
        }
