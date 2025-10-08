import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

from app.models.draft_models import TOCTopic, TOCSubtopic
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


class TOCManipulationService:
    """Service for handling natural language TOC modifications."""

    def __init__(self, llm: AzureLLM, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    def _ensure_int(self, value: Any, default: int = 1) -> int:
        """Convert a value to integer, handling various input types."""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        return default

    async def parse_toc_command(
        self,
        user_command: str,
        current_toc: List[Dict],
        conversation_history: List[Dict] = None,
        draft_context: Dict = None
    ) -> Dict[str, Any]:
        """
        Parse a natural language command to modify the TOC.

        Args:
            user_command: Natural language command from user
            current_toc: Current TOC structure
            conversation_history: Previous chat exchanges
            draft_context: Draft metadata (title, description, industry)

        Returns:
            Dictionary with parsed action and parameters
        """
        try:
            # Format current TOC for prompt
            toc_text = self._format_toc_for_prompt(current_toc)

            # Format conversation history
            history_text = ""
            if conversation_history:
                history_text = self._format_conversation_history(conversation_history[-3:])

            # Get the parsing prompt and format it with variables
            prompt = self.prompt_manager.get_prompt(
                "parse_toc_modification",
                current_toc=toc_text,
                user_command=user_command,
                conversation_history=history_text,
                draft_title=draft_context.get('title', 'Policy Document') if draft_context else 'Policy Document',
                draft_description=draft_context.get('description', '') if draft_context else '',
                company_name=draft_context.get('company_name', '') if draft_context else '',
                industry=draft_context.get('industry', '') if draft_context else '',
                city=draft_context.get('city', '') if draft_context else '',
                country=draft_context.get('country', '') if draft_context else ''
            )

            # Get LLM response
            response = await self.llm.generate(prompt)

            # Parse JSON response
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    logger.error(f"Failed to parse LLM response as JSON: {response}")
                    return self._error_response("I couldn't understand that command. Could you please rephrase it?")

            # Validate and enhance the parsed command
            validated = self._validate_parsed_command(parsed, current_toc)

            return validated

        except Exception as e:
            logger.error(f"Error parsing TOC command: {e}")
            return self._error_response(f"An error occurred while processing your command: {str(e)}")

    def execute_toc_operation(
        self,
        operation: Dict[str, Any],
        current_toc: List[Dict],
        preserve_content: bool = True
    ) -> Tuple[List[Dict], str]:
        """
        Execute a TOC modification operation.

        Args:
            operation: Parsed operation from parse_toc_command
            current_toc: Current TOC structure
            preserve_content: Whether to preserve existing content and conversation history

        Returns:
            Tuple of (modified_toc, success_message)
        """
        action = operation.get('action')
        params = operation.get('parameters', {})

        try:
            if action == 'suggest_topics':
                # Convert suggestions into add operations
                return self._apply_suggestions(current_toc, params)
            elif action == 'add_topic':
                return self._add_topic(current_toc, params, preserve_content)
            elif action == 'add_subtopic':
                return self._add_subtopic(current_toc, params, preserve_content)
            elif action == 'remove_topic':
                return self._remove_topic(current_toc, params, preserve_content)
            elif action == 'remove_topics':
                return self._remove_multiple_topics(current_toc, params, preserve_content)
            elif action == 'remove_subtopic':
                return self._remove_subtopic(current_toc, params, preserve_content)
            elif action == 'rename_topic':
                return self._rename_topic(current_toc, params)
            elif action == 'rename_subtopic':
                return self._rename_subtopic(current_toc, params)
            elif action == 'reorder_topics':
                return self._reorder_topics(current_toc, params)
            elif action == 'reorder_subtopics':
                return self._reorder_subtopics(current_toc, params)
            elif action == 'move_topic':
                return self._move_topic(current_toc, params)
            elif action == 'move_subtopic':
                return self._move_subtopic(current_toc, params)
            elif action == 'split_topic':
                return self._split_topic(current_toc, params, preserve_content)
            elif action == 'merge_topics':
                return self._merge_topics(current_toc, params, preserve_content)
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing TOC operation: {e}")
            raise

    def _apply_suggestions(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Apply AI-generated suggestions to the TOC."""
        suggestions = params.get('suggestions', [])

        if not suggestions:
            return toc, "No suggestions to apply"

        toc_copy = toc.copy()
        added_topics = []
        added_subtopics = []

        for suggestion in suggestions:
            if suggestion['type'] == 'topic':
                # Add as a new main topic
                new_topic = {
                    'topic_id': str(uuid.uuid4()),
                    'topic': suggestion['title'],
                    'order': len(toc_copy) + 1,
                    'content': '',
                    'summary': suggestion.get('description', ''),
                    'conversation_history': [],
                    'subtopics': []
                }

                # Add suggested subtopics if any
                if suggestion.get('subtopics'):
                    for i, subtopic_title in enumerate(suggestion['subtopics']):
                        new_topic['subtopics'].append({
                            'subtopic_id': str(uuid.uuid4()),
                            'topic': subtopic_title,
                            'order': i + 1,
                            'content': '',
                            'summary': '',
                            'conversation_history': []
                        })

                toc_copy.append(new_topic)
                added_topics.append(suggestion['title'])

            elif suggestion['type'] == 'subtopic':
                # Find parent topic and add subtopic
                parent_topic_name = suggestion.get('parent_topic')
                if parent_topic_name:
                    for topic in toc_copy:
                        if topic['topic'].lower() == parent_topic_name.lower():
                            new_subtopic = {
                                'subtopic_id': str(uuid.uuid4()),
                                'topic': suggestion['title'],
                                'order': len(topic.get('subtopics', [])) + 1,
                                'content': '',
                                'summary': suggestion.get('description', ''),
                                'conversation_history': []
                            }

                            if 'subtopics' not in topic:
                                topic['subtopics'] = []
                            topic['subtopics'].append(new_subtopic)
                            added_subtopics.append(f"{suggestion['title']} (under {parent_topic_name})")
                            break

        # Reorder all topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        # Create success message
        messages = []
        if added_topics:
            messages.append(f"Added {len(added_topics)} topics: {', '.join(added_topics)}")
        if added_subtopics:
            messages.append(f"Added {len(added_subtopics)} subtopics: {', '.join(added_subtopics)}")

        return toc_copy, '. '.join(messages) if messages else "No suggestions were applied"

    def _add_topic(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Add a new topic to the TOC."""
        # Ensure position is an integer
        position_value = self._ensure_int(params.get('position'), len(toc) + 1)

        new_topic = {
            'topic_id': str(uuid.uuid4()),
            'topic': params.get('title', 'New Topic'),
            'order': position_value,
            'content': '',
            'summary': '',
            'conversation_history': [],
            'subtopics': []
        }

        # Add subtopics if provided
        if 'subtopics' in params:
            for i, subtopic_title in enumerate(params['subtopics']):
                new_topic['subtopics'].append({
                    'subtopic_id': str(uuid.uuid4()),
                    'topic': subtopic_title,
                    'order': i + 1,
                    'content': '',
                    'summary': '',
                    'conversation_history': []
                })

        # Insert at specified position or append
        position = position_value - 1
        toc_copy = toc.copy()
        toc_copy.insert(max(0, min(position, len(toc_copy))), new_topic)

        # Reorder remaining topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        message = f"Added topic '{params.get('title', 'New Topic')}' at position {position + 1}"
        if 'subtopics' in params:
            message += f" with {len(params['subtopics'])} subtopics"

        return toc_copy, message

    def _add_subtopic(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Add a new subtopic to an existing topic."""
        topic_id = params.get('topic_id')
        toc_copy = toc.copy()

        for topic in toc_copy:
            if topic['topic_id'] == topic_id:
                # Ensure position is an integer
                position_value = self._ensure_int(params.get('position'), len(topic.get('subtopics', [])) + 1)

                new_subtopic = {
                    'subtopic_id': str(uuid.uuid4()),
                    'topic': params.get('title', 'New Subtopic'),
                    'order': position_value,
                    'content': '',
                    'summary': '',
                    'conversation_history': []
                }

                if 'subtopics' not in topic:
                    topic['subtopics'] = []

                position = position_value - 1
                topic['subtopics'].insert(max(0, min(position, len(topic['subtopics']))), new_subtopic)

                # Reorder subtopics
                for i, subtopic in enumerate(topic['subtopics']):
                    subtopic['order'] = i + 1

                return toc_copy, f"Added subtopic '{params.get('title', 'New Subtopic')}' to topic '{topic['topic']}'"

        raise ValueError(f"Topic with ID {topic_id} not found")

    def _remove_topic(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Remove a topic from the TOC."""
        topic_id = params.get('topic_id')
        topic_name = params.get('topic_name')

        # Find the topic to remove
        removed_topic = None
        toc_copy = []

        for topic in toc:
            should_remove = False

            # Match by ID
            if topic_id and topic['topic_id'] == topic_id:
                should_remove = True

            # Match by name
            elif topic_name:
                # Direct name match
                if topic['topic'].lower() == str(topic_name).lower():
                    should_remove = True
                # Position-based match (e.g., "topic 3", "3", "third topic")
                elif str(topic_name).lower().startswith('topic '):
                    try:
                        position = int(str(topic_name).lower().replace('topic ', '').strip())
                        if topic['order'] == position:
                            should_remove = True
                    except (ValueError, AttributeError):
                        pass
                # Just a number (e.g., "3")
                elif str(topic_name).isdigit():
                    if topic['order'] == int(topic_name):
                        should_remove = True

            if should_remove:
                removed_topic = topic
            else:
                toc_copy.append(topic)

        if not removed_topic:
            error_msg = f"Topic with ID {topic_id} not found" if topic_id else f"Topic '{topic_name}' not found"
            raise ValueError(error_msg)

        # Reorder remaining topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        message = f"Removed topic '{removed_topic.get('topic', 'Unknown')}'"

        if preserve_content and removed_topic.get('content'):
            message += " (content was preserved and can be restored)"

        return toc_copy, message

    def _remove_multiple_topics(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Remove multiple topics from the TOC."""
        topic_ids = params.get('topic_ids', [])
        topic_names = params.get('topic_names', [])

        # Build a list of topics to remove
        topics_to_remove = set()
        removed_names = []

        for topic in toc:
            should_remove = False

            # Check if topic ID matches any in the list
            if topic_ids and topic['topic_id'] in topic_ids:
                should_remove = True

            # Check if topic name matches any in the list
            if topic_names:
                for name in topic_names:
                    if name and topic['topic'].lower() == str(name).lower():
                        should_remove = True
                        break

            # Also check if topic matches by position (e.g., "topic 3" means the 3rd topic)
            if topic_names:
                for name in topic_names:
                    if name and str(name).lower().startswith('topic '):
                        try:
                            position = int(str(name).lower().replace('topic ', '').strip())
                            if topic['order'] == position:
                                should_remove = True
                                break
                        except (ValueError, AttributeError):
                            pass

            if should_remove:
                topics_to_remove.add(topic['topic_id'])
                removed_names.append(topic.get('topic', 'Unknown'))

        if not topics_to_remove:
            error_msg = f"No topics found matching: {', '.join(topic_names) if topic_names else 'specified IDs'}"
            raise ValueError(error_msg)

        # Create new TOC without the removed topics
        toc_copy = [topic for topic in toc if topic['topic_id'] not in topics_to_remove]

        # Reorder remaining topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        message = f"Removed {len(removed_names)} topic(s): {', '.join(removed_names)}"

        if preserve_content:
            message += " (content was preserved and can be restored)"

        return toc_copy, message

    def _remove_subtopic(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Remove a subtopic from a topic."""
        topic_id = params.get('topic_id')
        topic_name = params.get('topic_name')
        subtopic_id = params.get('subtopic_id')
        subtopic_name = params.get('subtopic_name')
        toc_copy = toc.copy()

        for topic in toc_copy:
            # Match topic by ID or name
            if (topic_id and topic['topic_id'] == topic_id) or \
               (topic_name and topic['topic'].lower() == topic_name.lower()):

                # Find and remove subtopic by ID or name
                original_subtopics = topic.get('subtopics', [])
                removed_subtopic = None
                new_subtopics = []

                for subtopic in original_subtopics:
                    if (subtopic_id and subtopic['subtopic_id'] == subtopic_id) or \
                       (subtopic_name and subtopic['topic'].lower() == subtopic_name.lower()):
                        removed_subtopic = subtopic
                    else:
                        new_subtopics.append(subtopic)

                if not removed_subtopic:
                    error_msg = f"Subtopic with ID {subtopic_id} not found" if subtopic_id else f"Subtopic '{subtopic_name}' not found"
                    raise ValueError(error_msg)

                topic['subtopics'] = new_subtopics

                # Reorder remaining subtopics
                for i, subtopic in enumerate(topic['subtopics']):
                    subtopic['order'] = i + 1

                return toc_copy, f"Removed subtopic '{removed_subtopic['topic']}' from topic '{topic['topic']}'"

        error_msg = f"Topic with ID {topic_id} not found" if topic_id else f"Topic '{topic_name}' not found"
        raise ValueError(error_msg)

    def _rename_topic(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Rename a topic."""
        topic_id = params.get('topic_id')
        topic_name = params.get('topic_name')
        new_name = params.get('new_name')
        toc_copy = toc.copy()

        for topic in toc_copy:
            # Match by ID or name
            if (topic_id and topic['topic_id'] == topic_id) or \
               (topic_name and topic['topic'].lower() == topic_name.lower()):
                old_name = topic['topic']
                topic['topic'] = new_name
                return toc_copy, f"Renamed topic '{old_name}' to '{new_name}'"

        error_msg = f"Topic with ID {topic_id} not found" if topic_id else f"Topic '{topic_name}' not found"
        raise ValueError(error_msg)

    def _rename_subtopic(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Rename a subtopic."""
        topic_id = params.get('topic_id')
        subtopic_id = params.get('subtopic_id')
        new_name = params.get('new_name')
        toc_copy = toc.copy()

        for topic in toc_copy:
            if topic['topic_id'] == topic_id:
                for subtopic in topic.get('subtopics', []):
                    if subtopic['subtopic_id'] == subtopic_id:
                        old_name = subtopic['topic']
                        subtopic['topic'] = new_name
                        return toc_copy, f"Renamed subtopic '{old_name}' to '{new_name}'"

                raise ValueError(f"Subtopic with ID {subtopic_id} not found")

        raise ValueError(f"Topic with ID {topic_id} not found")

    def _reorder_topics(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Reorder all topics based on new positions."""
        new_order = params.get('new_order', [])  # List of topic_ids in new order
        toc_copy = []

        # Build new TOC in specified order
        for topic_id in new_order:
            topic = next((t for t in toc if t['topic_id'] == topic_id), None)
            if topic:
                topic_copy = topic.copy()
                topic_copy['order'] = len(toc_copy) + 1
                toc_copy.append(topic_copy)

        # Add any topics not in new_order (shouldn't happen, but safety check)
        for topic in toc:
            if topic['topic_id'] not in new_order:
                topic_copy = topic.copy()
                topic_copy['order'] = len(toc_copy) + 1
                toc_copy.append(topic_copy)

        return toc_copy, "Reordered topics successfully"

    def _reorder_subtopics(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Reorder subtopics within a topic."""
        topic_id = params.get('topic_id')
        new_order = params.get('new_order', [])  # List of subtopic_ids in new order
        toc_copy = toc.copy()

        for topic in toc_copy:
            if topic['topic_id'] == topic_id:
                new_subtopics = []

                # Build new subtopic list in specified order
                for subtopic_id in new_order:
                    subtopic = next((s for s in topic.get('subtopics', []) if s['subtopic_id'] == subtopic_id), None)
                    if subtopic:
                        subtopic_copy = subtopic.copy()
                        subtopic_copy['order'] = len(new_subtopics) + 1
                        new_subtopics.append(subtopic_copy)

                # Add any subtopics not in new_order
                for subtopic in topic.get('subtopics', []):
                    if subtopic['subtopic_id'] not in new_order:
                        subtopic_copy = subtopic.copy()
                        subtopic_copy['order'] = len(new_subtopics) + 1
                        new_subtopics.append(subtopic_copy)

                topic['subtopics'] = new_subtopics
                return toc_copy, f"Reordered subtopics in topic '{topic['topic']}'"

        raise ValueError(f"Topic with ID {topic_id} not found")

    def _move_topic(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Move a topic to a new position."""
        topic_id = params.get('topic_id')
        # Ensure new_position is an integer
        new_position = self._ensure_int(params.get('new_position'), 1)

        # Find and remove the topic
        topic_to_move = None
        toc_copy = []
        for topic in toc:
            if topic['topic_id'] == topic_id:
                topic_to_move = topic.copy()
            else:
                toc_copy.append(topic)

        if not topic_to_move:
            raise ValueError(f"Topic with ID {topic_id} not found")

        # Insert at new position
        position = max(0, min(new_position - 1, len(toc_copy)))
        toc_copy.insert(position, topic_to_move)

        # Reorder all topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        return toc_copy, f"Moved topic '{topic_to_move['topic']}' to position {position + 1}"

    def _move_subtopic(self, toc: List[Dict], params: Dict) -> Tuple[List[Dict], str]:
        """Move a subtopic to a different topic or position."""
        source_topic_id = params.get('source_topic_id')
        target_topic_id = params.get('target_topic_id')
        subtopic_id = params.get('subtopic_id')
        # Ensure new_position is an integer
        new_position = self._ensure_int(params.get('new_position'), 1)

        toc_copy = toc.copy()
        subtopic_to_move = None

        # Find and remove the subtopic from source
        for topic in toc_copy:
            if topic['topic_id'] == source_topic_id:
                for subtopic in topic.get('subtopics', []):
                    if subtopic['subtopic_id'] == subtopic_id:
                        subtopic_to_move = subtopic.copy()
                        topic['subtopics'] = [s for s in topic['subtopics'] if s['subtopic_id'] != subtopic_id]
                        break

        if not subtopic_to_move:
            raise ValueError(f"Subtopic with ID {subtopic_id} not found")

        # Add to target topic
        for topic in toc_copy:
            if topic['topic_id'] == target_topic_id:
                if 'subtopics' not in topic:
                    topic['subtopics'] = []

                position = max(0, min(new_position - 1, len(topic['subtopics'])))
                topic['subtopics'].insert(position, subtopic_to_move)

                # Reorder subtopics in target
                for i, subtopic in enumerate(topic['subtopics']):
                    subtopic['order'] = i + 1

                break

        # Reorder subtopics in source
        for topic in toc_copy:
            if topic['topic_id'] == source_topic_id:
                for i, subtopic in enumerate(topic.get('subtopics', [])):
                    subtopic['order'] = i + 1
                break

        message = f"Moved subtopic '{subtopic_to_move['topic']}'"
        if source_topic_id != target_topic_id:
            message += " to a different topic"

        return toc_copy, message

    def _split_topic(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Split a topic into two topics."""
        topic_id = params.get('topic_id')
        split_point = params.get('split_point', None)  # Subtopic index to split at
        new_topic_name = params.get('new_topic_name', 'New Topic (Split)')

        toc_copy = []
        original_topic = None
        original_position = 0

        for i, topic in enumerate(toc):
            if topic['topic_id'] == topic_id:
                original_topic = topic.copy()
                original_position = i
            toc_copy.append(topic)

        if not original_topic:
            raise ValueError(f"Topic with ID {topic_id} not found")

        # Create new topic
        new_topic = {
            'topic_id': str(uuid.uuid4()),
            'topic': new_topic_name,
            'order': original_position + 2,
            'content': '',
            'summary': '',
            'conversation_history': [],
            'subtopics': []
        }

        # Split subtopics if split_point provided
        if split_point is not None and 'subtopics' in original_topic:
            subtopics = original_topic['subtopics']
            # Keep first part in original
            toc_copy[original_position]['subtopics'] = subtopics[:split_point]
            # Move second part to new topic
            new_topic['subtopics'] = subtopics[split_point:]

            # Reorder subtopics
            for i, subtopic in enumerate(toc_copy[original_position]['subtopics']):
                subtopic['order'] = i + 1
            for i, subtopic in enumerate(new_topic['subtopics']):
                subtopic['order'] = i + 1

        # Insert new topic after original
        toc_copy.insert(original_position + 1, new_topic)

        # Reorder all topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        return toc_copy, f"Split topic '{original_topic['topic']}' into two topics"

    def _merge_topics(self, toc: List[Dict], params: Dict, preserve_content: bool) -> Tuple[List[Dict], str]:
        """Merge two or more topics into one."""
        topic_ids = params.get('topic_ids', [])
        merged_name = params.get('merged_name', None)

        if len(topic_ids) < 2:
            raise ValueError("Need at least 2 topics to merge")

        toc_copy = []
        topics_to_merge = []
        first_position = None

        for i, topic in enumerate(toc):
            if topic['topic_id'] in topic_ids:
                topics_to_merge.append(topic)
                if first_position is None:
                    first_position = i
            else:
                toc_copy.append(topic)

        if len(topics_to_merge) < 2:
            raise ValueError("Could not find all topics to merge")

        # Create merged topic
        merged_topic = topics_to_merge[0].copy()
        if merged_name:
            merged_topic['topic'] = merged_name

        # Merge subtopics
        all_subtopics = []
        for topic in topics_to_merge:
            all_subtopics.extend(topic.get('subtopics', []))

        # Reorder merged subtopics
        for i, subtopic in enumerate(all_subtopics):
            subtopic['order'] = i + 1

        merged_topic['subtopics'] = all_subtopics

        # Merge content if preserve_content is True
        if preserve_content:
            contents = [t.get('content', '') for t in topics_to_merge if t.get('content')]
            if contents:
                merged_topic['content'] = '\n\n'.join(contents)

        # Insert merged topic at first position
        toc_copy.insert(first_position, merged_topic)

        # Reorder all topics
        for i, topic in enumerate(toc_copy):
            topic['order'] = i + 1

        topic_names = [t['topic'] for t in topics_to_merge]
        return toc_copy, f"Merged {len(topics_to_merge)} topics: {', '.join(topic_names)}"

    def _format_toc_for_prompt(self, toc: List[Dict]) -> str:
        """Format TOC for inclusion in prompt with detailed status."""
        if not toc:
            return "No topics in the table of contents yet."

        lines = []
        total_topics = 0
        completed_topics = 0

        for topic in toc:
            total_topics += 1
            has_content = bool(topic.get('content'))
            if has_content:
                completed_topics += 1

            status = "[COMPLETED]" if has_content else "[PENDING]"
            word_count = f"({len(topic.get('content', '').split())} words)" if has_content else ""

            lines.append(f"{topic['order']}. {topic['topic']} {status} {word_count} (ID: {topic['topic_id']})")

            # Include topic summary if available
            if topic.get('summary'):
                lines.append(f"   Summary: {topic['summary'][:100]}...")

            for subtopic in topic.get('subtopics', []):
                total_topics += 1
                sub_has_content = bool(subtopic.get('content'))
                if sub_has_content:
                    completed_topics += 1

                sub_status = "[COMPLETED]" if sub_has_content else "[PENDING]"
                sub_word_count = f"({len(subtopic.get('content', '').split())} words)" if sub_has_content else ""

                lines.append(f"   {topic['order']}.{subtopic['order']} {subtopic['topic']} {sub_status} {sub_word_count} (ID: {subtopic['subtopic_id']})")

        # Add summary
        lines.append(f"\nProgress: {completed_topics}/{total_topics} sections completed ({int(completed_topics*100/total_topics if total_topics else 0)}%)")
        return '\n'.join(lines)

    async def generate_topic_suggestions(
        self,
        user_query: str,
        current_toc: List[Dict],
        draft_context: Dict
    ) -> Dict[str, Any]:
        """
        Generate intelligent topic/subtopic suggestions based on user query.

        Args:
            user_query: User's request for suggestions
            current_toc: Current TOC structure
            draft_context: Draft metadata (title, description, industry, etc.)

        Returns:
            Dictionary with suggestions
        """
        try:
            # Format current TOC for the prompt
            toc_text = self._format_toc_for_prompt(current_toc)

            # Get the suggestion prompt
            prompt = self.prompt_manager.get_prompt(
                "suggest_toc_topics",
                draft_title=draft_context.get('title', 'Policy Document'),
                draft_description=draft_context.get('description', ''),
                industry=draft_context.get('industry', ''),
                city=draft_context.get('city', ''),
                country=draft_context.get('country', ''),
                current_toc=toc_text,
                user_query=user_query
            )

            # Get LLM response
            response = await self.llm.generate(prompt)

            # Parse JSON response
            try:
                suggestions = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    suggestions = json.loads(json_match.group())
                else:
                    logger.error(f"Failed to parse suggestions as JSON: {response}")
                    return {
                        "suggestions": [],
                        "rationale": "Failed to generate suggestions",
                        "error": "Could not parse AI response"
                    }

            return suggestions

        except Exception as e:
            logger.error(f"Error generating topic suggestions: {e}")
            return {
                "suggestions": [],
                "rationale": "An error occurred while generating suggestions",
                "error": str(e)
            }

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt."""
        lines = []
        for entry in history:
            lines.append(f"User: {entry.get('user_message', '')}")
            lines.append(f"Assistant: {entry.get('ai_response', '')}")
        return '\n'.join(lines)

    def _validate_parsed_command(self, parsed: Dict, current_toc: List[Dict]) -> Dict[str, Any]:
        """Validate and enhance the parsed command."""
        action = parsed.get('action')

        # Add interpretation for user confirmation
        if 'interpretation' not in parsed:
            parsed['interpretation'] = self._generate_interpretation(parsed)

        # Check if confirmation is needed
        if 'requires_confirmation' not in parsed:
            # Destructive operations need confirmation
            destructive_actions = ['remove_topic', 'remove_topics', 'remove_subtopic', 'merge_topics', 'split_topic']
            parsed['requires_confirmation'] = action in destructive_actions

        # Validate topic/subtopic IDs exist
        if 'parameters' in parsed:
            params = parsed['parameters']

            # Check topic_id
            if 'topic_id' in params:
                if not any(t['topic_id'] == params['topic_id'] for t in current_toc):
                    # Try to find by name
                    topic = next((t for t in current_toc if t['topic'].lower() == params.get('topic_name', '').lower()), None)
                    if topic:
                        params['topic_id'] = topic['topic_id']
                    else:
                        parsed['error'] = f"Could not find topic: {params.get('topic_name', params['topic_id'])}"

            # Check subtopic_id
            if 'subtopic_id' in params and 'topic_id' in params:
                topic = next((t for t in current_toc if t['topic_id'] == params['topic_id']), None)
                if topic:
                    if not any(s['subtopic_id'] == params['subtopic_id'] for s in topic.get('subtopics', [])):
                        # Try to find by name
                        subtopic = next((s for s in topic.get('subtopics', []) if s['topic'].lower() == params.get('subtopic_name', '').lower()), None)
                        if subtopic:
                            params['subtopic_id'] = subtopic['subtopic_id']

        return parsed

    def _generate_interpretation(self, parsed: Dict) -> str:
        """Generate human-readable interpretation of the command."""
        action = parsed.get('action')
        params = parsed.get('parameters', {})

        interpretations = {
            'add_topic': f"Add a new topic '{params.get('title', 'New Topic')}'",
            'add_subtopic': f"Add a new subtopic '{params.get('title', 'New Subtopic')}'",
            'remove_topic': f"Remove the topic '{params.get('topic_name', 'Unknown')}'",
            'remove_topics': f"Remove {len(params.get('topic_names', params.get('topic_ids', [])))} topics",
            'remove_subtopic': f"Remove the subtopic '{params.get('subtopic_name', 'Unknown')}'",
            'rename_topic': f"Rename topic to '{params.get('new_name', 'New Name')}'",
            'rename_subtopic': f"Rename subtopic to '{params.get('new_name', 'New Name')}'",
            'reorder_topics': "Reorder all topics",
            'reorder_subtopics': "Reorder subtopics",
            'move_topic': f"Move topic to position {params.get('new_position', 'unknown')}",
            'move_subtopic': "Move subtopic",
            'split_topic': f"Split topic into two separate topics",
            'merge_topics': f"Merge {len(params.get('topic_ids', []))} topics together"
        }

        return interpretations.get(action, f"Perform {action} operation")

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            'action': 'error',
            'error': message,
            'requires_confirmation': False,
            'interpretation': message
        }