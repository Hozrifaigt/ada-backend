import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


from app.models.draft_models import ClientMetadata, DraftMetadata, CompleteDraft
from app.services.generation_service import ContentGenerationService
from app.services.toc_generation_service import TOCGenerationService
from app.services.reflection_agent import ReflectionAgent
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DraftService:
    """Business logic service for draft management."""

    def __init__(self, storage_service):
        """Initialize draft service with storage dependency."""
        self.storage = storage_service

        # Initialize reflection agent for content critique
        prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        llm_service = AzureLLM(prompt_manager)
        reflection_agent = ReflectionAgent(llm_service)
        self.generation_service = ContentGenerationService(reflection_agent=reflection_agent)
    
    async def initialize_draft(
        self,
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        llm_service=None,
        vector_store=None,
        description: Optional[str] = None,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3
    ) -> Dict[str, Any]:
        """
        Initialize a new draft.
        If similar policies exist (>70% threshold), return them for user selection
        instead of immediately generating a TOC. Otherwise generate TOC with AI.
        """
        try:
            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=description,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source="pending_selection",
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level
            )

            draft_to_save = CompleteDraft(
                id=draft_id,
                metadata=draft_metadata,
                toc=[]
            )

            # Search for similar policies (top 3, threshold 0.7)
            similar_policies = []
            if vector_store:
                try:
                    description = draft_metadata.description or ""
                    search_query = f"{title} {description}".strip()
                    embedding = vector_store.embedder.embed_query(search_query)
                    matches = vector_store.search_similar_policies(
                        query_vector=embedding,
                        threshold=0.75,
                        top_k=3,
                        function=function
                    )
                    similar_policies = [
                        {
                            "policy_id": m["id"],
                            "filename": m["filename"],
                            "description": m["description"],
                            "web_url": m.get("web_url", ""),
                            "similarity_score": round(m["similarity_score"], 3),
                        }
                        for m in matches
                    ]
                    logger.info(f"Found {len(similar_policies)} similar policies above cosine 0.75 for '{title}'")
                except Exception as e:
                    import traceback
                    logger.error(f"Similar policy search FAILED (falling back to AI TOC): {e}")
                    logger.error(traceback.format_exc())

            # If similar policies found, save draft with empty TOC and return for user selection
            if similar_policies:
                draft_to_save.metadata.toc_source = "pending_selection"
                success = self.storage.create_draft(draft_to_save)
                if not success:
                    logger.error(f"Failed to save draft {draft_id}")
                    return None
                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_to_save.model_dump(),
                    "similar_policies_found": len(similar_policies),
                    "most_similar_policy_id": similar_policies[0]["policy_id"],
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": "pending_selection",
                    "needs_policy_selection": True,
                    "similar_policies": similar_policies,
                }

            # No similar policies — generate TOC with AI
            toc_source = "ai_generated"
            logger.info(f"No similar policies found, generating AI TOC for '{title}'")
            if llm_service:
                toc_generator = TOCGenerationService(llm_service)
                generated_toc = await toc_generator.generate_toc(
                    title=title,
                    client_metadata=client_metadata,
                    function=function,
                    client_specific_requests=client_specific_requests,
                    sector_specific_comments=sector_specific_comments,
                    regulations=regulations,
                    detail_level=detail_level
                )
                draft_to_save.toc = self._transform_ai_generated_toc(generated_toc)
                logger.info(f"Generated TOC with {len(draft_to_save.toc)} topics using AI")
            else:
                logger.warning("No LLM service available, using basic template")
                toc_source = "basic_template"

            draft_to_save.metadata.toc_source = toc_source
            if not draft_to_save.toc:
                draft_to_save.toc = self._get_basic_toc_template()

            success = self.storage.create_draft(draft_to_save)
            if not success:
                logger.error(f"Failed to save draft {draft_to_save.id}")
                return None

            return {
                "draft_id": draft_to_save.id,
                "draft": draft_to_save.model_dump(),
                "similar_policies_found": 0,
                "most_similar_policy_id": None,
                "client_metadata": client_metadata.model_dump(),
                "toc_source": toc_source,
                "needs_policy_selection": False,
                "similar_policies": [],
            }

        except Exception as e:
            logger.error(f"Failed to initialize draft: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    async def apply_policy_toc(
        self,
        draft_id: str,
        policy_id: Optional[str],
        vector_store=None,
        llm_service=None
    ) -> bool:
        """
        Apply a selected policy's TOC to an existing draft, or generate with AI.
        Called after the user picks a similar policy from the selection screen.
        """
        try:
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found for apply_policy_toc")
                return False

            if policy_id and vector_store:
                # Get topics for the selected policy from the vector store
                topics = vector_store.get_topics_by_policy(policy_id)
                if topics:
                    toc_input = self._build_toc_from_vector_topics(topics)
                    draft.toc = self._transform_toc_for_draft(toc_input)
                    draft.metadata.toc_source = "similar_policy"
                    draft.metadata.most_similar_policy_id = policy_id
                    logger.info(f"Applied policy {policy_id} TOC ({len(draft.toc)} topics) to draft {draft_id}")
                else:
                    logger.warning(f"Policy {policy_id} has no topics, falling back to AI")
                    policy_id = None

            if not policy_id:
                # Generate TOC with AI
                if llm_service:
                    toc_generator = TOCGenerationService(llm_service)
                    generated_toc = await toc_generator.generate_toc(
                        title=draft.metadata.title,
                        client_metadata=draft.metadata.client_metadata,
                        function=draft.metadata.function,
                        client_specific_requests=draft.metadata.client_specific_requests,
                        sector_specific_comments=draft.metadata.sector_specific_comments,
                        regulations=draft.metadata.regulations,
                        detail_level=draft.metadata.detail_level
                    )
                    draft.toc = self._transform_ai_generated_toc(generated_toc)
                else:
                    draft.toc = self._get_basic_toc_template()
                draft.metadata.toc_source = "ai_generated"
                logger.info(f"Generated AI TOC ({len(draft.toc)} topics) for draft {draft_id}")

            draft.metadata.modified_at = datetime.now(timezone.utc).isoformat()
            return self.storage.update_draft(draft_id, draft)

        except Exception as e:
            logger.error(f"Failed to apply policy TOC to draft {draft_id}: {e}")
            return False

    def _build_toc_from_vector_topics(self, topics: List[Dict]) -> List[Dict]:
        """Map vector store topic documents into the format _transform_toc_for_draft expects."""
        import json as _json
        result = []
        for idx, t in enumerate(topics, 1):
            raw_subs = t.get("subtopics", [])
            if isinstance(raw_subs, str):
                raw_subs = _json.loads(raw_subs)
            subtopics = []
            for sub_idx, sub in enumerate(raw_subs, 1):
                subtopics.append({
                    "subtopic_id": sub.get("id", sub.get("subtopic_id", str(uuid.uuid4()))),
                    "topic": sub.get("name", sub.get("topic", "")),
                    "order": sub_idx,
                })
            result.append({
                "topic_id": t.get("id", t.get("topic_id", str(uuid.uuid4()))),
                "topic": t.get("topic", ""),
                "order": idx,
                "subtopics": subtopics,
            })
        return result

    async def initialize_draft_from_upload(
        self,
        file_bytes: bytes,
        filename: str,
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3,
        previous_policy_bytes: Optional[bytes] = None,
        previous_policy_filename: Optional[str] = None,
        regulations_files: Optional[List[tuple]] = None  # list of (filename, bytes)
    ) -> Dict[str, Any]:
        """
        Initialize a new draft from an uploaded .docx file.
        Parses the document, extracts the TOC and content, and uses them as the
        starting point for the draft instead of AI generation.
        """
        try:
            from io import BytesIO
            from app.pipelines.document_processor import DocumentProcessor
            from azure.storage.blob import BlobServiceClient

            toc_source = "uploaded_policy"
            logger.info(f"Initializing draft '{title}' from uploaded file '{filename}'")

            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            # Upload files to Azure Blob Storage
            try:
                blob_service = BlobServiceClient.from_connection_string(
                    settings.AZURE_STORAGE_CONNECTION_STRING
                )
                container_client = blob_service.get_container_client(
                    settings.AZURE_STORAGE_CONTAINER_NAME
                )
                container_client.upload_blob(
                    name=f"uploaded-policies/{draft_id}/{filename}",
                    data=file_bytes, overwrite=True
                )
                if previous_policy_bytes and previous_policy_filename:
                    container_client.upload_blob(
                        name=f"uploaded-policies/{draft_id}/previous/{previous_policy_filename}",
                        data=previous_policy_bytes, overwrite=True
                    )
                if regulations_files:
                    for reg_filename, reg_bytes in regulations_files:
                        container_client.upload_blob(
                            name=f"uploaded-policies/{draft_id}/regulations/{reg_filename}",
                            data=reg_bytes, overwrite=True
                        )
                logger.info(f"Uploaded policy files to blob for draft {draft_id}")
            except Exception as e:
                logger.warning(f"Failed to upload files to blob storage (non-fatal): {e}")

            # Parse the main policy document
            processor = DocumentProcessor()
            processed = processor.process(BytesIO(file_bytes))
            topics = processed.get("topics", [])
            logger.info(f"Parsed document: {len(topics)} topics extracted")

            # Extract text from uploaded regulations files (.docx only)
            uploaded_regulations_text = None
            if regulations_files:
                reg_texts = []
                for reg_filename, reg_bytes in regulations_files:
                    if reg_filename.endswith(".docx"):
                        try:
                            reg_processed = processor.process(BytesIO(reg_bytes))
                            reg_text = reg_processed.get("full_text", "").strip()
                            if reg_text:
                                reg_texts.append(f"[{reg_filename}]\n{reg_text}")
                        except Exception as e:
                            logger.warning(f"Could not parse regulation file '{reg_filename}': {e}")
                if reg_texts:
                    uploaded_regulations_text = "\n\n---\n\n".join(reg_texts)
                    logger.info(f"Extracted text from {len(reg_texts)} regulation file(s)")

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=None,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source=toc_source,
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level,
                uploaded_regulations_text=uploaded_regulations_text
            )

            draft_to_save = CompleteDraft(
                id=draft_id,
                metadata=draft_metadata,
                toc=self._transform_uploaded_doc_toc(topics)
            )

            success = self.storage.create_draft(draft_to_save)

            if success:
                logger.info(f"Initialized draft {draft_id} from upload with {len(draft_to_save.toc)} topics")
                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_to_save.model_dump(),
                    "similar_policies_found": 0,
                    "most_similar_policy_id": None,
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": toc_source
                }
            else:
                logger.error(f"Failed to save uploaded draft {draft_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to initialize draft from upload: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _transform_uploaded_doc_toc(self, topics: List[Dict[str, Any]]) -> List:
        """
        Transform DocumentProcessor output topics into draft TOCTopic models.
        Content from the uploaded document is pre-filled into each topic/subtopic.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for idx, topic_data in enumerate(topics, 1):
            subtopics = []
            for sub_idx, sub in enumerate(topic_data.get("subtopics", []), 1):
                subtopic = TOCSubtopic(
                    subtopic_id=sub.get("id", str(uuid.uuid4())),
                    topic=sub.get("name", sub.get("topic", "")),
                    order=sub_idx,
                    content=sub.get("content", ""),
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            topic = TOCTopic(
                topic_id=topic_data.get("id", str(uuid.uuid4())),
                topic=topic_data.get("topic", ""),
                order=idx,
                content=topic_data.get("topic_content", ""),
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    async def initialize_draft_from_description(
        self,
        description: Optional[str],
        title: str,
        user_id: str,
        client_metadata: ClientMetadata,
        function: str,
        llm_service=None,
        client_specific_requests: Optional[str] = None,
        sector_specific_comments: Optional[str] = None,
        regulations: str = "",
        detail_level: int = 3
    ) -> Dict[str, Any]:
        """
        Initialize a new draft using AI-generated TOC based on policy information.
        TOC is always generated by AI using the policy title, description, and function.

        DEPRECATED: Use initialize_draft() instead for new implementations.
        """
        try:
            # Determine if we have a description
            has_description = description and description.strip()
            toc_source = "ai_generated"

            logger.info(f"Initializing draft with AI-generated TOC for function='{function}'")

            # Create draft structure
            draft_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            draft_metadata = DraftMetadata(
                id=draft_id,
                title=title,
                client_metadata=client_metadata,
                description=description,
                function=function,
                created_by=user_id,
                created_at=now.isoformat(),
                modified_at=now.isoformat(),
                toc_source=toc_source,
                client_specific_requests=client_specific_requests,
                sector_specific_comments=sector_specific_comments,
                regulations=regulations,
                detail_level=detail_level
            )

            draft_to_save = CompleteDraft(
                id=str(uuid.uuid4()),
                metadata=draft_metadata,
                toc=[]
            )

            # Generate TOC using AI (always use AI generation)
            if llm_service:
                # Initialize TOC generation service
                toc_generator = TOCGenerationService(llm_service)

                # Generate TOC based on policy information
                generated_toc = await toc_generator.generate_toc_from_description(
                    description=description if has_description else f"A policy for {function}",
                    title=title,
                    client_metadata=client_metadata,
                    function=function,
                    client_specific_requests=client_specific_requests,
                    sector_specific_comments=sector_specific_comments,
                    regulations=regulations,
                    detail_level=detail_level
                )

                # Transform to draft format
                draft_to_save.toc = self._transform_ai_generated_toc(generated_toc)
                logger.info(f"Generated TOC with {len(draft_to_save.toc)} topics using AI")
            else:
                # Fallback to basic structure if no LLM service available
                logger.warning("No LLM service available for TOC generation, using basic template")
                toc_source = "basic_template"
                draft_to_save.metadata.toc_source = toc_source
                draft_to_save.toc = self._get_basic_toc_template()

            success = self.storage.create_draft(draft_to_save)

            if success:
                logger.info(f"Initialized draft {draft_to_save.id} with {len(draft_to_save.toc)} suggested topics (source: {toc_source})")
                draft_dict = draft_to_save.model_dump()

                return {
                    "draft_id": draft_to_save.id,
                    "draft": draft_dict,
                    "similar_policies_found": 0,
                    "most_similar_policy_id": None,
                    "client_metadata": client_metadata.model_dump(),
                    "toc_source": toc_source
                }
            else:
                logger.error(f"Failed to save initialized draft {draft_to_save.id}")
                return None

        except Exception as e:
            logger.error(f"Failed to initialize draft: {e}")
            return None

    def _transform_toc_for_draft(self, toc_from_policy: List[Dict[str, Any]]) -> List:
        """
        Transform TOC from similar policy to draft format.
        Creates new IDs for draft while preserving source IDs from vector DB.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for topic_item in toc_from_policy:
            # Create subtopics with new IDs but preserve source IDs
            subtopics = []
            for sub in topic_item.get('subtopics', []):
                subtopic = TOCSubtopic(
                    subtopic_id=str(uuid.uuid4()),  # New ID for draft
                    topic=sub.get('topic', ''),
                    order=sub.get('order', 0),
                    source_subtopic_id=sub.get('subtopic_id'),  # Preserve vector DB ID
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            # Create topic with new ID but preserve source ID
            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),  # New ID for draft
                topic=topic_item.get('topic', ''),
                order=topic_item.get('order', 0),
                source_topic_id=topic_item.get('topic_id'),  # Preserve vector DB ID
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _transform_ai_generated_toc(self, ai_toc: List[Dict[str, Any]]) -> List:
        """
        Transform AI-generated TOC to draft format.
        AI-generated topics don't have source_topic_id or source_subtopic_id.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        transformed_toc = []
        for topic_item in ai_toc:
            # Create subtopics without source IDs
            subtopics = []
            for sub in topic_item.get('subtopics', []):
                subtopic = TOCSubtopic(
                    subtopic_id=sub.get('subtopic_id', str(uuid.uuid4())),
                    topic=sub.get('topic', ''),
                    order=sub.get('order', 0),
                    # No source_subtopic_id for AI-generated
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            # Create topic without source ID
            topic = TOCTopic(
                topic_id=topic_item.get('topic_id', str(uuid.uuid4())),
                topic=topic_item.get('topic', ''),
                order=topic_item.get('order', 0),
                # No source_topic_id for AI-generated
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _get_basic_toc_template(self) -> List:
        """
        Get a basic TOC template when no similar policies and no LLM available.
        """
        from app.models.draft_models import TOCTopic, TOCSubtopic

        basic_template = [
            {
                "topic": "Purpose and Scope",
                "subtopics": [
                    {"topic": "Policy Objectives"},
                    {"topic": "Applicability"}
                ]
            },
            {
                "topic": "Definitions and Terms",
                "subtopics": [
                    {"topic": "Key Terminology"},
                    {"topic": "Acronyms"}
                ]
            },
            {
                "topic": "Policy Requirements",
                "subtopics": [
                    {"topic": "Core Requirements"},
                    {"topic": "Implementation Guidelines"}
                ]
            },
            {
                "topic": "Compliance and Monitoring",
                "subtopics": [
                    {"topic": "Monitoring Procedures"},
                    {"topic": "Enforcement"}
                ]
            }
        ]

        transformed_toc = []
        for idx, template_item in enumerate(basic_template, 1):
            subtopics = []
            for sub_idx, sub in enumerate(template_item['subtopics'], 1):
                subtopic = TOCSubtopic(
                    subtopic_id=str(uuid.uuid4()),
                    topic=sub['topic'],
                    order=sub_idx,
                    content="",
                    summary="",
                    conversation_history=[]
                )
                subtopics.append(subtopic)

            topic = TOCTopic(
                topic_id=str(uuid.uuid4()),
                topic=template_item['topic'],
                order=idx,
                content="",
                summary="",
                conversation_history=[],
                subtopics=subtopics
            )
            transformed_toc.append(topic)

        return transformed_toc

    def _merge_tocs_from_policies(self, policies: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract and merge TOCs from similar policies.
        Prioritize topics that appear in multiple policies.
        """
        topic_frequency = {}
        topic_details = {}
        
        print(f"Number of policies: {len(policies)}")

        for i, policy in enumerate(policies):

            # if i != 0: break  # let's just keep only the first policy
        
            if not policy.toc:
                continue
                
            for topic in policy.toc:
                topic_key = topic.get('topic', '').lower().strip()
                if not topic_key:
                    continue
                
                if topic_key not in topic_frequency:
                    topic_frequency[topic_key] = 0
                    topic_details[topic_key] = {
                        "id": str(uuid.uuid4()),
                        "topic": topic.get('topic', ''),
                        "order": len(topic_details) + 1,
                        "source_topic_id": topic.get('topic_id', ''),  # Use topic_id from vector DB as source
                        "source_policy_id": policy.policy_id,  # PolicyMatch is a Pydantic model, use attribute access
                        "subtopics": []
                    }
                topic_frequency[topic_key] += 1
                
                # Merge subtopics if they exist
                if 'subtopics' in topic and topic['subtopics']:
                    for subtopic in topic['subtopics']:
                        subsection = {
                            "id": str(uuid.uuid4()),
                            "topic": subtopic.get('topic', ''),
                            "order": len(topic_details[topic_key]["subtopics"]) + 1,
                            "source_subtopic_id": subtopic.get('subtopic_id', '')  # Use subtopic_id from vector DB as source
                        }
                        # Avoid duplicate subtopics
                        existing_titles = [s['topic'].lower() for s in topic_details[topic_key]["subtopics"]]
                        if subsection['topic'].lower() not in existing_titles:
                            topic_details[topic_key]["subtopics"].append(subsection)
        
        # Sort by frequency and return as list
        sorted_topics = sorted(
            topic_details.items(),
            key=lambda x: topic_frequency[x[0]],
            reverse=True
        )
        
        # Reorder and return
        toc = []
        for idx, (_, topic) in enumerate(sorted_topics, 1):
            topic["order"] = idx
            toc.append(topic)
        
        return toc
    
    def _format_toc_for_model(self, raw_toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform raw TOC data to match Pydantic model structure.
        Handles different input formats and ensures required fields are present.
        """
        formatted_toc = []

        for idx, item in enumerate(raw_toc, 1):
            # Handle different possible input formats
            if isinstance(item, dict):
                # Extract title from 'topic' field or 'title' field
                title = item.get('topic', item.get('title', f'Topic {idx}'))

                formatted_topic = {
                    'topic_id': item.get('topic_id') or item.get('id') or str(uuid.uuid4()),  # Check both 'topic_id' and 'id'
                    'title': title.strip(),
                    'order': item.get('order', idx),
                    'source_topic_id': item.get('source_topic_id'),  # Preserve source_topic_id if it exists
                    'subtopics': []
                }

                # Handle subtopics if they exist
                subtopics = item.get('subtopics', [])
                if subtopics:
                    for sub_idx, subtopic in enumerate(subtopics, 1):
                        if isinstance(subtopic, dict):
                            sub_title = subtopic.get('title') or subtopic.get('topic', f'Subtopic {sub_idx}')

                            formatted_subtopic = {
                                'id': subtopic.get('subtopic_id') or subtopic.get('id') or str(uuid.uuid4()),  # Check both 'subtopic_id' and 'id'
                                'title': sub_title.strip(),
                                'order': subtopic.get('order', sub_idx),
                                'source_subtopic_id': subtopic.get('source_subtopic_id')  # Preserve source_subtopic_id if it exists
                            }
                            formatted_topic['subtopics'].append(formatted_subtopic)

                formatted_toc.append(formatted_topic)

        return formatted_toc
    
    def update_toc(self, draft_id: str, updated_toc: List[Dict[str, Any]]) -> bool:
        """
        Update the TOC structure of a draft from frontend.
        """
        try:
            logger.info(f"Starting TOC update for draft {draft_id}")

            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found for TOC update")
                return False

            logger.info(f"Successfully retrieved draft {draft_id}")

            # Transform TOC data to match Pydantic model structure
            formatted_toc = self._format_toc_for_model(updated_toc)

            # Convert formatted TOC to TOCTopic models
            from app.models.draft_models import TOCTopic, TOCSubtopic

            toc_topics = []
            for toc_item in formatted_toc:
                # Find existing topic to preserve content and conversation history
                existing_topic = None
                for existing in draft_model.toc:
                    if existing.topic_id == toc_item['topic_id']:
                        existing_topic = existing
                        break

                # Convert subtopics
                subtopics = []
                for sub_item in toc_item.get('subtopics', []):
                    # Find existing subtopic to preserve content and conversation history
                    existing_subtopic = None
                    if existing_topic:
                        for existing_sub in existing_topic.subtopics:
                            if existing_sub.subtopic_id == sub_item['id']:
                                existing_subtopic = existing_sub
                                break

                    subtopic = TOCSubtopic(
                        subtopic_id=sub_item['id'],
                        topic=sub_item['title'],
                        order=sub_item['order'],
                        source_subtopic_id=sub_item.get('source_subtopic_id'),
                        content=existing_subtopic.content if existing_subtopic else "",
                        summary=existing_subtopic.summary if existing_subtopic else "",
                        conversation_history=existing_subtopic.conversation_history if existing_subtopic else []
                    )
                    subtopics.append(subtopic)

                # Convert main topic
                topic = TOCTopic(
                    topic_id=toc_item['topic_id'],  # Use topic_id, not id
                    topic=toc_item['title'],
                    order=toc_item['order'],
                    source_topic_id=toc_item.get('source_topic_id'),
                    content=existing_topic.content if existing_topic else "",
                    summary=existing_topic.summary if existing_topic else "",
                    conversation_history=existing_topic.conversation_history if existing_topic else [],
                    subtopics=subtopics
                )
                toc_topics.append(topic)

            # Update TOC in the model
            draft_model.toc = toc_topics
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            logger.info(f"Updated TOC with {len(toc_topics)} topics")

            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully updated TOC for draft {draft_id} with {len(toc_topics)} topics")
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update TOC for draft {draft_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    
    def update_toc_chat_history(self, draft_id: str, conversation_history: List[Dict[str, Any]]) -> bool:
        """Update the TOC chat conversation history for a draft."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found for TOC chat history update")
                return False

            from app.models.draft_models import ConversationEntry
            entries = [ConversationEntry(**entry) for entry in conversation_history]
            draft_model.toc_chat_history = entries

            success = self.storage.update_draft(draft_id, draft_model)
            if success:
                logger.info(f"Updated TOC chat history for draft {draft_id} with {len(entries)} entries")
            return success
        except Exception as e:
            logger.error(f"Failed to update TOC chat history for draft {draft_id}: {e}")
            return False

    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete draft by ID."""
        draft_model = self.storage.get_draft(draft_id)
        if draft_model:
            return draft_model.model_dump()
        return None
    
    def list_drafts(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        created_by: Optional[str] = None,
        industry: Optional[str] = None,
        function: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all drafts with optional filtering.
        user_id parameter kept for compatibility but all drafts are visible to all users.
        """
        return self.storage.list_all_drafts(
            title=title,
            country=country,
            city=city,
            created_by=created_by,
            industry=industry,
            function=function
        )
    
    async def update_topic_content(
        self,
        draft_id: str,
        topic_id: str,
        content: str,
        subtopic_id: Optional[str] = None,
        llm_service = None,
        conversation_history: Optional[List] = None
    ) -> bool:
        """
        Update content for a topic or subtopic using typed models.
        """
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.warning(f"Draft {draft_id} not found")
                return False

            # Find and update in the TOC structure
            topic_found = False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    topic_found = True

                    if subtopic_id:
                        # Update subtopic content
                        subtopic_found = False
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.content = content
                                # Update conversation history if provided
                                if conversation_history is not None:
                                    toc_subtopic.conversation_history = conversation_history
                                    logger.info(f"Updated subtopic {subtopic_id} conversation history with {len(conversation_history)} entries")
                                # Generate LLM summary if content > 150 words and LLM service available
                                if llm_service:
                                    toc_subtopic.summary = await self.generation_service.generate_summary(
                                        content=content,
                                        title=toc_subtopic.topic,
                                        llm_service=llm_service
                                    )
                                else:
                                    toc_subtopic.summary = self._create_simple_summary_for_context(content)
                                subtopic_found = True
                                logger.info(f"Updated subtopic {subtopic_id} content")
                                break

                        if not subtopic_found:
                            logger.warning(f"Subtopic {subtopic_id} not found in topic {topic_id}")
                            return False
                    else:
                        # Update main topic content
                        toc_topic.content = content
                        # Update conversation history if provided
                        if conversation_history is not None:
                            toc_topic.conversation_history = conversation_history
                            logger.info(f"Updated topic {topic_id} conversation history with {len(conversation_history)} entries")
                        # Generate LLM summary if content > 150 words and LLM service available
                        if llm_service:
                            toc_topic.summary = await self.generation_service.generate_summary(
                                content=content,
                                title=toc_topic.topic,
                                llm_service=llm_service
                            )
                        else:
                            toc_topic.summary = self._create_simple_summary_for_context(content)
                        logger.info(f"Updated topic {topic_id} content")
                    break

            if not topic_found:
                logger.warning(f"Topic {topic_id} not found in draft {draft_id}")
                return False

            # Update modified timestamp
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            # Save the updated model
            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully saved content for topic {topic_id}" +
                           (f", subtopic {subtopic_id}" if subtopic_id else ""))
            else:
                logger.error(f"Storage update failed for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update topic content: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _create_simple_summary_for_context(self, content: str) -> str:
        """
        Create a simple summary for context building by truncating content > 150 words.
        This is used for context building when full content would be too long.
        """
        if not content:
            return ""

        word_count = len(content.split())
        if word_count > 150:
            # Take first 150 words as summary for context
            words = content.split()
            return " ".join(words[:150]) + "..."

        return ""  # No summary needed for short content

    def get_toc_item_by_id(self, draft: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
        """Get a TOC item (topic) by its ID."""
        # Convert to model if needed for compatibility
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        for toc_topic in draft_model.toc:
            if toc_topic.topic_id == item_id:
                return {
                    "id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "title": toc_topic.topic,  # For backward compatibility
                    "order": toc_topic.order,
                    "subtopics": [
                        {
                            "id": sub.subtopic_id,
                            "topic": sub.topic,
                            "title": sub.topic  # For backward compatibility
                        }
                        for sub in toc_topic.subtopics
                    ]
                }
        return None
    
    def build_sequential_context(
        self,
        draft: Dict[str, Any],
        target_topic_id: str,
        target_subtopic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context from all previous topics up to the target.
        Returns structured context for LLM consumption.
        Note: Still accepts dict for API compatibility but works with typed structure internally.
        """
        # Convert to model if needed
        if isinstance(draft, dict):
            draft_model = CompleteDraft(**draft)
        else:
            draft_model = draft

        context_parts = {
            "document_context": {
                "title": draft_model.metadata.title,
                "description": draft_model.metadata.description,
                "client_metadata": draft_model.metadata.client_metadata.model_dump() if draft_model.metadata.client_metadata else {}
            },
            "previous_topics": [],
            "current_topic_toc": None
        }

        # Find target position in TOC
        target_found = False

        for toc_topic in draft_model.toc:
            if target_found:
                break

            if toc_topic.topic_id == target_topic_id:
                context_parts["current_topic_toc"] = {
                    "topic_id": toc_topic.topic_id,
                    "topic": toc_topic.topic,
                    "order": toc_topic.order
                }

                if target_subtopic_id:
                    # Include main topic content if generating subtopic
                    if toc_topic.content:
                        context_parts["current_topic_content"] = {
                            "title": toc_topic.topic,
                            "content": toc_topic.content if not toc_topic.summary else toc_topic.summary
                        }

                    # Include previous subtopics of current topic
                    for toc_subtopic in toc_topic.subtopics:
                        if toc_subtopic.subtopic_id == target_subtopic_id:
                            break

                        if toc_subtopic.content:
                            context_parts["previous_topics"].append({
                                "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                                "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
                                "is_subtopic": True
                            })

                target_found = True
            else:
                # Add all content from topics before target
                if toc_topic.content:
                    context_parts["previous_topics"].append({
                        "title": toc_topic.topic,
                        "content": toc_topic.content if not toc_topic.summary else toc_topic.summary,
                        "is_subtopic": False
                    })

                # Add subtopics
                for toc_subtopic in toc_topic.subtopics:
                    if toc_subtopic.content:
                        context_parts["previous_topics"].append({
                            "title": f"{toc_topic.topic} - {toc_subtopic.topic}",
                            "content": toc_subtopic.content if not toc_subtopic.summary else toc_subtopic.summary,
                            "is_subtopic": True
                        })

        return context_parts
    
    def update_draft_metadata(self, draft_id: str, metadata_request) -> bool:
        """
        Update draft metadata (title, description, client info, etc.).
        """
        try:
            # Get the existing draft
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                logger.error(f"Draft {draft_id} not found")
                return False

            # Update metadata fields
            draft_model.metadata.title = metadata_request.title
            draft_model.metadata.description = metadata_request.description
            draft_model.metadata.client_metadata = metadata_request.client_metadata
            draft_model.metadata.function = metadata_request.function
            draft_model.metadata.client_specific_requests = metadata_request.client_specific_requests
            draft_model.metadata.sector_specific_comments = metadata_request.sector_specific_comments
            draft_model.metadata.regulations = metadata_request.regulations
            draft_model.metadata.detail_level = metadata_request.detail_level

            # Update modified timestamp
            from datetime import datetime, timezone
            draft_model.metadata.modified_at = datetime.now(timezone.utc).isoformat()

            # Save the updated draft
            success = self.storage.update_draft(draft_id, draft_model)

            if success:
                logger.info(f"Successfully updated metadata for draft {draft_id}")
            else:
                logger.error(f"Failed to update metadata for draft {draft_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating draft metadata: {e}")
            return False

    def delete_draft(self, draft_id: str) -> bool:
        """Delete a draft from storage."""
        try:
            success = self.storage.delete_draft(draft_id)
            if success:
                logger.info(f"Deleted draft {draft_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete draft {draft_id}: {e}")
            return False
    
    def delete_all_drafts(self) -> Dict[str, Any]:
        """Delete all drafts from storage."""
        try:
            # Get all drafts first
            all_drafts = self.storage.list_all_drafts()
            total_count = len(all_drafts)
            
            if total_count == 0:
                logger.info("No drafts found to delete")
                return {
                    "deleted_count": 0,
                    "total_count": 0,
                    "message": "No drafts found to delete"
                }
            
            # Delete each draft
            deleted_count = 0
            failed_deletions = []
            
            for draft in all_drafts:
                # The list_all_drafts returns 'draft_id' not 'id'
                draft_id = draft.get("draft_id")
                if draft_id:
                    try:
                        if self.storage.delete_draft(draft_id):
                            deleted_count += 1
                        else:
                            failed_deletions.append(draft_id)
                    except Exception as e:
                        logger.error(f"Failed to delete draft {draft_id}: {e}")
                        failed_deletions.append(draft_id)
            
            logger.info(f"Deleted {deleted_count}/{total_count} drafts")
            
            return {
                "deleted_count": deleted_count,
                "total_count": total_count,
                "failed_count": len(failed_deletions),
                "failed_draft_ids": failed_deletions,
                "message": f"Successfully deleted {deleted_count} out of {total_count} drafts"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete all drafts: {e}")
            return {
                "deleted_count": 0,
                "total_count": 0,
                "failed_count": 0,
                "failed_draft_ids": [],
                "message": f"Failed to delete all drafts: {str(e)}"
            }
    
    async def generate_content_for_topic(
        self,
        draft_id: str,
        topic_id: str,
        user_prompt: str,
        subtopic_id: Optional[str],
        vector_store,
        llm_service
    ) -> Dict[str, Any]:
        """
        Generate content for a topic or subtopic using the generation service.
        
        Returns:
            Dict with content and sources used
        """
        try:
            # Get the draft
            draft = self.storage.get_draft(draft_id)
            if not draft:
                logger.error(f"Draft {draft_id} not found")
                return None
            

            # Log conversation history BEFORE generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"BEFORE generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"BEFORE generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Generate content using the generation service
            content, source_ids, is_chat_response, critique = await self.generation_service.generate_topic_content(
                draft=draft,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                user_prompt=user_prompt,
                vector_store=vector_store,
                llm_service=llm_service
            )

            # Log conversation history AFTER generation
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"AFTER generation - subtopic {subtopic_id} has {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"AFTER generation - topic {topic_id} has {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Don't save content automatically - let the frontend handle saving
            # The conversation history is already updated by the generation service

            # Generate summary for response but don't save it
            word_count = len(content.split())
            summary = ""
            if word_count > 150 and not is_chat_response:
                # Get the topic title for summary generation
                toc_item = self.get_toc_item_by_id(draft.model_dump(), topic_id)

                if subtopic_id:
                    # Find subtopic title
                    for sub in toc_item.get("subtopics", []):
                        if sub["id"] == subtopic_id:
                            title = sub["title"]
                            break
                else:
                    title = toc_item["title"]

                summary = await self.generation_service.generate_summary(
                    content=content,
                    title=title,
                    llm_service=llm_service
                )
                # Don't save the summary - let the frontend handle it when saving content

            # Log conversation history state before saving (for debugging)
            for toc_topic in draft.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                logger.info(f"Saving subtopic {subtopic_id} with {len(toc_subtopic.conversation_history)} conversation entries")
                                break
                    else:
                        logger.info(f"Saving topic {topic_id} with {len(toc_topic.conversation_history)} conversation entries")
                    break

            # Save the modified draft (only conversation history is updated)
            success = self.storage.update_draft(draft_id, draft)

            if success:
                # Build critique dict if available
                critique_dict = None
                if critique:
                    critique_dict = {
                        "score": critique.score,
                        "alignment_score": critique.alignment_score,
                        "regulation_compliance_score": critique.regulation_compliance_score,
                        "strengths": critique.strengths,
                        "weaknesses": critique.weaknesses,
                        "improvements": critique.improvements,
                        "compliance_gaps": critique.compliance_gaps,
                        "recommendations": critique.recommendations,
                        "requires_revision": critique.requires_revision
                    }
                
                return {
                    "content": content,
                    "sources_used": source_ids,
                    "summary": summary,
                    "word_count": word_count,
                    "is_chat_response": is_chat_response,
                    "critique": critique_dict
                }
            else:
                logger.error(f"Failed to save generated content for topic {topic_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    def _update_summary(
        self,
        draft_id: str,
        topic_id: str,
        subtopic_id: Optional[str],
        summary: str
    ) -> bool:
        """Update the summary for a topic or subtopic."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return False

            for toc_topic in draft_model.toc:
                if toc_topic.topic_id == topic_id:
                    if subtopic_id:
                        for toc_subtopic in toc_topic.subtopics:
                            if toc_subtopic.subtopic_id == subtopic_id:
                                toc_subtopic.summary = summary
                                break
                    else:
                        toc_topic.summary = summary

                    return self.storage.update_draft(draft_id, draft_model)

            return False

        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
            return False
    
    def get_draft_progress(self, draft_id: str) -> Dict[str, Any]:
        """Calculate draft completion progress."""
        try:
            draft_model = self.storage.get_draft(draft_id)
            if not draft_model:
                return {"error": "Draft not found"}

            total_items = 0
            completed_items = 0

            for toc_topic in draft_model.toc:
                total_items += 1
                if toc_topic.content:
                    completed_items += 1

                for toc_subtopic in toc_topic.subtopics:
                    total_items += 1
                    if toc_subtopic.content:
                        completed_items += 1

            return {
                "total": total_items,
                "completed": completed_items,
                "percentage": (completed_items / total_items * 100) if total_items > 0 else 0,
                "remaining": total_items - completed_items
            }

        except Exception as e:
            logger.error(f"Failed to calculate progress for draft {draft_id}: {e}")
            return {"error": "Failed to calculate progress"}
