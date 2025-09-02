import logging
import time
import uuid
import json
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

from app.models.policy_models import PolicyGenerationRequest, PolicyGenerationResponse
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Service for AI-powered policy content generation."""
    
    def __init__(self, max_generation_time_seconds: int = 30):
        self.max_generation_time_seconds = max_generation_time_seconds
        self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        self.generation_prompt = self.prompt_manager.get_prompt("generate_policy_content")
    
    async def generate_policy_content(
        self,
        request: PolicyGenerationRequest,
        store: AzureVectorStore,
        llm_service: AzureLLM,
        request_id: str = None
    ) -> PolicyGenerationResponse:
        """Generate policy content based on topic/subtopic IDs."""
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # Validate that topic_id is provided
        if not request.topic_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="topic_id is required"
            )
        
        logger.info(
            f"Starting policy generation - Request ID: {request_id}, "
            f"Topic: {request.topic_id}, Subtopic: {request.subtopic_id}"
        )
        
        try:
            # Get content by ID (no fallback strategies needed)
            content_result = await self._get_content_by_id(
                request.topic_id,
                request.subtopic_id,
                store,
                request_id
            )
            
            # Generate content using prompt manager
            generated_content = await self._generate_content_with_prompt_manager(
                request.user_query, 
                content_result["content"], 
                llm_service, 
                request_id
            )
            
            # Calculate generation time
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Policy generation completed - Request ID: {request_id}, "
                f"Time: {generation_time_ms}ms, Content length: {len(generated_content)}"
            )
            
            return PolicyGenerationResponse(response=generated_content)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Policy generation failed - Request ID: {request_id}, Error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Policy generation failed. Please try again later."
            )

    async def _get_content_by_id(
        self, 
        topic_id: str,
        subtopic_id: Optional[str],
        store: AzureVectorStore,
        request_id: str
    ) -> Dict[str, Any]:
        """Get content by topic ID and optional subtopic ID."""
        
        # Get topic document by ID
        topic_doc = store.get_topic_by_id(topic_id)
        
        if not topic_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic with ID '{topic_id}' not found"
            )
        
        logger.info(f"Retrieved topic '{topic_doc.get('topic')}' - Request ID: {request_id}")
        
        # Build content based on whether subtopic is requested
        content = self._build_content_from_topic(topic_doc, subtopic_id)
        
        return {
            "content": content,
            "topic_id": topic_id,
            "topic_name": topic_doc.get('topic'),
            "source_method": "direct_id_lookup"
        }

    def _build_content_from_topic(
        self, 
        topic_doc: Dict[str, Any], 
        subtopic_id: Optional[str] = None
    ) -> str:
        """
        Build content from topic document.
        If subtopic_id is provided, return topic content + that subtopic's content.
        Otherwise, return topic content + all subtopics content.
        """
        
        # Get subtopics (already parsed in get_topic_by_id)
        subtopics = topic_doc.get("subtopics", [])
        
        # Start with topic information
        topic_title = topic_doc.get('topic', 'Unknown Topic')
        topic_content = topic_doc.get('topic_content', '')
        
        content_parts = [f"# {topic_title}\n\n{topic_content}"]
        
        # If specific subtopic requested, add only that subtopic
        if subtopic_id:
            subtopic_found = False
            for subtopic in subtopics:
                if subtopic.get("id") == subtopic_id:
                    subtopic_name = subtopic.get('name', 'Unknown Subtopic')
                    subtopic_content = subtopic.get('content', '')
                    content_parts.append(f"\n## {subtopic_name}\n\n{subtopic_content}")
                    subtopic_found = True
                    logger.info(f"Found requested subtopic: {subtopic_name}")
                    break
            
            if not subtopic_found:
                logger.warning(f"Subtopic {subtopic_id} not found in topic, returning topic content only")
        else:
            # Add all subtopics content
            for subtopic in subtopics:
                subtopic_name = subtopic.get('name', 'Unknown Subtopic')
                subtopic_content = subtopic.get('content', '')
                if subtopic_content:
                    content_parts.append(f"\n## {subtopic_name}\n\n{subtopic_content}")
        
        return "\n\n".join(content_parts)

    async def _generate_content_with_prompt_manager(
        self,
        user_query: str,
        topic_content: str,
        llm_service: AzureLLM,
        request_id: str
    ) -> str:
        """Generate content using the prompt manager system."""
        
        try:
            logger.debug(f"Sending request to LLM - Request ID: {request_id}")
            
            # Use the prompt manager to format the message
            messages = self.generation_prompt.format_messages(
                topic_content=topic_content,
                user_query=user_query
            )
            
            llm_response = await llm_service.client.ainvoke(messages)
            
            if not llm_response or not llm_response.content:
                raise Exception("LLM returned empty response")
            
            generated_content = llm_response.content.strip()
            content_len = len(generated_content.split())
            
            if content_len < 50:
                logger.warning(f"Generated content seems too short - Request ID: {request_id}")
            
            logger.debug(f"LLM response received - Request ID: {request_id}, Length: {content_len} words")
            return generated_content
            
        except Exception as e:
            logger.error(f"LLM generation failed - Request ID: {request_id}, Error: {str(e)}")
            raise Exception(f"Content generation failed: {str(e)}")