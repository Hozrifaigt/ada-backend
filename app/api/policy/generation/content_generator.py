import logging
import time
import uuid
import json
import re
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

from app.models.policy_models import PolicyGenerationRequest, PolicyGenerationResponse
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Service for AI-powered policy content generation with error handling."""
    
    def __init__(
        self, 
        max_generation_time_seconds: int = 30,
        max_retries: int = 3
    ):
        self.max_generation_time_seconds = max_generation_time_seconds
        self.max_retries = max_retries
        self.prompt_manager = PromptManager(settings.PROMPT_MANAGER_PATH)
        self.generation_prompt = self.prompt_manager.get_prompt("generate_policy_content")
        self.empty_content_prompt = self.prompt_manager.get_prompt("generate_from_empty")
    
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
        
        # Pydantic already validates the request
        
        logger.info(
            f"Starting policy generation - Request ID: {request_id}, "
            f"Topic: {request.topic_id}, Subtopic: {request.subtopic_id}"
        )
        
        try:
            # Get content (or context for empty topics)
            content_result = await self._get_content(
                request.topic_id,
                request.subtopic_id,
                request.user_query,
                store,
                request_id
            )
            
            # Generate content with retry logic
            generated_content = await self._generate_with_retry(
                request.user_query,
                content_result["content"],
                llm_service,
                request_id,
                content_result.get("generation_mode", "enhance_existing")
            )
            
            # Calculate metrics for logging
            generation_time_ms = int((time.time() - start_time) * 1000)
            word_count = len(generated_content.split())
            
            logger.info(
                f"Policy generation completed - Request ID: {request_id}, "
                f"Time: {generation_time_ms}ms, Words: {word_count}"
            )
            
            return PolicyGenerationResponse(response=generated_content)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Policy generation failed - Request ID: {request_id}, Error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Policy generation failed. Please try again."
            )
    
    def _validate_request(self, request: PolicyGenerationRequest):
        """Validate the generation request."""
        # Validate topic_id
        if not request.topic_id:
            raise ValueError("topic_id is required")
        # Validate UUID format
        try:
            uuid.UUID(request.topic_id)
        except ValueError:
            raise ValueError(f"Invalid topic_id format: {request.topic_id}")
        if request.subtopic_id:
            try:
                uuid.UUID(request.subtopic_id)
            except ValueError:
                raise ValueError(f"Invalid subtopic_id format: {request.subtopic_id}")
        
        # Validate user query
        if not request.user_query or not request.user_query.strip():
            raise ValueError("user_query cannot be empty")
        
        if len(request.user_query) > 5000:
            raise ValueError("user_query exceeds maximum length of 5000 characters")
    
    async def _get_content(
        self,
        topic_id: str,
        subtopic_id: Optional[str],
        user_query: str,
        store: AzureVectorStore,
        request_id: str
    ) -> Dict[str, Any]:
        """Get content from store, or prepare context for empty topics."""
        
        # Get topic document
        topic_doc = store.get_topic_by_id(topic_id)
        
        if not topic_doc:
            logger.error(f"Topic not found - Topic ID: {topic_id}, Request ID: {request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic with ID '{topic_id}' not found"
            )
        
        # Build content
        content = self._build_content(topic_doc, subtopic_id)
        topic_name = topic_doc.get('topic', 'Unknown Topic')
        
        # Check if content is empty/minimal
        word_count = len(content.split())
        is_empty = word_count < 10 or "[No content available]" in content
        
        if is_empty:
            logger.info(f"Empty content detected - generating from scratch for topic: {topic_name}")
            
            context_messages = self.empty_content_prompt.format_messages(
                topic_name=topic_name,
                user_query=user_query
            )
            # Extract the content from the messages for use as context
            context = context_messages[0].content if context_messages else f"Topic: {topic_name}\nUser Request: {user_query}"
            
            return {
                "content": context,
                "generation_mode": "create_from_scratch"
            }
        
        # Handle very long content
        if word_count > 50000:
            logger.warning(f"Truncating very long content - Words: {word_count}")
            words = content.split()[:50000]
            content = " ".join(words)
        
        return {
            "content": content,
            "generation_mode": "enhance_existing"
        }
    
    def _build_content(
        self,
        topic_doc: Dict[str, Any],
        subtopic_id: Optional[str]
    ) -> str:
        """Build content from topic document."""
        
        subtopics = topic_doc.get("subtopics", [])
        topic_title = topic_doc.get('topic', 'Unknown Topic')
        topic_content = topic_doc.get('topic_content', '')
        
        # Clean content
        topic_content = self._clean_content(topic_content)
        content_parts = [f"# {topic_title}\n\n{topic_content}"]
        
        if subtopic_id:
            # Find specific subtopic
            for subtopic in subtopics:
                if subtopic.get("id") == subtopic_id:
                    subtopic_name = subtopic.get('name', 'Unknown Subtopic')
                    subtopic_content = self._clean_content(subtopic.get('content', ''))
                    content_parts.append(f"\n## {subtopic_name}\n\n{subtopic_content}")
                    break
        else:
            # Add all subtopics
            for subtopic in subtopics:
                subtopic_name = subtopic.get('name', 'Unknown Subtopic')
                subtopic_content = self._clean_content(subtopic.get('content', ''))
                if subtopic_content:
                    content_parts.append(f"\n## {subtopic_name}\n\n{subtopic_content}")
        
        return "\n\n".join(content_parts)
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing artifacts."""
        if not content:
            return ""
        
        content = re.sub(r'\[No content available\]', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    async def _generate_with_retry(
        self,
        user_query: str,
        topic_content: str,
        llm_service: AzureLLM,
        request_id: str,
        generation_mode: str = "enhance_existing"
    ) -> str:
        """Generate content with retry logic."""
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generation attempt {attempt + 1}/{self.max_retries} - Mode: {generation_mode}")
                
                # Add timeout
                import asyncio
                generation_task = self._call_llm(
                    user_query, topic_content, llm_service, request_id, generation_mode
                )
                
                generated_content = await asyncio.wait_for(
                    generation_task,
                    timeout=self.max_generation_time_seconds
                )
                
                # Basic validation - just check it's not empty or an error
                if generated_content and len(generated_content.strip()) > 10:
                    if not self._is_error_response(generated_content):
                        return generated_content
                
                logger.warning(f"Invalid response - Attempt {attempt + 1}")
                last_error = "Invalid response format"
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout - Attempt {attempt + 1}")
                last_error = "Generation timeout"
            except Exception as e:
                logger.error(f"Error - Attempt {attempt + 1}: {str(e)}")
                last_error = str(e)
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")
    
    async def _call_llm(
        self,
        user_query: str,
        topic_content: str,
        llm_service: AzureLLM,
        request_id: str,
        generation_mode: str
    ) -> str:
        """Call LLM service."""
        
        try:
            messages = self.generation_prompt.format_messages(
                topic_content=topic_content,
                user_query=user_query
            )
            
            llm_response = await llm_service.client.ainvoke(messages)
            
            if not llm_response or not llm_response.content:
                raise Exception("Empty LLM response")
            
            return llm_response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM call failed - {str(e)}")
            raise
    
    def _is_error_response(self, content: str) -> bool:
        """Check if response is an error message."""
        error_patterns = [
            "error:", "exception:", "failed",
            "as an ai", "i cannot", "undefined", "null"
        ]
        
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in error_patterns)