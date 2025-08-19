import logging
import time
import uuid
from typing import Dict, Any
from fastapi import HTTPException, status
from langchain.prompts import ChatPromptTemplate

from app.models.policy_models import PolicyGenerationRequest, PolicyGenerationResponse
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from app.prompts.manager import PromptManager
from app.config.settings import settings
from ..search.search_service import SearchService
from ..management.chunk_service import ChunkService  # Import existing service

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
        """Generate policy content with multiple content retrieval strategies."""
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        logger.info(
            f"Starting policy generation - Request ID: {request_id}, "
            f"Policy: {request.policy_id}, Topic: {request.topic_id}"
        )
        
        try:
            # Step 1: Get content using multiple strategies
            content_result = await self._get_content_with_fallback(
                request, store, request_id
            )
            
            # Step 2: Generate content using prompt manager
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
            
            # Return simplified response
            return PolicyGenerationResponse(response=generated_content)
            
        except Exception as e:
            logger.error(f"Policy generation failed - Request ID: {request_id}, Error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Policy generation failed. Please try again later."
            )

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
            
            logger.debug(f"LLM response received - Request ID: {request_id}, Length: {content_len}")
            return generated_content
            
        except Exception as e:
            logger.error(f"LLM generation failed - Request ID: {request_id}, Error: {str(e)}")
            raise Exception(f"Content generation failed: {str(e)}")


    async def _get_content_with_fallback(
        self, 
        request: PolicyGenerationRequest, 
        store: AzureVectorStore,
        request_id: str
    ) -> Dict[str, Any]:
        """Get content using multiple fallback strategies."""
        
        # Strategy 1: Try direct chunk lookup using existing ChunkService
        try:
            chunk_info = await ChunkService.get_chunk_by_id(request.topic_id, store)
            if chunk_info:                
                logger.info(f"Retrieved content via existing ChunkService - Request ID: {request_id}")
                return {
                    "content": chunk_info.chunk_content,
                    "chunk_id": chunk_info.chunk_id,
                    "source_method": "chunk_service_lookup"
                }
        except HTTPException as e:
            if e.status_code != 404:
                logger.warning(f"ChunkService lookup failed - Request ID: {request_id}, Error: {e}")
        except Exception as e:
            logger.warning(f"ChunkService lookup failed - Request ID: {request_id}, Error: {e}")
        
        # Strategy 2: Search by policy + topic (treating topic_id as topic name)
        try:
            results = await SearchService.search_chunks_by_policy_and_topic(
                request.policy_id, request.topic_id, store
            )
            
            if results:
                chunk = results[0]
                logger.info(f"Retrieved content via topic search - Request ID: {request_id}")
                return {
                    "content": chunk["content"],
                    "chunk_id": chunk["id"],
                    "source_method": "topic_search"
                }
            
        except Exception as e:
            logger.warning(f"Topic search failed - Request ID: {request_id}, Error: {e}")
        
        # Strategy 3: Semantic search within policy
        try:
            results = await SearchService.semantic_search_within_policy(
                request.policy_id, request.user_query, store, top_k=1
            )
            
            if results:
                chunk = results[0]
                logger.info(f"Retrieved content via semantic search - Request ID: {request_id}")
                return {
                    "content": chunk["content"],
                    "chunk_id": chunk["id"],
                    "source_method": "semantic_search"
                }
                
        except Exception as e:
            logger.warning(f"Semantic search failed - Request ID: {request_id}, Error: {e}")
        
        # If all strategies fail
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No content found for policy '{request.policy_id}' and topic '{request.topic_id}'"
        )
