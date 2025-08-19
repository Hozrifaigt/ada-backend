import logging
from typing import List
from fastapi import HTTPException, status
from langchain.prompts import ChatPromptTemplate

from app.models.policy_models import PolicyDetail, ChunkDetail
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from .models import PolicyGenerationRequest, PolicyGenerationResponse

logger = logging.getLogger(__name__)

class PolicyService:
    """Service class for policy-related operations."""
    
    @staticmethod
    async def get_all_policies(store: AzureVectorStore) -> List[PolicyDetail]:
        """Retrieve all indexed policy documents."""
        logger.info("Fetching a list of all indexed policies.")
        try:
            results = await store.client.asimilarity_search(
                query="*",
                k=100,
                filters="record_type eq 'policy'"
            )
            return [
                PolicyDetail(
                    filename=doc.metadata.get("filename", "N/A"),
                    description=doc.metadata.get("description", "No description available."),
                ) for doc in results
            ]
        except Exception as e:
            logger.error(f"Listing policies failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not retrieve policies.")

    @staticmethod
    async def get_chunks_by_policy(policy_id: str, store: AzureVectorStore) -> List[ChunkDetail]:
        """Retrieve all chunks for a specific policy."""
        logger.info(f"Fetching all chunks for policy: '{policy_id}'")
        try:
            filter_string = f"record_type eq 'chunk' and source eq '{policy_id}'"
            results = await store.client.asimilarity_search(
                query="*",
                k=300,
                filters=filter_string
            )
            if not results:
                return []
                
            return [
                ChunkDetail(
                    topic=doc.metadata.get("topic", "N/A"),
                    content=doc.page_content,
                ) for doc in results
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for '{policy_id}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Chunk retrieval failed.")

    @staticmethod
    async def get_chunk_by_topic(policy_id: str, topic: str, store: AzureVectorStore) -> ChunkDetail:
        """Retrieve specific chunk by policy and topic."""
        logger.info(f"Fetching chunk for policy '{policy_id}' and topic '{topic}'.")
        try:
            filter_string = f"source eq '{policy_id}' and topic eq '{topic}'"
            results = await store.client.asimilarity_search(
                query="*",
                k=1,
                filters=filter_string
            )
            
            if not results:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="The specified policy file or topic could not be found."
                )
            
            doc = results[0]
            return ChunkDetail(
                topic=doc.metadata.get("topic", "N/A"),
                content=doc.page_content,
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Specific chunk retrieval failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Chunk retrieval failed.")

class PolicyGenerationService:
    """Service class for AI-powered policy generation."""
    
    @staticmethod
    async def generate_policy_content(
        request: PolicyGenerationRequest,
        store: AzureVectorStore,
        llm_service: AzureLLM
    ) -> PolicyGenerationResponse:
        """Generate policy content based on user query and existing topic content."""
        logger.info(f"Generating policy content for policy '{request.policy_id}', topic '{request.topic_id}'")
        
        try:
            # Step 1: Retrieve the topic content
            target_topic = request.subtopic_id if request.subtopic_id else request.topic_id
            topic_content = await PolicyGenerationService._get_topic_content(
                request.policy_id, target_topic, store
            )
            
            # Step 2: Generate new content using LLM
            generated_content = await PolicyGenerationService._generate_content_with_llm(
                request.user_query, topic_content, llm_service
            )
            
            # Step 3: Return response
            return PolicyGenerationResponse(
                user_query=request.user_query,
                policy_id=request.policy_id,
                topic_id=request.topic_id,
                subtopic_id=request.subtopic_id,
                topic_content=topic_content,
                generated_response=generated_content
            )
            
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Policy generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Policy generation failed: {str(e)}"
            )
    
    @staticmethod
    async def _get_topic_content(policy_id: str, topic: str, store: AzureVectorStore) -> str:
        """Retrieve content for a specific topic from a policy."""
        filter_string = f"record_type eq 'chunk' and source eq '{policy_id}' and topic eq '{topic}'"
        
        logger.info(f"Searching for content with filter: {filter_string}")
        
        results = await store.client.asimilarity_search(
            query="*",
            k=1,
            filters=filter_string
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No content found for policy '{policy_id}' and topic '{topic}'"
            )
        
        topic_content = results[0].page_content
        logger.info(f"Retrieved topic content: {len(topic_content)} characters")
        return topic_content
    
    @staticmethod
    async def _generate_content_with_llm(user_query: str, topic_content: str, llm_service: AzureLLM) -> str:
        """Generate new content using LLM based on user query and existing content."""
        logger.info("Generating LLM response")
        
        llm_prompt = f"""
You are an AI assistant helping to generate policy content based on existing policy templates and user requirements.

EXISTING POLICY CONTENT FOR REFERENCE:
{topic_content}

USER REQUEST:
{user_query}

Please generate improved or customized policy content that addresses the user's request while building upon the existing policy content provided above. Ensure the response is:
1. Professional and appropriate for policy documentation
2. Consistent with the structure and tone of the existing content
3. Responsive to the specific user request
4. Clear and actionable

GENERATED POLICY CONTENT:
"""
        
        try:
            prompt_template = ChatPromptTemplate.from_template(llm_prompt)
            messages = prompt_template.format_messages()
            
            llm_response = await llm_service.client.ainvoke(messages)
            generated_content = llm_response.content.strip() if llm_response.content else "No content generated"
            
            logger.info(f"Successfully generated {len(generated_content)} characters of content")
            return generated_content
            
        except Exception as llm_error:
            logger.error(f"LLM generation failed: {llm_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate content: {str(llm_error)}"
            )