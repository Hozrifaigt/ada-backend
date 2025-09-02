import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from app.models.policy_models import (
    PoliciesListResponse, SimilarPolicyRequest, PolicyMatch,
    PolicyGenerationRequest, PolicyGenerationResponse, TopicInfo
)
from .dependencies import get_vector_store, get_llm
from .search import SimilarityService
from .management import PolicyService, TopicService
from .generation import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter()
generation_service = GenerationService()

@router.get("/policies", response_model=PoliciesListResponse)
async def get_all_policies(store=Depends(get_vector_store)):
    """Retrieve a list of all indexed policy documents with their metadata."""
    return await PolicyService.get_all_policies(store)

@router.get("/topics/{topic_id}", response_model=TopicInfo)
async def get_topic_by_id(
    topic_id: str,
    subtopic_id: Optional[str] = Query(None),
    store=Depends(get_vector_store)
):
    """
    Get a specific topic by its ID.
    Optionally specify subtopic_id to get only that subtopic's content.
    """
    return await TopicService.get_topic_by_id(topic_id, subtopic_id, store)

@router.post("/policies/similar", response_model=List[PolicyMatch])
async def find_similar_policies_with_scores(
    request: SimilarPolicyRequest,
    store=Depends(get_vector_store)
):
    """
    Find policies similar to the user's description.
    This is the only vector-based search in the system.
    """
    return await SimilarityService.find_similar_policies(request, store)

@router.post("/policies/generate", response_model=PolicyGenerationResponse)
async def generate_policy_content(
    request: PolicyGenerationRequest,
    store=Depends(get_vector_store),
    llm_service=Depends(get_llm)
):
    """
    Generate policy content based on user query and topic/subtopic IDs.
    
    Requires:
    - user_query: The user's request for the LLM
    - topic_id: UUID of the topic to retrieve
    - subtopic_id (optional): UUID of specific subtopic
    """
    return await generation_service.generate_content(request, store, llm_service)