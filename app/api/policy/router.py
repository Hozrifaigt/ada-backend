import logging


from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query

from app.auth.entra_service import get_current_user
from app.models.policy_models import (
    PoliciesListResponse, SimilarPolicyRequest, PolicyMatch,
    PolicyGenerationRequest, PolicyGenerationResponse, TopicInfo
)

from .search import SimilarityService
from .generation import GenerationService
from .dependencies import get_vector_store, get_llm
from .management import PolicyService, TopicService


logger = logging.getLogger(__name__)
router = APIRouter()
generation_service = GenerationService()


@router.get("/policies", response_model=PoliciesListResponse)
async def get_all_policies(
    store=Depends(get_vector_store), 
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieve a list of all indexed policy documents with their metadata."""
    return await PolicyService.get_all_policies(store)

@router.get("/topics/{topic_id}", response_model=TopicInfo)
async def get_topic_by_id(
    topic_id: str,
    subtopic_id: Optional[str] = Query(None),
    store=Depends(get_vector_store),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get a specific topic by its ID.
    Optionally specify subtopic_id to get only that subtopic's content.
    """
    logger.info(f"Topic {topic_id} requested by user: {current_user.get('email')}")
    return await TopicService.get_topic_by_id(topic_id, subtopic_id, store)

@router.post("/policies/similar", response_model=List[PolicyMatch])
async def find_similar_policies_with_scores(
    request: SimilarPolicyRequest,
    store=Depends(get_vector_store),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Find policies similar to the user's description.
    This is the only vector-based search in the system.
    """
    logger.info(f"Similar policy search by user: {current_user.get('email')}")
    return await SimilarityService.find_similar_policies(request, store)

@router.post("/policies/generate", response_model=PolicyGenerationResponse)
async def generate_policy_content(
    request: PolicyGenerationRequest,
    store=Depends(get_vector_store),
    llm_service=Depends(get_llm),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate policy content based on user query and topic/subtopic IDs.
    
    Requires:
    - user_query: The user's request for the LLM
    - topic_id: UUID of the topic to retrieve
    - subtopic_id (optional): UUID of specific subtopic
    - current_user: The user requesting the policy generation
    """
    logger.info(f"Policy generation by user: {current_user.get('email')}")
    return await generation_service.generate_content(request, store, llm_service)