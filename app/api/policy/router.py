import logging
from typing import List
from fastapi import APIRouter, Depends

from app.models.policy_models import (
    PoliciesListResponse, SimilarPolicyRequest, PolicyMatch, ChunkInfo,
    PolicyGenerationRequest, PolicyGenerationResponse
)
from .dependencies import get_vector_store, get_llm
from .search import SimilarityService
from .management import PolicyService, ChunkService
from .generation import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter()
generation_service = GenerationService()

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "policy-api"}

@router.get("/policies", response_model=PoliciesListResponse)
async def get_all_policies(store=Depends(get_vector_store)):
    """Retrieve a list of all indexed policy documents with their metadata."""
    return await PolicyService.get_all_policies(store)

@router.get("/chunks/{chunk_id}", response_model=ChunkInfo)
async def get_chunk_by_id(chunk_id: str, store=Depends(get_vector_store)):
    """Get a specific chunk by its unique chunk ID."""
    return await ChunkService.get_chunk_by_id(chunk_id, store)

@router.post("/policies/similar_policy", response_model=List[PolicyMatch])
async def find_similar_policies_with_scores(
    request: SimilarPolicyRequest,
    store=Depends(get_vector_store)
):
    """Find policies similar to the user's description with similarity scores."""
    return await SimilarityService.find_similar_policies(request, store)


@router.post("/policies/generate", response_model=PolicyGenerationResponse)
async def generate_policy_content(
    request: PolicyGenerationRequest,
    store=Depends(get_vector_store),
    llm_service=Depends(get_llm)
):
    """
    Generate policy content based on user query and existing topic content.
    
    Supports multiple content retrieval strategies:
    - Topic-based search  
    - Semantic search within policy
    """
    return await generation_service.generate_content(request, store, llm_service)
