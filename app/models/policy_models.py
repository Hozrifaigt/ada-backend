import re

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional



class SimilarPolicyRequest(BaseModel):
    """Request for finding similar policies - simplified as requested."""
    description: str = Field(..., min_length=1, max_length=2000)
    threshold: float = Field(default=0.6, ge=0, le=1)

class PolicyMatch(BaseModel):
    """Single policy match from similarity search."""
    policy_id: str
    filename: str
    description: str
    similarity_score: float
    meets_threshold: bool
    toc: List[Dict[str, Any]]
    raw_azure_score: float
    rank_position: int
    threshold_used: float
    source: str

class ChunkInfo(BaseModel):
    """Chunk information"""
    chunk_id: str
    chunk_content: str

class PoliciesListResponse(BaseModel):
    """Response for getting all policies."""
    count: int
    policies: List[Dict[str, Any]]

class PolicyGenerationRequest(BaseModel):
    """Request for generating policy content - simplified as requested."""
    user_query: str = Field(..., min_length=1, max_length=5000)
    policy_id: str = Field(..., min_length=1, max_length=200)
    topic_id: str = Field(..., min_length=1, max_length=500)
    subtopic_id: Optional[str] = Field(None, max_length=500)
    
    @validator('policy_id')
    def validate_policy_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Policy ID contains invalid characters')
        return v

class PolicyGenerationResponse(BaseModel):
    """Response from policy generation - simplified as requested."""
    response: str
