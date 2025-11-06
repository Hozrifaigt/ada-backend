from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SimilarPolicyRequest(BaseModel):
    """Request for finding similar policies - simplified as requested."""
    description: Optional[str] = Field(None, max_length=2000)
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

class TopicInfo(BaseModel):
    """Information about a topic and its subtopics."""
    topic_id: str
    topic_name: str
    topic_content: str
    subtopics: List[Dict[str, Any]]  # List of subtopic objects with id, name, content

class PoliciesListResponse(BaseModel):
    """Response for getting all policies."""
    count: int
    policies: List[Dict[str, Any]]

class PolicyGenerationRequest(BaseModel):
    """Request for generating policy content - simplified."""
    user_query: str = Field(..., min_length=1, max_length=5000)
    topic_id: str = Field(..., min_length=1, max_length=200)
    subtopic_id: Optional[str] = Field(None, max_length=200)

class PolicyGenerationResponse(BaseModel):
    """Response from policy generation."""
    response: str