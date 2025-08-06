from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PolicyDetail(BaseModel):
    filename: str
    description: str

class ChunkDetail(BaseModel):
    topic: str
    content: str

class SearchDescriptionQuery(BaseModel):
    description: str
    threshold: float = 0.7


class RetrievalQuery(BaseModel):
    query: str 
    top_k: int
    company_context: Optional[Dict[str, Any]]

class ChunkResult(BaseModel):
    id: str
    content: str
    topic: str
    level: int
    parent_topic: Optional[str]
    is_subtopic: bool
    source_policy: str 
    relevance_score: float
    
class TopicResult(BaseModel):
    topic: str
    level: int
    parent_topic: Optional[str]
    related_chunks: List[ChunkResult]
    source_policies: List[str]
    
class PolicyMatchResult(BaseModel):
    policy_id: str
    filename: str
    description: str
    similarity_score: float
    toc: List[Dict[str, Any]]
    source: str

class RetrievalResponse(BaseModel):
    query: str
    results_count: int
    processing_time: float
    chunks: List[ChunkResult] 
    topics: List[TopicResult] 
    similar_policies: List[PolicyMatchResult]