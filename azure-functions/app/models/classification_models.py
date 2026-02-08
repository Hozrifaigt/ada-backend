"""
Query Classification Models
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryCategory(str, Enum):
    """Classification categories for user queries."""
    LEGITIMATE_REQUEST = "legitimate_request"
    SMALL_TALK = "small_talk"
    INAPPROPRIATE = "inappropriate"


class QueryClassification(BaseModel):
    """Result of query classification."""
    category: QueryCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reason: str = Field(..., description="Explanation for the classification")
    rule_name: Optional[str] = Field(None, description="Name of the rule that triggered this classification")
    suggested_response: Optional[str] = Field(None, description="Suggested response for non-legitimate queries")
    should_proceed: bool = Field(..., description="Whether to proceed with content generation")


class ClassificationResult(BaseModel):
    """Extended classification result with processing details."""
    classification: QueryClassification
    processing_time_ms: int
    method_used: str = Field(..., description="'rule_based' or 'llm_based'")
    rule_matched: Optional[str] = Field(None, description="Name of rule that matched (if rule-based)")