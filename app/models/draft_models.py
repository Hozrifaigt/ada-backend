import uuid

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ClientMetadata(BaseModel):
    name: str
    country: str 
    city: str 


class InitializeDraftRequest(BaseModel):
    """Request to initialize a new draft from description."""
    title: str = Field(..., min_length=1, max_length=200, description="Draft title")
    description: str = Field(..., min_length=1, max_length=2000, description="Policy description for RAG search")
    client_metadata: ClientMetadata = Field(..., description="Client metadata")


class UpdateTOCRequest(BaseModel):
    """Request to update/finalize TOC structure."""
    toc: List[Dict[str, Any]] = Field(..., min_items=1, description="Table of contents structure")


class GenerateContentRequest(BaseModel):
    """Request to generate content for a topic/subtopic."""
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="User requirements for content generation")
    subtopic_id: Optional[str] = Field(None, description="Subtopic ID if generating subtopic content")


class UpdateContentRequest(BaseModel):
    """Request to manually update topic/subtopic content."""
    content: str = Field(..., description="Content to update")
    subtopic_id: Optional[str] = Field(None, description="Subtopic ID if updating subtopic")


class RegenerateRequest(BaseModel):
    """Request to regenerate content with new prompt."""
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="New prompt for regeneration")
    subtopic_id: Optional[str] = Field(None, description="Subtopic ID if regenerating subtopic")


class FinalizeTopicRequest(BaseModel):
    """Request to finalize a topic."""
    generate_summary: bool = Field(default=True, description="Generate AI summary if content > 150 words")


class ExportRequest(BaseModel):
    """Request to export draft."""
    delete_after_export: bool = Field(default=True, description="Delete draft after successful export")


# ============= Response Models =============

class InitializeDraftResponse(BaseModel):
    """Response after initializing a draft."""
    draft_id: str
    similar_policies_found: int
    suggested_toc: List[Dict[str, Any]]
    message: str


class DraftSummary(BaseModel):
    """Summary information for a draft (used in list)."""
    draft_id: str
    title: str
    description: str
    created_by: str
    created_at: str
    modified_at: str
    most_similar_policy_id: Optional[str] = None
    client_metadata: Optional[Dict[str, str]] = None


class DraftListResponse(BaseModel):
    """Response containing list of drafts."""
    drafts: List[DraftSummary]
    total: int


class DraftProgress(BaseModel):
    """Draft completion progress."""
    total: int = Field(..., description="Total number of topics and subtopics")
    completed: int = Field(..., description="Number of completed items")
    percentage: float = Field(..., description="Completion percentage")
    remaining: int = Field(..., description="Number of remaining items")


class TopicContext(BaseModel):
    """Context information for a topic."""
    document_context: Dict[str, str] = Field(..., description="Document metadata")
    previous_topics: List[Dict[str, Any]] = Field(..., description="Previous topics content/summaries")
    current_topic_toc: Optional[Dict[str, Any]] = Field(None, description="Current topic TOC item")


class ContentGenerationResponse(BaseModel):
    """Response after content generation."""
    content: str = Field(..., description="Generated content")
    context_used: str = Field(..., description="Preview of context used for generation")
    message: str


# ============= Data Models (for internal use) =============

class TOCTopic(BaseModel):
    """Topic in table of contents."""
    id: str
    title: str
    order: int
    source_topic_id: Optional[str] = None
    subtopics: List['TOCSubtopic'] = []


class TOCSubtopic(BaseModel):
    """Subtopic in table of contents."""
    id: str
    title: str
    order: int
    source_subtopic_id: Optional[str] = None


class Topic(BaseModel):
    """Topic with content."""
    id: str
    content: str = ""
    summary: str = ""
    subtopics: List['Subtopic'] = []


class Subtopic(BaseModel):
    """Subtopic with content."""
    id: str
    content: str = ""
    summary: str = ""


class DraftMetadata(BaseModel):
    """Draft metadata."""
    title: str
    description: str
    created_by: str
    created_at: str
    modified_at: str
    client_metadata: ClientMetadata
    most_similar_policy_id: Optional[str] = None


class CompleteDraft(BaseModel):
    """Complete draft structure."""
    id: str = Field(..., default_factory=lambda: str(uuid.uuid4()))
    metadata: DraftMetadata
    toc: List[TOCTopic]

# Update forward references
TOCTopic.model_rebuild()
Topic.model_rebuild()