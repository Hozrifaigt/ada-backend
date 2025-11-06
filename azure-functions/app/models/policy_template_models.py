"""Models for policy templates imported from Excel."""

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from app.models.draft_models import TOCTopic, TOCSubtopic


class PolicyTemplate(BaseModel):
    """Policy template with pre-filled content from Excel."""
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    function: str = Field(..., description="Function category (e.g., 'HR', 'IT', 'Legal')")
    policy_type: str = Field(..., description="Policy type (e.g., 'Remote Work Policy')")
    toc: List[TOCTopic] = Field(default_factory=list, description="Table of contents with pre-filled content")
    is_template: bool = Field(default=True, description="Indicates this is a template")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    modified_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = Field(default="excel_import", description="Source of the template")


class PolicyTemplateListResponse(BaseModel):
    """Response containing list of policy templates."""
    templates: List[PolicyTemplate]
    total: int


class PolicyTemplateSummary(BaseModel):
    """Summary information for a policy template."""
    template_id: str
    function: str
    policy_type: str
    topic_count: int
    subtopic_count: int
    created_at: str


class UploadPolicyTemplatesRequest(BaseModel):
    """Request to upload policy templates from Excel."""
    # Will be handled as file upload in FastAPI
    pass


class UploadPolicyTemplatesResponse(BaseModel):
    """Response after uploading policy templates."""
    success: bool
    templates_imported: int
    functions: List[str]
    policy_types: List[str]
    message: str
    errors: List[str] = Field(default_factory=list)


class GetPolicyTypesResponse(BaseModel):
    """Response for getting available policy types."""
    policy_types: List[str]
    total: int


class CreateDraftFromTemplateRequest(BaseModel):
    """Request to create a draft from a template."""
    template_id: str = Field(..., description="Template ID to use")
    title: str = Field(..., min_length=1, max_length=200, description="Draft title")
    client_metadata: dict = Field(..., description="Client metadata")
    client_specific_requests: Optional[str] = Field(None, description="Client-specific requirements")
    regulations: str = Field(default="", description="Applicable regulation")
    detail_level: int = Field(default=3, ge=1, le=5, description="Level of detail")
