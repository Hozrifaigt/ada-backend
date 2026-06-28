from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class TOCChatMessage(BaseModel):
    """Request model for TOC chat interaction."""
    message: str = Field(..., description="Natural language command to modify TOC")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Previous conversation exchanges for context"
    )
    current_toc: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="The TOC the chatbot should operate on (the dropdown-selected one). Falls back to the working TOC when omitted."
    )
    client_toc: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reference: the client's Current TOC, passed when editing the Good TOC so the bot can merge both."
    )
    benchmark_toc: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reference: the Extracted/benchmark TOC, passed when editing the Good TOC so the bot can merge both."
    )


class TOCOperation(BaseModel):
    """Parsed TOC operation details."""
    action: str = Field(..., description="Type of operation: add_topic, remove_topic, rename_topic, etc.")
    parameters: Dict[str, Any] = Field(default={}, description="Parameters for the operation")
    interpretation: str = Field(..., description="Human-readable interpretation of the command")
    requires_confirmation: bool = Field(default=False, description="Whether user confirmation is required")
    error: Optional[str] = Field(default=None, description="Error message if parsing failed")


class TOCChatResponse(BaseModel):
    """Response model for TOC chat interaction."""
    success: bool = Field(..., description="Whether the command was understood and processed")
    operation: Optional[TOCOperation] = Field(default=None, description="Parsed operation details")
    preview_toc: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Preview of TOC after applying the operation"
    )
    current_toc: List[Dict[str, Any]] = Field(..., description="Current TOC before changes")
    message: str = Field(..., description="Response message to the user")
    follow_up_question: Optional[str] = Field(
        default=None,
        description="Follow-up question to clarify or enhance the command"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested next actions or commands"
    )


class TOCConfirmRequest(BaseModel):
    """Request to confirm and apply a TOC operation."""
    operation: TOCOperation = Field(..., description="Operation to apply")
    current_toc: List[Dict[str, Any]] = Field(..., description="Current TOC to modify")
    target: Optional[str] = Field(
        default=None,
        description="Which TOC to persist to: good (working, default) | client | benchmark"
    )


class TOCConfirmResponse(BaseModel):
    """Response after applying a confirmed TOC operation."""
    success: bool = Field(..., description="Whether the operation was successfully applied")
    updated_toc: List[Dict[str, Any]] = Field(..., description="Updated TOC after operation")
    message: str = Field(..., description="Success or error message")
    undo_available: bool = Field(default=True, description="Whether undo is available")


class TOCAnalysisRequest(BaseModel):
    """Request to analyze and suggest improvements to TOC."""
    current_toc: List[Dict[str, Any]] = Field(..., description="Current TOC to analyze")
    draft_context: Dict[str, Any] = Field(..., description="Draft metadata for context")


class TOCSuggestion(BaseModel):
    """Individual TOC improvement suggestion."""
    type: str = Field(..., description="Type of suggestion: add, remove, rename, reorder")
    description: str = Field(..., description="Description of the suggestion")
    operation: TOCOperation = Field(..., description="Operation to apply the suggestion")
    rationale: str = Field(..., description="Why this suggestion would improve the TOC")
    priority: str = Field(default="medium", description="Priority: high, medium, low")


class TOCAnalysisResponse(BaseModel):
    """Response with TOC analysis and suggestions."""
    analysis_summary: str = Field(..., description="Summary of TOC analysis")
    strengths: List[str] = Field(default=[], description="Current TOC strengths")
    weaknesses: List[str] = Field(default=[], description="Areas for improvement")
    suggestions: List[TOCSuggestion] = Field(default=[], description="Specific improvement suggestions")
    industry_compliance: Dict[str, bool] = Field(
        default={},
        description="Compliance with industry standards"
    )
    completeness_score: int = Field(
        default=0,
        description="TOC completeness score (0-100)"
    )