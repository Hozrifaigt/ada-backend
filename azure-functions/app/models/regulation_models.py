"""
Pydantic models for regulations and their components.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RegulationMetadata(BaseModel):
    """Metadata for a regulation document."""
    regulation_category: str = Field(..., description="Standardized regulation category (e.g., 'UAE Labor Law', 'ADCG', 'DIFC')")
    decree_name: str = Field(..., description="Full decree name (e.g., 'Federal Decree-Law No. 33 of 2021')")
    regulation_type: str = Field(..., description="Type: primary, amendment, or executive")
    effective_date: Optional[str] = Field(None, description="Effective date in ISO format")
    decree_number: Optional[str] = Field(None, description="Decree number (e.g., '33 of 2021')")
    file_path: str = Field(..., description="Path to the PDF file")


class RegulationChunk(BaseModel):
    """A chunk of regulation content."""
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The actual text content")
    regulation_category: str = Field(..., description="Standardized regulation category (e.g., 'UAE Labor Law', 'ADCG', 'DIFC')")
    decree_name: str = Field(..., description="Full decree name (e.g., 'Federal Decree-Law No. 33 of 2021')")
    article_number: Optional[str] = Field(None, description="Article number (e.g., 'Article 15')")
    regulation_type: str = Field(..., description="Type: primary, amendment, or executive")
    effective_date: Optional[str] = Field(None, description="Effective date")
    topics: List[str] = Field(default_factory=list, description="Auto-tagged topics")
    page_number: Optional[int] = Field(None, description="Page number in the source PDF")

    # For search results
    score: Optional[float] = Field(None, description="Similarity score from search")


class RegulationSearchRequest(BaseModel):
    """Request for searching regulations."""
    query_text: str = Field(..., description="The search query")
    regulation_name: Optional[str] = Field(None, description="Filter by regulation name")
    top_k: int = Field(default=7, ge=1, le=20, description="Number of results to return")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for search")
    min_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Minimum similarity score")


class RegulationSearchResponse(BaseModel):
    """Response from regulation search."""
    chunks: List[RegulationChunk] = Field(default_factory=list, description="Retrieved regulation chunks")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="The original query")


class CitedRegulation(BaseModel):
    """A regulation that was cited in generated content."""
    article: str = Field(..., description="Article number (e.g., 'Article 15')")
    regulation_name: str = Field(..., description="Full regulation name")
    text_snippet: str = Field(..., description="First 150 chars of the article")
    chunk_id: Optional[str] = Field(None, description="ID of the source chunk")


class RegulationProcessingResult(BaseModel):
    """Result of processing a regulation PDF."""
    regulation_name: str = Field(..., description="Name of the regulation")
    file_path: str = Field(..., description="Path to the PDF")
    chunks_created: int = Field(..., description="Number of chunks created")
    articles_found: int = Field(..., description="Number of articles detected")
    success: bool = Field(..., description="Whether processing succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_seconds: float = Field(..., description="Time taken to process")


class RegulationsIndexStatus(BaseModel):
    """Status of the regulations index."""
    total_regulations: int = Field(..., description="Total number of regulations indexed")
    total_chunks: int = Field(..., description="Total number of chunks")
    last_updated: datetime = Field(..., description="Last update timestamp")
    regulations: List[RegulationProcessingResult] = Field(default_factory=list, description="Processing results")
