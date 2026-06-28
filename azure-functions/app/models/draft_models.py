import uuid

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ClientMetadata(BaseModel):
    name: str
    country: str
    city: str
    industry: str 


class InitializeDraftRequest(BaseModel):
    """Request to initialize a new draft from description."""
    title: str = Field(..., min_length=1, max_length=200, description="Draft title")
    description: Optional[str] = Field(None, max_length=2000, description="Policy description (optional)")
    client_metadata: ClientMetadata = Field(..., description="Client metadata")
    function: str = Field(..., description="Function category (e.g., 'HR')")
    client_specific_requests: Optional[str] = Field(None, max_length=5000, description="Client-specific requirements")
    sector_specific_comments: Optional[str] = Field(None, max_length=5000, description="Sector-specific comments")
    regulations: str = Field(default="", description="Applicable regulation (e.g., 'UAE Labor Law')")
    detail_level: int = Field(default=3, ge=1, le=5, description="Level of detail: 1 (light) to 5 (comprehensive)")


class UpdateTOCRequest(BaseModel):
    """Request to update/finalize TOC structure."""
    toc: List[Dict[str, Any]] = Field(..., description="Table of contents structure")


class UpdateDraftMetadataRequest(BaseModel):
    """Request to update draft metadata."""
    title: str = Field(..., min_length=1, max_length=200, description="Draft title")
    description: Optional[str] = Field(None, max_length=2000, description="Policy description (optional)")
    client_metadata: ClientMetadata = Field(..., description="Client metadata")
    function: str = Field(..., description="Function category (e.g., 'HR')")
    client_specific_requests: Optional[str] = Field(None, max_length=5000, description="Client-specific requirements")
    sector_specific_comments: Optional[str] = Field(None, max_length=5000, description="Sector-specific comments")
    regulations: str = Field(default="", description="Applicable regulation (e.g., 'UAE Labor Law')")
    detail_level: int = Field(default=3, ge=1, le=5, description="Level of detail: 1 (light) to 5 (comprehensive)")


class GenerateContentRequest(BaseModel):
    """Request to generate content for a topic/subtopic."""
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="User requirements for content generation")
    subtopic_id: Optional[str] = Field(None, description="Subtopic ID if generating subtopic content")


class UpdateContentRequest(BaseModel):
    """Request to manually update topic/subtopic content."""
    content: str = Field(..., description="Content to update")
    subtopic_id: Optional[str] = Field(None, description="Subtopic ID if updating subtopic")
    conversation_history: Optional[List['ConversationEntry']] = Field(None, description="Updated conversation history")





class ValidateDraftRequest(BaseModel):
    """Request to validate draft creation parameters."""
    title: str = Field(..., min_length=1, max_length=200, description="Draft title")
    description: Optional[str] = Field(None, max_length=2000, description="Policy description (optional)")
    client_metadata: ClientMetadata = Field(..., description="Client metadata")
    function: str = Field(..., description="Function category (e.g., 'HR')")
    client_specific_requests: Optional[str] = Field(None, max_length=5000, description="Client-specific requirements")
    sector_specific_comments: Optional[str] = Field(None, max_length=5000, description="Sector-specific comments")
    regulations: str = Field(default="", description="Applicable regulation (e.g., 'UAE Labor Law')")
    detail_level: int = Field(default=3, ge=1, le=5, description="Level of detail: 1 (light) to 5 (comprehensive)")


class ExportRequest(BaseModel):
    """Request to export draft."""
    delete_after_export: bool = Field(default=True, description="Delete draft after successful export")


# ============= Response Models =============

class ValidateDraftResponse(BaseModel):
    """Response for draft validation."""
    is_valid: bool = Field(..., description="Whether the draft parameters are valid")
    issues: List[str] = Field(default=[], description="List of validation issues")
    suggestions: List[str] = Field(default=[], description="Suggestions for improvement")
    description_quality_score: int = Field(..., description="Quality score for description (0-100)")
    improved_description: Optional[str] = Field(None, description="Suggested improved description")

class SimilarPolicy(BaseModel):
    """A similar policy found during draft initialization."""
    policy_id: str
    filename: str
    description: str
    web_url: str = ""
    similarity_score: float


class ApplyPolicyRequest(BaseModel):
    """Request to apply a similar policy's TOC to a draft, or generate with AI if policy_id is None."""
    policy_id: Optional[str] = Field(None, description="Policy ID to apply. None = generate TOC with AI.")


class InitializeDraftResponse(BaseModel):
    """Response after initializing a draft."""
    draft_id: str
    similar_policies_found: int
    suggested_toc: List[Dict[str, Any]]
    message: str
    toc_source: Optional[str] = Field(default="similarity_search", description="Source of TOC: similarity_search or ai_generated")
    needs_policy_selection: bool = Field(default=False, description="True when similar policies were found and user must choose one")
    similar_policies: List[SimilarPolicy] = Field(default_factory=list, description="Up to 3 similar policies for user to choose from")


class DraftSummary(BaseModel):
    """Summary information for a draft (used in list)."""
    draft_id: str
    title: str
    description: Optional[str] = None
    created_by: str
    created_at: str
    modified_at: str
    most_similar_policy_id: Optional[str] = None
    client_metadata: Optional[Dict[str, str]] = None
    function: Optional[str] = None


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


class CritiqueResponse(BaseModel):
    """Reflection agent critique of generated content."""
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0-1")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alignment with client requirements 0-1")
    strengths: List[str] = Field(default_factory=list, description="Positive aspects of content")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    improvements: List[str] = Field(default_factory=list, description="Specific improvements suggested")
    compliance_gaps: List[str] = Field(default_factory=list, description="Regulatory or requirement gaps")
    recommendations: str = Field(..., description="Comprehensive recommendations")
    requires_revision: bool = Field(default=False, description="Whether content needs revision")
    fixed_content: Optional[str] = Field(None, description="Auto-improved content if revision was performed")


class GapFinding(BaseModel):
    """A single gap-assessment finding for a topic."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable id used as the checklist key")
    type: str = Field(..., description="missing | non_compliant | enhancement")
    source: str = Field(..., description="benchmark | regulation | previous")
    severity: str = Field(default="medium", description="high | medium | low")
    description: str = Field(..., description="What the gap is")
    suggested_change: str = Field(..., description="The concrete change to apply")
    # Evidence (Stage 3d): missing/non_compliant findings must quote the benchmark/regulation
    # text they are based on, so gaps can't be "made up".
    evidence: Optional[str] = Field(None, description="Verbatim quote from the benchmark/regulation supporting this gap")
    evidence_source: Optional[str] = Field(None, description="Locator for the evidence (benchmark section / regulation article)")
    # De-dup (Stage 3c): findings that say the same thing across sections share a canonical_id;
    # the cross-section pass keeps one per cluster, in its most-relevant section.
    canonical_id: Optional[str] = Field(None, description="Cluster key assigned by the cross-section de-dup pass")


class GapReport(BaseModel):
    """Gap assessment of one client topic vs the benchmark + regulations."""
    topic_id: str
    findings: List[GapFinding] = Field(default_factory=list)
    benchmark_topic_matched: Optional[str] = Field(None, description="Name of the benchmark topic used for comparison")
    overall_summary: str = Field(default="", description="Short narrative summary of the gaps")


class GapReportResponse(BaseModel):
    """Response after generating a topic gap report."""
    gap_report: GapReport
    message: str


class ApplyGapRequest(BaseModel):
    """Request to apply confirmed gap findings to a topic."""
    confirmed_finding_ids: List[str] = Field(default_factory=list, description="Finding ids the reviewer kept")
    edited_findings: Optional[List[GapFinding]] = Field(None, description="Reviewer-edited findings (overrides matching ids)")
    preview: bool = Field(default=False, description="Generate the proposed rewrite without persisting (preview step)")
    final_content: Optional[str] = Field(None, description="Reviewer-approved content to commit directly (confirm step); skips LLM generation")
    subtopic_id: Optional[str] = Field(None, description="Target subtopic; when set the gap unit is the subtopic, not the topic")


class BatchUnitRef(BaseModel):
    """Reference to a gap unit (topic, or one of its subtopics)."""
    topic_id: str
    subtopic_id: Optional[str] = None


class AssessBatchRequest(BaseModel):
    """Request to assess several units in one call (server-side parallel)."""
    units: List[BatchUnitRef] = Field(default_factory=list)


class ApplyGapResponse(BaseModel):
    """Response after applying confirmed gap findings to a topic."""
    content: str
    summary: str = ""
    word_count: int = 0
    message: str


class ConsistencyIssue(BaseModel):
    """A single whole-policy consistency issue."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(..., description="contradiction | terminology | duplicate | undefined_term | cross_reference")
    severity: str = Field(default="medium", description="high | medium | low")
    description: str = ""
    locations: List[str] = Field(default_factory=list, description="Section titles involved")
    suggestion: str = ""


class ConsistencyReport(BaseModel):
    """Result of a whole-policy consistency pass."""
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    summary: str = ""
    generated_at: str = ""


class GapSummaryTopic(BaseModel):
    """Per-unit entry in the gap summary."""
    title: str
    status: str = "pending"
    findings_count: int = 0
    overall_summary: str = ""
    findings: List[GapFinding] = Field(default_factory=list)


class GapSummary(BaseModel):
    """Aggregated, client-facing summary of a gap-assessment review."""
    narrative: str = ""
    total_units: int = 0
    assessed: int = 0
    applied: int = 0
    pending: int = 0
    benchmark_coverage: int = 0
    pct_preserved: Optional[float] = None
    total_findings: int = 0
    by_severity: Dict[str, int] = Field(default_factory=dict)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_source: Dict[str, int] = Field(default_factory=dict)
    topics: List[GapSummaryTopic] = Field(default_factory=list)


class DraftVersion(BaseModel):
    """Metadata for a saved draft snapshot (content not included)."""
    version_id: str
    created_at: str
    created_by: str = ""
    label: str = ""
    reason: str = ""
    size: int = 0


class AuditEntry(BaseModel):
    """A single audit-trail entry for a draft."""
    ts: str
    actor: str = ""
    action: str
    target_id: str = ""
    target_title: str = ""
    summary: str = ""


class SaveVersionRequest(BaseModel):
    """Request to snapshot the current draft as a version."""
    label: Optional[str] = None
    reason: Optional[str] = None


class ContentGenerationResponse(BaseModel):
    """Response after content generation."""
    content: str = Field(..., description="Generated content")
    sources_used: List[str] = Field(default_factory=list, description="Source IDs used for generation")
    summary: str = Field(default="", description="Summary of generated content if > 150 words")
    word_count: int = Field(default=0, description="Word count of generated content")
    message: str
    is_chat_response: bool = Field(default=False, description="True if this is a chat response, not actual policy content")
    critique: Optional[CritiqueResponse] = Field(None, description="Reflection agent critique (if available)")


# ============= Data Models (for internal use) =============

class TOCTopic(BaseModel):
    """Topic in table of contents."""
    topic_id: str
    topic: str
    order: int
    source_topic_id: Optional[str] = None
    content: str = Field(default="", description="Generated content for this topic")
    summary: str = Field(default="", description="Summary if content > 150 words")
    conversation_history: List['ConversationEntry'] = Field(default_factory=list, description="Conversation history for this topic")
    subtopics: List['TOCSubtopic'] = Field(default_factory=list)
    # Gap-assessment review fields (Review Policy / review_mode drafts)
    gap_report_json: Optional[str] = Field(None, description="Serialized GapReport for this topic")
    gap_status: str = Field(default="pending", description="Gap review status: pending | assessed | applied")
    added_from_benchmark: bool = Field(default=False, description="True if this topic was auto-added because the client policy lacked it")
    benchmark_match: Optional[str] = Field(None, description="Benchmark section this topic maps to (set at TOC reconciliation, reused by gap-report)")
    reconcile_note: Optional[str] = Field(None, description="Short rationale for a reconciliation merge/rename/add")


class TOCSubtopic(BaseModel):
    """Subtopic in table of contents."""
    subtopic_id: str
    topic: str
    order: int
    source_subtopic_id: Optional[str] = None
    content: str = Field(default="", description="Generated content for this subtopic")
    summary: str = Field(default="", description="Summary if content > 150 words")
    conversation_history: List['ConversationEntry'] = Field(default_factory=list, description="Conversation history for this subtopic")
    # Gap-assessment review fields (per-subtopic gap units)
    gap_report_json: Optional[str] = Field(None, description="Serialized GapReport for this subtopic")
    gap_status: str = Field(default="pending", description="Gap review status: pending | assessed | applied")
    benchmark_match: Optional[str] = Field(None, description="Benchmark subtopic this subtopic maps to")


class ConversationEntry(BaseModel):
    """Single conversation entry."""
    timestamp: str
    user_message: str
    ai_response: str


class Topic(BaseModel):
    """Topic with content."""
    id: str
    content: str = ""
    summary: str = ""
    conversation_history: List['ConversationEntry'] = []
    subtopics: List['Subtopic'] = []


class Subtopic(BaseModel):
    """Subtopic with content."""
    id: str
    content: str = ""
    summary: str = ""
    conversation_history: List['ConversationEntry'] = []


class DraftMetadata(BaseModel):
    """Draft metadata."""
    title: str
    description: Optional[str] = None
    created_by: str
    created_at: str
    modified_at: str
    client_metadata: ClientMetadata
    function: str
    most_similar_policy_id: Optional[str] = None
    toc_source: Optional[str] = Field(default="ai_generated", description="Source of TOC: ai_generated")
    client_specific_requests: Optional[str] = Field(None, description="Client-specific requirements")
    sector_specific_comments: Optional[str] = Field(None, description="Sector-specific comments")
    regulations: str = Field(default="", description="Applicable regulation")
    detail_level: int = Field(default=3, description="Level of detail: 1 (light) to 5 (comprehensive)")
    policy_type: Optional[str] = Field(None, description="Policy type (e.g., 'Leave Policy'). Falls back to title if not set.")
    uploaded_regulations_text: Optional[str] = Field(None, description="Concatenated text from uploaded regulation files (Review Policy flow)")
    previous_policy_text: Optional[str] = Field(None, description="Full text of the previous client policy (Review Policy flow)")
    # Gap-assessment review fields
    review_mode: bool = Field(default=False, description="True = topic-by-topic gap review draft")
    benchmark_policy_id: Optional[str] = Field(None, description="Library policy_id used as the good-standard benchmark")
    benchmark_source: Optional[str] = Field(None, description="Where the benchmark came from: library | uploaded")
    benchmark_topics_json: Optional[str] = Field(None, description="Serialized parsed topics when the benchmark is an uploaded file")
    review_intensity: str = Field(default="preserve", description="How much the review may change the policy: preserve (keep >=50-60% baseline) | rebuild (heavy rewrite for weak policies)")
    toc_reconciliation_note: Optional[str] = Field(None, description="LLM hint from TOC reconciliation (e.g. 'policy looks light, consider Rebuild')")
    needs_manual_toc: bool = Field(default=False, description="True when no structure could be extracted from the upload — the consultant should build/fix the TOC manually")
    consistency_report_json: Optional[str] = Field(None, description="Serialized last whole-policy ConsistencyReport")
    # Perfect-TOC-first stage (v2): the TOC tab shows three distinct TOCs and is an explicit gate
    # before any per-section content work.
    benchmark_toc_json: Optional[str] = Field(None, description="Titles-only benchmark TOC for the read-only 'Extracted' column (set for any benchmark source)")
    client_toc_json: Optional[str] = Field(None, description="Snapshot of the client's parsed TOC (read-only 'Current' column)")
    good_toc_json: Optional[str] = Field(None, description="The reconciled recommendation (read-only 'Good' column); the editable working TOC is `toc`")
    good_toc_basis_json: Optional[str] = Field(None, description="Snapshot of the Current + Extracted TOCs the Good TOC was last derived from; the 'Regenerate' action shows only when one has since changed")
    toc_approved: bool = Field(default=False, description="True once the reviewer approves the good/working TOC; unlocks the per-section pipeline")

    @property
    def effective_policy_type(self) -> str:
        """Return policy_type if set, otherwise fall back to title."""
        return self.policy_type or self.title


class CompleteDraft(BaseModel):
    """Complete draft structure."""
    id: str = Field(..., default_factory=lambda: str(uuid.uuid4()))
    metadata: DraftMetadata
    toc: List[TOCTopic]
    toc_chat_history: List[ConversationEntry] = Field(
        default_factory=list,
        description="Conversation history for TOC chat assistant"
    )

# Update forward references
TOCTopic.model_rebuild()
Topic.model_rebuild()