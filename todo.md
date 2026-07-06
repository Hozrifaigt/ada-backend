# Accuracy audit (2026-07-06) — retrieval + generation logic bugs
Full-pipeline review of retrieval/generation accuracy (vector_store, generation_service,
reflection_agent, query_classification, draft_service matching). draft_service's benchmark-matching
layer was left alone (already hardened in the 2026-07-02 audit rounds). Four defects fixed:
1. **vector_store.search_relevant_regulations ignored `regulation_name` AND `min_score`** — every
   regulation search ran unfiltered across ALL indexed regulations, and no relevance floor was
   applied (the configured 0.55 could never work: Azure hybrid returns RRF scores ~0.01-0.06).
   Now: category filter with unfiltered fallback (so a non-matching free-text name can't zero out
   results) + wide hybrid candidate pool (top_k*3) + true-cosine rerank (batch-embedded, bounded
   module cache `_REG_RERANK_EMBED_CACHE`) + min_score applied to the cosine; chunk.score is now a
   real cosine. Rerank failure degrades to hybrid order (non-fatal).
2. **vector_store.search_similar_topics compared raw Azure score to a cosine threshold** — raw 0.7
   is really cosine ≈0.57, so weak matches passed. Now inverts (2 − 1/score), same as
   search_similar_policies.
3. **reflection_agent: every critique on a topic with conversation history CRASHED** —
   `msg.get('user_prompt')` on ConversationEntry models (AttributeError) → default score-0.0
   "Critique failed" critique with requires_revision=True → content force-regenerated from
   meaningless feedback. Fixed keys (user_message/ai_response) + model_dump-or-dict normalization.
   Plus: critic saw only the FIRST 1000 chars of the content it judged (now up to 12k, truncation
   noted); strict json.loads failed on ```json fences (now fence-tolerant like _parse_json_object);
   a FAILED critique no longer triggers auto-fix (new `critique_failed` flag; requires_revision
   False on error) — the original content is kept instead of being rewritten from noise.
4. **generation_service: conversation history never reached the generators** — computed for
   classification but both _adapt_existing_content and _generate_from_scratch got "" (Step 5 was
   commented out), so follow-ups like "make it shorter" had no memory of prior turns. Re-enabled
   (last 3 legitimate entries, existing filter).
## Review
Files: app/pipelines/components/vector_store.py, app/services/reflection_agent.py,
app/services/generation_service.py. No API-shape or model changes; RegulationChunk.score semantics
changed from hybrid-RRF to cosine (only used for logging/ordering downstream). Follow-ups: consider
a min_score for search_uploaded_regulations (currently 0.0); `_match_benchmark_topic` tier-1
substring rule can false-match very short titles ("IT" ⊂ "exit") — left as-is, recently audited.
UNTESTED here (no build/test in this env).

# Pre-production fix (2026-07-06) — ETag optimistic concurrency on draft saves
Closes the audit's top HIGH: every save was read→modify→upsert(REPLACE) with no condition, so two
concurrent editors silently overwrote each other (last write wins). Scope kept minimal per user
("fix it for now only") — backend only, drafts partition only (versions/audit rows are append-only
and don't need it).
1. **draft_storage.py:** `get_draft` records the entity etag per instance (`_etags`; the service is
   per-request, so the map spans exactly one read→modify→write cycle). `update_draft` now writes with
   `update_entity(etag, MatchConditions.IfNotModified)` when an etag is known and raises the new
   `DraftConflictError` on 412 (unconditional upsert fallback when the draft wasn't read through this
   instance — unchanged behavior). Etag refreshed after every successful write so multi-save requests
   keep working.
2. **Overflow-blob race fixed alongside:** the >450KB overflow blob is uploaded BEFORE the entity
   write and previously used the shared name `draft_{id}.json`, so a LOSING writer would still have
   clobbered the winner's payload. Blob names are now unique per write; the entity write is the
   commit point. Replaced blobs are deleted after a successful write (`_remember_write`), a rejected
   write deletes its own blob, and `delete_draft` resolves the pointer from the entity (legacy fixed
   name still attempted for old blobs).
3. **main.py:** global `DraftConflictError` handler → 409 with the reload-and-retry message in both
   `error` and `detail`. Caveat: endpoints with broad `except Exception → HTTPException(500,
   detail=str(e))` intercept the exception first, so some conflicts surface as 500 — but the
   actionable message still reaches the UI via `detail`. Per-endpoint 409 mapping = follow-up.
## Review
Two files changed: `app/services/draft_storage.py`, `app/main.py`. No frontend changes (panels
already display `detail`). No model/API-shape changes. Follow-ups identified: re-raise
DraftConflictError (or HTTPException) before the broad excepts in drafts.py for clean 409s
everywhere; frontend could offer an explicit "reload draft" action on conflict. UNTESTED here
(no build/test in this env).

# Feature update (2026-07-02) — variants × formats + format-aware preview
1. **Shared module `review_diff.py`** (diff_ops/review_units/removed_sections/filter_review_draft) —
   Word AND PDF builders use it; word_export_service's private copies removed.
2. **PDF variants:** pdf_export_service gained `export_review_redline` + `export_changes_overview`
   (reportlab inline markup: green backColor for additions, red backColor + <strike> for removals;
   text XML-escaped). PDF **final policy now applies filter_review_draft** — closes the audit's
   "PDF export leaks unreviewed sections" HIGH that was previously deprioritized.
3. **Endpoints generalized:** `POST /{draft_id}/export/{redline|changes}/{fmt}` (fmt = word | pdf),
   shared `_review_variant_response` helper; 400 on bad fmt / non-review drafts.
4. **Frontend:** draftService exportRedline/exportChangesOverview take format; ExportReviewPanel now has
   TWO choices — DOCUMENT (Final policy / Redline / Changes overview, review drafts) × FILE FORMAT
   (Word / PDF); labels/filenames follow both. **Preview is variant-aware**: Redline → in-browser
   word-level diff (wordDiff util, green/red spans, "(new section)" tags; removed-sections note),
   Changes → counts + per-section status rows; Final → the original clean preview. New components
   RedlineSpans/RedlinePreviewBody/ChangesPreviewBody in DraftEditPage.
UNTESTED (no build/test in this env).

# Feature (2026-07-02) — review-draft export variants + Export tab sizing
1. **Redline .docx** (`export_review_redline` in word_export_service + `POST /{draft_id}/export/redline/word`):
   per section, word-level diff (difflib) of frozen `baseline_content` vs `content` — additions
   highlighted BRIGHT_GREEN, removals RED + strikethrough (WD_COLOR_INDEX); added sections tagged
   "(new section)"; appendix of REMOVED sections (original snapshot units absent from the final TOC by
   id AND normalized title, fully struck). Uses the UNFILTERED toc (documents the change set).
2. **Changes overview .docx** (`export_changes_overview` + `POST /{draft_id}/export/changes/word`):
   rollup counts + per-section Added/Modified(+x/−y words)/Unchanged bullets (reconcile_note italic)
   + removed list. Both endpoints share `_review_export_prereqs` (404/400 gate: review drafts only;
   original snapshot via storage.get_original_version, model_dump'd).
3. **Frontend**: draftService `exportRedline`/`exportChangesOverview` (blob); ExportReviewPanel format
   union +'redline'|'changes'; review drafts see 4 option cards (Final policy / Redline green /
   Changes overview amber / PDF); export-button + preview-modal labels mapped.
4. **Sizing**: ExportReviewPanel `isCompact` forced to true — the XL layout (h3/h4, p:3, 32px icons)
   dwarfed the other tabs on large screens; dense variant is now the only variant.
UNTESTED (no build/test in this env). Note: "Final policy" Word export keeps the closed-sections
filter; redline/changes intentionally include every section.

# UX (2026-07-02) — approve → jump to Review; benchmark step decluttered
1. Approving the TOC now lands the reviewer on the Review tab: `onApproved` prop threaded
   DraftEditPage → TocComparePanel (+ RedistributeButton "Save & approve") → parent reloads then
   `setTabValue(4)`. `loadDraft`'s review-draft bounce to the TOC tab now fires on FIRST load only
   (`reviewTabInitRef`) so reloads/apply/restore no longer yank the user back.
2. ReviewPipelinePanel benchmark step decluttered: match picker + paste-text override live inside a
   `Collapse` behind a single "Adjust match" text button (Tune icon); match status (Manual text /
   Pinned / No benchmark matched) shows as one chip on the Run row; auto-opens when a run returns no
   benchmark content. Default view = Run button → findings → continue → content.

# UI polish pass revision (2026-07-02) — less text, more color
Per user: REMOVED the "AI reconciliation note" line and the "Drag, edit and save…" explainer caption
from TocComparePanel. Instead the three columns are differentiated visually: ColumnShell takes
accent/headerTint/bodyTint — 3px accent top border + tinted header + faint body tint per column
(Current = slate #64748b/#f1f5f9/#f8fafc, Benchmark = purple #764ba2/#f5f0fa/#fbf9fe, Good = indigo
#667eea/#eef2ff/#f8f9ff). TocEditor's white cards sit on the tinted body. Workflow strip + "+N added"
chip + dynamic benchmark subtitle kept.

# UI polish pass (2026-07-02) — make the workflow visible
Frontend only. (1) TocComparePanel: 4-step workflow strip (Attach benchmark → Build Good TOC → Map
client content → Approve TOC) with done/current/upcoming states + a right-aligned hint; AI
reconciliation note (metadata.toc_reconciliation_note) now displayed; Good column header shows a green
"+N added" chip (topics+subtopics added from benchmark); Benchmark column subtitle reflects the actual
source (library/uploaded/none). (2) ReviewPipelinePanel: green LinearProgress (closed/total) under the
SECTIONS header; section rows now show the CURRENT pipeline step (benchmark/regulation/editing) in
indigo for in-progress sections (provenance stays as the colored dot; label falls back to kept/added
for untouched rows); legend explains the step label. New `WorkflowStep` mini-component + `STEP_SHORT`
map. No behavior changes — display only.

# Reconcile match evidence (2026-07-02) — snippet 400 → 1100 chars + middle sampling
`_format_topics_for_reconcile` snippet raised to 1100 chars; long sections sampled head (70%) + a
middle slice, because section openers are boilerplate and carried little distinguishing signal — the
main cause of same/different mis-judgements in reconcile. Token cost ≈ +6-7k per reconcile call (runs
once per upload/regenerate). Embedding texts (title+content[:1500] in _build_reconciled_toc /
_best_benchmark_unit) unchanged — candidate follow-up if matching still misfires.

# UX (2026-07-02) — benchmark source choice made explicit on NewDraftPage
The Benchmark Policy card used a Switch whose label showed the CURRENT mode ("Upload a file" /
"Browse from library"), so users couldn't tell what flipping it would do. Replaced with an explicit
two-option ToggleButtonGroup ("Upload a file" | "Choose from library") ABOVE the input, matching the
Rewrite-intensity pattern on the same card; header caption now explains both options. Same
state/clearing behavior (switching modes clears the other mode's selection). Switch/FormControlLabel
imports left in place (possible other usages; grep unavailable in this env).

# Pre-production fixes — audit round 2 MEDIUM issues (2026-07-02)
1. **Stale finding ids fail loudly:** `apply_gap_for_topic` raises ValueError when confirmed ids match
   nothing in the stored report (ids regenerate every assess); endpoint → 409; ReviewPipelinePanel
   `tweakSelected` surfaces `response.data.detail`.
2. **Orphan promotion keeps review state:** `commit_final_structure` promoted-subtopic branch now copies
   content/summary/conversation/gap/step/matches/manual fields from `ex_subs` instead of rebuilding bare.
3. **Regulation check failure ≠ "none apply":** `review_regulation_check` returns `check_failed` on JSON
   parse failure; ReviewPipelinePanel shows an amber "check didn't complete — re-check" state instead of
   the false-negative "no applicable regulation found".
4. **Export placeholder:** review-mode exports never print "*[No content generated yet]*" — empty closed
   sections emit heading only (both Word + PDF builders).
5. **Filtered-export numbering:** `_filter_review_draft` renumbers survivors sequentially (topics 1..n,
   subtopics 1..m; subs deep-copied before mutation).
6. **Audit:** append failures now logged at ERROR with action/actor (callers discard the bool; the log is
   the only signal). Known accepted limitation: list_audit scans the partition and caps at 200.
7. **TOC-extraction coverage guard:** the LLM printed-TOC rebuild is only adopted when ≥70% of extracted
   titles were located in the body; otherwise the deterministic structure is kept (mis-located titles
   silently folded their content into the preceding section).
UNTESTED here (no build/test in this env).

# Pre-production fixes — audit round 2 HIGH issues (2026-07-02)
Scope per client: .docx-only uploads everywhere (NO PDF/OCR), plus highs #2-#6. PDF *export* of review
drafts still skips `_filter_review_draft` (user chose to deprioritize; one-line fix if needed later).
1. **.docx-only uploads:** `_require_docx` helper in drafts.py applied to ALL uploads — initialize-from-
   upload now validates previous_policy/benchmark/regulations files too (was policy only), extract-from-
   file + regulations/upload share the helper (case-insensitive). NewDraftPage regulations input got
   accept=".docx" + extension check (was unvalidated).
2. **Word export leak:** `_filter_review_draft` blanks a NON-closed parent's content/summary when it's
   kept only because a subtopic is closed (heading-only).
3. **Manual benchmark fields preserved:** `manual_benchmark_content`/`manual_benchmark_match` now carried
   in `update_toc`, `apply_toc_structure`, `_client_topics_from_snapshot` (which also now carries
   gap_report_json/gap_status). TOC edits no longer wipe pins/pasted benchmark text.
4. **No more whole-regulations prompt dumps:** the four `uploaded_regulations_text` appends in
   generation_service (apply_gap_to_content, consolidate_theme_content, generate_gap_report,
   _adapt_existing_content) are gone; callers (`apply_gap_for_topic`, `assess_batch`,
   `generate_gap_report_for_topic`, `consolidate_theme_for_topic`) now use `_regulation_chunks_for_unit`
   (library index + retrieval over uploaded text).
5. **Gap summary:** `build_gap_summary` derives per-unit status via `_unit_status` (review_step
   regulation/review/closed → applied; benchmark → assessed; else legacy gap_status) so completed review
   drafts no longer roll up as pending.
6. **Version safety:** original snapshot at upload checks create_version's return + retries once + ERROR
   log; `restore_version` ABORTS when the pre-restore backup fails; `propose_redistribution` returns
   `original_missing` and RedistributeButton shows a warning Alert when mapping from the live draft.
UNTESTED here (no build/test in this env).

# Good-TOC de-duplication: A (content-grounded LLM hints) + C (semantic dedup guardrail)

## Problem
`_build_reconciled_toc` resolves the LLM's client<->benchmark mapping by EXACT title match, and has no
content-level dedup. So a client topic and a benchmark-added topic that mean the same thing but are
titled differently (client "Fixed Assets" vs benchmark "Property, Plant & Equipment") end up as TWO
topics. The LLM sees content so it often merges, but nothing guarantees it, and the exact-title
resolution + preserve-mode safety loop can reintroduce the duplicate.

## Fix
- **A — content-grounded hints (better merges):** before `reconcile_toc`, embed each client topic
  (title+content) and benchmark topic (title+content), find semantically-similar client<->benchmark
  pairs, and feed them to the prompt as explicit "treat as the SAME section" hints.
- **C — deterministic semantic guardrail (the guarantee):** AFTER reconciliation, for every
  `added_from_benchmark` topic, compare (by embedding cosine, i.e. by MEANING) against the kept client
  topics; if it clears a conservative threshold, MERGE into the client topic (keep client title+content,
  set `benchmark_match`, add a `reconcile_note`) instead of leaving a duplicate.

Both reuse the existing embedding helpers (`_embed`, `_cosine`, `_benchmark_topic_text`). No new storage.

### Thresholds (tunable)
- HINT (A): cosine >= 0.75 — permissive; it's only a suggestion to the LLM.
- MERGE (C): cosine >= 0.82 — conservative; it auto-removes a benchmark section, so err toward keeping
  both. Every merge is visible via `reconcile_note` and reversible at the 3-column TOC / Approve gate.

## Changes
- [x] `prompts.v1.yml` `reconcile_toc`: add a `{similarity_hints}` block + an instruction to map flagged
      pairs as ONE entry (`source:both`, set client_match+benchmark_match), not a duplicate.
- [x] `generation_service.reconcile_toc`: add `similarity_hints: str = ""` param → prompt.
- [x] `draft_service._reconcile_vector_store()`: best-effort access to the shared vector store (lazy
      import of the singleton); returns None if unavailable → reconciliation stays LLM/title-only.
- [x] `draft_service._build_reconciled_toc(..., vector_store=None)`:
      - compute client + benchmark embeddings; build `similarity_hints`; pass to `reconcile_toc` (A);
      - after the safety loop, run the dedup guardrail merging added->client topics (C).
- [x] upload call site (initialize_draft_from_upload): pass `vector_store=vector_store`.

## Pre-production fixes — 5 MEDIUM issues (2026-07-02)
4. **Empty-content rule no longer step-gated:** ReviewPipelinePanel `workingContentOf` blanks whenever
   content merely mirrors the baseline and the section isn't closed (running a benchmark advances the
   step without producing text); closed sections always show their real content (that's what exports).
5. **Embedding cost:** module-level `_EMBED_CACHE` (FIFO, 1024) in draft_service so benchmark units stay
   warm across requests (DraftService is per-request); `_best_benchmark_unit` title-only pass is now
   LAZY (only runs when the combined title+content pass fails the 0.78 threshold).
6. **Reconcile duplicate guards:** the added-entry branch skips a title that duplicates a client section
   that will be present (reused or preserve-mode safety loop; in rebuild mode only if actually reused)
   and skips a benchmark title already added this run (`added_titles`). `_naive_benchmark_union` now
   dedups by `_norm_title` (and won't add the same benchmark title twice).
7. **Approve gate server-side:** `approve_toc` raises ValueError when `content_mapped` is False; the
   endpoint returns 409 with the message (UI gate unchanged, API can no longer bypass map-first).
8. **Export-vs-display honesty:** CURRENT CONTENT empty-state now says the client baseline is kept for
   export until changes are applied/saved.
UNTESTED here (no build/test in this env).

## Pre-production fixes — 3 HIGH issues (2026-07-02)
1. **Rename no longer wipes content:** `apply_toc_structure` now matches existing topics/subtopics by
   STABLE ID first (entry `id`/`topic_id`, subtopic `id`/`subtopic_id`), title fallback for id-less
   callers (extract/paste previews, /toc/use). Frontend `toStructureItems` (TocComparePanel) sends ids;
   `TocStructureItem` type extended ({id?, title, subtopics:(string|{id?,title})[]}); preview render
   handles both shapes. Synthetic ids (client/benchmark columns, new_… sections) miss the id map and
   fall back to title — unchanged behavior. NOTE: `_client_topics_from_snapshot` (regenerate) is still
   title-based (snapshot has no ids) — known, lower risk.
2. **Embed clause regression:** ReviewPipelinePanel builds the embed preview on
   `editContent || selected.content || selected.baseline` so the empty-seed editor can't cause the
   clause to REPLACE the section's stored client text.
3. **Duplicate benchmark names:** pins are qualified `parent|name`. Picker MenuItem value =
   `parent|name` for subtopics (renderValue shows "name — parent"); backend `_flatten_benchmark_units`
   carries `parent`; `_resolve_benchmark_unit_by_name` resolves qualified pins (parent+name → name-only
   fallback if parent renamed → plain resolution for old/unqualified pins & persisted benchmark_match);
   new `_pin_name` helper used in both `benchmark_source` computations. Old plain-name pins keep the
   previous first-occurrence behavior (back-compat).
UNTESTED here (no build/test in this env).

## Benchmark matching v2 + match picker + empty working content (2026-07-02)
Three asks: better section↔benchmark matching, a picker instead of paste-text-only manual matching, and
"current content" empty when a section has no gaps applied / no baseline.
1. **Matching (draft_service):** `_benchmark_content_for_unit` resolution order is now
   manual TEXT → PINNED section (`manual_benchmark_match`, new field on TOCTopic/TOCSubtopic, resolved
   by `_resolve_benchmark_unit_by_name`, no similarity) → persisted `benchmark_match` (reconciliation /
   previous run, trusted before search) → auto. `_best_benchmark_unit` is layered: normalized exact title
   (`_norm_title`) → title+content embedding ≥0.78 → title-only embedding ≥0.86 (same-named section whose
   content drifted). `_match_benchmark_topic` tier-1 normalized. `assess_benchmark_for_unit` now persists
   the resolved match for TOPICS too (was subtopics only; never persists "Manual benchmark") and returns
   `benchmark_source` (manual|pinned|matched|none). `_get_benchmark_topics` falls back to the titles-only
   `benchmark_toc_json` (TOC-tab upload/paste benchmarks) so name resolution + the picker still work.
2. **Picker:** new `manual_benchmark_match` (pin) — `set_manual_benchmark_match` (pin clears manual text;
   CLEARING the pin also clears persisted benchmark_match so auto truly re-runs; setting text clears pin),
   `list_benchmark_units` + endpoints `GET /{draft_id}/benchmark-units`,
   `POST /{draft_id}/topics/{topic_id}/benchmark-match` (`SetBenchmarkMatchRequest`). Frontend:
   ReviewPipelinePanel "BENCHMARK MATCH" Select (Auto + topics bold / subtopics indented with parent,
   "(no content)" suffix), pin → save + auto re-run, status chips from benchmark_source; paste-text
   override kept as secondary ("Paste benchmark text instead"). draftService `getBenchmarkUnits`/
   `setBenchmarkMatch`; TS types +manual_benchmark_content/manual_benchmark_match.
3. **Empty working content:** ReviewPipelinePanel `workingContentOf` — the SECTION CONTENT editor and
   CURRENT CONTENT panel are empty for an untouched ('pending') section whose content merely mirrors the
   frozen baseline or that has no baseline; fills only once the reviewer applies/saves something. Display
   seed only — unit.content is untouched server-side; "Save edits" disabled-compare uses the same seed so
   an accidental click can't wipe content.
UNTESTED here (no build/test in this env).

## UX (2026-07-02) — approve the mapping from inside the "Map client content" dialog
The reviewer had to Save the mapping, close the dialog, then click "Approve TOC & continue" separately.
`RedistributeButton` now takes `showApprove` (parent passes `!metadata.toc_approved`) and shows a
"Save & approve TOC" button in the dialog footer: commits the mapping (`commitRedistribution`, which sets
`content_mapped`) then calls `approveToc` in the same flow — enabled even with no edits, so reopening just
to approve works. Existing Save/"Save all" unchanged (`save()` now takes an `approve` flag; call sites
wrapped so the click event isn't passed as the flag). Frontend-only: RedistributeButton.tsx,
TocComparePanel.tsx.

## Fix #3 (2026-07-02) — additions still never shown: stale claims, strict guard, no subtopics, no UI badge
Even after Rebalance #2 the reviewer never SAW an added topic or subtopic. Four independent causes, all fixed:
1. **Stale-claim fold (the regenerate killer):** guardrail C(a) built `claimed` from every kept client
   topic's PERSISTED `benchmark_match` (carried from a previous reconcile by `_client_topics_from_snapshot`
   and the safety re-add loop). On "Regenerate Good TOC" nearly every benchmark section was already
   "claimed", so every LLM addition was folded away → Good TOC == client TOC. Now `claimed` = only the
   mappings the LLM emitted THIS run (`run_claims`, recorded when an entry reuses a client topic).
2. **Exact-title anti-hallucination guard:** an added entry was dropped unless its title/benchmark_match
   EXACTLY equaled a top-level benchmark title (lowercase only). New `_norm_title` (lowercase, &→and,
   strip numbering/punctuation, collapse spaces) used across the guard, `client_by_title`, embedding-cache
   keys, and guardrail C; the allowed set now also includes benchmark SUBTOPIC titles.
3. **Subtopics were structurally impossible:** the reconcile prompt had no subtopic field and the LLM never
   even saw subtopic titles. Now `_format_topics_for_reconcile` lists subtopics (client dicts include them
   too), the prompt has rule 1d + per-entry `add_subtopics`, and `_build_reconciled_toc` (a) gives an added
   benchmark topic its own benchmark subtopics (`_benchmark_subtopic_names`), (b) appends guarded
   `add_subtopics` to matched client topics via `_append_added_subtopics` (added_from_benchmark=True,
   normalized dedup, validated against real benchmark titles). `_naive_benchmark_union` carries subtopics
   too, and `_fold` moves a folded addition's subtopics into the owning client topic.
4. **Invisible + wiped flags:** the UI never rendered provenance — TocEditor now shows a green "added"
   chip (tooltip = reconcile_note / benchmark_match) on topic AND subtopic rows. `update_toc` and
   `apply_toc_structure` subtopic branches now preserve `added_from_benchmark` (they silently reset it).
Files: draft_service.py, generation_service.py, prompts.v1.yml, ada-frontend TocEditor.tsx.
Re-run "Regenerate" on the Good TOC to see additions. UNTESTED here (no build/test in this env).
Follow-ups: consider surfacing "added" counts in toc_reconciliation_note; C(b) 0.82 fold still folds
same-meaning additions (intended).

## Rebalance #2 (2026-07-02) — hints suppressed all additions (still a copy)
Even after softening 1c, the Good TOC stayed an exact client copy. Cause: the A similarity hints were
ONE-DIRECTIONAL — "each benchmark → its single best client if ≥0.78" — so in a same-domain policy almost
every benchmark topic got hinted onto some client, and the LLM merged everything → nothing added. Fix:
hints now require a MUTUAL best match (benchmark and client are each other's nearest) at ≥0.82, so a
genuinely-missing benchmark section (no reciprocal client) is added, not merged away. The C guardrail's
threshold raised 0.78 → 0.82 (shared constant) so it doesn't undo legitimate additions either. Re-run
reconciliation (Regenerate / re-upload) to apply.

## Rebalance (2026-07-02) — "keep, don't add" over-suppressed additions
Symptom: with a real benchmark attached, the Good TOC came out an EXACT copy of the client (nothing
added), across multiple drafts. Cause: rule 1c ("even loosely... when unsure, keep") pushed the LLM to
map EVERY benchmark topic onto a client section, so genuine gaps were never added. Fix: rewrote 1c to
"ADD what's missing; DON'T DUPLICATE what's there" — add a benchmark topic when the client genuinely
lacks it; merge only on a CLEAR match, and then represent it exactly once (source:both). Added an intro
line telling the LLM to understand the policy purpose + current structure and improve toward the benchmark
(not an unchanged copy, not a rewrite). The no-duplicate guarantee still comes from the C guardrail +
topic_id-based safety loop. Re-run reconciliation (re-upload OR the Good-TOC "Regenerate" button) to apply.

## Revision (2026-07-02) — "keep, don't add"
Reported regression: good TOC filled with EMPTY added-benchmark topics, client content buried. Cause:
when multiple benchmark topics map to the same client topic, the rebuild's used-client guard turns the
extras into empty added sections; the 0.82 guardrail was too high to remove them.
- Prompt: added rule 1c "KEEP, DON'T ADD" — only `source:benchmark` when the client has NOTHING
  corresponding; never duplicate a client section as a new benchmark one.
- C guardrail now drops an added topic when a kept client topic already covers it, via
  (a) EXACT same benchmark_match (no embeddings needed — catches the many-to-one empties), or
  (b) semantic match. Client topics are never dropped → no client content lost.
- Thresholds aligned: hint (A) and merge (C) both 0.78, so nothing hinted can survive as an empty add.

## Bugfix (2026-07-02) — subtopic content collapses onto the parent topic [EXTRACTION]
Root cause: `_map_content_to_toc_entries` located each heading by scanning for its FIRST match from a
fixed start. With repeated subtopic titles ("Policy Objective"/"Scope"/... under every topic), every
occurrence resolved to the FIRST one in the doc, so later topics' subtopics never got their own body
position and all their content collapsed onto the parent topic (subtopics ended up empty). Fix: locate
headings SEQUENTIALLY with a moving cursor in TOC order → repeated titles map to successive occurrences,
each subtopic lands under its own parent. Parse-time fix → re-upload the doc to re-extract.
RedistributeButton `statusOf`: now ORIGIN-based for the 0-source case — a client-origin section with no
mapped content is "kept" (it's from the client, just empty here); only benchmark-added or brand-new
sections are "added". (1 source → kept, ≥2 → merged.) The earlier "everything kept / content gone" was
the extraction bug above, not this labeling — with extraction fixed, origin-based is correct.

## Bugfix (2026-07-02) — "Map content" dialog mislabels client sections "added" (superseded/reverted)
The RedistributeButton ("Map client content → sections") computed added/kept/merged purely from the
number of client CONTENT SOURCES mapped to a section (`statusOf`: 0→added, 1→kept, ≥2→merged). Client
subtopics whose text lives under the parent (or wasn't extracted as a separate source) map 0 sources →
shown "added", even though they're client-origin. Fix: label by ORIGIN. Backend `propose_redistribution`
now tags each target with `added_from_benchmark`; frontend `statusOf` treats a 0-source section as "kept"
when it's a client-origin section, "added" only when it's benchmark-added or a brand-new section.
NOTE: the `redistribute_content` prompt is NOT involved — this was label logic, not the LLM mapping.

## Bugfix (2026-07-02) — client sections mislabeled "added" (Review panel)
Review panel decided kept/added by baseline_content presence, so a client section with empty baseline
(e.g. a parent topic whose content is in its subtopics) showed as "added". Fixed: provenance now uses
the `added_from_benchmark` flag (origin), not content presence. Added `added_from_benchmark` to
TOCSubtopic (default False) + the TS type; panel `provOf` now reads the flag.

## Bugfix (2026-07-02) — merged topic shown TWICE
`used_client` tracked reused client topics by TITLE, but a "both" mapping renames `existing.topic` to the
benchmark title; the preserve-mode safety loop then compared the RENAMED title (not in used_client) and
re-appended the same client topic → duplicate. Fixed: track reused topics by stable `topic_id` (both the
reuse guard and the safety loop), so a reused-and-renamed client topic is placed exactly once.

## Over-merge safety
- C only merges an `added_from_benchmark` topic INTO a client topic — never two client topics, never
  two benchmark topics. Preserve mode keeps all client topics.
- Conservative threshold + requires title+content embedding similarity.
- The reviewer sees every merge (`reconcile_note`) in the 3-column TOC and can undo before Approve.

## Verification
1. Client "Fixed Assets" + benchmark "PPE" (different titles, same content) → ONE topic (client's), with
   `benchmark_match` = the PPE benchmark section, not two topics.
2. Genuinely distinct benchmark section the client lacks → still added (below threshold).
3. No vector store available → behaves exactly as today (LLM/title-only), no crash.
4. reconcile_note explains each merge; reviewer can split/edit in the TOC tab.
