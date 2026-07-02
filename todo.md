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
