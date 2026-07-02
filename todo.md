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
