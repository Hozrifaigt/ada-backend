# D1 fix — retrieve the RIGHT regulations in the Review tab's Regulation step

## Problem (confirmed in code)
The per-section **Regulation** step does not actually search the uploaded regulation. Two paths exist and both are wrong:

1. `draft_service._regulation_chunks_for_unit` → `generation_service._search_relevant_regulations`, which **returns `[]` unless `draft.metadata.regulations` is set** — and that field is the *name of a prebuilt library index*, not the file the consultant uploaded.
2. The file the consultant uploaded is stored in `draft.metadata.uploaded_regulations_text` and is **dumped whole** into the prompt by `generation_service._regulations_context` (and the content-gen prompts). For a long decree this overflows / gets truncated, and nothing is *retrieved* — the model just gets a wall of text or nothing.

Net effect: for a typical review draft (uploaded reg, no library name), the Regulation step either dumps the whole text or shows "no applicable regulation" for every section — indistinguishable from "checked, nothing applies."

## Fix = real draft-scoped retrieval over the uploaded text (chunk → embed → top-k), plus an honest empty-state.

Data is already in place (`uploaded_regulations_text` set at upload). No new storage, no Azure index. Embeddings computed at retrieval time, cached in-process by text hash. `embedder.embed_documents` (batch) already exists.

### Backend
- [ ] `generation_service.py`: add module-level `_cosine(a,b)` + in-process cache `_UPLOADED_REG_EMBED_CACHE: {md5(text): [vectors]}` (bounded, ~8 entries).
- [ ] `generation_service.py`: add `_chunk_regulation_text(text, target_chars=1200, overlap_chars=150)` — split on blank lines / article markers, greedily pack to ~target_chars with small overlap.
- [ ] `generation_service.py`: add `async search_uploaded_regulations(query_text, uploaded_text, vector_store, top_k=5, min_score=0.0) -> List[RegulationChunk]` — chunk, batch-embed (cached), embed query, cosine top-k, wrap each hit as a `RegulationChunk` (decree_name="Uploaded Regulation", article_number=f"Excerpt {rank}", score).
- [ ] `draft_service._regulation_chunks_for_unit`: keep the library search, AND when `uploaded_regulations_text` is present, append `search_uploaded_regulations(query, ...)` results. Query = `unit.topic` + first ~600 chars of the section's current/baseline content (better recall than title alone).
- [ ] `generation_service._regulations_context`: **stop appending the whole `uploaded_regulations_text`** — return only `_format_regulations_for_prompt(regulation_chunks)`. The relevant uploaded passages now arrive as chunks. (Scope: this only affects the review pipeline — `review_regulation` + `review_chat`.)
- [ ] `draft_service.check_regulation_for_unit`: add `has_regulations` to the response = `bool(regulations name OR uploaded_regulations_text)`, so the UI can tell "none loaded" from "checked, none apply."

### Frontend
- [ ] `draftService.ts` `reviewRegulation` return type: add `has_regulations: boolean`.
- [ ] `ReviewPipelinePanel.tsx` Regulation step: if `has_regulations === false` → distinct "No regulations loaded for this draft" state (not "no applicable regulation"); else keep the applies/extract/suggestion rendering.

## Out of scope for this pass (flag, don't do unless agreed)
- Content-generation flow still dumps `uploaded_regulations_text` whole in `_adapt_existing_content` / `generate_gap_report` / `apply_gap_to_content` / `consolidate_theme_content`. Same retrieval helper can replace those later (Decision A below).
- A **re-upload regulations** endpoint (add/replace regs on an existing review draft). Only needed if a draft has NO `uploaded_regulations_text` yet (Decision B below).

## Decisions (CONFIRMED with user 2026-06-29)
- **A. Scope = Review step only.** Content-gen/tweak prompts keep their current behavior this pass.
- **B. Re-upload = YES.** Reviewers can retrieve from regs uploaded at draft creation AND re-upload new regulation files in the Review tab (under "Regulations"). Need a re-upload endpoint + a Review-tab upload control. Re-upload APPENDS to `uploaded_regulations_text` (reuse `DocumentProcessor().process().full_text`, `.docx` only, `[filename]` header — same as initial upload).
- **C. Query = section title + FULL content** (current content if any, else baseline; not truncated, modulo a safety cap to stay under the embedding token limit).

### Re-upload (new)
- [ ] `draft_service.add_uploaded_regulations(draft_id, files)` — extract .docx full_text, append to `uploaded_regulations_text`, store raw files in blob (parity), save. Returns `{has_regulations, added, message}`.
- [ ] `api/drafts.py`: `POST /drafts/{draft_id}/regulations/upload` (multipart .docx) near the review endpoints.
- [ ] `draftService.ts`: `uploadRegulations(draftId, files)` (FormData, long timeout).
- [ ] `ReviewPipelinePanel.tsx` Regulation step: an "Upload regulations" control (draft-scoped); when none loaded, the empty-state offers it.

> Note (out of scope, flagged): appending reg text to metadata grows the Azure table entity (G1 1MB risk). Same risk the initial upload already has — handle under G1.

## Verification
1. Review draft with a long uploaded reg → Regulation step returns a short, *relevant* extract (not the whole text), per section.
2. Draft with NO regulations → step shows "No regulations loaded," not "no applicable regulation."
3. Token usage per Regulation click is bounded regardless of reg length.
4. Non-review (content-gen) drafts unchanged.

---

## Review (changes made)

**Backend**
- `generation_service.py`: added module-level `_cosine` + bounded process cache `_UPLOADED_REG_EMBED_CACHE`; new `_chunk_regulation_text` (blank-line/article packing + overlap + hard-split), `_embed_reg_chunks` (cached batch embed), and `search_uploaded_regulations(query, uploaded_text, vector_store, top_k=5)` returning the top-k relevant passages as `RegulationChunk`s. Rewrote `_regulations_context` to **stop dumping the whole uploaded text** — it now only formats the (now-retrieved) chunks. Scope: this only affects the review pipeline (`review_regulation`, `review_chat`).
- `draft_service.py`: `_regulation_chunks_for_unit` now merges library-index hits **and** `search_uploaded_regulations` (query = section title + full content, current else baseline). `check_regulation_for_unit` returns `has_regulations`. New `add_uploaded_regulations(draft_id, files)` — extract .docx full_text, append to `uploaded_regulations_text`, store raw files in blob, save.
- `api/drafts.py`: new `POST /{draft_id}/regulations/upload` (multipart .docx), audited.

**Frontend**
- `draftService.ts`: `reviewRegulation` return type gains `has_regulations`; new `uploadRegulations(draftId, files)` (FormData).
- `ReviewPipelinePanel.tsx`: Regulation step gets an "Upload regulations" button (hidden .docx multi-input) + success line; empty-state now distinguishes **"No regulations loaded"** (has_regulations=false) from **"No applicable regulation found"** (checked, none apply). Added `UploadFile` icon, `regsMsg` state, `regFileRef`, `uploadRegs` handler.

**Side effects / flags**
- `_regulations_context` no longer appends the whole uploaded text → review regulation/chat prompts are now retrieval-only (intended). Content-gen/tweak prompts still dump whole text (out of scope this pass).
- Re-upload appends reg text to draft metadata → grows the Azure table entity (G1 1MB risk; pre-existing, tracked separately).

**Follow-ups**
- Optional: apply the same retrieval to the content-gen/tweak prompts (Decision A "Review + content-gen").
- Optional: surface the list of attached regulation filenames in the Review tab.
- Cannot run backend/frontend here (no shell) — user rebuilds containers to test.
