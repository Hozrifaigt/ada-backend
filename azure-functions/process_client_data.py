"""Dev tool — batch-run the new DocumentProcessor over a tree of client .docx files and write the
extracted TOC + the full content of each topic/subtopic into a mirror folder, so the new extraction
can be confirmed manually before relying on it.

It mirrors the LIVE upload flow (draft_service.initialize_draft_from_upload): it runs the
deterministic parse, and for low-confidence docs it runs the SAME LLM heading-leveling pass the
backend uses, so you can verify the end-to-end result locally before deploying. For any doc that gets
re-leveled it also writes a `*.deterministic.*` copy so you can diff before/after.

It is SELF-LOCATING: drop this file anywhere (e.g. inside client_data/data) and run it. It finds the
backend `app` package by walking up the directory tree (or via the APP_ROOT env var), then processes
every .docx under its own folder (recursively) and writes results to ./processed_data next to it.

    python process_client_data.py [input_root] [output_root]

Defaults:  input_root = this script's folder    output_root = <input_root>/processed_data
If it can't find `app`, set:  APP_ROOT=/path/to/ada-backend/azure-functions

LLM leveling is ON by default (needs the same Azure OpenAI settings the backend uses). To run a pure
deterministic pass with NO LLM calls, set  USE_LLM=0.
"""
import sys
import os
import glob
import json
import asyncio

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_app_root(start: str):
    """Walk up from `start` looking for a folder containing app/pipelines/document_processor.py."""
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "app", "pipelines", "document_processor.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _bootstrap_app_path():
    for cand in (os.environ.get("APP_ROOT"), _THIS_DIR, os.getcwd()):
        if cand:
            root = _find_app_root(cand)
            if root:
                sys.path.insert(0, root)
                return root
    sys.exit(
        "ERROR: could not locate the backend 'app' package.\n"
        "Run this from inside ada-backend/azure-functions, or set "
        "APP_ROOT=/path/to/ada-backend/azure-functions"
    )


_APP_ROOT = _bootstrap_app_path()
from app.pipelines.document_processor import DocumentProcessor  # noqa: E402

_USE_LLM = os.environ.get("USE_LLM", "1").strip().lower() not in ("0", "false", "no", "off", "")


def _build_llm():
    """Build the same generation LLM the live upload flow uses, or (None, None) if disabled/unavailable."""
    if not _USE_LLM:
        print("USE_LLM=0 → deterministic only (no LLM leveling).")
        return None, None
    try:
        from app.services.generation_service import ContentGenerationService
        from app.pipelines.components.llm import AzureLLM
        from app.prompts.manager import PromptManager
        from app.config.settings import settings
        llm = AzureLLM(PromptManager(settings.PROMPT_MANAGER_PATH))
        gen = ContentGenerationService()  # reflection agent not needed for leveling
        print("LLM leveling ENABLED — low-confidence docs get re-leveled exactly like the upload flow.")
        return gen, llm
    except Exception as e:
        print(f"WARNING: could not init the LLM ({e}).\n         Running deterministic only. Set USE_LLM=0 to silence this.")
        return None, None


class _RecordingLLM:
    """Wraps the real LLM to capture the EXACT prompt sent and response received (dev debug only)."""

    def __init__(self, inner):
        self._inner = inner
        self.last_prompt = None
        self.last_response = None

    async def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.last_response = await self._inner.generate(prompt)
        return self.last_response


async def process_one(path: str, gen=None, llm=None):
    """Run the deterministic parse, then the gated LLM leveling (when configured and the doc is
    low-confidence). Returns (deterministic_result, final_result, leveled, debug). `final` ==
    deterministic unless leveling ran and produced topics — same decision as draft_service. `debug`
    is a dict (or None) capturing exactly what was sent to / returned from the LLM."""
    proc = DocumentProcessor()
    with open(path, "rb") as f:
        # keep_empty_sections=True == the review/upload behaviour (empty sections kept as gaps)
        det = proc.process(f, keep_empty_sections=True)
    conf = (det.get("metadata") or {}).get("confidence")
    if not (gen and llm) or conf in ("high", "empty", None):
        return det, det, False, None

    doc_context = proc.toc_context_for_llm(300)
    hints = proc.get_heading_hints()
    rec = _RecordingLLM(llm)
    debug = {"confidence": conf, "first_words": doc_context, "hints": hints}
    try:
        updates, drops = await gen.infer_heading_levels(doc_context, hints, rec)
        debug.update({
            "prompt": rec.last_prompt,
            "response": rec.last_response,
            "updates": dict(updates),
            "drops": sorted(drops),
        })
        if updates or drops:
            rebuilt = proc.rebuild_with_levels(updates, drops)
            if rebuilt.get("topics"):  # guard: don't let a degenerate result wipe the TOC
                return det, rebuilt, True, debug
    except Exception as e:
        debug["error"] = str(e)
        print(f"    LLM leveling failed ({os.path.basename(path)}): {e}")
    return det, det, False, debug


def _wc(text) -> int:
    return len((text or "").split())


def _subtopic_count(res: dict) -> int:
    return sum(len(t.get("subtopics", [])) for t in res.get("topics", []))


def render_txt(name: str, res: dict, leveled: bool = False) -> str:
    topics = res.get("topics", [])
    meta = res.get("metadata", {})
    tag = "  [LLM-RELEVELED]" if leveled else ""
    out = [
        f"FILE: {name}{tag}",
        f"topics={len(topics)}  subtopics={_subtopic_count(res)}  "
        f"confidence={meta.get('confidence')}  weak_ratio={meta.get('weak_ratio')}  "
        f"extraction_method={meta.get('extraction_method')}",
        "=" * 80,
        "",
        "[TABLE OF CONTENTS]",
    ]
    for i, t in enumerate(topics, 1):
        out.append(f"{i}. {t.get('topic')}")
        for j, s in enumerate(t.get("subtopics", []), 1):
            out.append(f"   {i}.{j} {s.get('name')}")

    out += ["", "=" * 80, "[CONTENT]"]
    for i, t in enumerate(topics, 1):
        out += ["", "-" * 80, f"# {i}. {t.get('topic')}  ({_wc(t.get('topic_content'))} words)",
                (t.get("topic_content") or "").strip() or "(no content)"]
        for j, s in enumerate(t.get("subtopics", []), 1):
            out += ["", f"## {i}.{j} {s.get('name')}  ({_wc(s.get('content'))} words)",
                    (s.get("content") or "").strip() or "(no content)"]
    return "\n".join(out)


def render_llm_debug(name: str, debug: dict) -> str:
    """Full transparency dump: what we sent to the LLM and what it returned, per heading."""
    fw = debug.get("first_words", "")
    hints = debug.get("hints", [])
    updates = debug.get("updates", {})
    drops = set(debug.get("drops", []))

    # What the LLM itself proposed (before our guard), parsed from its raw response.
    proposed = {}
    try:
        data = json.loads((debug.get("response") or "").strip()
                          [(debug.get("response") or "").find("{"):])
    except Exception:
        data = {}
    for it in (data.get("levels", []) if isinstance(data, dict) else []):
        try:
            proposed[int(it.get("index"))] = (it.get("level"), it.get("is_heading"))
        except (TypeError, ValueError):
            pass

    out = [
        f"FILE: {name}   (confidence={debug.get('confidence')})",
        "=" * 80,
        f"[1] DOCUMENT START SENT TO LLM  (first ~300 words; line breaks + TOC indentation preserved)",
        "-" * 80,
        fw or "(empty)",
        "",
        f"[2] HEADING LIST SENT TO LLM  ({len(hints)} headings)",
        "-" * 80,
    ]
    for h in hints:
        out.append(f"{h['index']}: [{h.get('hint','')}] {h['title']}  (current_level {h['level']})")

    out += ["", "[3] RAW LLM RESPONSE", "-" * 80, (debug.get("response") or "(none)"), ""]
    if debug.get("error"):
        out += [f"[!] LLM ERROR: {debug['error']}", ""]

    out += ["[4] PER-HEADING: current -> LLM-proposed -> APPLIED (after guard)", "-" * 80]
    for h in hints:
        i = h["index"]
        cur = h["level"]
        prop_lvl, prop_is_h = proposed.get(i, (None, None))
        if i in drops:
            applied = "DROPPED (not a heading)"
        elif i in updates:
            applied = f"level {updates[i]}" + (" (changed)" if updates[i] != cur else "")
        else:
            blocked = prop_lvl is not None and prop_lvl != cur
            applied = "kept current" + ("  <-- LLM proposed a change, GUARD BLOCKED it" if blocked else "")
        prop_str = f"lvl={prop_lvl} is_heading={prop_is_h}" if i in proposed else "(no entry returned)"
        out.append(f"{i}: [{h.get('hint','')}] cur={cur}  | LLM: {prop_str}  | -> {applied}")

    out += ["", "[5] NET CHANGES", "-" * 80,
            f"applied level updates: {updates or '{}'}",
            f"dropped headings: {sorted(drops) or '[]'}"]
    return "\n".join(out)


def _debug_counts(debug: dict):
    """(llm_proposed_changes, applied_changes) — to spot when the guard is eating everything."""
    if not debug:
        return "", ""
    hints = {h["index"]: h["level"] for h in debug.get("hints", [])}
    proposed = 0
    try:
        resp = (debug.get("response") or "").strip()
        data = json.loads(resp[resp.find("{"):])
        for it in data.get("levels", []):
            idx, lvl, is_h = it.get("index"), it.get("level"), it.get("is_heading")
            if is_h is False or (isinstance(lvl, int) and lvl != hints.get(idx)):
                proposed += 1
    except Exception:
        proposed = ""
    applied = len(debug.get("updates", {})) + len(debug.get("drops", []))
    return proposed, applied


def _json_payload(res: dict) -> dict:
    """TOC structure + full topic/subtopic content + metadata."""
    return {"toc": res.get("toc"), "topics": res.get("topics"), "metadata": res.get("metadata")}


def _write_outputs(base_path: str, name: str, res: dict, leveled: bool) -> None:
    with open(base_path + ".txt", "w", encoding="utf-8") as f:
        f.write(render_txt(name, res, leveled))
    with open(base_path + ".json", "w", encoding="utf-8") as f:
        json.dump(_json_payload(res), f, ensure_ascii=False, indent=2)


async def main() -> None:
    in_root = sys.argv[1] if len(sys.argv) > 1 else _THIS_DIR
    out_root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(in_root, "processed_data")
    out_root_abs = os.path.abspath(out_root)

    paths = [
        p for p in sorted(glob.glob(os.path.join(in_root, "**", "*.docx"), recursive=True))
        if not os.path.basename(p).startswith("~$")              # skip Word temp/lock files
        and not os.path.abspath(p).startswith(out_root_abs)      # don't reprocess prior output
    ]
    print(f"app package: {_APP_ROOT}")
    gen, llm = _build_llm()
    print(f"Found {len(paths)} .docx under '{in_root}'")

    ok = fail = 0
    rows = []  # confidence summary, one row per file
    cols = ["file", "confidence", "reason", "weak_ratio", "top_ratio", "leveled",
            "llm_proposed", "llm_applied",
            "topics_before", "subtopics_before", "topics_after", "subtopics_after",
            "headings_total", "reliable", "heuristic", "mixed", "levels", "method"]
    for path in paths:
        rel = os.path.relpath(path, in_root)
        base = os.path.splitext(rel)[0]
        out_base = os.path.join(out_root, base)
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        try:
            det, final, leveled, debug = await process_one(path, gen, llm)
            # Final (post-LLM, if any) result is the canonical .txt/.json
            _write_outputs(out_base, os.path.basename(path), final, leveled)
            # When leveling changed things, also keep the deterministic version for before/after diff
            if leveled:
                _write_outputs(out_base + ".deterministic", os.path.basename(path), det, False)
            # Whenever the LLM was actually invoked, dump exactly what was sent / returned.
            if debug is not None:
                with open(out_base + ".llm_debug.txt", "w", encoding="utf-8") as f:
                    f.write(render_llm_debug(os.path.basename(path), debug))

            m = final.get("metadata", {})
            dm = det.get("metadata", {})
            llm_proposed, llm_applied = _debug_counts(debug)
            rows.append({
                "file": rel,
                "confidence": dm.get("confidence"),
                "reason": dm.get("low_reason"),
                "weak_ratio": dm.get("weak_ratio"),
                "top_ratio": dm.get("top_level_ratio"),
                "leveled": "yes" if leveled else "",
                "llm_proposed": llm_proposed,
                "llm_applied": llm_applied,
                "topics_before": len(det.get("topics", [])),
                "subtopics_before": _subtopic_count(det),
                "topics_after": len(final.get("topics", [])),
                "subtopics_after": _subtopic_count(final),
                "headings_total": dm.get("headings_total"),
                "reliable": dm.get("headings_reliable"),
                "heuristic": dm.get("headings_heuristic"),
                "mixed": dm.get("mixed_signals"),
                "levels": "/".join(str(x) for x in (dm.get("levels_present") or [])),
                "method": m.get("extraction_method"),
            })
            ok += 1
            flag = " -> LLM" if leveled else ""
            print(f"  ok  {rel}  (conf={dm.get('confidence')}, weak={dm.get('weak_ratio')}, "
                  f"{len(final.get('topics', []))} topics{flag})")
        except Exception as e:
            fail += 1
            rows.append({"file": rel, "confidence": "ERROR", "method": str(e)[:60]})
            with open(out_base + ".txt", "w", encoding="utf-8") as f:
                f.write(f"FAILED: {e}")
            print(f"  FAIL {rel}  -> {e}")

    # Write a single scannable summary (tab-separated) + print it.
    summary_path = os.path.join(out_root, "_confidence_summary.tsv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print("\n==== CONFIDENCE SUMMARY ====")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    print(f"\nDone. ok={ok} fail={fail}. Output + {os.path.basename(summary_path)} under '{out_root}'.")


if __name__ == "__main__":
    asyncio.run(main())
