"""Shared pure-data helpers for the review-draft export variants (redline / changes overview).
Used by both the Word and PDF builders so the diff semantics can't drift between formats."""
import difflib
from typing import Any, Dict, List, Optional


def norm_key(s: Optional[str]) -> str:
    return " ".join((s or "").strip().lower().split())


def diff_ops(old: str, new: str) -> List[tuple]:
    """Word-level diff → [(op, text)] with op in equal | removed | added."""
    a, b = (old or "").split(), (new or "").split()
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            ops.append(("equal", " ".join(a[i1:i2])))
        else:
            if i2 > i1:
                ops.append(("removed", " ".join(a[i1:i2])))
            if j2 > j1:
                ops.append(("added", " ".join(b[j1:j2])))
    return ops


def review_units(draft_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the working TOC into diffable units: the frozen client baseline vs the reviewed
    content, per topic and subtopic."""
    units = []
    for t in draft_data.get("toc", []) or []:
        units.append({
            "label": f"{t.get('order', '')}. {t.get('topic', '')}", "level": 1,
            "old": (t.get("baseline_content") or "").strip(),
            "new": (t.get("content") or "").strip(),
            "added": bool(t.get("added_from_benchmark")),
            "note": (t.get("reconcile_note") or "").strip(),
        })
        for s in t.get("subtopics") or []:
            units.append({
                "label": f"{t.get('order', '')}.{s.get('order', '')} {s.get('topic', '')}", "level": 2,
                "old": (s.get("baseline_content") or "").strip(),
                "new": (s.get("content") or "").strip(),
                "added": bool(s.get("added_from_benchmark")),
                "note": "",
            })
    return units


def filter_review_draft(draft_data: Dict[str, Any]) -> Dict[str, Any]:
    """For review_mode drafts, keep only banked (review_step == 'closed') sections in an export.
    A topic is kept if it is closed or has any closed subtopic; a NON-closed parent kept only for its
    subtopics emits heading-only (its unreviewed body is blanked). Survivors are renumbered
    sequentially. If no section is closed yet, the draft is returned unchanged (export everything)."""
    metadata = draft_data.get('metadata') or {}
    if not metadata.get('review_mode'):
        return draft_data
    toc = draft_data.get('toc') or []

    def is_closed(unit):
        return (unit or {}).get('review_step') == 'closed'

    any_closed = any(
        is_closed(t) or any(is_closed(s) for s in (t.get('subtopics') or []))
        for t in toc
    )
    if not any_closed:
        return draft_data

    filtered = []
    for t in toc:
        closed_subs = [dict(s) for s in (t.get('subtopics') or []) if is_closed(s)]
        if is_closed(t) or closed_subs:
            kept = dict(t)
            kept['subtopics'] = closed_subs
            if not is_closed(t):
                kept['content'] = ''
                kept['summary'] = ''
            filtered.append(kept)

    for i, t in enumerate(filtered, 1):
        t['order'] = i
        for j, s in enumerate(t['subtopics'], 1):
            s['order'] = j

    out = dict(draft_data)
    out['toc'] = filtered
    return out


def removed_sections(draft_data: Dict[str, Any], original_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Original-snapshot sections with no counterpart (by unit id or normalized title) in the
    final TOC — the sections the review dropped."""
    if not original_data:
        return []
    final_ids, final_titles = set(), set()
    for t in draft_data.get("toc", []) or []:
        final_ids.add(t.get("topic_id"))
        final_titles.add(norm_key(t.get("topic")))
        for s in t.get("subtopics") or []:
            final_ids.add(s.get("subtopic_id"))
            final_titles.add(norm_key(s.get("topic")))
    removed = []
    for t in original_data.get("toc", []) or []:
        if t.get("topic_id") not in final_ids and norm_key(t.get("topic")) not in final_titles:
            removed.append({"title": t.get("topic", ""), "content": (t.get("content") or "").strip()})
        for s in t.get("subtopics") or []:
            if s.get("subtopic_id") not in final_ids and norm_key(s.get("topic")) not in final_titles:
                removed.append({"title": f"{t.get('topic', '')} › {s.get('topic', '')}",
                                "content": (s.get("content") or "").strip()})
    return removed
