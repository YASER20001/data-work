# backend/agents/agent_therapist.py

from __future__ import annotations
import re
import time
from typing import Dict, List, Any

from backend.core.state import AppState, CaseNotes
from backend.core.llm_gateway import json_out
from backend.core.utils import last_user_text, guess_language_from_messages, serialize_toon, \
    history_window  # <--- IMPORTED


# =============================================================================
# PART 1: CASE NOTES HELPERS
# =============================================================================

def _as_notes_dict(cn: Any) -> Dict[str, Any]:
    """Ensures case notes are in a dictionary format with required keys."""
    if isinstance(cn, CaseNotes):
        d = cn.model_dump()
    elif isinstance(cn, dict):
        d = dict(cn)
    else:
        d = {}

    d.setdefault("concerns", [])
    d.setdefault("emotions", [])
    d.setdefault("risks", [])
    d.setdefault("actions_now", [])
    d.setdefault("next_24h", [])

    cp = d.get("coping_plan", {})
    if isinstance(cp, list):
        cp = {"now": list(cp), "next_24h": []}
    elif not isinstance(cp, dict):
        cp = {}
    cp.setdefault("now", [])
    cp.setdefault("next_24h", [])
    d["coping_plan"] = cp
    return d


def _merge_lists(a: List[str], b: List[str]) -> List[str]:
    """Combines two lists of strings while maintaining uniqueness."""
    out, seen = [], set()
    for x in (a or []):
        if isinstance(x, str) and x not in seen:
            seen.add(x);
            out.append(x)
    for x in (b or []):
        if isinstance(x, str) and x not in seen:
            seen.add(x);
            out.append(x)
    return out


def _merge_case_notes(old_any: Any, patch_any: Any) -> CaseNotes:
    """Consolidates old notes with new updates into a CaseNotes object."""
    old = _as_notes_dict(old_any)
    patch = _as_notes_dict(patch_any)

    merged = {
        "concerns": _merge_lists(old.get("concerns", []), patch.get("concerns", [])),
        "emotions": _merge_lists(old.get("emotions", []), patch.get("emotions", [])),
        "risks": _merge_lists(old.get("risks", []), patch.get("risks", [])),
        "actions_now": _merge_lists(old.get("actions_now", []), patch.get("actions_now", [])),
        "next_24h": _merge_lists(old.get("next_24h", []), patch.get("next_24h", [])),
        "coping_plan": {
            "now": _merge_lists(old["coping_plan"].get("now", []),
                                patch["coping_plan"].get("now", [])),
            "next_24h": _merge_lists(old["coping_plan"].get("next_24h", []),
                                     patch["coping_plan"].get("next_24h", [])),
        },
    }

    return CaseNotes(
        concerns=merged["concerns"],
        emotions=merged["emotions"],
        risks=merged["risks"],
        actions_now=merged["actions_now"],
        next_24h=merged["next_24h"],
        coping_plan=merged["coping_plan"],
    )


# =============================================================================
# PART 2: RAG UTILITIES
# =============================================================================

def _get_therapist_rag_snippets(rag_pipeline, user_text: str, k: int = 5) -> str:
    """Fetches relevant clinical knowledge snippets from the vector database."""
    if not rag_pipeline:
        return "(Therapist RAG offline.)"

    if not isinstance(user_text, str) or not user_text.strip():
        return "(No user text for RAG.)"

    try:
        results = rag_pipeline.search_therapist(user_text, k=k)
        snippets: List[str] = []
        for meta in results:
            if not meta or meta.get("source") == "error":
                continue
            txt = (meta.get("text") or "").replace("\n", " ").strip()
            tag = meta.get("tag") or meta.get("source") or "example"
            if txt:
                snippets.append(f"- [{tag}] {txt[:350]}")
        return "\n".join(snippets) if snippets else "(No therapist RAG matches found.)"
    except Exception as e:
        print(f"[Therapist RAG] query failed: {e}")
        return "(Therapist RAG error.)"


# =============================================================================
# PART 3: PROMPTS AND TEMPLATES
# =============================================================================

SYSTEM = (
    "You are a bilingual (Arabic/English) trauma-informed well-being coach for domestic abuse. "
    "Listen carefully to the whole situation, validate the user's reality, and offer light psycho-education "
    "and simple coping options when helpful. Use the contextual signals you are given (risk, history, "
    "personality, case notes, language, time since last contact) to decide what is most relevant. "
    "When the user seems overwhelmed, dissociated, or heavily activated based on the overall meaning of what "
    "they say (not specific trigger words), you may briefly suggest grounding or breathing exercises as options, "
    "never as commands. "
    "Do NOT provide diagnoses, medical advice, or definitive outcomes. "
    "Keep the user safe, respected, and in control of choices. "
    "### SUPERVISOR FEEDBACK RULES (CRITICAL) ### "
    "If you receive input in 'SUPERVISOR FEEDBACK': "
    "1. **Check Relevance:** Does this legal info answer the User's *current* specific question? "
    "   - If NO (it's off-topic), IGNORE the legal feedback to avoid confusion. "
    "   - If YES, you MUST include it. "
    "2. **How to Cite:** Do NOT just list Article numbers. You must explain the **ruling** or **content** of the law naturally. "
    "   (e.g., 'Regarding your question, Article 126 states that custody priority generally goes to the mother first...'). "
    "Return STRICT JSON only."
)

TEMPLATE = """
Language: {lang}

Context snapshot:
- risk_score: {risk_score}
- risk_band: {risk_band}
- immediate_danger: {immediate_danger}
- detected_personality_label: {personality}
- personality_agent_hint: {personality_probe}
- history_window (Last 5 messages): {history}
- action_summary: {action_summary}
- transcribed_user_input: {transcribed_user_input}
- time_since_last_chat: {time_since_last_chat}

*** SUPERVISOR FEEDBACK (From Legal Review) ***
(If populated, you MUST follow these instructions to fix your previous draft)
{supervisor_feedback}
***********************************************

Existing case_notes:
{case_notes}

Therapist knowledge snippets (RAG; DO NOT copy facts verbatim, use only for ideas/psycho-education):
{therapist_rag_snippets}

User says (current focus):
{user_input}

OUTPUT JSON (single object):
{{
  "ai_message": string,                     // empathetic, 3–6 sentences, max one tiny list (<=3 bullets). Language = {lang}.
  "case_notes_patch": {{
    "concerns": [string],
    "emotions": [string],
    "coping_plan": {{
      "now": [string],
      "next_24h": [string]
    }}
  }},
  "follow_up_question": string,             // ≤20 words, open, agency-preserving, {lang}.
  "safety_nudge": string                    // 0–1 short sentence. If (immediate_danger=true OR risk_band in [high,critical] OR risk_score >= 0.80) include; else "".
}}

HARD RULES:
- No diagnoses. No medical advice. No definitive outcomes or guarantees.
- No phone numbers, agencies, or URLs. No instructions that escalate risk.
- Use the context snapshot to decide what to focus on. Do not repeat everything.
- Brief and concrete. Respect autonomy. Cultural sensitivity for KSA.
"""


# =============================================================================
# PART 4: TEXT AND TIME UTILITIES
# =============================================================================

def _trim_words(s: str, n: int) -> str:
    """Truncates a string to a specific word count."""
    if not isinstance(s, str): return ""
    w = s.strip().split()
    return " ".join(w[:n])


def _time_since_last_chat(state: AppState) -> str:
    """Calculates human-readable time elapsed since last activity."""
    last = getattr(state, "last_activity_at", None)
    if not isinstance(last, (int, float)):
        return "unknown"

    try:
        delta = max(0.0, time.time() - float(last))
    except Exception:
        return "unknown"

    days = int(delta // 86400)
    if days > 0:
        return f"{days} day(s) ago"
    hours = int((delta % 86400) // 3600)
    if days == 0 and hours > 0:
        return f"{hours} hour(s) ago"
    minutes = int((delta % 3600) // 60)
    return f"{minutes} minute(s) ago"


def _sanitize_output(obj: Any) -> Dict[str, Any]:
    """Validates and cleans the LLM's JSON output structure."""
    if not isinstance(obj, dict): obj = {}
    ai_message = obj.get("ai_message") or ""
    if not isinstance(ai_message, str): ai_message = str(ai_message)

    patch = obj.get("case_notes_patch") or {}
    if not isinstance(patch, dict): patch = {}

    cp = patch.get("coping_plan") or {}
    if not isinstance(cp, dict): cp = {}
    cp.setdefault("now", [])
    cp.setdefault("next_24h", [])
    if not isinstance(cp["now"], list):      cp["now"] = []
    if not isinstance(cp["next_24h"], list): cp["next_24h"] = []
    patch["coping_plan"] = cp

    for k in ("concerns", "emotions"):
        v = patch.get(k) or []
        patch[k] = [x for x in v if isinstance(x, str)]

    follow = obj.get("follow_up_question") or ""
    if not isinstance(follow, str): follow = str(follow)

    nudge = obj.get("safety_nudge") or ""
    if not isinstance(nudge, str): nudge = str(nudge)

    return {
        "ai_message": ai_message.strip(),
        "case_notes_patch": patch,
        "follow_up_question": follow.strip(),
        "safety_nudge": nudge.strip(),
    }


# =============================================================================
# PART 5: CORE AGENT RUNNER
# =============================================================================

def run(state: AppState) -> Dict[str, Any]:
    """Main execution point for the Therapist Agent."""
    messages: List[Dict[str, str]] = list(getattr(state, "messages", []) or [])

    # Resolve language
    lang = getattr(state, "lang", None)
    if lang not in ("ar", "en"):
        lang = "ar" if guess_language_from_messages(messages) == "ar" else "en"

    # Input aggregation
    raw_user_input = getattr(state, "user_input", "") or ""
    transcribed = getattr(state, "transcribed_text", "") or ""
    user_last_msg = last_user_text(messages)

    user = raw_user_input or transcribed or user_last_msg

    # Context gathering
    risk_score = float(getattr(state, "risk_score", 0.0) or 0.0)
    risk_band = (getattr(state, "risk_band", None) or ("critical" if risk_score >= 0.80 else "low")).lower()
    immediate = bool(getattr(state, "immediate_danger", False))
    personality = getattr(state, "personality_label", None) or ""
    probe = getattr(state, "personality_probe", None) or "(no hint provided)"
    history_str = history_window(messages, n=5)
    cn_snapshot = _as_notes_dict(getattr(state, "case_notes", None))
    action_summary = getattr(state, "action_summary", "") or ""
    if not isinstance(action_summary, str):
        action_summary = str(action_summary)

    time_since_last = _time_since_last_chat(state)

    # RAG Search
    rag_service = getattr(state, "rag_pipeline", None)
    therapist_rag_snippets = _get_therapist_rag_snippets(rag_service, user, k=5)

    print("[Therapist] RAG snippets preview:", therapist_rag_snippets[:200])

    # Feedback Loop Logic
    retry_count = getattr(state, "legal_retry_count", 0)
    prev_reason = getattr(state, "legal_reason", "")

    supervisor_feedback = ""
    if retry_count > 0 and prev_reason:
        supervisor_feedback = (
            f"!!! ALERT: PREVIOUS DRAFT REJECTED BY LEGAL SUPERVISOR !!!\n"
            f"REASON/INSTRUCTION: {prev_reason}\n"
            f"YOU MUST REWRITE THE MESSAGE TO COMPLY WITH THIS INSTRUCTION."
        )
    else:
        supervisor_feedback = "(No feedback - First draft)"

    # LLM Interaction
    prompt = TEMPLATE.format(
        lang=("Arabic" if lang == "ar" else "English"),
        risk_score=f"{risk_score:.2f}",
        risk_band=risk_band,
        immediate_danger=str(immediate).lower(),
        personality=personality,
        personality_probe=probe,
        history=history_str,
        action_summary=action_summary,
        transcribed_user_input=transcribed or "",
        time_since_last_chat=time_since_last,
        case_notes=serialize_toon(cn_snapshot),
        therapist_rag_snippets=therapist_rag_snippets,
        supervisor_feedback=supervisor_feedback,
        user_input=user,
    )

    raw = json_out(SYSTEM, prompt, temperature=0.25) or {}
    data = _sanitize_output(raw)

    ai_msg = data["ai_message"]
    follow_q = data["follow_up_question"]
    nudge = data["safety_nudge"]
    patch = data["case_notes_patch"]

    # Safety gate logic
    if immediate or risk_band in ("critical",) or risk_score >= 0.80:
        if not nudge:
            nudge = (
                "If safety feels uncertain now, take a small step to a safer spot—we can sketch a short safety plan."
                if lang == "en"
                else "لو الأمان الآن مو واضح، خذي خطوة صغيرة لمكان أكثر أمانًا—نقدر نرسم معًا خطة أمان قصيرة."
            )

    if not follow_q:
        follow_q = getattr(state, "personality_probe", None) or ""
    follow_q = _trim_words(follow_q, 20)

    # Message assembly
    final_message = (f"{nudge} " if nudge else "") + ai_msg
    final_message = final_message.strip()
    if follow_q:
        final_message = f"{final_message}\n\n{follow_q}"

    # Clinical record update
    try:
        merged_model = _merge_case_notes(getattr(state, "case_notes", None), patch)
        new_case_notes = merged_model.model_dump()
    except Exception:
        new_case_notes = _as_notes_dict(getattr(state, "case_notes", None))

    # Construct state updates
    updates: Dict[str, Any] = {
        "ai_message_draft": final_message,
        "therapist_reply": final_message,
        "follow_up_question": (follow_q or None),
        "case_notes": new_case_notes,
        "route": "legal_review",
        "phase": "agent",
        "done": False,
    }

    # Metadata cleanup
    updates.pop("session_id", None)
    updates.pop("user_id", None)
    return updates


print("\n[Agent] Loaded Therapist agent.")
