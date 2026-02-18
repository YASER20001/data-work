# backend/agents/agent_translation.py

from __future__ import annotations
import json
import os
import regex as re
from typing import Dict, List, Any, Tuple

from backend.core.llm_gateway import json_out
from backend.core.state import AppState
from backend.core.utils import serialize_toon, history_window  # <--- IMPORTED UTILS
from backend.core.env_config import flag as env_flag

# =============================================================================
# PART 1: TONE & PERSONA DEFINITIONS
# =============================================================================

# System persona for converting clinical drafts into humanized language
SYSTEM = (
    "You are a compassionate Saudi language specialist and therapist. Your task is to rewrite the input text "
    "into the target language using a highly natural, human persona:\n\n"
    "1. **English**: Use a **'Friendly, Human Therapist'** tone. \n"
    "   - Be warm, tangible, and sensible. Do NOT sound like a robotic AI.\n"
    "   - Use contractions (e.g., 'I'm' instead of 'I am') and natural conversational flow.\n"
    "   - Show genuine care without being overly formal.\n\n"
    "2. **Arabic**: Use **'Natural Saudi Dialect' (اللهجة السعودية الطبيعية)**. \n"
    "   - **STRICTLY FORBIDDEN**: Do NOT use Modern Standard Arabic (Fusha) or robotic AI phrasing (e.g., 'أنا نموذج ذكاء اصطناعي', 'أتفهم مشاعرك').\n"
    "   - **REQUIRED**: Speak like a supportive local Saudi sister/friend. Use white Saudi dialect (اللهجة البيضاء).\n"
    "   - Use natural fillers and compassionate phrases: 'يا قلبي'، 'الله يعينك'، 'أنا معك'، 'سلامتك'، 'عسى ما شر'، 'ما عليك باس'.\n"
    "   - Make the text feel 'human' and 'warm'.\n\n"
    "HARD RULES: \n"
    "- Do NOT add new facts, medical advice, or promises.\n"
    "- **Do NOT change the original meaning or advice.** Your job is ONLY tone/dialect conversion.\n"
    "- Preserve all numbers/links/citations.\n"
    "- Return STRICT JSON only."
)

TEMPLATE = """
Target Language: {lang}
Risk Level: {risk_band}
Personality: {personality}
Glossary:
{glossary}

SOURCE TEXT (Preserve specific details like URLs and numbers):
{source}

CONTEXT (For tone reference only):
{history}

OUTPUT JSON (single object):
{{
  "ai_message": string  // The final rewritten message in {lang}.
}}
"""

RETRY_TEMPLATE = """
The previous output was incorrect. Rewrite the following text into {lang}.
- If English: Use 'Friendly, Human' tone.
- If Arabic: Use 'Natural Saudi Dialect' (avoid Fusha).

Text:
{text}

OUTPUT JSON ONLY:
{{
  "ai_message": string
}}
"""

# =============================================================================
# PART 2: TEXT PROCESSING HELPERS (MARKERS & REGEX)
# =============================================================================

_AR_CHARS = re.compile(r"[\u0600-\u06FF]")
_URL = re.compile(r"https?://\S+")
_PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
_CITATION = re.compile(r"\[\d+\]")
_NUMERIC = re.compile(r"(?<!\w)(?:\d{1,4}(?:[.,]\d{3})*|\d+)(?!\w)")
_JSON_OBJ = re.compile(r"\{(?:[^{}]|(?R))*\}")


def _source_draft(state: AppState) -> str:
    """Retrieves the most recent draft content from state or history."""
    draft = (
        getattr(state, "ai_message_reviewed", None)
        or getattr(state, "ai_message_draft", None)
        or getattr(state, "ai_message", None)
        or ""
    ).strip()
    if draft:
        return draft
    for m in reversed(getattr(state, "messages", []) or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _coerce_json(obj: Any) -> Dict[str, Any]:
    """Ensures LLM output is correctly parsed into a JSON dictionary."""
    if isinstance(obj, dict): return obj
    s = str(obj).strip()
    if not s: return {}
    if s.startswith("```"):
        s = s.strip("` \n")
        s = re.sub(r"^\s*(json|JSON)\s*", "", s)
    m = _JSON_OBJ.search(s)
    if m: s = m.group(0)
    s = re.sub(r"(?P<key>\"[^\"]+\"\s*:\s*)'([^']*)'", r'\g<key>"\2"', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s)
    except Exception:
        return {}


def _extract_markers(text: str) -> Dict[str, List[str]]:
    """Finds URLs, placeholders, citations, and numbers to ensure integrity."""
    return {
        "urls": _URL.findall(text or ""),
        "placeholders": _PLACEHOLDER.findall(text or ""),
        "citations": _CITATION.findall(text or ""),
        "numbers": _NUMERIC.findall(text or ""),
    }


def _ensure_markers(final_msg: str, source: str) -> str:
    """Checks that all functional tokens from the source exist in the output."""
    f = final_msg or ""
    src_m = _extract_markers(source)
    dst_m = _extract_markers(f)
    missing: List[str] = []

    for key in ("urls", "placeholders", "citations"):
        for token in src_m[key]:
            if token not in dst_m[key]:
                missing.append(token)
    for n in set(src_m["numbers"]):
        if n and n not in dst_m["numbers"]:
            missing.append(n)

    if missing:
        tail = " " + " ".join(missing)
        f = (f + tail).strip()
    return f


# =============================================================================
# PART 3: GLOSSARY & LANGUAGE SANITIZATION
# =============================================================================

_PREF_GLOSSARY: List[Tuple[str, str]] = [
    (r"\bdomestic\s+violence\b", "domestic abuse"),
    (r"\bvictim\b", "survivor"),
    (r"\bwhy\s+did\s+you\b", "it makes sense that you"),
    (r"\bالعنف\s+المنزلي\b", "العنف الأسري"),
    (r"\bضحية\b", "ناجية"),
]


def _apply_glossary(text: str, glossary_items: List[Dict[str, str]] | List[Tuple[str, str]]) -> str:
    """Replaces restricted clinical terms with preferred user-friendly terms."""
    s = text or ""
    for item in (glossary_items or []):
        if isinstance(item, dict):
            src = (item.get("source") or "").strip()
            pref = (item.get("preferred") or "").strip()
            if src and pref:
                s = re.sub(rf"(?i)(?<!\w){re.escape(src)}(?!\w)", pref, s)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pat, repl = item
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    for pat, repl in _PREF_GLOSSARY:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _ensure_language(final_msg: str, target_lang: str) -> str:
    """Appends directionality markers if needed for mixed RTL/LTR environments."""
    if not final_msg: return final_msg
    is_ar = bool(_AR_CHARS.search(final_msg))
    if target_lang == "ar" and not is_ar: return "\u200F" + final_msg
    if target_lang == "en" and is_ar: return "\u200E" + final_msg
    return final_msg


def _clip_sentences(s: str, max_sentences: int = 7, max_chars: int = 900) -> str:
    """Limits message length for readability."""
    if not s: return s
    parts = re.split(r'(?<=[.!؟\?])\s+', s.strip())
    s = " ".join(parts[:max_sentences]).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def _language_mismatch(text: str, target: str) -> bool:
    """Detects if the LLM generated output in the wrong script."""
    if not text: return False
    is_ar = bool(_AR_CHARS.search(text))
    return (target == "ar" and not is_ar) or (target == "en" and is_ar)


# =============================================================================
# PART 4: MAIN AGENT ENTRYPOINT (RUN)
# =============================================================================

def run(state: AppState) -> Dict[str, Any]:
    """Primary logic for the translation and humanization agent."""
    # 1. Gather Context
    lang_code = getattr(state, "lang", "en")
    risk_band = (getattr(state, "risk_band", "") or "low").lower()
    personality = (
        getattr(state, "personality_label", None)
        or getattr(state, "personality", "")
        or ""
    )
    glossary = getattr(state, "glossary", []) or []
    source = _source_draft(state)

    msgs = getattr(state, "messages", []) or []
    history = history_window(msgs, n=5)

    # 2. Prepare Prompt
    prompt = TEMPLATE.format(
        lang=("Arabic" if lang_code == "ar" else "English"),
        risk_band=risk_band,
        personality=personality,
        glossary=serialize_toon(glossary),
        source=source,
        history=history,
    )

    # 3. Call Gateway
    try:
        raw_dict = json_out(SYSTEM, prompt, temperature=0.3)
    except Exception:
        raw_dict = {}

    final_msg = (raw_dict.get("ai_message") or "").strip()

    # 4. Fallbacks & Post-processing
    if not final_msg:
        final_msg = source

    final_msg = _ensure_markers(final_msg, source)
    final_msg = _apply_glossary(final_msg, glossary)

    # 5. Retry Logic for Incorrect Script
    if _language_mismatch(final_msg, lang_code):
        retry_prompt = RETRY_TEMPLATE.format(
            lang=("Arabic" if lang_code == "ar" else "English"),
            text=final_msg or source
        )
        try:
            retry_dict = json_out(SYSTEM, retry_prompt, temperature=0.0)
            retry_msg = (retry_dict.get("ai_message") or "").strip()
            if retry_msg:
                retry_msg = _ensure_markers(retry_msg, source)
                retry_msg = _apply_glossary(retry_msg, glossary)
                final_msg = retry_msg
        except Exception:
            pass

    # 6. Final Polish & Formatting
    final_msg = _ensure_language(final_msg, lang_code)
    final_msg = _clip_sentences(final_msg, max_sentences=7, max_chars=900)

    if env_flag("TERMINOLOGY_SANITIZE", True):
        final_msg = re.sub(r"\s+", " ", final_msg).strip()

    # Update history and route
    msgs = list(getattr(state, "messages", []) or [])
    msgs.append({"role": "assistant", "content": final_msg})

    return {
        "ai_message": final_msg,
        "messages": msgs,
        "route": "report_maker"
    }

print("[Agent] Loaded Translation agent.")
