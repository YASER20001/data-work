# backend/core/utils.py
# =============================================================================
# RIFD — UTILS (SafeAssist Edition)
# =============================================================================

from typing import Any, Dict, List

# Type alias for message objects used across the pipeline
Message = Dict[str, Any]

# =============================================================================
# PART 1: BASIC MESSAGE UTILITIES
# =============================================================================

def last_user_text(messages: List[Message]) -> str:
    """Return the most recent user message in the history."""
    try:
        for m in reversed(messages or []):
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
    except:
        pass
    return ""


def guess_language_from_messages(messages: List[Message], default: str = "auto") -> str:
    """
    Lightweight Arabic vs English detection based on character set frequency.
    Checks the most recent user message to determine script priority.
    """
    txt = last_user_text(messages)
    if not txt:
        return default

    # Unicode range check for Arabic script
    ar = sum(1 for ch in txt if '\u0600' <= ch <= '\u06FF')
    en = sum(1 for ch in txt if ch.isascii())

    if ar == 0 and en == 0:
        return default

    return "ar" if ar > en else "en"


def history_window(messages: List[Message], n: int = 5, style: str = "role") -> str:
    """
    Returns the last N messages formatted as a string.
    Used by agents to maintain conversational context within token limits.
    """
    msgs = messages[-n:] if messages else []
    lines = []

    for m in msgs:
        role = m.get("role", "").strip()
        content = (m.get("content") or "").strip().replace("\n", " ")

        if style == "role":
            lines.append(f"{role}: {content}")
        else:
            lines.append(content)

    return "\n".join(lines)


# =============================================================================
# PART 2: CASE NOTES NORMALIZATION
# =============================================================================

def normalize_case_notes(state):
    """
    Standardizes the state.case_notes attribute into the CaseNotes v3 Pydantic structure.
    Handles transitions from raw dictionaries to validated objects.
    """
    try:
        from backend.core.state import CaseNotes
    except Exception:
        return state

    cn = getattr(state, "case_notes", None)

    if isinstance(cn, CaseNotes):
        return state

    # Dict conversion logic: mapping raw keys to CaseNotes attributes
    if isinstance(cn, dict):
        state.case_notes = CaseNotes(
            physical_abuse=list(cn.get("physical_abuse", [])),
            verbal_abuse=list(cn.get("verbal_abuse", [])),
            threat=list(cn.get("threat", [])),
            control=list(cn.get("control", [])),
            fear=list(cn.get("fear", [])),
            emotion=list(cn.get("emotion", [])),
            risk=list(cn.get("risk", [])),
            context=list(cn.get("context", [])),
            patterns=list(cn.get("patterns", [])),
        )
        return state

    # Fallback to a fresh CaseNotes object if input is unrecognized
    state.case_notes = CaseNotes()
    return state


# =============================================================================
# PART 3: TOON SERIALIZATION
# =============================================================================

def serialize_toon(obj, indent=0) -> str:
    """
    Converts complex dictionaries and lists into a clean, YAML-like TOON format.
    Optimized for token efficiency in LLM prompts by removing unnecessary JSON syntax.
    """
    if hasattr(obj, 'dict'):
        obj = obj.dict()

    prefix = "  " * indent

    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            # Skip empty fields to save tokens
            if v in [None, "", [], {}]:
                continue
            formatted_v = serialize_toon(v, indent + 1)
            if "\n" in formatted_v or len(formatted_v) > 50:
                lines.append(f"{prefix}{k}:\n{formatted_v}")
            else:
                lines.append(f"{prefix}{k}: {formatted_v.strip()}")
        return "\n".join(lines)

    elif isinstance(obj, list):
        # Inline short primitive lists
        if all(isinstance(x, (str, int, float)) for x in obj):
            return prefix + "[" + ", ".join(str(x) for x in obj) + "]"
        # Multi-line bulleted list for complex objects
        return "\n".join([f"{prefix}- {serialize_toon(x, indent + 1).strip()}" for x in obj])

    return f"{prefix}{str(obj)}"
