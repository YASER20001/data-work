# backend/core/tracing.py
from __future__ import annotations
import os, time, re
from typing import Any, Dict


# =============================================================================
# PART 1: CONFIGURATION & WHITELISTING
# =============================================================================

def _enabled() -> bool:
    """Checks environment variable at runtime to enable/disable tracing."""
    return os.getenv("RIFD_TRACE", "0").strip() == "1"


_NUM = re.compile(r"\b\d{3,}\b")
_WS = re.compile(r"\s+")

# Authorized keys for state tracking to ensure data privacy and log clarity
_WHITELIST_KEYS = {
    # 1. Router
    "intent", "confidence", "route_reason", "immediate_danger", "lang",

    # 2. STT & Personality
    "detected_tone", "personality_label", "personality_probe",

    # 3. Risk Assessment
    "risk_score", "risk_level", "risk_reasons_ar",

    # 4. Therapist Logic
    "therapist_reply", "follow_up_question", "case_notes",

    # 5. Legal & Translation
    "legal_reason", "ai_message_reviewed", "ai_message", "glossary",

    # 6. Report Maker
    "final_report", "final_report_pdf",

    # 7. TTS (Text-to-Speech)
    "tts_text", "audio_path", "tts_audio_b64"
}

# Optional: module initialization log
print(f"[trace] (imported) current env says: {'ENABLED' if _enabled() else 'disabled'}")


# =============================================================================
# PART 2: SANITIZATION HELPERS
# =============================================================================

def _safe_text(s: Any, max_len: int = 200) -> str:
    """Sanitizes strings by removing long numbers and normalizing whitespace."""
    if not isinstance(s, str): return ""
    t = _WS.sub(" ", _NUM.sub("", s.strip()))
    return t[:max_len]


def _safe_messages(msgs):
    """Extracts a sanitized tail of the conversation history for the trace."""
    out = []
    for m in (msgs or [])[-2:]:
        out.append({"role": m.get("role", ""), "text": _safe_text(m.get("content", ""), 160)})
    return out


def safe_snapshot(state) -> Dict[str, Any]:
    """Captures a whitelisted, sanitized snapshot of the current AppState."""
    d = {}
    for k in _WHITELIST_KEYS:
        v = getattr(state, k, None)
        d[k] = _safe_text(v, 200) if isinstance(v, str) else v
    try:
        d["_tail"] = _safe_messages(getattr(state, "messages", []))
    except:
        d["_tail"] = []
    return d


# =============================================================================
# PART 3: DIFFING & EMISSION LOGIC
# =============================================================================

def _shallow_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Identifies differences between 'before' and 'after' state snapshots."""
    diff = {}
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            diff[k] = {"before": a.get(k), "after": b.get(k)}
    return diff


def emit(state, node_name: str, before: Dict[str, Any], after: Dict[str, Any]):
    """Records a trace entry into the state's debug buffer."""
    if not _enabled():
        return
    try:
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node": node_name,
            "diff": _shallow_diff(before, after)
        }
        buf = getattr(state, "debug_trace", [])
        buf.append(entry)
        state.debug_trace = buf[-50:]  # Keep only the last 50 transitions
    except:
        pass


# =============================================================================
# PART 4: GRAPH NODE WRAPPER
# =============================================================================

def wrap(node_name: str, fn):
    """
    Decorator used in graph construction to automatically trace
    the state delta for a specific node execution.
    """

    def _wrapped(state):
        before = safe_snapshot(state)
        out = fn(state)
        # Handle nodes that return updates vs nodes that modify state directly
        target = out if out is not None else state
        after = safe_snapshot(target)
        emit(target, node_name, before, after)
        return out

    setattr(_wrapped, "_trace_wrapped_from", getattr(fn, "__name__", "fn"))
    setattr(_wrapped, "_trace_node_name", node_name)
    return _wrapped
