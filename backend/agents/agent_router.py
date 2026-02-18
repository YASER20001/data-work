# backend/agents/agent_router.py
from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, List

# Internal system imports
from backend.core.llm_gateway import model, json_out
from backend.core.state import AppState
from backend.core.utils import serialize_toon, history_window  # <--- IMPORTED

# =============================================================================
# PART 1: CONFIGURATION & CONSTANTS
# =============================================================================

# Debug and Safety Flags from environment variables
ROUTER_DEBUG = bool(int(os.getenv("ROUTER_DEBUG", "0")))
ENABLE_HARD_RISK_TRIPWIRE = bool(int(os.getenv("ROUTER_ENABLE_DANGER_FLAG", "1")))

# Complete list of system intents
INTENTS: List[str] = [
    "therapist", "risk_assessment", "digital_safety", "legal",
    "translation_access", "scheduler", "report_maker", "privacy_shell",
    "services_handoff", "personality",
]

# Core set for the classification engine
PRIMARY_ROUTE_SET = ("risk_assessment", "personality", "therapist")


# =============================================================================
# PART 2: HELPER FUNCTIONS
# =============================================================================

def _as_recommended_question(intent: str, lang_code: str | None, text_hint: str | None) -> str | None:
    """Returns a localized follow-up question if the personality agent is chosen."""
    if intent != "personality": return None
    is_ar = (lang_code == "ar") or bool(re.search(r"[\u0600-\u06FF]", text_hint or ""))
    return "باختصار، كيف جعلك ذلك تشعر؟" if is_ar else "In a few words, how did that make you feel?"


def _compose_user_payload(text: str, ctx: Dict[str, Any]) -> str:
    """Packages and serializes user input and context for LLM processing."""
    hw = ctx.get("history_window") or []

    # Get conversation context
    hw_formatted_str = history_window(hw, n=5)

    # Normalize Case Notes structure
    case_raw = ctx.get("case_notes")
    if hasattr(case_raw, "model_dump"):
        try:
            case = case_raw.model_dump()
        except Exception:
            case = {}
    elif isinstance(case_raw, dict):
        case = dict(case_raw)
    elif isinstance(case_raw, list):
        case = {"_legacy_list": list(case_raw)}
    else:
        case = {}

    banned = list(ctx.get("ban_intents") or [])

    # Final payload preparation
    payload_data = {
        "message": (text or "").strip(),
        "signals": {
            "time_since_last_chat": ctx.get("time_since_last_chat"),
            "tone": ctx.get("tone"),
            "lang": ctx.get("lang"),
        },
        "history_context": hw_formatted_str,
        "case_notes_snapshot": case,
        "ban_intents": [b for b in banned if b in PRIMARY_ROUTE_SET],
    }
    return serialize_toon(payload_data)


# =============================================================================
# PART 3: HARD-RISK TRIPWIRE (REGEX-BASED DETECTION)
# =============================================================================

# Patterns for immediate English risk keywords
_EN_RISK_PATTERNS = [
    r"\b(kill myself|suicide|end my life|take my life|i don'?t want to live)\b",
    r"\b(overdose|took\s+\d+\s*(pills?|tablets?)|cut myself|self[-\s]?harm)\b",
    r"\b(kill|hurt)\s+(him|her|them|someone|people)\b",
    r"\b(i am in danger right now|he is outside my door|she is outside my door)\b",
]
_EN_RISK_RE = re.compile("|".join(_EN_RISK_PATTERNS), re.IGNORECASE)

# Patterns for immediate Arabic risk keywords
_AR_RISK_RE = re.compile(
    r"(انتحار|أنتحر|اقتُل نفسي|اقتل نفسي|أقتل نفسي|"
    r"ما ?أبغى أعيش|لا أريد العيش|أنهي حياتي|"
    r"أؤذي نفسي|جرحت نفسي|قطعت نفسي|"
    r"سأقتل|بقتله|بقتلها|أقتله|أقتلها|أؤذيه|أؤذيها|"
    r"هو عند الباب|واقف عند الباب|برا الباب|خارج الباب|"
    r"يضربني(?: الآن| الحين)?)"
)


def _hard_risk_signal(text: str) -> bool:
    """Bypasses LLM classification if critical danger keywords are detected."""
    if not ENABLE_HARD_RISK_TRIPWIRE: return False
    t = text or ""
    return bool(_EN_RISK_RE.search(t) or _AR_RISK_RE.search(t))


# =============================================================================
# PART 4: ROUTER LLM CONFIGURATION
# =============================================================================

# System persona for the dispatcher
_SYSTEM = """
Role: You are a careful dispatcher that understands context and chooses ONE next agent.

Valid next agents (choose exactly one):
- risk_assessment     → when the user's message implies danger, self-harm, or a change in safety status.
- personality         → when the user exhibits a distinct INTERPERSONAL STYLE or EMOTIONAL STATE.
                        TRIGGERS:
                        1. Conflict (Defensiveness, Stonewalling, Anger, Contempt).
                        2. Abnormal Affect (Euphoria/Mania, Deep Depression, Detachment, Dissociation).
                        3. Humor/Sarcasm or Ambiguity.
                        DO NOT SKIP just because the user seems "happy" (Euphoria is a signal).
- therapist           → when the conversation is supportive and stable, with no new risk or personality shifts.

Hard rules:
- All risk and emergency assessments are handled exclusively by the 'risk_assessment' agent.
- If a route is in 'ban_intents', DO NOT select it.
- Input provided in TOON format.

Output STRICT JSON with:
{
  "intent": "risk_assessment" | "personality" | "therapist",
  "confidence": <number between 0 and 1>,
  "route_reason": "<short reason focusing on *what changed* and why this route>",
  "impact": "risk" | "personality" | "none"
}

Keep the reason short and factual; no user-facing advice.
"""

# Validation schema for structured output
_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(PRIMARY_ROUTE_SET)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "route_reason": {"type": "string"},
        "impact": {"type": "string", "enum": ["risk", "personality", "none"]},
    },
    "required": ["intent", "confidence", "route_reason"],
}


# =============================================================================
# PART 5: CORE ROUTING LOGIC (classify)
# =============================================================================

def classify(text: str, **ctx) -> Dict[str, Any]:
    """Primary classification engine for incoming messages."""

    # 1. Immediate exit for empty input
    if not text or not str(text).strip():
        return {
            "intent": "therapist",
            "confidence": 0.50,
            "route_reason": "Empty/whitespace input → fall back to therapist.",
        }

    # 2. Safety Bypass check
    if _hard_risk_signal(text):
        out = {
            "intent": "risk_assessment",
            "confidence": 0.95,
            "route_reason": "Hard-risk keywords detected → route to risk_assessment.",
            "impact": "risk",
        }
        if ROUTER_DEBUG: out["router_debug"] = {"tripwire": True}
        return out

    # 3. Request LLM Classification
    payload_toon = _compose_user_payload(text, ctx)

    data: Dict[str, Any]
    try:
        data = json_out(_SYSTEM, payload_toon, temperature=0.0, schema=_SCHEMA)
    except TypeError:
        try:
            data = json_out(_SYSTEM, payload_toon, temperature=0.0) or {}
        except Exception:
            data = {}
    except Exception:
        data = {}

    # 4. Fallback logic for LLM failures
    if not isinstance(data, dict) or not data:
        mdl = model(system=_SYSTEM, response_mime_type="application/json", temperature=0.0)
        try:
            resp = mdl.generate_content(payload_toon)
            raw = getattr(resp, "text", "") or "{}"
            data = json.loads(raw)
        except Exception:
            data = {
                "intent": "therapist", "confidence": 0.55,
                "route_reason": "LLM tie-break fallback.", "impact": "none",
            }

    # 5. Output Normalization
    intent = (data.get("intent") or "").strip()
    if intent not in PRIMARY_ROUTE_SET: intent = "therapist"

    try:
        confidence = float(data.get("confidence", 0.6))
    except Exception:
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    route_reason = str(data.get("route_reason") or "Routing by LLM.").strip()
    if len(route_reason) > 240: route_reason = route_reason[:237] + "..."

    impact = data.get("impact")
    if impact not in ("risk", "personality", "none"):
        impact = "risk" if intent == "risk_assessment" else ("personality" if intent == "personality" else "none")

    # 6. Apply context-based ban lists
    banned = set(ctx.get("ban_intents") or [])
    if intent in banned:
        original = intent
        intent = "therapist"
        route_reason = f"Selected intent was banned by context; falling back to therapist. (original: {original})"
        impact = "none"
        confidence = min(confidence, 0.60)

    # 7. Construct final result
    out: Dict[str, Any] = {
        "intent": intent,
        "confidence": confidence,
        "route_reason": route_reason,
        "impact": impact,
    }

    # Add localized prompts if applicable
    rq = _as_recommended_question(intent, ctx.get("lang"), text)
    if rq:
        out["recommended_question"] = rq

    return out


print("[Agent] Loaded router agent.")
