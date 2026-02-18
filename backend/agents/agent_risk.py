# backend/agents/agent_risk.py

from __future__ import annotations
import os, json, re
from typing import Dict, Any, List

from backend.core.state import AppState
from backend.core.llm_gateway import json_out
from backend.core.utils import last_user_text, history_window, serialize_toon, guess_language_from_messages

# =========================================================================
# PART 1: CONFIGURATION & DEBUGGING
# =========================================================================

def _flag(name: str, default: str = "0") -> bool:
    """Helper to convert environment variables to boolean flags."""
    try:
        return bool(int(os.getenv(name, default)))
    except Exception:
        return False

# Flag used to enable/disable detailed console logging for the risk agent
_DEBUG = _flag("RISK_AGENT_DEBUG", "0")

# =========================================================================
# PART 2: SYSTEM PROMPTS & TEMPLATES
# =========================================================================

# Strict clinical persona focused solely on immediate physical threat detection
_SYSTEM = (
    "You are an agent assessing immediate physical danger in domestic violence contexts. "
    "Evaluate only the current physical risk. Ignore the past and emotions unless they "
    "directly indicate an active physical threat. Do not escalate general medical issues "
    "unless history links them to a current physical act by an abuser. "
    "Return only a single JSON object."
)

# Detailed assessment template with tiered logic for risk scoring
_TEMPLATE = """
You are a bot assessing **IMMEDIATE PHYSICAL DANGER** specifically within **domestic abuse** contexts. Ignore emotions, past events, and non-physical abuse unless they directly
indicate a *physical threat happening NOW*. Decide if emergency intervention is needed *right now*. **Your most important task** is to determine if the danger is *caused by domestic abuse*
by linking the User Input to the context in the `History Window` and `Case Notes`. Do NOT escalate general medical emergencies unless they are clearly and directly linked to an abuser's
*current* physical action.

LOGIC (Strict Order - STOP at first match):

1) **Needs Info** (score = null) IF:
   - Input is insufficient to understand the *current* situation (e.g., vague: "He's here", call cut).
   - OR urgent third-party threat but unverified (e.g., “They have my daughter”).
   - OR severe medical symptom (“can’t breathe”, “bleeding”, “vomiting”, suspected poisoning) **without a clear current abuser action** in History/Notes.

2) **Low (0.1–0.4)** IF:
   - Context is clearly not real (dream/nightmare/movie/panic attack description).
   - OR anything else without active physical threat now (past abuse, verbal control, fear/whispering without clear escalation, vague threats, seeking info, minor medical issues not linked to current abuse).

3) **If real & clear, assess ACTIVE PHYSICAL threat (linked to abuse)**:
   - **Critical (0.8–1.0):** Weapon + threat/use; active assault/strangulation; credible suicidal threat; OR severe medical emergency *clearly linked now* to abuser action (“He hit me and now I’m bleeding”).
     → escalation_needed: true
   - **Medium (0.5–0.7):** Active physical escalation or confinement now (breaking things, user hiding from an active threat, pushing/throwing at user, forced entry).
     → escalation_needed: your judgment (true only if immediate external help is required now).
   - **Low:** Otherwise (as above)
     → escalation_needed: false

Available Data  (User data may be Arabic):
1. detected_tone: {detected_tone}
2. action_summary: {action_summary}
3. language: {language}
4. user_input: {user_input}
5. risk_score_initial: {risk_score_initial}
6. personality: {personality}

[CASE NOTES & HISTORY]
history_window:
{history_window}

case_notes:
{case_notes}

Use all fields to determine risk except `language` — it only controls output language elsewhere.


OUTPUT (single JSON object only):
{{
  "risk_score": number|null,          // null for Needs Info; else 0.1..1.0
  "escalation_needed": boolean,       // your decision per rules above
  "confidence_score": number,         // 0.0..1.0
  "risk_reasons": string              // brief justification in ENGLISH
}}
"""

# =========================================================================
# PART 3: HELPER UTILITIES
# =========================================================================

_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_WESTERN_DIGITS = "0123456789"
_TRANS_TABLE = str.maketrans(_ARABIC_DIGITS, _WESTERN_DIGITS)

def _norm_num(val):
    """Normalizes both Western and Arabic digit strings to a standard float format."""
    if isinstance(val, str):
        s = val.strip().translate(_TRANS_TABLE)
        s = s.replace("،", ".")
        if "." not in s and s.count(",") == 1:
            s = s.replace(",", ".")
        return s
    return val

def _band_from_score(score: float | None) -> str:
    """Categorizes a raw risk score into clinical severity bands."""
    if score is None: return "needs_info"
    if score >= 0.8: return "critical"
    if score >= 0.5: return "medium"
    return "low"

# =========================================================================
# PART 4: CORE AGENT LOGIC
# =========================================================================

def run(state: AppState) -> Dict[str, Any]:
    """
    Main execution function for immediate physical risk assessment.
    """
    messages: List[Dict[str, Any]] = list(getattr(state, "messages", []) or [])

    # Step 1: Resolve language for the session
    lang = getattr(state, "lang", None)
    if lang not in ("ar", "en"):
        lang = guess_language_from_messages(messages)

    # Step 2: Aggregate state context and case history for the prompt
    input_data = {
        "detected_tone": getattr(state, "detected_tone", "") or "",
        "action_summary": getattr(state, "action_summary", "") or "",
        "case_notes": dict(getattr(state, "case_notes", {}) or {}),
        "language": lang,
        "user_input": last_user_text(messages) or "",
        "history_window": history_window(messages, n=8),
        "risk_score_initial": getattr(state, "risk_score", None),
        "personality": getattr(state, "personality", "") or "",
    }

    # Step 3: Serialize data using TOON for token efficiency
    prompt = _TEMPLATE.format(**{
        **input_data,
        "case_notes": serialize_toon(input_data["case_notes"])
    })

    # Step 4: Execute LLM call for assessment
    data: Dict[str, Any] = json_out(_SYSTEM, prompt, temperature=0.05) or {}

    # Step 5: Process and normalize output values
    score = data.get("risk_score", None)
    risk_level = _band_from_score(score)
    if score is not None:
        try:
            score = float(_norm_num(score))
            if not (0.1 <= score <= 1.0): score = None
        except Exception:
            score = None

    conf = data.get("confidence_score", 0.5)
    try:
        conf = float(_norm_num(conf))
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.5

    # Step 6: Finalize risk justification and escalation flag
    esc = bool(data.get("escalation_needed", False))
    reasons = data.get("risk_reasons")
    if not isinstance(reasons, str) or not reasons.strip():
        reasons = "Insufficient information provided." if score is None else "No reason provided."

    # Step 7: Construct state updates
    updates: Dict[str, Any] = {
        "risk_score": float(score) if score is not None else getattr(state, "risk_score", 0.0),
        "risk_level": risk_level,
        "escalation_needed": esc,
        "risk_reasons": reasons,
        "confidence_score": conf,
        "route": "therapist",
        "risk_seen_for_msg": True
    }

    if _DEBUG:
        print(f"[Risk] score={updates['risk_score']}, level={risk_level}, esc={esc}")

    return updates

print("[Agent] Loaded risk agent.")
