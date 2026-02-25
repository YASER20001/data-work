# backend/agents/agent_personality.py
# =========================================================================
# SECTION 1: IMPORTS & RAG INITIALIZATION
# =========================================================================
from __future__ import annotations
import os
from typing import Dict, Any
import json

try:
    from backend.rag.rag_service import RagPipeline
except ImportError:
    print("Warning (PersonalityAgent): Could not import RagPipeline.")
    RagPipeline = None

from backend.core.state import AppState
from backend.core.llm_gateway import json_out
from backend.core.utils import last_user_text, guess_language_from_messages, history_window, serialize_toon

# =========================================================================
# SECTION 2: CONFIGURATION & CLINICAL SCHEMAS
# =========================================================================

# Environment-based debug and threshold flags
_DEBUG = os.getenv("PERSONALITY_AGENT_DEBUG", "0") == "1"
_MIN_CONF = float(os.getenv("PERSONALITY_MIN_CONF", "0.60"))

# Authorized clinical labels for classification
PERSONALITY_LABELS = [
    "STYLE_COOPERATIVE", "STYLE_DEFENSIVE", "STYLE_STONEWALLING",
    "STYLE_CONTEMPT", "STYLE_DISTRESSED", "STYLE_ANGER",
    "STYLE_DEPRESSED", "STYLE_DEFLECTION_HUMOR", "STYLE_NEUTRAL",
    "STYLE_EUPHORIC", "STYLE_DISSOCIATIVE",
    "STYLE_UNCERTAIN"
]

_SYSTEM = (
    "You are an expert Clinical Classification Agent specializing in Cross-Cultural Psychology. "
    "Your goal is to detect the user's emotional style and interpersonal dynamics, "
    "accounting for cultural nuances (e.g., expressions of guilt, shame, or resilience in Arab/Western contexts). "
    "Be intellectually honest: if the input is vague, lower your confidence."
)

_TEMPLATE = """
## Task
Analyze the 'user_input' and match it to a single Clinical Style ID from the allowed list.

## Allowed Output Labels
{labels_list}

## Retrieved Clinical Standards (Source of Truth)
{rag_criteria}

## User Context
{payload}

## Analysis Rules
1. **Cultural Nuance:** Look beyond literal words. Interpret idioms and cultural expressions of distress (e.g., "I feel suffocated" -> Distressed/Depressed).
2. **Tie-Breaker:** If the user exhibits multiple styles (e.g., Anger and Distress), prioritize the one that represents the **primary** current state.
3. **Context:** Use the 'history_window' to see if this is a sudden change or a pattern.

## Scoring Rules
- **High Confidence (0.85 - 1.0):** User clearly matches the specific keywords or core definition of a style.
- **Medium Confidence (0.60 - 0.84):** User matches the behavior/intent, but uses indirect language.
- **Low Confidence (0.0 - 0.59):** Input is too short, ambiguous, or lacks emotional content.
  -> FORCE 'detected_personality' to "STYLE_UNCERTAIN".

## Output (STRICT JSON)
{{
  "detected_personality": "STYLE_...",
  "confidence": <float between 0.0 and 1.0>, 
  "reasoning": "Brief explanation of why this style fits the user's language.",
  "recommended_strategy": "Specific therapeutic tip (e.g., 'Validate their feelings', 'Use grounding techniques')."
}}
"""


# =========================================================================
# SECTION 3: DATA PROCESSING HELPERS
# =========================================================================

def _mk_payload(state: AppState) -> Dict[str, Any]:
    """Prepares the user context payload for the LLM analysis."""
    lang = getattr(state, "lang", None)
    if lang not in ("ar", "en"):
        lang = guess_language_from_messages(getattr(state, "messages", []))

    messages = list(getattr(state, "messages", []) or [])

    return {
        "language": lang,
        "user_input": last_user_text(messages) or "",
        "history_window": history_window(messages, n=5, style="role"),
        "case_notes": getattr(state, "case_notes", {}) or {},
        "previous_style": getattr(state, "personality_label", None),
    }


def _validate_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures LLM output conforms to schema and confidence thresholds."""
    if not isinstance(obj, dict): obj = {}

    label = obj.get("detected_personality", "STYLE_UNCERTAIN")
    conf = obj.get("confidence", 0.5)

    try:
        conf = float(conf)
    except (ValueError, TypeError):
        conf = 0.5

    # Strict Label Validation
    if label not in PERSONALITY_LABELS:
        label = "STYLE_UNCERTAIN"

    # Force UNCERTAIN if confidence is too low
    if conf < _MIN_CONF:
        label = "STYLE_UNCERTAIN"

    strat = obj.get("recommended_strategy", "Ask clarifying questions.")

    return {"detected_personality": label, "confidence": conf, "recommended_strategy": strat}


def _get_rag_criteria(
    rag_pipeline: Any,
    user_text: str,
    history_context: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve clinical personality criteria from the vector database.

    (C) Uses a context-enriched query (user text + recent history) so that
        criteria matching benefits from conversational context, not just the
        latest single message.
    """
    if not rag_pipeline or not user_text.strip():
        return "(No RAG context.)"

    # (C) Enrich the query with recent conversation context
    enriched_query = user_text
    if history_context and history_context.strip():
        enriched_query = f"{user_text}\n{history_context}"

    try:
        results = rag_pipeline.search_personality(enriched_query, k=k)
        formatted = []
        for meta in results:
            if not meta or meta.get("source") == "error":
                continue
            formatted.append(
                f"- ID: {meta.get('id')}\n"
                f"  Def: {meta.get('definition')}\n"
                f"  Keywords: {', '.join(meta.get('keywords', []))}\n"
            )
        return "\n".join(formatted) if formatted else "(No criteria found.)"
    except Exception:
        return "(RAG Error)"


# =========================================================================
# SECTION 4: CORE AGENT EXECUTION
# =========================================================================

def run(state: AppState) -> Dict[str, Any]:
    """
    Analyzes user personality and emotional style to provide therapeutic hints.
    """
    payload = _mk_payload(state)
    user_text = payload.get("user_input", "")
    messages  = list(getattr(state, "messages", []) or [])

    # 1. RAG Retrieval — enriched with recent conversation context (C)
    rag_service = getattr(state, "rag_pipeline", None)
    history_context = history_window(messages, n=2, style="role")
    rag_criteria = _get_rag_criteria(rag_service, user_text, history_context=history_context)

    # 2. LLM Inference
    prompt = _TEMPLATE.format(
        payload=serialize_toon(payload),
        rag_criteria=rag_criteria,
        labels_list=json.dumps(PERSONALITY_LABELS)
    )

    raw = json_out(_SYSTEM, prompt, temperature=0.1) or {}
    clean = _validate_json(raw)

    label = clean["detected_personality"]
    conf = clean["confidence"]
    strategy = clean["recommended_strategy"]

    # Handle low-confidence detection
    if label == "STYLE_UNCERTAIN":
        strategy = "AMBIGUITY DETECTED: Ask a gentle clarifying question."

    # 3. State Updates
    updates: Dict[str, Any] = {
        "personality_label": label,
        "personality_confidence": conf,
        "therapist_hint": f"Style: {label} (Conf: {conf:.2f}). Advice: {strategy}",
        "route": "therapist",
    }

    if _DEBUG:
        print(f"[Personality] {label} (Conf: {conf:.2f})")

    return updates


print("[Agent] Personality Agent.")
