# =============================================================================
# RIFD — UNIFIED GRAPH ENGINE (SafeAssist v2025)
# backend/pipeline_bootstrap.py
# =============================================================================

from __future__ import annotations
import hashlib
import json
# LangGraph Core imports
from langgraph.graph import StateGraph, START, END

# State Engine and Trace Components
from backend.core.state import (
    AppState,
    metric_wrap,
    push_debug,
    push_flow,
    print_user_input
)

# Shared Utilities
from backend.core.utils import (
    last_user_text,
    guess_language_from_messages
)

# Retrieval-Augmented Generation (RAG) Service
from backend.rag.rag_service import RagPipeline

# Global Configuration and Model Gateway
from backend.core.env_config import load as load_env
from backend.core.llm_gateway import configure_gemini

# =============================================================================
# PART 1: SYSTEM INITIALIZATION
# =============================================================================

# Load environment variables and configure Gemini LLM
env = load_env()
llm = configure_gemini()

# Initialize the shared RAG pipeline with the multilingual embedding model
rag_pipeline_instance = RagPipeline(model_name="paraphrase-multilingual-mpnet-base-v2")

# =============================================================================
# PART 2: AGENT ADAPTERS (ADAPTER PATTERN)
# =============================================================================

# --- 1. CASE NOTES (FAST SCRIBE) ---
from backend.agents.agent_case_notes import run as case_notes_run
class CaseNotesAdapter:
    def run(self, state: AppState):
        return case_notes_run(state)
case_notes_agent = CaseNotesAdapter()

# --- 2. ROUTER ---
from backend.agents.agent_router import classify
class RouterAdapter:
    def run(self, messages, user_input, lang, case_notes):
        ctx = {
            "history_window": messages,
            "case_notes": case_notes,
            "lang": lang
        }
        return classify(user_input, **ctx)
router = RouterAdapter()

# --- 3. RISK ---
from backend.agents.agent_risk import run as risk_run
class RiskAdapter:
    def run(self, state: AppState):
        return risk_run(state)
risk = RiskAdapter()

# --- 4. THERAPIST ---
from backend.agents.agent_therapist import run as therapist_run
class TherapistAdapter:
    def __init__(self, rag_pipeline: RagPipeline):
        self.rag_pipeline = rag_pipeline
    def run(self, state: AppState):
        if getattr(state, "rag_pipeline", None) is None:
            state.rag_pipeline = self.rag_pipeline
        return therapist_run(state)
therapist = TherapistAdapter(rag_pipeline_instance)

# --- 5. LEGAL REVIEW ---
from backend.agents.agent_legal_review import run as legal_run
class LegalReviewAdapter:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    def run(self, state: AppState):
        state.rag_pipeline = self.rag
        return legal_run(state)
legal = LegalReviewAdapter(rag_pipeline_instance)

# --- 6. TRANSLATION ---
from backend.agents.agent_translation import run as translation_run
class TranslationAdapter:
    def __init__(self, llm):
        self.llm = llm
    def run(self, state: AppState):
        return translation_run(state)
translation_final = TranslationAdapter(llm)

# --- 7. PERSONALITY ---
from backend.agents.agent_personality import run as personality_run
class PersonalityAdapter:
    def __init__(self, rag_pipeline: RagPipeline):
        self.rag_pipeline = rag_pipeline
    def run(self, state: AppState):
        # Ensure RAG is available before personality agent runs (it runs before therapist)
        if getattr(state, "rag_pipeline", None) is None:
            state.rag_pipeline = self.rag_pipeline
        return personality_run(state)
personality = PersonalityAdapter(rag_pipeline_instance)

# --- 8. STT ---
from backend.agents.agent_stt import run as stt_run
class STTAdapter:
    def __init__(self, env):
        self.env = env
    def run(self, state: AppState):
        return stt_run(state)
stt = STTAdapter(env)

# --- 9. TTS ---
from backend.agents.agent_tts import run as tts_run
class TTSAdapter:
    def __init__(self, env):
        self.env = env
    def run(self, state: AppState):
        return tts_run(state)
tts = TTSAdapter(env)


# =============================================================================
# PART 3: HELPER NODES (ROUTER, STT, NOTES)
# =============================================================================



def _fingerprint(txt: str) -> str:
    """Generates a short hash of input text to prevent redundant node execution."""
    if not txt: return ""
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]


# --- ROUTER NODE ---
def router_node(state: AppState):
    """Classifies user intent and prepares state for specialized agents."""
    push_debug(state, "info", "[Router] Determining intent…")

    # Inject RAG pipeline early so all downstream agents have access,
    # and clear the per-turn embedding cache for a fresh turn.
    if getattr(state, "rag_pipeline", None) is None:
        state.rag_pipeline = rag_pipeline_instance
    if hasattr(rag_pipeline_instance, "clear_turn_cache"):
        rag_pipeline_instance.clear_turn_cache()

    txt = last_user_text(state.messages)
    lang = guess_language_from_messages(state.messages, default="auto")

    r = router.run(
        messages=state.messages,
        user_input=txt,
        lang=lang,
        case_notes=state.case_notes
    )
    r.pop("case_notes", None)
    r.pop("messages", None)

    intent = r.get("intent") or "therapist"
    confidence = float(r.get("confidence", 0.0))

    if intent not in INTENT_TO_NODE:
        intent = "therapist"

    new_fp = _fingerprint(txt)

    updates = {
        "intent": intent,
        "confidence": confidence,
        "route_reason": r.get("route_reason"),
        "immediate_danger": bool(r.get("immediate_danger", False)),
        "lang": lang,
        "target_agent": INTENT_TO_NODE.get(intent),
        "recommended_question": r.get("recommended_question"),
        "msg_fp": new_fp,
    }

    if new_fp != state.msg_fp:
        updates.update({
            "risk_seen_for_msg": False,
            "personality_seen_for_msg": False
        })
    state.user_input = txt

    return updates


router_node_wrapped = metric_wrap("router", router_node)


# --- CASE NOTES NODE (FAST) ---
def case_notes_node(state: AppState):
    """Fast-sync scribe to ensure clinical memory is updated before output."""
    push_debug(state, "info", "[CaseNotes] Updating memory (Fast Scribe)...")

    r = case_notes_agent.run(state)

    # Clinical Memory Debug Output
    notes = r.get("case_notes", {})
    print("\n\033[93m" + "=" * 20 + " [MEMORY DUMP] " + "=" * 20)
    print(json.dumps(notes, indent=2, ensure_ascii=False))
    print("=" * 55 + "\033[0m\n")

    push_debug(state, "info", "[CaseNotes] Memory updated.")

    # Proceed to Translation
    r["route"] = "translation_final"
    return r


case_notes_node_wrapped = metric_wrap("case_notes", case_notes_node)

# --- STT NODE ---
def stt_node(state: AppState):
    """Handles speech-to-text processing for voice mode interaction."""
    push_debug(state, "info", "[STT] Processing audio…")

    r = stt.run(state)
    r.pop("case_notes", None)

    text = r.get("user_input", "")
    if text and not (state.messages and state.messages[-1].get("content") == text):
        state.messages.append({"role": "user", "content": text})

    push_debug(state, "info", f"[STT output] {r}")

    r["route"] = "router"
    return r

stt_node_wrapped = metric_wrap("stt", stt_node)


# =============================================================================
# CHUNK C — MAIN AGENT NODES
# =============================================================================

# --- RISK NODE: Evaluates safety and immediate danger ---
def risk_node(state: AppState):
    push_debug(state, "info", "[Risk] Evaluating safety…")
    r = risk.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    push_debug(state, "info", f"[risk_level] {r.get('risk_level')}")

    updates = r.copy()
    updates["risk_level"] = r.get("risk_level")
    updates["risk_seen_for_msg"] = True
    updates["route"] = "therapist"
    return updates

risk_node_wrapped = metric_wrap("risk", risk_node)


# --- PERSONALITY NODE: Analyzes emotional state and interpersonal style ---
def personality_node(state: AppState):
    push_debug(state, "info", "[Personality] Analyzing emotional tone…")
    r = personality.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    updates = r.copy()
    updates["personality_seen_for_msg"] = True
    updates["route"] = "therapist"
    return updates

personality_node_wrapped = metric_wrap("personality", personality_node)


# --- THERAPIST NODE: Generates core supportive response based on context ---
def therapist_node(state: AppState):
    push_debug(state, "info", "[Therapist] Generating supportive reply…")
    r = therapist.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    push_debug(state, "info", f"[therapist reply] {r.get('therapist_reply')}")

    if r.get("therapist_reply"):
        state.messages.append({"role": "assistant", "content": r["therapist_reply"]})

    r["route"] = "legal"
    return r

therapist_node_wrapped = metric_wrap("therapist", therapist_node)


# --- LEGAL REVIEW NODE: Audits the response for legal compliance ---
def legal_node(state: AppState):
    push_debug(state, "info", "[Legal] Running Legal compliance check…")
    r = legal.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    return r

legal_node_wrapped = metric_wrap("legal", legal_node)


# --- TRANSLATION NODE: Localizes and humanizes the final text ---
def translation_node(state: AppState):
    push_debug(state, "info", "[Translation] Finalizing output language…")
    r = translation_final.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    push_debug(state, "info", f"[translated] {r.get('ai_message')}")

    if r.get("ai_message"):
        state.messages.append({"role": "assistant", "content": r["ai_message"]})

    r["route"] = "tts"
    return r

translation_node_wrapped = metric_wrap("translation_final", translation_node)


# --- TTS NODE: Synthesizes final audio if voice mode is enabled ---
def tts_node(state: AppState):
    push_debug(state, "info", "[TTS] Converting final text to speech…")
    r = tts.run(state)
    r.pop("case_notes", None)
    r.pop("messages", None)

    if r.get("audio_path"):
        push_debug(state, "info", "[TTS audio] <PATH_ONLY>")

    r["route"] = "END"
    return r

tts_node_wrapped = metric_wrap("tts", tts_node)

# =============================================================================
# CHUNK D — CONDITIONAL LOGIC & ROUTING RULES
# =============================================================================

# Mapping of intent strings to internal graph node names
INTENT_TO_NODE = {
    "therapist": "therapist",
    "risk_assessment": "risk",
    "personality": "personality",
    "legal_review": "legal",
    "translation": "translation_final",
    "tts": "tts",
}


def _is_voice_mode_enabled(state: AppState) -> bool:
    """Detects if the application should output audio."""
    return (state.output_mode or "").strip().lower() == "voice"


def _from_start(state: AppState) -> str:
    """Initial graph entry: Routes to STT for audio or Router for text."""
    has_audio = bool(state.audio_bytes or state.stt_raw_audio)
    if has_audio or state.input_mode == "voice":
        push_flow(state, "[Graph] START → STT")
        return "stt"
    push_flow(state, "[Graph] START → Router")
    return "router"


def _after_stt(state: AppState) -> str:
    """
    Checks if the STT node requested a specific shortcut (like 'tts').
    Otherwise, proceeds to the standard 'router'.
    """
    # If the STT agent logic returned "route": "tts", follow it.
    if getattr(state, "route", None) == "tts":
        push_flow(state, "[Graph] STT → TTS (Fast Clarification)")
        return "tts"

    push_flow(state, "[Graph] STT → Router")
    return "router"


def _after_router(state: AppState) -> str:
    """Main routing logic including loop guards for risk and personality assessment."""
    intent = state.intent or "therapist"
    already_risk = state.risk_seen_for_msg
    already_personality = state.personality_seen_for_msg

    if intent == "risk_assessment":
        if already_risk:
            push_flow(state, "ROUTER → THERAPIST (Loop Guard)")
            return "therapist"
        push_flow(state, "ROUTER → RISK")
        return "risk"

    if intent == "personality":
        if already_personality:
            push_flow(state, "ROUTER → THERAPIST (Loop Guard)")
            return "therapist"
        push_flow(state, "ROUTER → PERSONALITY")
        return "personality"

    push_flow(state, "ROUTER → THERAPIST")
    return "therapist"


def _after_legal(state: AppState) -> str:
    """Feedback loop: Re-routes to therapist if legal audit fails."""
    route = state.route

    if route == "therapist":
        push_flow(state, "[Graph] LEGAL → THERAPIST (Feedback Loop)")
        return "therapist"

    push_flow(state, "[Graph] LEGAL → CASE NOTES")
    return "case_notes"


def _after_translation(state: AppState) -> str:
    """Terminal routing: Directs to audio synthesis or concludes the graph."""
    if _is_voice_mode_enabled(state):
        push_flow(state, "[Graph] TRANS → TTS")
        return "tts"

    push_flow(state, "[Graph] TRANS → END")
    return END


# =============================================================================
# CHUNK E — GRAPH BUILDING (LANGGRAPH ORCHESTRATION)
# =============================================================================

# Initialize the StateGraph with the AppState schema
graph_builder = StateGraph(AppState)

# Register All Nodes to the Graph
graph_builder.add_node("router", router_node_wrapped)
graph_builder.add_node("case_notes", case_notes_node_wrapped)
graph_builder.add_node("risk", risk_node_wrapped)
graph_builder.add_node("personality", personality_node_wrapped)
graph_builder.add_node("therapist", therapist_node_wrapped)
graph_builder.add_node("legal", legal_node_wrapped)
graph_builder.add_node("translation_final", translation_node_wrapped)
graph_builder.add_node("tts", tts_node_wrapped)
graph_builder.add_node("stt", stt_node_wrapped)

# 1. ENTRY POINT: Direct to STT (Voice) or Router (Text) based on input type
graph_builder.set_conditional_entry_point(
    _from_start,
    {
        "stt": "stt",
        "router": "router"
    }
)

# 2. DYNAMIC ROUTING: if STT didn't hear -> TTS, other than that -> Router
graph_builder.add_conditional_edges(
    "stt",
    _after_stt,
    {
        "router": "router",
        "tts": "tts"
    }
)

# 3. DYNAMIC ROUTING: Branches based on Intent Classification
graph_builder.add_conditional_edges(
    "router",
    _after_router,
    {
        "risk": "risk",
        "personality": "personality",
        "therapist": "therapist"
    }
)

# 4. CORE FLOW: Specialists route to Therapist; Therapist routes to Legal Audit
graph_builder.add_edge("risk", "therapist")
graph_builder.add_edge("personality", "therapist")
graph_builder.add_edge("therapist", "legal")

# 5. FEEDBACK LOOP: Legal routes back to Therapist if audit fails, else to Case Notes
graph_builder.add_conditional_edges(
    "legal",
    _after_legal,
    {
        "therapist": "therapist",
        "case_notes": "case_notes"
    }
)

# 6. PERSISTENCE: Case Notes updates flow into final Translation/Humanization
graph_builder.add_edge("case_notes", "translation_final")

# 7. TERMINATION: Routes to TTS for voice mode or concludes the graph execution
graph_builder.add_conditional_edges(
    "translation_final",
    _after_translation,
    {
        "tts": "tts",
        END: END
    }
)

# 8. AUDIO TERMINATION: TTS synthesis leads to final END
graph_builder.add_edge("tts", END)

# =============================================================================
# COMPILE
# =============================================================================



# Compiles the definition into an executable graph object
graph = graph_builder.compile()
print("\n[Pipeline] RIFD Graph successfully built")


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def debug_run_pipeline(user_message: str):
    """
    Utility to execute a full graph pass for debugging.
    Initializes a fresh state and invokes the compiled graph.
    """
    print("\n=====================================================")
    print("  SAFEASSIST — DEBUG RUN")
    print("=====================================================")
    print_user_input(user_message)

    state = AppState()
    state.messages.append({"role": "user", "content": user_message})
    state.user_input = user_message
    state.input_mode = "text"
    state.output_mode = "text"

    # Invoke the compiled LangGraph
    result = graph.invoke(state)

    print("\n========== FINAL STATE (SUMMARY) ==========")
    print(f"Intent: {result.intent}")
    print(f"Risk Level: {result.risk_level}")
    print(f"AI Reply: {result.ai_message or result.ai_message_reviewed}")
    print(f"Case Notes Updated: {getattr(result, 'case_notes_updated', False)}")

    # Visualization of the execution path
    node_trace = result.get("node_trace", [])
    print("\n" + "━" * 55)
    if node_trace:
        flow_str = " -> ".join(node_trace)
        print(f"\033[93m[FLOW]: {flow_str}\033[0m")
    else:
        print("\033[91m[FLOW]: No trace recorded\033[0m")
    print("━" * 55 + "\n")

    return result


# Module Exports
__all__ = [
    "graph",
    "rag_pipeline_instance",
    "debug_run_pipeline",
    "case_notes_node_wrapped"
]
