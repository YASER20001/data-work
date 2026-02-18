# backend/core/state.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator
import time

# =============================================================================
# PART 1: CASE NOTES DATA MODEL (PYDANTIC)
# =============================================================================

class CaseNotes(BaseModel):
    """Normalized structure for clinical patient records."""
    physical_abuse: List[str] = Field(default_factory=list)
    verbal_abuse: List[str] = Field(default_factory=list)
    threat: List[str] = Field(default_factory=list)
    control: List[str] = Field(default_factory=list)
    fear: List[str] = Field(default_factory=list)
    emotion: List[str] = Field(default_factory=list)
    risk: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)


# =============================================================================
# PART 2: MAIN APPLICATION STATE
# =============================================================================

class AppState(BaseModel):
    """
    Global state object for the RIFD LangGraph pipeline.
    Manages inputs, agent outputs, RAG context, and orchestration routing.
    """
    # --- Input & Transcription ---
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_input: str = ""
    lang: str = "auto"
    input_mode: str = "text"
    output_mode: str = "text"
    transcribed_text: str = ""
    detected_tone: str = ""
    action_summary: str = ""
    personality: str = ""

    audio_bytes: Optional[bytes] = None
    stt_raw_audio: Optional[bytes] = None

    # --- RAG & Orchestration ---
    rag_pipeline: Any = None
    intent: Optional[str] = None
    confidence: float = 0.0
    route_reason: Optional[str] = None
    immediate_danger: bool = False
    risk_seen_for_msg: bool = False
    personality_seen_for_msg: bool = False
    msg_fp: Optional[str] = None
    ban_intents: List[str] = Field(default_factory=list)
    route: str = ""  # Control variable for conditional graph edges

    # --- Risk Assessment Outputs ---
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    escalation_confidence: Optional[float] = None
    risk_reasons_ar: Optional[str] = None

    # --- Personality Analysis Outputs ---
    personality_label: Optional[str] = None
    personality_confidence: Optional[float] = None
    personality_confidence_band: Optional[str] = None
    personality_probe: Optional[str] = None
    therapist_hint: Optional[str] = None

    # --- Therapist Drafting ---
    ai_message_draft: Optional[str] = None
    therapist_reply: Optional[str] = None
    follow_up_question: Optional[str] = None
    phase: Optional[str] = None

    # --- Legal & Supervisor Review ---
    ai_message_reviewed: Optional[str] = None
    legal_reason: Optional[str] = None
    legal_retry_count: int = 0
    legal_context_injection: Optional[str] = None

    # --- Final Output & Translation ---
    ai_message: Optional[str] = None
    glossary: List[Dict[str, str]] = Field(default_factory=list)

    # --- Persistent Records ---
    case_notes: Dict[str, Any] = Field(default_factory=dict)

    # --- Reporting & Audio Assets ---
    final_report: Optional[str] = None
    final_report_pdf: Optional[str] = None
    tts_text: Optional[str] = None
    tts_audio: Optional[bytes] = None
    tts_audio_b64: Optional[str] = None
    audio_path: Optional[str] = None

    # --- Debugging & Traceability ---
    backend_debug: List[str] = Field(default_factory=list)
    streamlit_steps: List[Any] = Field(default_factory=list)
    full_flow: List[str] = Field(default_factory=list)
    node_trace: List[str] = Field(default_factory=list)
    agent_thoughts: Dict[str, str] = Field(default_factory=dict)
    rag_debug: List[Any] = Field(default_factory=list)
    waterfall_summary: List[str] = Field(default_factory=list)

    # --- Data Normalization Validator ---
    @field_validator("case_notes", mode="before")
    def normalize_case_notes(cls, v):
        """Ensures case_notes always follow a consistent list-of-strings format."""
        if not isinstance(v, dict):
            return {}
        fixed = {}
        for k, vv in v.items():
            if isinstance(vv, dict):
                fixed[k] = [f"{a}: {b}" for a, b in vv.items()]
            elif vv is None:
                fixed[k] = []
            elif isinstance(vv, list):
                fixed[k] = [str(x) for x in vv]
            else:
                fixed[k] = [str(vv)]
        return fixed


# =============================================================================
# PART 3: DEBUG & TELEMETRY HELPERS
# =============================================================================

DEBUG_COLORS = {
    "info": "\033[94m[INFO]\033[0m",
    "warn": "\033[93m[WARN]\033[0m",
    "err": "\033[91m[ERR]\033[0m",
}

def push_debug(state: AppState, level: str, msg: Any):
    """Logs sanitized debug information to the state and console."""
    if isinstance(msg, (bytes, bytearray)):
        msg = "<AUDIO_BYTES_REMOVED>"
    if isinstance(msg, str) and "\\x" in msg:
        msg = "<BINARY_REMOVED>"

    safe = f"{DEBUG_COLORS.get(level, '[INFO]')} {msg}"
    state.backend_debug.append(safe)
    print("• " + safe)

def push_step(state: AppState, name: str, ms: float):
    """Tracks the execution time of individual graph nodes."""
    state.streamlit_steps.append({"step": name, "ms": ms})

def push_agent_thought(state: AppState, agent: str, thought: str):
    """Captures internal reasoning for display in the UI debug panel."""
    print(f"\033[95m[{agent} THOUGHT]\033[0m {thought}")
    state.agent_thoughts[agent] = thought

def push_flow(state: AppState, text: str):
    """Records high-level process flow changes."""
    state.full_flow.append(text)

def push_waterfall(state: AppState, node: str, content: Any):
    """Maintains a summary of content passed between nodes."""
    try:
        preview = str(content).replace("\n", " ")[:300]
    except:
        preview = "<unprintable>"
    state.waterfall_summary.append(f"{node}: {preview}")


# =============================================================================
# PART 4: LANGGRAPH NODE WRAPPER (METRICS & TRACING)
# =============================================================================

def metric_wrap(name: str, fn):
    """
    Decorator for LangGraph nodes to automate tracing,
    timing, and sanitized console logging.
    """
    def wrapped(state: AppState):
        print("\n\033[96m" + "━" * 55 + "\033[0m")
        print(f"\033[92m▶ NODE: {name}\033[0m")
        print("\033[96m" + "━" * 55 + "\033[0m")

        current_trace = state.node_trace.copy()
        current_trace.append(name)

        push_debug(state, "info", f"ENTER {name}")
        start = time.time()

        result = fn(state) or {}
        ms = int((time.time() - start) * 1000)

        # Sanitized output logging loop
        for k, v in result.items():
            if k in ["messages", "case_notes", "node_trace"]:
                continue
            if isinstance(v, (bytes, bytearray)):
                print(f"• [{k}] <AUDIO_BYTES_REMOVED>")
                continue
            if isinstance(v, str) and len(v) > 5000:
                print(f"• [{k}] <LARGE_STRING_REMOVED (len={len(v)})>")
                continue
            if isinstance(v, str) and "\\x" in v:
                print(f"• [{k}] <BINARY_REMOVED>")
                continue
            print(f"• [{k}] {v}")

        push_step(state, name, ms)
        print(f"• TIME: {ms} ms")
        print("\033[96m" + "━" * 55 + "\033[0m")

        next_route = result.get("route") or result.get("target_agent")
        if next_route:
            print(f"\033[94m--> NEXT ROUTE: {next_route}\033[0m")

        result["node_trace"] = current_trace
        return result

    wrapped._trace_node_name = name
    return wrapped


# =============================================================================
# PART 5: STATE UPDATE UTILITIES
# =============================================================================

def print_user_input(text: str):
    """Formatted console print for primary user interaction."""
    print("\n\n")
    print("\033[96m" + "━" * 55 + "\033[0m")
    print(f"\033[95m USER INPUT:\033[0m  \033[92m{text}\033[0m")
    print("\033[96m" + "━" * 55 + "\033[0m")

def apply_updates(state: AppState, updates: Dict[str, Any]) -> AppState:
    """Safely applies a dictionary of updates to the AppState instance."""
    if not updates:
        return state

    # Inline normalization for case notes updates
    if "case_notes" in updates and isinstance(updates["case_notes"], dict):
        fixed = {}
        for k, v in updates["case_notes"].items():
            if isinstance(v, dict):
                fixed[k] = [f"{a}: {b}" for a, b in v.items()]
            elif v is None:
                fixed[k] = []
            elif isinstance(v, str):
                fixed[k] = [v]
            elif isinstance(v, list):
                fixed[k] = [str(x) for x in v]
            else:
                fixed[k] = [str(v)]
        updates["case_notes"] = fixed

    for k, v in updates.items():
        try:
            setattr(state, k, v)
        except Exception as e:
            push_debug(state, "warn", f"Failed update '{k}': {e}")

    return state


print("[System] SAFEASSIST Engine Initialized\n")
