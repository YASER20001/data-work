# =============================================================================
# data_generation/generate_dataset.py
#
# Main synthetic data generator for SafeAssist training data.
#
# ARCHITECTURE:
#   - Persona LLM  : Gemini instance with persona system prompt → Arabic user messages
#   - SafeAssist   : Real production pipeline (graph.invoke) → assistant responses
#   - Output       : 1000 multi-turn sessions (5–20 user turns each), ALL Arabic
#
# TRAINING GOALS:
#   Task A — Therapist response generation:
#             (conversation history + user_msg) → assistant_response
#   Task D — Case notes extraction:
#             (full conversation) → structured clinical case notes JSON
#
# USAGE:
#   python -m data_generation.generate_dataset --sessions 1000
#   python -m data_generation.generate_dataset --sessions 1000 --start-from 250
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running as a module from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# SafeAssist pipeline imports (initializes RAG, Gemini, LangGraph on import)
# ---------------------------------------------------------------------------
from backend.core.llm_gateway import configure_gemini, model as make_llm_model
from backend.core.state import AppState
from backend.pipeline_bootstrap import graph, rag_pipeline_instance

# ---------------------------------------------------------------------------
# Data generation config
# ---------------------------------------------------------------------------
from data_generation.personas import PERSONAS
from data_generation.session_config import (
    EMOTIONAL_STATES,
    DISCLOSURE_STAGES,
    build_session_plan,
)

# =============================================================================
# OUTPUT PATHS
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "output"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
ALL_SESSIONS_JSONL = OUTPUT_DIR / "all_sessions.jsonl"

# =============================================================================
# INTER-CALL DELAYS (seconds) — respect Gemini quota limits
# =============================================================================

DELAY_BETWEEN_TURNS = 2.5      # Between each persona+pipeline turn pair
DELAY_BETWEEN_SESSIONS = 4.0   # Between sessions


# =============================================================================
# PART 1: PERSONA LLM — generates the user side of the conversation
# =============================================================================

def _build_persona_turn_prompt(
    persona: Dict,
    emotional_state_key: str,
    disclosure_stage_key: str,
    conversation_history: List[Dict],
    turn_index: int,
    total_turns: int,
) -> str:
    """
    Build the instruction prompt for the persona LLM to generate the next
    Arabic user message, given the current session context and history.
    """
    emotional_state = EMOTIONAL_STATES[emotional_state_key]
    disclosure_stage = DISCLOSURE_STAGES[disclosure_stage_key]

    # Format conversation history (last 12 messages to keep context tight)
    history_lines = ""
    for msg in conversation_history[-12:]:
        role_label = "المستشارة" if msg["role"] == "assistant" else "أنتِ (المستخدمة)"
        history_lines += f"{role_label}: {msg['content']}\n\n"

    # Turn position cue helps the LLM know whether to open, deepen, or wrap up
    if turn_index == 0:
        position_cue = "هذه أولى رسائلك في هذه الجلسة — ابدئي المحادثة."
    elif turn_index >= total_turns - 1:
        position_cue = f"هذه آخر رسالة لكِ في هذه الجلسة (رقم {turn_index + 1} من {total_turns}). أنهي بشكل طبيعي."
    else:
        position_cue = f"أنتِ في منتصف المحادثة (رسالة {turn_index + 1} من {total_turns})."

    prompt = f"""
=== معلومات الجلسة ===
{position_cue}

الحالة العاطفية لهذه الجلسة:
{emotional_state['instruction_ar']}

مستوى الإفصاح:
{disclosure_stage['instruction_ar']}

=== سجل المحادثة حتى الآن ===
{history_lines if history_lines else "(لم تبدأ المحادثة بعد — هذه أول رسالة)"}

=== تعليمات ===
الآن اكتبي رسالتك التالية باللغة العربية فقط.
- كوني طبيعية وواقعية في لهجتك وأسلوبك.
- تنوعي في طول الرسالة: أحياناً جملة أو جملتين فقط، وأحياناً فقرة كاملة.
- لا تكوني مثالية أو رسمية أكثر من اللازم — أنتِ إنسانة تمر بظروف صعبة.
- لا تكتبي بالإنجليزية. لا تخرجي عن الشخصية.
""".strip()

    return prompt


def generate_user_turn(
    persona_model,
    persona: Dict,
    emotional_state_key: str,
    disclosure_stage_key: str,
    conversation_history: List[Dict],
    turn_index: int,
    total_turns: int,
) -> str:
    """Call the persona LLM to generate the next Arabic user message."""
    prompt = _build_persona_turn_prompt(
        persona, emotional_state_key, disclosure_stage_key,
        conversation_history, turn_index, total_turns,
    )
    try:
        response = persona_model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        return text if text else "لا أعرف كيف أبدأ."
    except Exception as e:
        print(f"    [Persona LLM Error] {type(e).__name__}: {e}")
        return "أحتاج مساعدة."


# =============================================================================
# PART 2: SAFEASSIST PIPELINE — generates the assistant side
# =============================================================================

def _build_app_state(
    messages: List[Dict],
    user_input: str,
    previous_state: Optional[AppState],
) -> AppState:
    """
    Construct an AppState for the next pipeline invocation, carrying forward
    accumulated case notes and assessments from the previous turn.
    """
    kwargs: Dict[str, Any] = {
        "rag_pipeline": rag_pipeline_instance,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "user_input": user_input,
        "input_mode": "text",
        "output_mode": "text",
        "lang": "ar",
    }

    if previous_state is not None:
        # Carry forward clinical memory and assessments
        kwargs["case_notes"] = previous_state.case_notes
        for field in ("risk_level", "personality_label", "personality_confidence_band", "legal_reason"):
            val = getattr(previous_state, field, None)
            if val is not None:
                kwargs[field] = val

    return AppState(**kwargs)


def run_pipeline_turn(
    messages: List[Dict],
    user_input: str,
    previous_state: Optional[AppState],
) -> Optional[AppState]:
    """
    Invoke the SafeAssist LangGraph pipeline for one turn.
    Returns the resulting AppState, or None on failure.
    """
    state = _build_app_state(messages, user_input, previous_state)
    try:
        return graph.invoke(state)
    except Exception as e:
        print(f"    [Pipeline Error] {type(e).__name__}: {e}")
        return None


def _extract_assistant_response(state: AppState) -> str:
    """Pull the final assistant response from pipeline state (priority order)."""
    response = (
        getattr(state, "ai_message", None) or
        getattr(state, "ai_message_reviewed", None) or
        getattr(state, "therapist_reply", None) or
        ""
    )
    return response.strip()


def _extract_system_state(state: AppState) -> Dict:
    """Extract key system state fields for recording in the session JSON."""
    return {
        "intent": getattr(state, "intent", None),
        "risk_score": getattr(state, "risk_score", None),
        "risk_level": getattr(state, "risk_level", None),
        "personality_label": getattr(state, "personality_label", None),
        "personality_confidence": getattr(state, "personality_confidence", None),
        "node_trace": list(getattr(state, "node_trace", []) or []),
    }


def _extract_case_notes(state: Optional[AppState]) -> Dict:
    """Pull accumulated case notes from the final pipeline state."""
    if state is None:
        return {}
    cn = getattr(state, "case_notes", {})
    return cn if isinstance(cn, dict) else {}


# =============================================================================
# PART 3: SESSION GENERATOR
# =============================================================================

def generate_session(
    session_id: str,
    persona: Dict,
    emotional_state_key: str,
    disclosure_stage_key: str,
    num_turns: int,
) -> Dict:
    """
    Generate one complete multi-turn session.

    Returns a session dict containing:
      - Full turn log (user + assistant messages with system state metadata)
      - Accumulated clinical case notes from the pipeline
      - Session metadata (persona, emotional state, disclosure stage, etc.)
    """
    print(f"  Emotional state : {emotional_state_key}")
    print(f"  Disclosure stage: {disclosure_stage_key}")
    print(f"  Turn count      : {num_turns} user turns")

    # Create a dedicated persona LLM instance for this session
    # High temperature for diverse, realistic, non-repetitive responses
    persona_model = make_llm_model(
        system=persona["system_prompt"],
        temperature=0.88,
    )

    conversation_history: List[Dict] = []
    turns_data: List[Dict] = []
    pipeline_state: Optional[AppState] = None

    for turn_idx in range(num_turns):
        print(f"    Turn {turn_idx + 1}/{num_turns} ...", end=" ", flush=True)

        # ------------------------------------------------------------------
        # Step 1: Generate Arabic user message via persona LLM
        # ------------------------------------------------------------------
        user_msg = generate_user_turn(
            persona_model,
            persona,
            emotional_state_key,
            disclosure_stage_key,
            conversation_history,
            turn_idx,
            num_turns,
        )
        conversation_history.append({"role": "user", "content": user_msg})

        # ------------------------------------------------------------------
        # Step 2: Run SafeAssist pipeline to generate assistant response
        # ------------------------------------------------------------------
        time.sleep(DELAY_BETWEEN_TURNS)

        pipeline_result = run_pipeline_turn(
            conversation_history,
            user_msg,
            pipeline_state,
        )

        if pipeline_result is not None:
            pipeline_state = pipeline_result
            assistant_response = _extract_assistant_response(pipeline_result)
            system_state = _extract_system_state(pipeline_result)
        else:
            # Graceful fallback if pipeline fails for this turn
            assistant_response = "أنا هنا معكِ. هل يمكنكِ أن تخبريني أكثر؟"
            system_state = {}

        if not assistant_response:
            assistant_response = "أنا أسمعكِ. تابعي من فضلكِ."

        conversation_history.append({"role": "assistant", "content": assistant_response})

        # ------------------------------------------------------------------
        # Step 3: Record both turns in the session log
        # ------------------------------------------------------------------
        user_turn_index = len(turns_data)
        turns_data.append({
            "turn_index": user_turn_index,
            "role": "user",
            "content": user_msg,
        })
        turns_data.append({
            "turn_index": user_turn_index + 1,
            "role": "assistant",
            "content": assistant_response,
            "system_state": system_state,
        })

        risk = system_state.get("risk_level", "—")
        intent = system_state.get("intent", "—")
        print(f"risk={risk}  intent={intent}")

    # --------------------------------------------------------------------------
    # Build final session record
    # --------------------------------------------------------------------------
    final_case_notes = _extract_case_notes(pipeline_state)

    return {
        "session_id": session_id,
        "persona_id": persona["id"],
        "persona_name_ar": persona["name_ar"],
        "persona_name_en": persona["name_en"],
        "language": "ar",
        "emotional_state": emotional_state_key,
        "disclosure_stage": disclosure_stage_key,
        "user_turns": num_turns,
        "total_turns": len(turns_data),
        "generated_at": time.time(),
        "turns": turns_data,
        "final_case_notes": final_case_notes,
    }


# =============================================================================
# PART 4: MAIN ORCHESTRATOR
# =============================================================================

def generate_dataset(
    total_sessions: int = 1000,
    start_from: int = 0,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Generate `total_sessions` synthetic multi-turn sessions.

    Sessions are distributed evenly across 10 personas (100 each by default).
    Each session uses a weighted-random emotional state and disclosure stage.
    Individual session JSON files are saved incrementally so the process can
    be resumed with --start-from if interrupted.

    Args:
        total_sessions : Total number of sessions to generate (default 1000).
        start_from     : Skip the first N sessions (for resuming).
        output_dir     : Override for the default output directory.
    """
    out_dir = output_dir or OUTPUT_DIR
    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sessions_per_persona = total_sessions // len(PERSONAS)
    print(f"\n[Generator] Plan: {total_sessions} sessions | "
          f"{len(PERSONAS)} personas | {sessions_per_persona} per persona")
    print(f"[Generator] Output: {out_dir}\n")

    # Build the full session plan (shuffled, weighted emotional/disclosure distribution)
    session_plan = build_session_plan(PERSONAS, sessions_per_persona=sessions_per_persona)

    all_jsonl_path = out_dir / "all_sessions.jsonl"
    errors: List[Dict] = []

    for idx, cfg in enumerate(session_plan):

        if idx < start_from:
            continue

        persona = PERSONAS[cfg["persona_id"]]
        session_id = f"s{idx + 1:04d}_{cfg['persona_id']}"
        session_file = sessions_dir / f"{session_id}.json"

        # Skip already-generated sessions (supports resuming)
        if session_file.exists():
            print(f"[{idx + 1}/{len(session_plan)}] SKIP {session_id} (already exists)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(session_plan)}] Generating: {session_id}")
        print(f"  Persona: {persona['name_en']}")

        session_start = time.time()
        try:
            session = generate_session(
                session_id=session_id,
                persona=persona,
                emotional_state_key=cfg["emotional_state"],
                disclosure_stage_key=cfg["disclosure_stage"],
                num_turns=cfg["num_turns"],
            )

            # Save individual session file
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)

            # Append to master JSONL (one line per session)
            with open(all_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(session, ensure_ascii=False) + "\n")

            elapsed = time.time() - session_start
            print(f"  [Saved] {session_id} | {session['total_turns']} total turns | {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\n[Interrupted] Progress saved. Resume with --start-from", idx)
            break
        except Exception as e:
            print(f"  [ERROR] Session {session_id} failed: {e}")
            traceback.print_exc()
            errors.append({"session_id": session_id, "error": str(e), "index": idx})
            # Continue to next session rather than aborting the whole run
            continue

        time.sleep(DELAY_BETWEEN_SESSIONS)

    # --------------------------------------------------------------------------
    # Save error log if any sessions failed
    # --------------------------------------------------------------------------
    if errors:
        error_path = out_dir / "generation_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n[Warning] {len(errors)} sessions failed. See: {error_path}")

    completed = len(list(sessions_dir.glob("*.json")))
    print(f"\n[Done] {completed} sessions saved to {sessions_dir}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic SafeAssist training data (Tasks A + D)"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=1000,
        help="Total number of sessions to generate (default: 1000)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        dest="start_from",
        help="Skip the first N sessions (for resuming interrupted runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Override default output directory",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    generate_dataset(
        total_sessions=args.sessions,
        start_from=args.start_from,
        output_dir=out,
    )
