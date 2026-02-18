# =============================================================================
# data_generation/format_training.py
#
# Post-processes raw session JSON files into two training-ready JSONL files:
#
#   Task A — Therapist response generation
#   ----------------------------------------
#   One example per assistant turn.
#   Input : full conversation history up to and including the user message
#   Output: the assistant's response
#   Format: {"messages": [...], "response": "...", "metadata": {...}}
#
#   Task D — Case notes extraction (clinical scribe)
#   ----------------------------------------
#   One example per session.
#   Input : full conversation (all user + assistant turns)
#   Output: structured clinical case notes JSON
#   Format: {"conversation": [...], "case_notes": {...}, "metadata": {...}}
#
# USAGE:
#   python -m data_generation.format_training
#   python -m data_generation.format_training --sessions-dir /path/to/sessions
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# PATHS
# =============================================================================

_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# PART 1: LOADERS
# =============================================================================

def load_sessions(sessions_dir: Path) -> List[Dict]:
    """Load all individual session JSON files from the sessions directory."""
    sessions = []
    files = sorted(sessions_dir.glob("*.json"))
    if not files:
        print(f"[Warning] No session files found in {sessions_dir}")
        return sessions
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                sessions.append(json.load(fp))
        except Exception as e:
            print(f"[Warning] Could not load {f.name}: {e}")
    return sessions


# =============================================================================
# PART 2: TASK A — THERAPIST RESPONSE GENERATION
# =============================================================================

def session_to_task_a_pairs(session: Dict) -> List[Dict]:
    """
    Convert one session into Task A training pairs.

    For each user turn that has a following assistant turn, produce one pair:
      - "messages": the full conversation history including the user's message
      - "response": the assistant's reply to that message
      - "metadata": session/turn context for analysis and filtering

    This means a session with N user turns produces N training pairs.
    """
    pairs = []
    turns = session.get("turns", [])

    for i, turn in enumerate(turns):
        # We only process user turns that are followed by an assistant turn
        if turn["role"] != "user":
            continue
        if i + 1 >= len(turns) or turns[i + 1]["role"] != "assistant":
            continue

        assistant_turn = turns[i + 1]
        assistant_response = (assistant_turn.get("content") or "").strip()

        if not assistant_response:
            continue

        # Build the messages list: all previous turns + this user message
        messages: List[Dict[str, str]] = []
        for prev in turns[:i]:
            messages.append({
                "role": prev["role"],
                "content": prev["content"],
            })
        messages.append({
            "role": "user",
            "content": turn["content"],
        })

        system_state = assistant_turn.get("system_state", {}) or {}

        pairs.append({
            "messages": messages,
            "response": assistant_response,
            "metadata": {
                "session_id": session["session_id"],
                "persona_id": session["persona_id"],
                "persona_name_ar": session.get("persona_name_ar", ""),
                "language": session.get("language", "ar"),
                "emotional_state": session.get("emotional_state"),
                "disclosure_stage": session.get("disclosure_stage"),
                "turn_index": i,
                "conversation_length_at_turn": len(messages),
                "risk_level": system_state.get("risk_level"),
                "risk_score": system_state.get("risk_score"),
                "intent": system_state.get("intent"),
                "personality_label": system_state.get("personality_label"),
                "node_trace": system_state.get("node_trace", []),
            },
        })

    return pairs


# =============================================================================
# PART 3: TASK D — CASE NOTES EXTRACTION
# =============================================================================

def session_to_task_d_example(session: Dict) -> Optional[Dict]:
    """
    Convert one session into a Task D training example.

    Input  : the full conversation (all turns)
    Output : the final accumulated case notes JSON from the pipeline

    Returns None if the session has no useful case notes.
    """
    case_notes = session.get("final_case_notes", {}) or {}

    # Only include sessions where the pipeline actually captured clinical data
    has_content = any(
        isinstance(v, list) and len(v) > 0
        for v in case_notes.values()
    )
    if not has_content:
        return None

    # Build clean conversation list (no system metadata)
    conversation = [
        {"role": t["role"], "content": t["content"]}
        for t in session.get("turns", [])
        if t.get("content")
    ]

    return {
        "conversation": conversation,
        "case_notes": case_notes,
        "metadata": {
            "session_id": session["session_id"],
            "persona_id": session["persona_id"],
            "persona_name_ar": session.get("persona_name_ar", ""),
            "language": session.get("language", "ar"),
            "emotional_state": session.get("emotional_state"),
            "disclosure_stage": session.get("disclosure_stage"),
            "total_turns": session.get("total_turns", len(conversation)),
            "user_turns": session.get("user_turns"),
        },
    }


# =============================================================================
# PART 4: STATS COMPUTATION
# =============================================================================

def compute_stats(
    sessions: List[Dict],
    task_a_pairs: List[Dict],
    task_d_examples: List[Dict],
) -> Dict:
    """Compute summary statistics over the generated dataset."""
    persona_counts: Dict[str, int] = {}
    emotional_state_counts: Dict[str, int] = {}
    disclosure_stage_counts: Dict[str, int] = {}
    risk_level_counts: Dict[str, int] = {}
    turn_lengths: List[int] = []

    for s in sessions:
        pid = s.get("persona_id", "unknown")
        persona_counts[pid] = persona_counts.get(pid, 0) + 1

        es = s.get("emotional_state", "unknown")
        emotional_state_counts[es] = emotional_state_counts.get(es, 0) + 1

        ds = s.get("disclosure_stage", "unknown")
        disclosure_stage_counts[ds] = disclosure_stage_counts.get(ds, 0) + 1

        turn_lengths.append(s.get("user_turns", 0))

    for pair in task_a_pairs:
        rl = (pair.get("metadata") or {}).get("risk_level") or "unknown"
        risk_level_counts[rl] = risk_level_counts.get(rl, 0) + 1

    avg_turns = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0

    return {
        "total_sessions": len(sessions),
        "task_a_pairs": len(task_a_pairs),
        "task_d_examples": len(task_d_examples),
        "avg_user_turns_per_session": round(avg_turns, 2),
        "min_user_turns": min(turn_lengths) if turn_lengths else 0,
        "max_user_turns": max(turn_lengths) if turn_lengths else 0,
        "sessions_per_persona": persona_counts,
        "sessions_per_emotional_state": emotional_state_counts,
        "sessions_per_disclosure_stage": disclosure_stage_counts,
        "task_a_pairs_per_risk_level": risk_level_counts,
    }


# =============================================================================
# PART 5: MAIN FORMATTER
# =============================================================================

def format_training_data(
    sessions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Load all session files, format them into Task A and Task D JSONL files,
    and write a stats summary.

    Returns the stats dict.
    """
    out_dir = output_dir or _DEFAULT_OUTPUT_DIR
    src_dir = sessions_dir or (out_dir / "sessions")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Formatter] Loading sessions from: {src_dir}")
    sessions = load_sessions(src_dir)
    print(f"[Formatter] Loaded {len(sessions)} sessions")

    # ------------------------------------------------------------------
    # Task A — therapist response generation
    # ------------------------------------------------------------------
    task_a_pairs: List[Dict] = []
    for session in sessions:
        task_a_pairs.extend(session_to_task_a_pairs(session))

    task_a_path = out_dir / "train_task_a.jsonl"
    with open(task_a_path, "w", encoding="utf-8") as f:
        for pair in task_a_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[Task A] {len(task_a_pairs)} training pairs → {task_a_path}")

    # ------------------------------------------------------------------
    # Task D — case notes extraction
    # ------------------------------------------------------------------
    task_d_examples: List[Dict] = []
    skipped_d = 0
    for session in sessions:
        example = session_to_task_d_example(session)
        if example is not None:
            task_d_examples.append(example)
        else:
            skipped_d += 1

    task_d_path = out_dir / "train_task_d.jsonl"
    with open(task_d_path, "w", encoding="utf-8") as f:
        for example in task_d_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"[Task D] {len(task_d_examples)} examples → {task_d_path}")
    if skipped_d:
        print(f"         ({skipped_d} sessions skipped — no clinical case notes captured)")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats = compute_stats(sessions, task_a_pairs, task_d_examples)
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[Stats]  Summary → {stats_path}")

    # Print a human-readable summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total sessions   : {stats['total_sessions']}")
    print(f"  Task A pairs     : {stats['task_a_pairs']}")
    print(f"  Task D examples  : {stats['task_d_examples']}")
    print(f"  Avg turns/session: {stats['avg_user_turns_per_session']}")
    print(f"  Turn range       : {stats['min_user_turns']}–{stats['max_user_turns']}")
    print("\n  Sessions per persona:")
    for pid, count in sorted(stats["sessions_per_persona"].items()):
        print(f"    {pid}: {count}")
    print("\n  Sessions per emotional state:")
    for es, count in sorted(stats["sessions_per_emotional_state"].items()):
        print(f"    {es}: {count}")
    print("=" * 50 + "\n")

    return stats


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format raw SafeAssist sessions into Task A and Task D training files"
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default=None,
        dest="sessions_dir",
        help="Path to directory containing individual session JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory for JSONL training files (default: data_generation/output/)",
    )
    args = parser.parse_args()

    src = Path(args.sessions_dir) if args.sessions_dir else None
    out = Path(args.output_dir) if args.output_dir else None
    format_training_data(sessions_dir=src, output_dir=out)
