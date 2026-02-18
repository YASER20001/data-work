# backend/agents/agent_case_notes.py
from __future__ import annotations
import json
import time
from typing import Dict, Any
from backend.core.state import AppState
from backend.core.llm_gateway import json_out


# =========================================================================
# SECTION 1: UTILITIES & LLM PROMPT CONFIGURATION
# =========================================================================

def now_ts():
    """Generates a current unix timestamp for record keeping."""
    return int(time.time())


UNIFIED_SCRIBE_PROMPT = """
You are the OFFICIAL CLINICAL SCRIBE for a domestic violence intervention system.
Your job is to UPDATE the "Patient Record" based on the new message and system findings.

### INPUTS PROVIDED:
1. **CURRENT_RECORD**: The existing patient history (JSON).
2. **NEW_USER_MSG**: The latest message from the user.
3. **SYSTEM_FINDINGS**: Technical tags (Risk Level, Intent) detected by other agents.

### INSTRUCTIONS:
1. **ANALYZE**: Read the User Message and System Findings.
2. **DECIDE**: Is there *new* relevant info? (Abuse, Threats, Control, Fear, Clarifications).
   - If NO new info (e.g., "Thanks", "Hello"): Return `is_relevant: false`.
3. **UPDATE (If Relevant)**:
   - **PRESERVE**: Keep existing data in `CURRENT_RECORD` unless you are refining it.
   - **APPEND**: Add new specific details to the correct categories.
   - **REFINE**: If the user clarifies (e.g., "He" is "Mohammed"), update the old entries.
   - **TRUST SYSTEM**: If `SYSTEM_FINDINGS` says "Critical Risk" or "Weapon", you MUST log that in `risk`.

### CATEGORIES:
- `physical_abuse` (Hitting, weapons, choking)
- `verbal_abuse` (Insults, screaming)
- `threat` (Death threats, suicide threats)
- `control` (Isolation, phone taking, financial)
- `fear` (User expressions of terror)
- `risk` (High-level signals like 'Weapon involved', 'Escalation')

### OUTPUT FORMAT (JSON):
{
  "is_relevant": true/false,
  "updated_record": {
      "physical_abuse": ["..."],
      "verbal_abuse": ["..."],
      "threat": ["..."],
      "control": ["..."],
      "fear": ["..."],
      "risk": ["..."],
      "patterns": ["..."],
      "timeline_update": { "event": "...", "date": "Today" }  // Optional, null if no specific event
  }
}
"""

# =========================================================================
# SECTION 2: CORE AGENT LOGIC
# =========================================================================

def run(state: AppState) -> Dict[str, Any]:
    """
    Main execution function for the Case-Notes Agent.
    Processes user input and updates the persistent clinical record.
    """
    msg = state.user_input or ""

    # --- Step 1: Context Aggregation ---
    system_findings = {
        "risk_level": state.risk_level or "unknown",
        "risk_reasons": state.route_reason or "none",
        "legal_status": state.legal_reason or "none"
    }

    # --- Step 2: Historical Record Normalization ---
    old_notes = state.case_notes
    if hasattr(old_notes, 'model_dump'):
        old = old_notes.model_dump()
    elif isinstance(old_notes, dict):
        old = old_notes
    else:
        old = {}

    clean_old_record = {
        "physical_abuse": old.get("physical_abuse", []),
        "verbal_abuse": old.get("verbal_abuse", []),
        "threat": old.get("threat", []),
        "control": old.get("control", []),
        "fear": old.get("fear", []),
        "risk": old.get("risk", []),
        "patterns": old.get("patterns", []),
        "timeline": old.get("timeline", [])[-3:]  # Limit history tokens
    }

    # --- Step 3: LLM Communication ---
    scribe_packet = f"""
    --- CURRENT_RECORD (HISTORY) ---
    {json.dumps(clean_old_record, indent=2)}

    --- SYSTEM_FINDINGS (TRUST THESE) ---
    {json.dumps(system_findings, indent=2)}

    --- NEW_USER_MSG ---
    "{msg}"
    """

    result = json_out(
        UNIFIED_SCRIBE_PROMPT,
        scribe_packet,
        temperature=0.0
    )

    # --- Step 4: Output Validation ---
    if not isinstance(result, dict) or not result.get("is_relevant", False):
        if system_findings["risk_level"] in ["critical", "high"]:
            pass  # Force update logic could be placed here if needed
        else:
            return {"case_notes_updated": False}

    updated_data = result.get("updated_record", {})

    # --- Step 5: State Merging & Final Assembly ---
    merged = {}

    # Map keys directly from the "updated_record"
    keys_to_transfer = ["physical_abuse", "verbal_abuse", "threat", "control", "fear", "risk", "patterns"]

    for k in keys_to_transfer:
        new_val = updated_data.get(k)
        if new_val is not None:
            merged[k] = new_val
        else:
            merged[k] = clean_old_record.get(k, [])

    full_timeline = old.get("timeline", [])
    timeline_update = updated_data.get("timeline_update")

    if timeline_update and isinstance(timeline_update, dict) and timeline_update.get("event"):
        timeline_update["ts"] = now_ts()
        full_timeline.append(timeline_update)

    merged["timeline"] = full_timeline
    merged["context"] = old.get("context", [])
    merged["emotion"] = old.get("emotion", [])

    return {
        "case_notes": merged,
        "case_notes_updated": True
    }

print("[Agent] Loaded Case-Notes agent.")
