# backend/agents/agent_legal_review.py
from __future__ import annotations
from typing import Dict, Any, List

from backend.core.state import AppState
from backend.core.llm_gateway import json_out
from backend.core.utils import last_user_text, history_window

# =============================================================================
# SECTION 1: SYSTEM PROMPTS & TEMPLATES
# =============================================================================

# --- GATEKEEPER: Orchestrates the decision to use legal RAG or approve as-is ---
GATEKEEPER_SYSTEM = (
    "You are a Legal Orchestrator for 'Rift' (Saudi Law context). "
    "Analyze the conversation and the Therapist's Draft.\n\n"

    "### TASK 1: DECISION\n"
    "1. **APPROVE (Safe/Casual):** If user is venting, telling a story, or asking for general support.\n"
    "2. **REJECT (Needs Law):** If user asks a specific legal question (e.g. 'Who takes custody?', 'What is my alimony?') "
    "AND the draft lacks specific legal citation.\n\n"

    "### TASK 2: QUERY OPTIMIZATION (Only if REJECT)\n"
    "- Generate a precise search query for the Vector DB.\n"
    "- **RULE:** Remove conversational context. Focus on LEGAL ENTITIES.\n"
    "  * Example: 'Custody after divorce' -> Query: 'شروط الحضانة نظام الأحوال الشخصية' (Remove 'Divorce' distraction).\n\n"

    "OUTPUT JSON:\n"
    "{\n"
    "  \"decision\": \"APPROVE\" | \"REJECT\",\n"
    "  \"optimized_query\": \"(If REJECT) The cleaned Arabic legal query\",\n"
    "  \"legal_intent\": \"(If REJECT) Brief note of what to look for\"\n"
    "}"
)

GATEKEEPER_TEMPLATE = """
[CHAT HISTORY (Last 5)]:
{history}

[USER INPUT]: {user_input}

[THERAPIST DRAFT]:
{ai_message}
"""

# --- SELECTOR: Validates retrieved laws against user intent ---
SELECTOR_SYSTEM = (
    "You are a Legal Research Assistant. You have a 'User Legal Intent' and 'Raw Laws'.\n"
    "Your Job: Select ONLY the Articles that answer the intent.\n\n"
    "OUTPUT JSON:\n"
    "{\n"
    "  \"found_relevant_law\": boolean, \n"
    "  \"instruction\": \"If found, write: 'Regarding [Intent], [Article Name] states...'. If NOT found, write: 'No relevant law found.'\"\n"
    "}"
)

SELECTOR_TEMPLATE = """
[USER LEGAL INTENT]: {legal_intent}

--- [RAW LAWS FROM DATABASE] ---
{rag_dump}
--- [END RAW LAWS] ---

TASK: Do these laws answer the intent?
"""

# =============================================================================
# SECTION 2: HELPER UTILITIES
# =============================================================================

def _extract_tag_and_content(result: Any) -> tuple[str, str]:
    """
    Normalizes metadata extraction from various RAG result formats.
    Prioritizes Article References followed by Source file names.
    """
    if isinstance(result, dict):
        meta = result.get('metadata') if 'metadata' in result else result
        content = result.get('text') or result.get('page_content') or ""
    else:
        meta = getattr(result, 'metadata', {})
        content = getattr(result, 'page_content', "")

    ref = meta.get('article_ref')
    src = meta.get('source')

    if src and isinstance(src, str):
        src = src.replace('.pdf', '').replace('_', ' ').strip()

    if ref and src:
        tag = f"{ref} - {src}"
    elif ref:
        tag = ref
    elif src:
        tag = src
    else:
        tag = "Law Context"

    return tag, content

# =============================================================================
# SECTION 3: CORE AGENT EXECUTION
# =============================================================================

def run(state: AppState) -> Dict[str, Any]:
    """
    Main execution loop for legal compliance and Islamic law citation.
    """
    ai_draft = (getattr(state, "ai_message_draft", None) or "").strip()
    retry_count = int(getattr(state, "legal_retry_count", 0) or 0)
    messages = list(getattr(state, "messages", []) or [])
    user_input = last_user_text(messages)
    rag_pipeline = getattr(state, "rag_pipeline", None)

    # --- Step 1: Loop Prevention ---
    # Auto-approves the draft if a correction loop has already occurred once.
    if retry_count >= 1:
        return {
            "route": "translation_final",
            "ai_message_reviewed": ai_draft,
            "legal_retry_count": 0,
            "legal_reason": "Approved (Retry)"
        }

    # --- Step 2: Gatekeeper Decision ---
    # Determines if the message requires legal verification.
    history_str = history_window(messages, n=5)
    gate_prompt = GATEKEEPER_TEMPLATE.format(
        history=history_str, user_input=user_input, ai_message=ai_draft
    )
    gate_out = json_out(GATEKEEPER_SYSTEM, gate_prompt, temperature=0.0) or {}
    decision = gate_out.get("decision", "APPROVE").upper()

    if decision == "APPROVE":
        return {
            "route": "translation_final",
            "ai_message_reviewed": ai_draft,
            "legal_retry_count": 0,
            "legal_reason": "Approved by Gatekeeper"
        }

    # --- Step 3: Legal RAG Search ---
    # Performs vector search based on optimized legal queries.
    search_query = gate_out.get("optimized_query", user_input)
    legal_intent = gate_out.get("legal_intent", "Check legal facts")

    rag_dump = ""
    if rag_pipeline:
        try:
            results = rag_pipeline.search_legal_review(search_query, k=8)
            formatted_results = [f"[{t}]: {c[:500]}..." for t, c in [_extract_tag_and_content(r) for r in results]]
            rag_dump = "\n\n".join(formatted_results)
        except Exception as e:
            rag_dump = "(Database Error)"

    # Fallback if no laws are found.
    if not rag_dump:
        return {
            "route": "translation_final",
            "ai_message_reviewed": ai_draft,
            "legal_retry_count": 0,
            "legal_reason": "Approved (RAG Empty)"
        }

    # --- Step 4: Fact Validation (Selector) ---
    # Cross-references found laws with the user's specific legal intent.
    selector_prompt = SELECTOR_TEMPLATE.format(legal_intent=legal_intent, rag_dump=rag_dump)
    sel_out = json_out(SELECTOR_SYSTEM, selector_prompt, temperature=0.0) or {}
    instruction = sel_out.get("instruction", "No relevant law found.")
    found_law = sel_out.get("found_relevant_law", False)

    if not found_law or "no relevant law" in instruction.lower():
        return {
            "route": "translation_final",
            "ai_message_reviewed": ai_draft,
            "legal_retry_count": 0,
            "legal_reason": "Approved (Laws retrieved were irrelevant)"
        }

    # --- Step 5: Routing ---
    # Returns the instruction to the therapist agent for draft correction.
    return {
        "route": "therapist",
        "legal_retry_count": retry_count + 1,
        "legal_reason": f"Gatekeeper rejected previous draft.\nINTENT: {legal_intent}\nINSTRUCTION: {instruction}"
    }

print("[Agent] Loaded Legal agent")
