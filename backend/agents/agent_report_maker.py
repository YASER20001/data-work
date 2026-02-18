# backend/agents/agent_report_maker.py

from __future__ import annotations
import os
import uuid
import datetime
import base64  # <--- NEW IMPORT
from typing import Dict, List, Any
import pdfkit

# Backend Imports
from backend.core.state import AppState
from backend.core.llm_gateway import text as lm_text


# =========================================================================
# SECTION 1: DATA EXTRACTION UTILITIES
# =========================================================================

def _g(notes: Dict[str, List[str]], key: str) -> List[str]:
    """ Helper to safely extract lists from the clinical notes dictionary. """
    val = notes.get(key, [])
    return val if isinstance(val, list) else []


def build_raw_structure(notes: Dict[str, Any], state: AppState) -> Dict[str, str]:
    """
    Translates raw case notes and state variables into a structured dictionary
    of clinical report sections.
    """
    sections = {}

    # --- 1) Case Overview Generation ---
    # Maps specific indicator flags to narrative clinical summaries.
    overview_lines = []
    if _g(notes, "physical_abuse"): overview_lines.append("The user reports ongoing or repeated physical aggression.")
    if _g(notes, "verbal_abuse"):   overview_lines.append("Reports indicate verbal hostility.")
    if _g(notes, "threat"):         overview_lines.append("There are reported threats.")
    if _g(notes, "control"):        overview_lines.append("Behaviors aligning with coercive control.")

    sections["overview_raw"] = (" ".join(overview_lines) if overview_lines else "No clinical indicators documented.")

    # --- 2) Patterns & Behavioral Analysis ---
    patterns = _g(notes, "patterns")
    sections["patterns_raw"] = "\n".join(patterns) if patterns else "No behavioral patterns detected."

    # --- 3) Chronological Timeline ---
    # Formats history events into a readable list with dates.
    t = notes.get("timeline", [])
    timeline_lines = []
    if t and isinstance(t, list):
        for item in t:
            if isinstance(item, dict):
                event = item.get("event") or item.get("text") or ""
                date = item.get("date", "Unspecified date")
            else:
                event = str(item)
                date = "Unspecified date"
            timeline_lines.append(f"{date}: {event}")
        sections["timeline_raw"] = "\n".join(timeline_lines)
    else:
        sections["timeline_raw"] = "No chronological events documented."

    # --- 4) Clinical Context & Emotion ---
    context_items = _g(notes, "context")
    sections["context_raw"] = ("\n".join(context_items) if context_items else "No contextual psychological notes.")

    emotional_items = _g(notes, "emotion")
    sentiment_list = notes.get("sentiment", [""])
    sentiment = sentiment_list[0] if sentiment_list else ""
    emotional_text = "\n".join(emotional_items) if emotional_items else "No direct emotional expressions."
    emotional_text += f"\nSentiment reading: {sentiment}"
    sections["emotion_raw"] = emotional_text

    # --- 5) Legal & Administrative ---
    legal = getattr(state, "legal_reason", "") or "No legal considerations documented."
    sections["legal_raw"] = legal

    # --- 6) System-Derived Truths ---
    # Robust retrieval for Risk and Personality markers used for narrative grounding.
    risk = getattr(state, "risk_level", None)
    sections["system_risk_level"] = risk if risk else "needs_info"

    p_label = getattr(state, "personality_label", None)
    p_input = getattr(state, "personality", None)
    tone = getattr(state, "detected_tone", None)

    # Resolution Logic: Prefer Label -> Input -> Tone -> Unknown
    if p_label and p_label != "None":
        final_p = p_label
    elif p_input and p_input != "None":
        final_p = p_input
    elif tone and tone != "Tone: —":
        final_p = tone
    else:
        final_p = "Unknown"

    sections["system_personality"] = final_p

    return sections


# =========================================================================
# SECTION 2: LLM NARRATIVE GENERATION
# =========================================================================

def generate_llm_report(sections: Dict[str, str], language: str = "en") -> Dict[str, str]:
    """
    Utilizes LLM to transform raw extracted data into a professional
    clinical narrative overview for the final PDF.
    """
    prompts = {
        "llm_overview": "Write a clinical narrative overview summarizing the user's situation.",
        "llm_patterns": "Describe the behavioral patterns observed in the case.",
        "llm_timeline": "Summarize the chronological course of events in a structured clinical manner.",
        "llm_emotion": "Analyze emotional indicators and their clinical significance.",
        "llm_risk": "Provide a risk assessment narrative. You MUST use the system risk level exactly as provided, with no changes.",
        "llm_legal": "Summarize the legal considerations relevant to this case."
    }

    results = {}
    for key, instruction in prompts.items():
        system_prompt = f"""
You are a licensed clinical psychologist writing a section of a formal report.
Your tone must be clinical, objective, and structured.

STRICT RULES:
- Output MUST be in English only.
- Do NOT invent or modify risk or personality values.
- USE the system risk & personality exactly as provided.
- Follow the instruction given for this section.

System Risk Level: {sections['system_risk_level']}
System Personality: {sections['system_personality']}
"""
        user_prompt = f"""
Instruction for this section:
{instruction}

RAW CONTENT:
CASE OVERVIEW: {sections["overview_raw"]}
PATTERNS: {sections["patterns_raw"]}
TIMELINE: {sections["timeline_raw"]}
CONTEXT: {sections["context_raw"]}
EMOTION: {sections["emotion_raw"]}
LEGAL: {sections["legal_raw"]}
"""
        results[key] = lm_text(system_prompt, user_prompt, temperature=0.1).strip()

    return results


# =========================================================================
# SECTION 3: DOCUMENT RENDERING & PDF EXPORT
# =========================================================================

def load_html_template(path="backend/templates/clinical_report_template.html"):
    """ Reads the HTML template file used for document structure. """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[Template] Failed to load HTML template: {e}")
        return None


def get_base64_logo():
    """Reads the logo from disk and converts it to a Base64 string."""
    # Define your specific path here
    logo_path = r"C:\Users\Abdul\Downloads\rifd_logo.jpeg"

    if not os.path.exists(logo_path):
        print(f"[ReportMaker] Warning: Logo not found at {logo_path}")
        return ""

    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Return the full data URI
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"[ReportMaker] Failed to encode logo: {e}")
        return ""


def render_html_template(template: str, sections: Dict[str, str]) -> str:
    """ Injects clinical data into HTML placeholders for rendering. """
    if not template: return ""

    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    case_id = str(uuid.uuid4())[:8].upper()

    # NEW: Get the logo string
    logo_source = get_base64_logo()

    full_data = sections.copy()
    full_data["report_date"] = report_date
    full_data["case_id"] = case_id
    full_data["logo_source"] = logo_source  # <--- Injecting the variable

    # Pattern replacement for all sections
    for key, value in full_data.items():
        placeholder = "{{" + key + "}}"
        # Replace and handle None values safely
        template = template.replace(placeholder, str(value) if value is not None else "")

    return template


def save_report_to_pdf(html_text: str, filename="clinical_report.pdf"):
    """ Uses pdfkit and wkhtmltopdf to generate a permanent PDF file. """
    out_dir = "output_reports"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, filename)

    wkhtml_path = os.getenv("WKHTMLTOPDF_PATH")

    if not wkhtml_path:
        print("[ReportMaker] ERROR: WKHTMLTOPDF_PATH not found in environment.")
        return None

    if not os.path.exists(wkhtml_path):
        print(f"[ReportMaker] ERROR: Configured path does not exist: {wkhtml_path}")
        return None

    try:
        # options to allow local file access (sometimes needed even with base64)
        options = {
            'enable-local-file-access': None,
            'encoding': "UTF-8"
        }
        config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)
        pdfkit.from_string(html_text, pdf_path, configuration=config, options=options)
        return pdf_path
    except Exception as e:
        print(f"[ReportMaker] PDF Generation Failed: {e}")
        return None


# =========================================================================
# SECTION 4: CORE AGENT RUNNER
# =========================================================================

def run(state: AppState) -> Dict[str, Any]:
    """
    Main entry point for generating the PDF Report.
    Processes state, generates LLM summaries, and renders the final PDF file.
    """
    print("[ReportMaker] Starting background PDF generation...")

    # --- 1. Preparation ---
    case_notes = getattr(state, "case_notes", {}) or {}
    if hasattr(case_notes, 'dict'): case_notes = case_notes.dict()

    # --- 2. Narrative Generation ---
    try:
        raw_sections = build_raw_structure(case_notes, state)
        llm_sections = generate_llm_report(raw_sections, language="en")
    except Exception as e:
        print(f"[ReportMaker] Logic failed: {e}")
        return {"error": str(e)}

    # --- 3. Rendering ---
    template = load_html_template()
    if not template:
        print("[ReportMaker] Template missing.")
        return {"error": "Template missing"}

    # Aggregates LLM narrative and raw data for full injection
    content = {
        "llm_overview": llm_sections.get("llm_overview", ""),
        "llm_patterns": llm_sections.get("llm_patterns", ""),
        "llm_timeline": llm_sections.get("llm_timeline", ""),
        "llm_emotion": llm_sections.get("llm_emotion", ""),
        "llm_risk": llm_sections.get("llm_risk", ""),
        "llm_legal": llm_sections.get("llm_legal", ""),
        "overview": raw_sections.get("overview_raw", ""),
        "patterns": raw_sections.get("patterns_raw", ""),
        "timeline": raw_sections.get("timeline_raw", ""),
        "emotion": raw_sections.get("emotion_raw", ""),
        "system_personality": raw_sections.get("system_personality", ""),
        "system_risk_level": raw_sections.get("system_risk_level", ""),
        "legal": raw_sections.get("legal_raw", ""),

        # Field-specific injections for the clinical audit table
        "physical_abuse": ", ".join(case_notes.get("physical_abuse", [])),
        "verbal_abuse": ", ".join(case_notes.get("verbal_abuse", [])),
        "threat": ", ".join(case_notes.get("threat", [])),
        "control": ", ".join(case_notes.get("control", [])),
        "fear": ", ".join(case_notes.get("fear", [])),
        "risk": ", ".join(case_notes.get("risk", [])),
    }

    # --- 4. Export ---
    final_text = render_html_template(template, content)
    pdf_path = save_report_to_pdf(final_text)

    print(f"[ReportMaker] PDF Generated successfully: {pdf_path}")

    return {
        "final_report": final_text,
        "final_report_pdf": pdf_path,
    }


print("[Agent] Loaded Report Maker agent.")
