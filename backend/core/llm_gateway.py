# core/llm_gateway.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional, Callable, TypeVar
import re, random

import google.generativeai as genai
from google.generativeai import generative_models
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
from .env_config import load as load_env, get
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# =============================================================================
# PART 1: MODULE CONFIGURATION & STATE
# =============================================================================

_GEMINI_READY = False
GEMINI_MODEL_NAME = None

def configure_gemini() -> None:
    """Initializes the Gemini API with credentials and model selection."""
    global _GEMINI_READY, GEMINI_MODEL_NAME
    load_env()
    api_key = get("GEMINI_API_KEY", required=True)
    model_name = get("GEMINI_MODEL", "gemini-2.5-flash")
    os.environ["GOOGLE_API_KEY"] = str(api_key)
    genai.configure(api_key=api_key)
    GEMINI_MODEL_NAME = model_name
    _GEMINI_READY = True

def configure():
    """Public wrapper for Gemini configuration."""
    return configure_gemini()

def _ensure_configured() -> None:
    """Internal check to ensure Gemini is ready before processing calls."""
    if not _GEMINI_READY or not GEMINI_MODEL_NAME:
        configure_gemini()


# =============================================================================
# PART 2: INTERNAL JSON PARSING UTILITIES
# =============================================================================

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

def _safe_json(text: str) -> dict:
    """Safely extracts and parses JSON data from LLM text responses."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        # Fallback: regex search if the LLM includes conversational filler
        m = _JSON_RE.search(text.strip())
        return json.loads(m.group(0)) if m else {}


# =============================================================================
# PART 3: MODEL FACTORY (WITH SAFETY OVERRIDES)
# =============================================================================

def model(system: str | None = None,
          response_mime_type: str | None = None,
          temperature: float = 0.2,
          **kwargs):
    """
    Creates a GenerativeModel instance with specific safety thresholds
    tailored for clinical and legal sensitivity in domestic abuse contexts.
    """
    mime = kwargs.pop("mime", None)
    mime_type = response_mime_type or mime

    _ensure_configured()

    generation_config = {"temperature": float(temperature)}
    if mime_type:
        generation_config["response_mime_type"] = mime_type

    # High-sensitivity bypass: Allowing critical discussion of abuse/harassment
    # for assessment purposes while maintaining high blocks for sexually explicit content.
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        system_instruction=system,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )


# =============================================================================
# PART 4: CONVENIENCE WRAPPERS (TEXT & JSON)
# =============================================================================

def text(system: str, user: str, temperature: float = 0.3) -> str:
    """Simple wrapper for text-only generation with error handling."""
    try:
        m = model(system=system, temperature=temperature)
        r = m.generate_content(user)
        return (getattr(r, "text", "") or "").strip()
    except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded):
        return ""
    except ValueError:
        # Catch content blocking errors
        return ""
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return ""


def json_out(system: str, user: str, temperature: float = 0.2) -> dict:
    """
    Advanced JSON output helper with exponential backoff and quota retry logic.
    Specifically handles safety blocks and 429 Resource Exhausted errors.
    """
    m = model(system=system, response_mime_type="application/json", temperature=temperature)
    max_retries, base_wait = 4, 6.0

    for attempt in range(1, max_retries + 1):
        try:
            r = m.generate_content(user)
            return _safe_json(getattr(r, "text", "") or "{}")

        except ResourceExhausted as e:
            msg = str(e)
            mobj = re.search(r"retry in (\d+(\.\d+)?)s", msg)
            wait_s = float(mobj.group(1)) if mobj else base_wait
            wait_s += random.uniform(0.5, 1.5)

            if "PerDayPerProjectPerModel" in msg or "PerDay" in msg:
                print("[Gemini] Daily quota exhausted; returning empty JSON.")
                return {}

            if attempt == max_retries:
                return {}

            print(f"[Retry {attempt}/{max_retries}] Gemini quota hit. Waiting {wait_s:.1f}s...")
            time.sleep(wait_s)
            base_wait = max(base_wait, wait_s) + 2.0

        except ValueError as e:
            # Handle empty candidates (Google safety block triggers)
            if "candidates" in str(e) or "quick accessor" in str(e):
                print(f"[Gemini Blocked] Safety filters triggered despite settings. Returning empty.")
                return {}

            print(f"[Error {attempt}] {type(e).__name__}: {e}")
            if attempt == max_retries: return {}
            time.sleep(1)

        except Exception as e:
            print(f"[Error {attempt}] {type(e).__name__}: {e}")
            if attempt == max_retries:
                return {}
            time.sleep(base_wait)
            base_wait += 2.0

    return {}
