# backend/agents/agent_tts.py

from __future__ import annotations

import os
import base64
from typing import Dict, Any

# =============================================================================
# PART 1: GOOGLE TTS BASE IMPORTS & CONFIGURATION
# =============================================================================
try:
    from google.cloud import texttospeech
    from google.oauth2 import service_account

    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    print("WARNING: google-cloud-texttospeech is not installed.")
    texttospeech = None
    service_account = None
    GOOGLE_TTS_AVAILABLE = False

# Path to the Google Cloud service account JSON file
_CREDENTIALS_PATH = os.getenv(
    "SAFEASSIST_GOOGLE_CREDENTIALS_PATH",
    r"C:\Users\Abdul\OneDrive\Desktop\RIFD\backend\rag\gen-lang-client-0375881764-f3537a7fde3b.json"
)


# =============================================================================
# PART 2: CLIENT INITIALIZER
# =============================================================================
def _google_tts_client():
    """
    Creates and returns a Google TTS client.
    Returns None if setup fails.
    """
    if not GOOGLE_TTS_AVAILABLE:
        print("[TTS] google-cloud-texttospeech not installed.")
        return None

    creds = None
    if os.path.exists(_CREDENTIALS_PATH):
        try:
            creds = service_account.Credentials.from_service_account_file(_CREDENTIALS_PATH)
        except Exception as e:
            print(f"[TTS] Failed loading service account: {e}")
            return None
    else:
        print(f"[TTS] Credential file missing at: {_CREDENTIALS_PATH}")

    try:
        return texttospeech.TextToSpeechClient(credentials=creds)
    except Exception as e:
        print(f"[TTS] Failed to create TTS client: {e}")
        return None


# =============================================================================
# PART 3: AUDIO SYNTHESIS LOGIC
# =============================================================================
def _synthesize_audio(text: str, lang_code: str) -> bytes | None:
    """
    Synthesizes speech from text using Google TTS.
    Returns raw WAV bytes or None.
    """
    client = _google_tts_client()
    if not client:
        print("[TTS] Client not available.")
        return None

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Normalize language and select Wavenet voices
    lang_code = lang_code.lower().strip()
    if lang_code.startswith("ar"):
        language_code = "ar-XA"
        voice_name = "ar-XA-Wavenet-B"
    else:
        language_code = "en-US"
        voice_name = "en-US-Wavenet-D"

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    # Configure audio to LINEAR16 (WAV)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        audio_data = response.audio_content

        if not audio_data or len(audio_data) < 100:
            print("[TTS] Response received but empty/too small.")
            return None

        return audio_data

    except Exception as e:
        print(f"[TTS] API error: {e}")
        return None


# =============================================================================
# PART 4: MAIN AGENT ENTRYPOINT
# =============================================================================
def run(state: Any) -> Dict[str, Any]:
    """
    TTS Agent entrypoint.
    Called by the pipeline graph to convert AI text responses into speech.
    """

    # Local import to avoid circular dependency
    from backend.core.env_config import get

    # Check if VOICE MODE is enabled in environment settings
    voice_mode = ((get("VOICE_MODE") or "false").strip().lower() == "true")
    if not voice_mode:
        print("[TTS] Skipped (VOICE_MODE disabled).")
        return {}

    # Safely retrieve text from AppState by checking multiple potential keys
    tts_text = (
        getattr(state, "ai_message", None) or
        getattr(state, "ai_message_reviewed", None) or
        getattr(state, "therapist_reply", None) or
        ""
    )

    lang_code = getattr(state, "lang", "en")

    if not tts_text or len(tts_text.strip()) < 4:
        print("[TTS] No valid text to synthesize.")
        return {
            "tts_text": "",
            "tts_audio": None,
            "tts_audio_b64": None,
            "audio_path": None
        }

    print(f"[TTS] Synthesizing ({lang_code}): {tts_text[:50]}...")

    # Generate raw audio bytes
    audio_wav = _synthesize_audio(tts_text, lang_code)

    if not audio_wav:
        print("[TTS] Synthesis failed.")
        return {
            "tts_text": tts_text,
            "tts_audio": None,
            "tts_audio_b64": None,
            "audio_path": None
        }

    # Convert to Base64 for easier transport to the web UI
    audio_b64 = base64.b64encode(audio_wav).decode("utf-8")

    print(f"[TTS] Generated {len(audio_wav)} bytes of audio.")

    return {
        "tts_text": tts_text,
        "tts_audio": audio_wav,
        "tts_audio_b64": audio_b64,
        "audio_path": "output.wav"
    }

print("[Agent] Loaded tts agent.")
