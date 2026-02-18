from __future__ import annotations

# --- CRITICAL FIX: Allow Nested Event Loops ---
import nest_asyncio
nest_asyncio.apply()
# ---------------------------------------------

import os
import io
import wave
import asyncio
import json
import ssl
import base64
import websockets
from typing import Optional, List, Dict, Any, Tuple, cast

from backend.core.state import AppState
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG / HELPERS ---

_CREDENTIALS_PATH = os.getenv(
    "SAFEASSIST_GOOGLE_CREDENTIALS_PATH",
    r"C:\Users\Abdul\OneDrive\Desktop\RIFD\backend\rag\gen-lang-client-0375881764-f3537a7fde3b.json"
)

_DEBUG = True 
TARGET_SR_DEFAULT = 44100 
CHANNELS = 1

_AR_RANGES = [
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)
]

def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a, b in _AR_RANGES)

def _detect_lang_from_text(text: str) -> str:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters: return "en"
    arabic_count = sum(1 for c in letters if _is_arabic_char(c))
    frac = arabic_count / max(1, len(letters))
    return "ar" if frac >= 0.25 else "en"

def _pcm_to_wav(raw_bytes: bytes, sample_rate: int) -> bytes:
    """Wraps raw PCM data into a valid WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframesraw(raw_bytes or b"")
    return buf.getvalue()

# --- GOOGLE STT ---

def _google_speech_client_and_types():
    try:
        from google.cloud import speech_v1p1beta1 as speech_beta
    except ImportError:
        try:
            from google.cloud import speech_v1 as speech_beta
        except ImportError:
            raise RuntimeError("google-cloud-speech is not installed.")

    creds = None
    if os.path.exists(_CREDENTIALS_PATH):
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(_CREDENTIALS_PATH)
        except Exception as e:
            if _DEBUG: print(f"[STT] Failed to load explicit credentials: {e}")

    try:
        client = speech_beta.SpeechClient(credentials=creds)
    except Exception as e:
        raise RuntimeError(f"Failed to create Google SpeechClient: {e}")

    return speech_beta, client

def _recognize_linear16_google(raw_bytes: bytes, sample_rate: int) -> Tuple[str, float]:
    try:
        speech, client = _google_speech_client_and_types()
    except RuntimeError as e:
        raise RuntimeError(f"google-cloud-speech is not ready. {e}")

    audio = speech.RecognitionAudio(content=raw_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        alternative_language_codes=["ar-SA"],
        enable_automatic_punctuation=True,
        model="latest_long",
    )

    try:
        req = speech.RecognizeRequest(config=config, audio=audio)
        resp = client.recognize(request=req)
    except Exception as e:
        print(f"[STT] Google recognize request failed: {e}")
        raise

    if resp is None: return "", 0.0

    texts, confs = [], []
    for res in cast(Any, resp).results or []:
        if getattr(res, "alternatives", None):
            best = res.alternatives[0]
            txt = (getattr(best, "transcript", "") or "").strip()
            if txt: texts.append(txt)
            conf_val = getattr(best, "confidence", None)
            if conf_val is not None:
                try: confs.append(float(conf_val))
                except: pass

    final_text = " ".join(texts)
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    return final_text, avg_conf

# --- HUME PROSODY ---

async def _hume_analyze_wav(wav_bytes: bytes) -> str:
    """Corrected Hume Streaming logic."""
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        print("[STT] ❌ No HUME_API_KEY found.")
        return "Tone: — (No Key)"

    # Note: Ensure the URL matches the documentation for streaming
    url = f"wss://api.hume.ai/v0/stream/models?api_key={api_key}"
    ssl_context = ssl.create_default_context()

    try:
        async with websockets.connect(url, ssl=ssl_context, open_timeout=10) as socket:
            b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

            # 1. Send the Audio Data
            payload = {
                "models": {"prosody": {}},
                "data": b64_audio
            }
            await socket.send(json.dumps(payload))

            # 2. Send flush signal (empty data)
            # CRITICAL FIX: You must include 'models' here too!
            flush_payload = {
                "models": {"prosody": {}},
                "data": ""
            }
            await socket.send(json.dumps(flush_payload))

            for i in range(20):
                try:
                    response = await asyncio.wait_for(socket.recv(), timeout=3.0)
                    data = json.loads(response)

                    if "error" in data:
                        print(f"[STT-HUME] ❌ API Error: {data['error']}")
                        return "Tone: (API Error)"

                    if "prosody" in data:
                        preds = data["prosody"].get("predictions", [])
                        if preds:
                            emotions = preds[0].get("emotions", [])
                            if emotions:
                                emotions.sort(key=lambda x: x["score"], reverse=True)
                                top_3 = [f"{e['name']} {e['score']:.2f}" for e in emotions[:3]]
                                return "Tone: " + " • ".join(top_3)

                except asyncio.TimeoutError:
                    continue

            return "Tone: Neutral (No speech detected)"

    except Exception as e:
        print(f"[STT-HUME] ❌ Connection Exception: {e}")
        return "Tone: (Conn Error)"

def _analyze_tone_sync(raw_bytes: bytes, sample_rate: int) -> str:
    """Synchronous wrapper for the async Hume analyzer."""
    try:
        wav = _pcm_to_wav(raw_bytes, sample_rate)
        return asyncio.run(_hume_analyze_wav(wav))
    except Exception as e:
        print(f"[STT] Tone Wrapper Fail: {e}")
        return f"Tone: (Error: {type(e).__name__})"

# ------------------------------- AGENT MAIN --------------------------------

def run(state: AppState) -> Dict[str, Any]:
    print("\n========= STT AGENT RUNNING =========")
    
    sr = getattr(state, "audio_sample_rate", TARGET_SR_DEFAULT) or TARGET_SR_DEFAULT
    print(f"[STT] Audio Sample Rate: {sr} Hz")

    raw: bytes = getattr(state, "audio_bytes", b"") or b""
    
    if not raw:
        print("[STT] ⚠️ No audio bytes in state.")
        return {
            "user_input": "",
            "detected_tone": "Tone: —",
            "messages": list(getattr(state, "messages", []) or [])
        }

    # 1) Google STT
    print("[STT] Calling Google...")
    text, conf = "", 0.0
    try:
        text, conf = _recognize_linear16_google(raw, sr)
    except Exception as e:
        print(f"[STT] Google Failed: {e}")
        raise RuntimeError(f"STT Error: {e}")

    # 2) Language Detection
    lang_current = getattr(state, "lang", "auto")
    lang_guess = _detect_lang_from_text(text) if text else None
    lang_new = lang_current
    if lang_current not in ("ar", "en"):
        lang_new = lang_guess if lang_guess else "en"

    # 3) Hume Tone
    print("[STT] Calling Hume...")
    tone = _analyze_tone_sync(raw, sr)
    
    if not tone:
        tone = "Tone: — (Empty)"

    # 4) Build Messages
    base_msgs = list(getattr(state, "messages", []) or [])
    if text:
        base_msgs.append({"role": "user", "content": text})

    note = "Audio transcribed." if lang_new == "en" else "تم نسخ الصوت."
    if text and conf:
        note += f" (conf≈{conf:.2f})"
    
    # Append tone to Assistant message
    if "Tone: —" not in tone:
        base_msgs.append({"role": "assistant", "content": f"{note} | {tone}"})
    else:
        base_msgs.append({"role": "assistant", "content": note})

    updates = {
        "user_input": text or "",
        "detected_tone": tone,
        "messages": base_msgs,
    }
    if lang_new != lang_current:
        updates["lang"] = lang_new

    # Cleanup internal state keys
    updates.pop("session_id", None)
    updates.pop("user_id", None)
    
    print(f"[STT] Finished. Tone result: {tone}")
    print("=====================================\n")
    return updates

def stt_healthcheck() -> Dict[str, Any]:
    return {"status": "ok", "method": "websockets"}

print("[Agent] Loaded stt agent.")
