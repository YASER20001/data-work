# =============================================================================
# RIFD — STREAMLENT FRONTEND (SafeAssist UI)
# =============================================================================

import streamlit as st
import sys
import os
import time
import wave
import subprocess
import datetime
from typing import Dict, Any

# --- Path Configuration ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Backend Integration ---
try:
    from backend.core.state import AppState, print_user_input
    from backend.pipeline_bootstrap import graph, rag_pipeline_instance
    from backend.agents.agent_stt import _recognize_linear16_google
    from backend.agents.agent_report_maker import run as generate_pdf_report
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.stop()

# =============================================================================
# PART 1: UI CONFIGURATION & SAFETY FEATURES
# =============================================================================

st.set_page_config(page_title="RIFD", layout="centered")
st.title("⚡ RIFD: Intelligent Assistant")


def quick_exit():
    """
    Emergency 'Quick Exit' function.
    Immediately hides the UI, redirects the browser to Google,
    and terminates the backend process for user safety.
    """
    hide_streamlit_style = """
        <style>
            [data-testid="sidebar"] { display: none !important; }
            header { display: none !important; }
            footer { display: none !important; }
            html, body, #root, #main { background-color: white !important; height: 100vh; margin: 0; padding: 0; }
            #root > div:nth-child(1) > div { display: none !important; }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.components.v1.html("""
        <script>
            window.open('https://www.google.com', '_blank');
            document.body.innerHTML = '';
            document.body.style.backgroundColor = 'white';
        </script>
    """, height=0)
    time.sleep(0.2)
    os._exit(0)


if st.sidebar.button("Quick Exit"):
    quick_exit()

# =============================================================================
# PART 2: SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_detected_tone" not in st.session_state:
    st.session_state.last_detected_tone = "Tone: —"

if "latest_state" not in st.session_state:
    st.session_state.latest_state = None

# --- FIX: Initialize Audio Reset Key ---
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# --- Voice Mode Configuration ---
VOICE_MODE_RAW = os.getenv("VOICE_MODE", "false")
VOICE_MODE_ENABLED = VOICE_MODE_RAW.strip().lower() in ("true", "1", "yes")

if VOICE_MODE_ENABLED:
    st.sidebar.info("Voice Mode is active. TTS audio will be generated.")

st.sidebar.markdown("---")
st.sidebar.subheader("Audio Signals")
tone_sidebar_placeholder = st.sidebar.empty()
tone_sidebar_placeholder.info(st.session_state.last_detected_tone)

#st.sidebar.markdown("---")
# st.sidebar.subheader("Live Voice Call")

# if st.sidebar.button("Start Live Call"):
#     try:
#         # Spawns the live call process in a separate console window
#         subprocess.Popen([sys.executable, "run_live_call.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
#         st.sidebar.success("Live call started! A new terminal window opened.")
#     except Exception as e:
#         st.sidebar.error(f"Failed to start live call: {e}")


# =============================================================================
# PART 3: UTILITY FUNCTIONS
# =============================================================================

def process_audio_input(audio_file):
    """Extracts raw frames and sample rate from recorded audio files."""
    try:
        with wave.open(audio_file, 'rb') as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate()
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None


def render_steps_log(steps_list):
    """Displays the execution path of the multi-agent graph."""
    for step_text in steps_list:
        st.write(step_text)


# =============================================================================
# PART 4: CORE PIPELINE ORCHESTRATION
# =============================================================================


def process_turn(user_text, audio_data=None, is_batch=False):
    """
    Manages a single conversation turn:
    1. Synchronizes local session history with the backend AppState.
    2. Persists clinical case notes across turns.
    3. Invokes the LangGraph pipeline with streaming updates.
    4. Renders the AI response and synthesized audio.
    """
    print_user_input(user_text)

    # UI Update: User Message
    with st.chat_message("user"):
        st.markdown(user_text)

    st.session_state.messages.append({"role": "user", "content": user_text})

    # Prepare AppState for the backend
    clean_messages_for_backend = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    state_init_kwargs: Dict[str, Any] = {
        "rag_pipeline": rag_pipeline_instance,
        "messages": clean_messages_for_backend,
        "user_input": user_text,
        "input_mode": "text"
    }

    # PERSISTENCE: Restore memory from previous turn
    if st.session_state.latest_state:
        previous = st.session_state.latest_state
        state_init_kwargs["case_notes"] = previous.case_notes

        # Maintain high-level assessments across the conversation turn
        for attr in ["risk_level", "personality", "personality_label", "legal_reason"]:
            if getattr(previous, attr, None):
                state_init_kwargs[attr] = getattr(previous, attr)

    # Audio Handling Logic
    if audio_data:
        raw_bytes, sample_rate = audio_data
        state_init_kwargs.update({
            "input_mode": "voice",
            "audio_bytes": raw_bytes,
            "audio_sample_rate": sample_rate
        })

    if VOICE_MODE_ENABLED:
        state_init_kwargs["output_mode"] = "voice"

    current_state = AppState(**state_init_kwargs)

    # Execute Pipeline & Render
    with st.chat_message("assistant"):
        current_steps_log = []
        label_text = "Processing..." if is_batch else "RAFIDA is thinking..."
        steps_container = st.status(label_text, expanded=not is_batch)

        with steps_container:
            total_start_time = time.time()
            last_step_time = total_start_time
            final_response_text = None
            last_state = current_state

            # STREAMING: Track node transitions in real-time
            for output in graph.stream(current_state):
                for node_name, node_update in output.items():
                    now = time.time()
                    step_duration = now - last_step_time
                    last_step_time = now

                    log_entry = f"Passed: **{node_name.upper()}** ({step_duration:.2f}s)"
                    st.write(log_entry)
                    current_steps_log.append(log_entry)

                    # Update local state snapshot from node delta
                    update_dict = dict(node_update) if hasattr(node_update, "items") else {}
                    current_state = current_state.model_copy(update=update_dict)
                    last_state = current_state

                    # Priority logic for final message selection
                    if "ai_message" in node_update:
                        final_response_text = node_update["ai_message"]
                    elif "ai_message_reviewed" in node_update:
                        final_response_text = node_update["ai_message_reviewed"]
                    elif "therapist_reply" in node_update:
                        final_response_text = node_update["therapist_reply"]

            total_duration = time.time() - total_start_time

            # Synchronize sidebar telemetry
            final_detected_tone = getattr(last_state, "detected_tone", "Tone: —")
            st.session_state.last_detected_tone = final_detected_tone
            if not is_batch:
                tone_sidebar_placeholder.info(final_detected_tone)

            st.session_state.latest_state = last_state
            steps_container.update(label=f"Response Ready! (Total: {total_duration:.2f}s)", state="complete",
                                   expanded=False)

        # Render Output Content
        if final_response_text:
            st.markdown(final_response_text)

            # --- FILE LOGGING (Engineering Record) ---
            try:
                with open("conversation_log.txt", "a", encoding="utf-8") as f:
                    f.write("--------------------------------------------------\n")
                    f.write(f"**User:** {user_text}\n\n")
                    f.write(f"**AI:** {final_response_text}\n\n")
                    f.write(f"**--> Stats:** (Took {total_duration:.2f}s)\n")
                    f.write("--------------------------------------------------\n\n")

            except Exception as e:
                print(f"Error saving extended log: {e}")

            final_tts_audio = getattr(current_state, "tts_audio", None)
            if final_tts_audio:
                st.audio(final_tts_audio, format="audio/wav", autoplay=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response_text,
                "steps_log": current_steps_log,
                "execution_time": f"{total_duration:.2f}",
                "tts_audio": final_tts_audio,
            })
        else:
            st.warning("No text output detected from agents.")


# =============================================================================
# PART 5: CLINICAL ADMINISTRATION & SIMULATION TOOLS
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Administration")

# REPORT GENERATION
if st.sidebar.button("Generate Report"):
    if st.session_state.latest_state is None:
        st.sidebar.warning("No conversation data available.")
    else:
        with st.sidebar.status("Generating Clinical PDF...", expanded=True) as status:
            try:
                report_result = generate_pdf_report(st.session_state.latest_state)
                pdf_path = report_result.get("final_report_pdf")

                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.sidebar.download_button(
                            label="Download Final Report",
                            data=f,
                            file_name="Clinical_Case_Report.pdf",
                            mime="application/pdf"
                        )
                    status.update(label="Report Generated!", state="complete")
                else:
                    status.update(label="Failed to create PDF", state="error")
            except Exception as e:
                st.sidebar.error(f"Report Error: {e}")
                status.update(label="Error", state="error")

# SYSTEM RESET
if st.sidebar.button("Start New Case (Reset)"):
    st.session_state.messages = []
    st.session_state.latest_state = None
    st.session_state.last_detected_tone = "Tone: —"
    st.session_state.audio_key = 0 # Reset audio key too
    st.rerun()

# # BATCH SIMULATION
# st.sidebar.markdown("---")
# st.sidebar.subheader("Batch Simulation")
# st.sidebar.caption("Run messages from batch_inputs.txt")
#
# if st.sidebar.button("Run Batch Simulation"):
#     if os.path.exists("batch_inputs.txt"):
#         with open("batch_inputs.txt", "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f.readlines() if line.strip()]
#
#         st.toast(f"Loaded {len(lines)} messages. Starting simulation...")
#         st.session_state["is_running_batch"] = True
#
#         for i, batch_msg in enumerate(lines):
#             st.markdown(f"**--- Processing Batch Message {i + 1}/{len(lines)} ---**")
#             process_turn(batch_msg, audio_data=None, is_batch=True)
#             time.sleep(1)
#
#         st.session_state["is_running_batch"] = False
#         st.success("Batch Simulation Complete!")
#     else:
#         st.sidebar.error("'batch_inputs.txt' not found.")

# =============================================================================
# PART 6: MAIN INTERACTION LOOP
# =============================================================================

# Display Message History
if not st.session_state.get("is_running_batch", False):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tts_audio"):
                st.audio(msg["tts_audio"], format='audio/wav', autoplay=True)
            if msg["role"] == "assistant" and "steps_log" in msg:
                with st.expander(f"Details ({msg.get('execution_time', '0.00')}s)"):
                    render_steps_log(msg["steps_log"])

# Input Handlers
final_user_input = None
raw_bytes = None
sample_rate = None
processed_audio = False # Flag to track audio success

with st.popover("Voice Note"):
    # FIX: Added dynamic key to allow auto-reset
    audio_value = st.audio_input("Record your message", key=f"audio_rec_{st.session_state.audio_key}")

text_input = st.chat_input("Type your message here...")

if audio_value:
    if text_input: text_input = None
    with st.spinner("Transcribing..."):
        raw_bytes, sample_rate = process_audio_input(audio_value)
        if raw_bytes and sample_rate:
            try:
                transcript, conf = _recognize_linear16_google(raw_bytes, sample_rate)
                if transcript:
                    final_user_input = transcript
                    processed_audio = True # Mark as successful audio
                    st.toast(f"Confidence: {conf:.2f}")
                else:
                    st.warning("Could not understand audio.")
            except Exception as e:
                st.error(f"STT Error: {e}")
elif text_input:
    final_user_input = text_input

# Trigger Logic
if final_user_input:
    # Deduplication check
    if st.session_state.messages and st.session_state.messages[-1]["content"] == final_user_input:
        # If duplicated but it was audio, force a reset anyway
        if processed_audio:
            st.session_state.audio_key += 1
            st.rerun()
        st.stop()

    audio_pkg = (raw_bytes, sample_rate) if (raw_bytes and sample_rate) else None
    process_turn(final_user_input, audio_data=audio_pkg, is_batch=False)

    # FIX: Auto-reset logic for voice notes
    if processed_audio:
        st.session_state.audio_key += 1
        st.rerun()
