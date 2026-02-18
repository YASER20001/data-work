# backend/api.py

# =============================================================================
# RIFD — FASTAPI ENTRYPOINT
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel
from backend.pipeline_bootstrap import graph, AppState

# Initialize FastAPI application
app = FastAPI(title="SafeAssist RIFD API", version="1.0")


# =============================================================================
# PART 1: DATA MODELS
# =============================================================================

class MessageInput(BaseModel):
    """Schema for incoming API requests."""
    message: str
    mode: str = "text"
    output: str = "text"


# =============================================================================
# PART 2: ENDPOINT DEFINITIONS
# =============================================================================

@app.post("/safeassist")
def run_safeassist(payload: MessageInput):
    """
    Primary API endpoint to trigger the RIFD intelligent pipeline.

    1. Initializes a fresh AppState.
    2. Injects the user payload.
    3. Invokes the LangGraph orchestration.
    4. Returns a condensed summary of the agent results.
    """
    # Create a fresh internal state for the request
    state = AppState()

    # Populate state with user input and configuration
    state.messages.append({"role": "user", "content": payload.message})
    state.user_input = payload.message
    state.input_mode = payload.mode
    state.output_mode = payload.output

    # Execute the LangGraph workflow
    # Result is returned as a dictionary of updates from the graph nodes
    result = graph.invoke(state)

    # Construct the JSON response for the client
    return {
        "intent": result.get("intent"),
        "risk_level": result.get("risk_level"),
        "reply": result.get("ai_message") or result.get("ai_message_reviewed"),
        "raw": result,  # Includes full state output for detailed backend debugging
    }
