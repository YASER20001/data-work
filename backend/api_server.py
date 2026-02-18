import asyncio
import contextlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import hashlib

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core.state import AppState
from backend.pipeline_bootstrap import graph, report_maker_node_wrapped # Logical error will be fixed next semester

load_dotenv()

# =============================================================================
# PART 1: GLOBAL CONFIGURATION & APP INITIALIZATION
# =============================================================================

# Session management constraints
SESSION_IDLE_SECONDS = 5 * 60  # Auto-expire sessions after 5 minutes of inactivity
PRUNE_INTERVAL_SECONDS = 60  # Background cleanup frequency

app = FastAPI(title="Rifd Chat API")

# Middleware: Cross-Origin Resource Sharing (CORS) configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory volatile storage for sessions, users, and feedback
SESSIONS: Dict[str, AppState] = {}
USERS: Dict[str, AppState] = {}
USER_FEEDBACK: List[Dict[str, Any]] = []
THERAPIST_FEEDBACK: List[Dict[str, Any]] = []


# =============================================================================
# PART 2: SESSION & USER STATE MANAGEMENT HELPERS
# =============================================================================

def _ensure_session_id(s: AppState) -> None:
    """Assigns a unique identifier to a session if not already present."""
    if not getattr(s, "session_id", None):
        s.session_id = str(uuid4())


def _get_user_state(user_id: str) -> AppState:
    """Retrieves an existing user state or initializes a new one with clinical fields."""
    if user_id in USERS:
        return USERS[user_id]
    u = AppState()
    u.user_id = user_id
    if not getattr(u, "case_notes", None):
        u.case_notes = []
    if not getattr(u, "history_window", None):
        u.history_window = []
    if not getattr(u, "action_summary", None):
        u.action_summary = []
    if not getattr(u, "sessions_log", None):
        u.sessions_log = []
    USERS[user_id] = u
    return u


def _messages_as_tuples(s: AppState) -> List[dict]:
    """Formats the last 10 messages into chronological tuples for history preservation."""
    out: List[dict] = []
    ts = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(getattr(s, "last_activity_at", time.time())),
    )
    for m in getattr(s, "messages", [])[-10:]:
        out.append(
            {
                "role": m.get("role", ""),
                "text": (m.get("content") or "")[:500],
                "ts": ts,
            }
        )
    return out


def _finalize_session_into_user(user: AppState, s: AppState) -> None:
    """Archives session metrics and history into the persistent user profile."""
    sessions_log = getattr(user, "sessions_log", [])
    sessions_log.append(
        {
            "session_id": getattr(s, "session_id", ""),
            "started_at": getattr(s, "current_session_started_at", 0.0) or 0.0,
            "ended_at": getattr(s, "last_activity_at", 0.0) or 0.0,
            "events": len(getattr(s, "report_events", []) or []),
            "last_paragraph_len": len(getattr(s, "report_text", "") or ""),
        }
    )
    user.sessions_log = sessions_log
    user.history_window = (
        getattr(user, "history_window", []) + _messages_as_tuples(s)
    )[-10:]
    USERS[user.user_id] = user


def _seed_session_from_user(user: AppState) -> AppState:
    """Initializes a new session context pre-loaded with historical user data."""
    s = AppState()
    _ensure_session_id(s)
    s.user_id = user.user_id
    s.case_notes = list(getattr(user, "case_notes", []))
    s.history_window = list(getattr(user, "history_window", []))
    s.action_summary = list(getattr(user, "action_summary", []))
    now = time.time()
    s.current_session_started_at = now
    s.last_activity_at = now
    if not getattr(s, "messages", None):
        s.messages = []
    SESSIONS[s.session_id] = s
    return s


def _get_or_create_session(payload_user_id: str, session_id: Optional[str]) -> AppState:
    """Resumes an existing session or handles creation/expiry logic."""
    user = _get_user_state(payload_user_id or "anon")
    now = time.time()
    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
        last = getattr(s, "last_activity_at", None)
        if last and (now - last) <= SESSION_IDLE_SECONDS:
            return s
        _finalize_session_into_user(user, s)
    return _seed_session_from_user(user)


# =============================================================================
# PART 3: BACKGROUND WORKERS & LIFECYCLE HOOKS
# =============================================================================

async def _prune_sessions_loop() -> None:
    """Continuous background loop to clean up idle sessions."""
    try:
        while True:
            try:
                now = time.time()
                to_delete = []
                for sid, s in list(SESSIONS.items()):
                    last = getattr(s, "last_activity_at", None)
                    if last and (now - last) > SESSION_IDLE_SECONDS:
                        to_delete.append(sid)
                for sid in to_delete:
                    del SESSIONS[sid]
            except Exception:
                traceback.print_exc()
            await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        pass


@app.on_event("startup")
async def _startup():
    """Initializes pruning task on server startup."""
    app.state._prune_task = asyncio.create_task(_prune_sessions_loop())


@app.on_event("shutdown")
async def _shutdown():
    """Cleanly cancels background tasks on server shutdown."""
    task: asyncio.Task = getattr(app.state, "_prune_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# =============================================================================
# PART 4: DATA SCHEMAS (PYDANTIC MODELS)
# =============================================================================

class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = "anon"


class ChatOut(BaseModel):
    session_id: str
    intent: Optional[str] = None
    confidence: float = 0.0
    route_reason: Optional[str] = None
    reply: str


class AdminStats(BaseModel):
    total_users: int
    total_sessions: int
    total_feedback: int


class UserFeedbackIn(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: int
    comment: Optional[str] = None


class TherapistSessionOut(BaseModel):
    session_id: str
    user_id: str
    scheduled_for: str
    status: str


class TherapistReportOut(BaseModel):
    case_id: str
    user_id: str
    created_at: str
    report_url: str


class TherapistNotificationOut(BaseModel):
    id: str
    title: str
    body: str
    created_at: str


class TherapistFeedbackIn(BaseModel):
    therapist_id: str
    session_id: Optional[str] = None
    risk_level: Optional[str] = None
    notes: Optional[str] = None


# =============================================================================
# PART 5: AUTHENTICATION & ACCOUNT PERSISTENCE
# =============================================================================

USERS_DB_PATH = Path("data/accounts.json")
USERS_DB_PATH.parent.mkdir(exist_ok=True)


class Account(BaseModel):
    id: str
    email: str
    password_hash: str
    role: str  # 'user' | 'therapist' | 'admin'


class LoginIn(BaseModel):
    email: str
    password: str


class LoginOut(BaseModel):
    user_id: str
    role: str


class AdminUserIn(BaseModel):
    email: str
    password: str
    role: str


class AdminUserOut(BaseModel):
    id: str
    email: str
    role: str


def _hash_password(pw: str) -> str:
    """Hashes passwords using SHA-256 for secure storage."""
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _load_accounts() -> Dict[str, Account]:
    """Loads credentials from local JSON database."""
    if not USERS_DB_PATH.exists():
        return {}
    data = json.loads(USERS_DB_PATH.read_text(encoding="utf-8") or "[]")
    accounts: Dict[str, Account] = {}
    for raw in data:
        acc = Account(**raw)
        accounts[acc.id] = acc
    return accounts


def _save_accounts(accounts: Dict[str, Account]) -> None:
    """Saves credentials to local JSON database."""
    USERS_DB_PATH.write_text(
        json.dumps([a.model_dump() for a in accounts.values()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


ACCOUNTS: Dict[str, Account] = _load_accounts()


def _ensure_seed_admin():
    """Bootstraps a master administrator account from environment variables."""
    if any(a.role == "admin" for a in ACCOUNTS.values()):
        return

    admin_email = os.getenv("RIFD_ADMIN_EMAIL")
    admin_password = os.getenv("RIFD_ADMIN_PASSWORD")

    if not admin_email or not admin_password:
        print("[Auth] No RIFD_ADMIN_EMAIL/RIFD_ADMIN_PASSWORD set – no seed admin created.")
        return

    admin_id = str(uuid4())
    admin = Account(
        id=admin_id,
        email=admin_email,
        password_hash=_hash_password(admin_password),
        role="admin",
    )
    ACCOUNTS[admin.id] = admin
    _save_accounts(ACCOUNTS)
    print(f"[Auth] Seed admin created: {admin_email}")


_ensure_seed_admin()


# =============================================================================
# PART 6: CORE API ENDPOINTS (CHAT, AUDIO, LIFECYCLE)
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {"ok": True}


@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    """Primary entry point for text-based chat interactions."""
    s = _get_or_create_session(payload.user_id or "anon", payload.session_id)
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if not getattr(s, "messages", None):
        s.messages = []
    s.messages.append({"role": "user", "content": msg})
    s.last_activity_at = time.time()

    try:
        # Invoke the LangGraph orchestration pipeline
        s = graph.invoke(s, config={"recursion_limit": 12})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {type(e).__name__}: {e}")

    last_assistant = next(
        (m for m in reversed(getattr(s, "messages", [])) if m.get("role") == "assistant"),
        {},
    )
    reply = (last_assistant.get("content") or "").strip() or "…"

    return ChatOut(
        session_id=s.session_id,
        intent=getattr(s, "intent", None),
        confidence=float(getattr(s, "confidence", 0.0) or 0.0),
        route_reason=getattr(s, "route_reason", None),
        reply=reply,
    )


@app.post("/api/audio", response_model=ChatOut)
async def audio(
    request: Request,
    session_id: Optional[str] = None,
    sr: int = 44100,
    user_id: Optional[str] = "anon",
):
    """Processes raw audio input through STT and the RAG pipeline."""
    s = _get_or_create_session(user_id or "anon", session_id)
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "Empty audio body")
    s.audio_bytes = raw
    s.audio_sample_rate = int(sr)
    s.last_activity_at = time.time()

    try:
        s = graph.invoke(s)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {type(e).__name__}: {e}")

    last_assistant = next(
        (m for m in reversed(getattr(s, "messages", [])) if m.get("role") == "assistant"),
        {},
    )
    reply = (last_assistant.get("content") or "").strip() or "…"

    return ChatOut(
        session_id=s.session_id,
        intent=getattr(s, "intent", None),
        confidence=float(getattr(s, "confidence", 0.0) or 0.0),
        route_reason=getattr(s, "route_reason", None),
        reply=reply,
    )


@app.post("/api/end/{session_id}", response_model=ChatOut)
def end_session(session_id: str, user_id: Optional[str] = "anon"):
    """Terminates a session, triggers report generation, and finalizes user data."""
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    s.session_end = True
    if not getattr(s, "messages", None):
        s.messages = []
    s.messages.append({"role": "user", "content": "[session ended by user]"})

    try:
        s = report_maker_node_wrapped(s)
    finally:
        user = _get_user_state(user_id or getattr(s, "user_id", "anon"))
        _finalize_session_into_user(user, s)
        SESSIONS.pop(session_id, None)

    last_assistant = next(
        (m for m in reversed(getattr(s, "messages", [])) if m.get("role") == "assistant"),
        {},
    )
    reply = (last_assistant.get("content") or "").strip() or "…"
    return ChatOut(session_id=session_id, reply=reply)


# =============================================================================
# PART 7: AUTHENTICATION, ADMIN, & THERAPIST PORTAL ENDPOINTS
# =============================================================================

@app.post("/api/login", response_model=LoginOut)
def api_login(payload: LoginIn):
    """Validates credentials and returns session user identity and role."""
    email = payload.email.strip().lower()
    pw_hash = _hash_password(payload.password)

    for acc in ACCOUNTS.values():
        if acc.email.lower() == email and acc.password_hash == pw_hash:
            return LoginOut(user_id=acc.id, role=acc.role)

    raise HTTPException(status_code=401, detail="Invalid email or password")


@app.get("/api/admin/stats", response_model=AdminStats)
def api_admin_stats():
    """Returns high-level system metrics for the administrative dashboard."""
    total_users = len(ACCOUNTS)
    total_sessions = len(SESSIONS)
    total_feedback = len(USER_FEEDBACK) + len(THERAPIST_FEEDBACK)
    return AdminStats(
        total_users=total_users,
        total_sessions=total_sessions,
        total_feedback=total_feedback,
    )


@app.get("/api/admin/users", response_model=List[AdminUserOut])
def api_admin_list_users():
    # NOTE: no auth check here – frontend role check only (capstone-level)
    return [
        AdminUserOut(id=a.id, email=a.email, role=a.role)
        for a in ACCOUNTS.values()
    ]


@app.post("/api/admin/users", response_model=AdminUserOut, status_code=201)
def api_admin_create_user(payload: AdminUserIn):
    """Registers a new user or staff member via the Admin interface."""
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if any(a.email.lower() == email for a in ACCOUNTS.values()):
        raise HTTPException(status_code=400, detail="Email already exists")

    user_id = str(uuid4())
    acc = Account(
        id=user_id,
        email=email,
        password_hash=_hash_password(payload.password),
        role=payload.role,
    )
    ACCOUNTS[acc.id] = acc
    _save_accounts(ACCOUNTS)

    return AdminUserOut(id=acc.id, email=acc.email, role=acc.role)


@app.post("/api/user/feedback")
def api_user_feedback(payload: UserFeedbackIn):
    """Captures qualitative user feedback and ratings post-session."""
    if payload.rating < 1 or payload.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    entry = {
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "rating": payload.rating,
        "comment": payload.comment,
        "ts": time.time(),
    }
    USER_FEEDBACK.append(entry)
    return {"ok": True}


@app.get("/api/therapist/schedule", response_model=List[TherapistSessionOut])
def api_therapist_schedule(therapist_id: str):
    """Fetches upcoming appointments for the Therapist portal view."""
    return [
        TherapistSessionOut(
            session_id="demo-session-1",
            user_id="user-1",
            scheduled_for="2025-01-01T10:00:00Z",
            status="upcoming",
        ),
        TherapistSessionOut(
            session_id="demo-session-2",
            user_id="user-2",
            scheduled_for="2025-01-01T11:00:00Z",
            status="upcoming",
        ),
    ]


@app.get("/api/therapist/reports", response_model=List[TherapistReportOut])
def api_therapist_reports(therapist_id: str):
    """Aggregates generated clinical reports from finished sessions."""
    out: List[TherapistReportOut] = []
    for user_id, state in USERS.items():
        for log in getattr(state, "sessions_log", []):
            out.append(
                TherapistReportOut(
                    case_id=log.get("session_id") or "",
                    user_id=user_id,
                    created_at=time.strftime(
                        "%Y-%m-%d",
                        time.localtime(log.get("ended_at") or time.time()),
                    ),
                    report_url=f"/output_reports/{log.get('session_id')}.pdf",
                )
            )
    return out


@app.get("/api/therapist/notifications", response_model=List[TherapistNotificationOut])
def api_therapist_notifications(therapist_id: str):
    """Retrieves critical risk alerts for professional oversight."""
    return [
        TherapistNotificationOut(
            id="notif-1",
            title="حالة ذات خطورة عالية",
            body="تم رصد مستوى خطورة عالي في آخر جلسة.",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
    ]


@app.post("/api/therapist/feedback")
def api_therapist_feedback(payload: TherapistFeedbackIn):
    """Records professional therapist assessment and clinical notes."""
    entry = {
        "therapist_id": payload.therapist_id,
        "session_id": payload.session_id,
        "risk_level": payload.risk_level,
        "notes": payload.notes,
        "ts": time.time(),
    }
    THERAPIST_FEEDBACK.append(entry)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
