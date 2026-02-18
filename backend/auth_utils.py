import os
import json
import hashlib
import random
import re
import smtplib
from email.mime.text import MIMEText

# ======================================================================
# PATHS
# ======================================================================
DATA_ROOT = "data"
ACCOUNTS_FILE = os.path.join(DATA_ROOT, "accounts.json")
USERS_ROOT = os.path.join(DATA_ROOT, "users")

PENDING_SIGNUPS_FILE = os.path.join(DATA_ROOT, "pending_signups.json")
PENDING_RESETS_FILE = os.path.join(DATA_ROOT, "pending_resets.json")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(USERS_ROOT, exist_ok=True)

# ======================================================================
# SMTP CONFIG
# ======================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "3marshnb@gmail.com"
SMTP_PASSWORD = "yglwxinfmmwhcdkw"  # GOOGLE APP PASSWORD

# ======================================================================
# HELPERS: LOAD / SAVE JSON
# ======================================================================
def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ======================================================================
# VALIDATION
# ======================================================================
def validate_password_strength(password: str) -> str | None:
    if len(password) < 6:
        return "Password must be at least 6 characters long."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    return None

def validate_email(email: str) -> str | None:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    if not re.match(pattern, email):
        return "Invalid email address format."
    return None

# ======================================================================
# HASH
# ======================================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# ======================================================================
# ACCOUNTS
# ======================================================================
def load_accounts() -> dict:
    return _load_json(ACCOUNTS_FILE)

def save_accounts(data: dict):
    _save_json(ACCOUNTS_FILE, data)

# ======================================================================
# EMAIL
# ======================================================================
def send_email_code(email: str, code: str) -> bool:
    try:
        msg = MIMEText(f"Your RIFD verification code is: {code}")
        msg["Subject"] = "RIFD Verification Code"
        msg["From"] = SMTP_EMAIL
        msg["To"] = email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, [email], msg.as_string())
        server.quit()
        return True

    except Exception as e:
        print("[SMTP ERROR]", e)
        return False

# ======================================================================
# SIGNUP (PERSISTENT)
# ======================================================================
def signup(username: str, password: str, email: str) -> dict:
    accounts = load_accounts()
    pending = _load_json(PENDING_SIGNUPS_FILE)

    if username in accounts:
        return {"success": False, "error": "Username already exists."}

    email_error = validate_email(email)
    if email_error:
        return {"success": False, "error": email_error}

    pw_error = validate_password_strength(password)
    if pw_error:
        return {"success": False, "error": pw_error}

    code = str(random.randint(1000, 9999))

    if not send_email_code(email, code):
        return {"success": False, "error": "Failed to send verification email."}

    pending[username] = {
        "email": email,
        "password_hash": hash_password(password),
        "code": code,
    }

    _save_json(PENDING_SIGNUPS_FILE, pending)

    return {"success": True, "verify_required": True}

def verify_signup(username: str, code_entered: str) -> dict:
    pending = _load_json(PENDING_SIGNUPS_FILE)

    if username not in pending:
        return {"success": False, "error": "No signup request found."}

    info = pending[username]

    if info["code"] != code_entered:
        return {"success": False, "error": "Incorrect code."}

    # Save account
    accounts = load_accounts()
    accounts[username] = {
        "username": username,
        "email": info["email"],
        "password_hash": info["password_hash"],
    }
    save_accounts(accounts)

    # Create user folder
    user_dir = os.path.join(USERS_ROOT, username)
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir, "reports"), exist_ok=True)

    with open(os.path.join(user_dir, "account.json"), "w", encoding="utf-8") as f:
        json.dump({"username": username, "email": info["email"]}, f, indent=2)

    with open(os.path.join(user_dir, "case_notes.json"), "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

    # Remove pending
    pending.pop(username, None)
    _save_json(PENDING_SIGNUPS_FILE, pending)

    return {"success": True}

# ======================================================================
# LOGIN
# ======================================================================

def login(identifier: str, password: str) -> dict:
    """
    identifier can be USERNAME or EMAIL
    """
    accounts = load_accounts()

    # Try direct username match first
    user = None
    if identifier in accounts:
        user = identifier
    else:
        # Otherwise search by email
        for u, data in accounts.items():
            if identifier == data.get("email"):
                user = u
                break

    if not user:
        return {"success": False, "error": "User not found."}

    if hash_password(password) != accounts[user]["password_hash"]:
        return {"success": False, "error": "Incorrect password."}

    return {"success": True, "username": user}


# ======================================================================
# PASSWORD RESET (PERSISTENT)
# ======================================================================
def start_password_reset(identifier: str) -> dict:
    accounts = load_accounts()
    pending = _load_json(PENDING_RESETS_FILE)

    user = None
    for u, data in accounts.items():
        if identifier == u or identifier == data.get("email"):
            user = u
            break

    if not user:
        return {"success": False, "error": "User not found."}

    email = accounts[user]["email"]
    code = str(random.randint(1000, 9999))

    if not send_email_code(email, code):
        return {"success": False, "error": "Failed to send email."}

    pending[user] = code
    _save_json(PENDING_RESETS_FILE, pending)

    return {"success": True, "username": user}

def finish_password_reset(username: str, code: str | None, new_password: str | None) -> dict:
    pending = _load_json(PENDING_RESETS_FILE)

    # Step 1: verify code
    if code is not None:
        if username not in pending:
            return {"success": False, "error": "No reset request found."}
        if pending[username] != code:
            return {"success": False, "error": "Incorrect verification code."}
        return {"success": True}

    # Step 2: update password
    pw_error = validate_password_strength(new_password)
    if pw_error:
        return {"success": False, "error": pw_error}

    accounts = load_accounts()
    if username not in accounts:
        return {"success": False, "error": "User not found."}

    accounts[username]["password_hash"] = hash_password(new_password)
    save_accounts(accounts)

    pending.pop(username, None)
    _save_json(PENDING_RESETS_FILE, pending)

    return {"success": True}
