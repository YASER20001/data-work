# backend/core/env.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PART 1: CONFIGURATION DEFAULTS
# =============================================================================

# Default filename for the environment configuration file
_DEFAULT_ENV_FILE = ".env"


# =============================================================================
# PART 2: ENVIRONMENT LOADING LOGIC
# =============================================================================

def load(env_path: str | None = None) -> None:
    """
    Load environment variables from .env (project root by default).
    Automatically handles both direct and subfolder runs.
    """
    root = Path.cwd()
    env_file = Path(env_path) if env_path else root / _DEFAULT_ENV_FILE

    if not env_file.exists():
        # Fallback: Try project root relative to this file's location
        # (folder containing rifd_main.py)
        maybe = Path(__file__).resolve().parent.parent / _DEFAULT_ENV_FILE
        if maybe.exists():
            env_file = maybe

    load_dotenv(dotenv_path=env_file)


# =============================================================================
# PART 3: VARIABLE RETRIEVAL & VALIDATION
# =============================================================================

def get(name: str, default: str | None = None, required: bool = False) -> str | None:
    """
    Read an environment variable with optional 'required' enforcement.
    Useful for critical keys like API_KEYs.
    """
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# =============================================================================
# PART 4: BOOLEAN FLAG PARSING
# =============================================================================

def _flag(name: str, default: str = "0") -> bool:
    """
    Read an environment variable as a boolean flag.
    Converts '1' to True and '0' to False.
    """
    try:
        # Standard integer-to-boolean conversion used throughout the system
        return bool(int(os.getenv(name, default)))
    except Exception:
        return False

def flag(name: str, default: str = "0") -> bool:
    """Public wrapper for the boolean flag parser."""
    return _flag(name, default)
