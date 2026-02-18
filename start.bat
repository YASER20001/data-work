@echo off
REM ===== start.bat =====
REM One-click launcher: venv (if missing) → install deps → start API (8000) + static site (5500) → open chat.

setlocal enableextensions
pushd "%~dp0"

echo [1/4] Checking Python...
where python >nul 2>nul || (echo Python not found. Install Python 3.10+ and rerun.& pause & exit /b 1)

echo [2/4] Creating virtualenv if missing...
if not exist ".venv" (
  python -m venv .venv || (echo Failed to create venv.& pause & exit /b 1)
)

echo [3/4] Installing/Updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if exist "requirements.txt" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
  echo (No requirements.txt found — skipping.)
)

REM --- Ports (change if you need) ---
set "API_PORT=8000"
set "WEB_PORT=5500"

echo [4/4] Starting servers...

REM --- Start backend API (FastAPI / Uvicorn via web_server.py) ---
start "RIFD API" cmd /k """.venv\Scripts\python.exe"" web_server.py"

REM --- Start static web server for Website/ (serves chat.html, CSS, JS) ---
pushd "Website"
start "RIFD WEB" cmd /k ""..\ .venv\Scripts\python.exe"" -m http.server %WEB_PORT% --bind 127.0.0.1
popd

REM --- Open browser to chat page ---
REM small delay so servers come up before opening the page
timeout /t 2 >nul
start "" "http://127.0.0.1:%WEB_PORT%/chat.html"

echo Done. Two terminal windows were opened:
echo   - "RIFD API" on http://127.0.0.1:%API_PORT%  (check /health)
echo   - "RIFD WEB" on http://127.0.0.1:%WEB_PORT%  (serving Website/)
echo Your browser should open to the chat page automatically.
popd
exit /b 0
