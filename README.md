## Team Git Workflow (Visual Studio)

**One-time after cloning**
1) Open Developer PowerShell in the repo
2) Run:  .\tools\setup-git.ps1
   - Sets pull-first defaults + enables shared hooks
3) In Visual Studio:
   - Git → Settings → Git Global Settings
     - Rebase current branch when pulling (On)
     - Prune remote branches during fetch (On)
     - Fetch automatically (On)

**Daily (feature branch)**
1) Pull (Rebase)
2) Do work → Commit
3) Push → Open Pull Request
4) After merge, switch to main → Pull (Rebase)
5) Delete your branch (local + remote)

**If push is blocked**
- VS auto-runs the pre-push hook. Do:
  - Pull (Rebase), fix any conflicts, push again.

Make a file ".env"
GEMINI_API_KEY=your_real_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
VOICE_MODE=false

# 1- Create Venv
python -m venv "$env:LOCALAPPDATA\venvs\rifd"
pip install -r requirements.txt

# 2- Set Venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "$env:LOCALAPPDATA\venvs\rifd\Scripts\Activate.ps1"
$env:RESPONSE_LANG="en"

# 3- Activate Script
python -m backend.pipeline_bootstrap


# 4- Run Streamlit
streamlit run app.py

# 5- Important
- make sure to download "wkhtmltopdf" into your device
- put this path into your .env:
  - WKHTMLTOPDF_PATH="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"

# 6- Voice
tts $env:VOICE_MODE = "true"

uvicorn backend.api_server:app --reload --port 8000


python -m py_compile backend\pipeline_bootstrap.py

python maintenance_repo_dump.py

# 7- RAG Files
https://pmqu-my.sharepoint.com/:u:/g/personal/4210361_upm_edu_sa/EbUtumiVBNROsQIVSnghDQUBa3N0GezITaNwWGMwfngUvw?e=BNGAnH

https://pmqu-my.sharepoint.com/:f:/g/personal/4210361_upm_edu_sa/Elb9yxQlL7dPq18r5T5Gp18BGhtTcKkWskdhraAlFCSu5w?e=fkTrH7


