# Run from repo root:  .\tools\setup-git.ps1
Write-Host "Configuring Git (global) and enabling shared hooks..." -ForegroundColor Cyan

# 1) Pull-first defaults (global)
git config --global pull.rebase true
git config --global rebase.autoStash true
git config --global fetch.prune true
git config --global pull.ff only

# 2) Point this repo at the shared hooks directory
git config core.hooksPath .githooks

# 3) Make pre-push executable on Windows Git (safe even if already executable)
if (Test-Path ".githooks\pre-push") {
  try {
    # Mark as executable bit for compat; Git for Windows respects it even without chmod
    git update-index --add --chmod=+x .githooks/pre-push | Out-Null
  } catch {}
}

Write-Host "✅ Done. Hooks enabled, pull-first enforced, prune on fetch enabled."
