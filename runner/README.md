# L runner
Right-click `START_L_RUNNER.ps1` → Run with PowerShell, once. Keep the window open; close it to stop.
- First start installs a Python environment with GPU PyTorch into `..\venv` (all on D:).
- Runs only `..\code\*.py` scripts listed in `..\jobs\*.json`; results/logs go to `..\results`.
- Up to 2 GPU jobs + 1 CPU job at a time.
