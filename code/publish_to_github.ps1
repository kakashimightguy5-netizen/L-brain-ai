# Publish project L to GitHub: copies code, jobs, runner, small results, cloud edits and vault notes into a clean folder, scans for secrets/big files, commits, pushes.
# Large files (checkpoints *.pt, data, venv) are EXCLUDED on purpose. Run in a NEW PowerShell window (not the one running training).
$ErrorActionPreference = "Stop"
$SRC  = "D:\AI\SpikingBrain-study\L-architecture"
$VAULT = "D:\devops learning obsidian\DevOps Learning\40 - Brain-Inspired Model"
$PUB  = "D:\AI\L-brain-ai-publish"
$REPO = "https://github.com/kakashimightguy5-netizen/L-brain-ai.git"
if (-not (Test-Path $PUB)) { git clone $REPO $PUB }
Set-Location $PUB
git pull --rebase origin main 2>$null
$xd = @("venv","data","__pycache__",".git","wandb")
$xf = @("*.pt","*.pth","*.ckpt","*.npy","*.npz","*.bin","*.safetensors","*.gguf","*.zip","*.pyc")
foreach ($d in @("code","jobs","runner","results","cloud_edits")) {
  if (Test-Path "$SRC\$d") { robocopy "$SRC\$d" "$PUB\$d" /E /XD $xd /XF $xf /NFL /NDL /NJH /NJS | Out-Null }
}
robocopy $VAULT "$PUB\notes" *.md /S /XD assets "SpikingBrain Assets" gpu-kit /NFL /NDL /NJH /NJS | Out-Null
Get-ChildItem $SRC -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -lt 5MB -and $_.Extension -in ".md",".txt",".json",".ps1",".py" } | Copy-Item -Destination $PUB -Force
@"
__pycache__/
*.pyc
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.safetensors
*.gguf
*.bin
venv/
data/
.env
"@ | Set-Content "$PUB\.gitignore"
@"
# L-brain-ai - Project L

A brain-inspired, attention-free (non-transformer) language-model research project, built by a solo founder. Goal: strong quality per unit of compute/energy/memory, with a long-term target of dedicated hardware.

**Honest status.** All evidence is small scale (models under ~10M parameters, byte-level, mostly toy tasks plus Python-text bits/byte). Nothing here supports any claim about AGI. L is brain-INSPIRED, not a human brain. Negative results are kept in the notes on purpose.

## Layout
- code/ - model code (L5: spiking diagonal recurrence + delta-rule fast-weight memory + hash recall), training, experiments, energy tools
- cloud_edits/ - newer experimental edits made in the cloud workspace (barcode/sleep/wide/COT flags, NOMAT sparse ternary wiring) and their small result files
- jobs/, runner/ - job files and the GPU job runner
- results/ - small JSON/log results (no checkpoints)
- notes/ - the full research log (numbered notes; latest: 144-149 energy fix, fair comparison, NOMAT, L-pure spec)

## Key measured results (laptop RTX 3050 6GB)
- Training energy, both models compiled + bf16: transformer 0.41 mJ/token, L 0.84 mJ/token (L is ~2x worse on a GPU). Notes 144-146.
- Pruning trained L5 to sparse ternary wiring: +17% (K=32) to +76% (K=4) bits/byte, fails the 0.5% gate. Note 148.
- Wide ternary L5 (d=192) beats same-recipe fp d=128 by 2.2% bits/byte at half the weight memory. Note 136.

Checkpoints and datasets are not included (size); see notes for how they are produced.
"@ | Set-Content "$PUB\README.md"
Write-Host "--- scanning for big files and secrets ---"
$big = Get-ChildItem $PUB -Recurse -File -Force | Where-Object { $_.FullName -notmatch "\\.git\\" -and $_.Length -gt 50MB }
if ($big) { $big | ForEach-Object { Write-Host "TOO BIG:" $_.FullName }; throw "Remove big files first" }
$hits = Get-ChildItem $PUB -Recurse -File -Force | Where-Object { $_.FullName -notmatch "\\.git\\" -and $_.Length -lt 5MB } | Select-String -Pattern "ghp_[A-Za-z0-9]{20,}","github_pat_","sk-[A-Za-z0-9]{20,}","AKIA[0-9A-Z]{16}","-----BEGIN (RSA|OPENSSH|EC) PRIVATE" -List
if ($hits) { $hits | ForEach-Object { Write-Host "POSSIBLE SECRET:" $_.Path }; throw "Possible secret found - review before pushing" }
git add -A
git status --short | Select-Object -First 15
git config user.name  | Out-Null; if ($LASTEXITCODE -ne 0) { git config user.name "kakashi" }
git config user.email | Out-Null; if ($LASTEXITCODE -ne 0) { git config user.email "kakashimightguy5@gmail.com" }
git commit -m "Project L: code, experiments, results and research notes (energy fix, fair comparison, NOMAT, L-pure spec)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BVroPqCEbtk6DopjyMXaLS"
git branch -M main
git push -u origin main
Write-Host "DONE: https://github.com/kakashimightguy5-netizen/L-brain-ai"
