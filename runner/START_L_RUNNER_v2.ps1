# L-architecture job runner v2 (adds stall/time-limit watchdog; original START_L_RUNNER.ps1 untouched).
# L-architecture job runner. Start once (right-click > Run with PowerShell). Close this window to stop it.
# It ONLY runs Python scripts from ..\code that are listed in ..\jobs\*.json, and writes logs/results to ..\results.
# Everything (Python environment, downloads, caches, temp files, results) stays inside D:\AI\SpikingBrain-study\L-architecture.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$code = Join-Path $root 'code'; $jobs = Join-Path $root 'jobs'; $res = Join-Path $root 'results'; $data = Join-Path $root 'data'; $venv = Join-Path $root 'venv'
foreach ($d in @($res, (Join-Path $root 'temp'), (Join-Path $root 'pip-cache'), (Join-Path $root 'torch-cache'))) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
$env:TEMP = Join-Path $root 'temp'; $env:TMP = $env:TEMP; $env:PIP_CACHE_DIR = Join-Path $root 'pip-cache'; $env:TORCH_HOME = Join-Path $root 'torch-cache'
$env:HF_HOME = Join-Path $root 'torch-cache'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:LDATA = $data; $env:LOUT = $res; $env:LSCAN_DISABLE = '1'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
  Write-Host 'First start: creating Python environment on D: and installing PyTorch with GPU support (about 2.5 GB, 10-20 min)...'
  & py -3.12 -m venv $venv
  & $py -m pip install --upgrade pip
  & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  & $py -m pip install numpy scikit-learn
}
$train = Join-Path $data 'py_big_train.txt'
if (-not (Test-Path $train)) {
  $parts = Get-ChildItem (Join-Path $data 'py_big_train.part*') | Sort-Object Name
  $out = [System.IO.File]::Create($train)
  foreach ($p in $parts) { $b = [System.IO.File]::ReadAllBytes($p.FullName); $out.Write($b, 0, $b.Length) }
  $out.Close()
}
& $py -B (Join-Path $code 'gpu_check.py')
$running = @{}; $maxGpu = 3; $maxCpu = 1; $stallMin = 90   # minutes without any new log/err output before a job counts as hung
Write-Host 'L runner is running. Leave this window open. Close it to stop.'
while ($true) {
  foreach ($k in @($running.Keys)) {
    $p = $running[$k].proc
    if (-not $p.HasExited) {
      # watchdog: kill a job whose log AND err files have not changed for $stallMin minutes, or that exceeds its hard time cap
      $age = ((Get-Date) - $running[$k].start).TotalMinutes
      $lastW = @((Join-Path $res "$k.log"), (Join-Path $res "$k.err")) | Where-Object { Test-Path $_ } | ForEach-Object { (Get-Item $_).LastWriteTime } | Sort-Object -Descending | Select-Object -First 1
      $idle = if ($lastW) { ((Get-Date) - $lastW).TotalMinutes } else { $age }
      if ($idle -gt $stallMin -or $age -gt $running[$k].cap) {
        Write-Host "WATCHDOG killing $k (idle $([int]$idle) min, age $([int]$age) min)"
        Set-Content (Join-Path $res "$k.killed") "idle=$([int]$idle) age=$([int]$age) at $(Get-Date -Format o)"
        try { Stop-Process -Id $p.Id -Force } catch {}
      }
    }
    if ($p.HasExited) { Set-Content -Path (Join-Path $res "$k.done") -Value $p.ExitCode; Write-Host "finished $k (exit $($p.ExitCode))"; $running.Remove($k) }
  }
  $gpuN = @($running.Values | Where-Object { $_.dev -eq 'gpu' }).Count; $cpuN = @($running.Values | Where-Object { $_.dev -eq 'cpu' }).Count
  foreach ($f in (Get-ChildItem (Join-Path $jobs '*.json') -ErrorAction SilentlyContinue | Sort-Object Name)) {
    $name = $f.BaseName
    if (Test-Path (Join-Path $res "$name.started")) { continue }
    try { $j = Get-Content $f.FullName -Raw | ConvertFrom-Json } catch { continue }
    $script = [string]$j.script
    if ($script -notmatch '^[A-Za-z0-9_]+\.py$' -or -not (Test-Path (Join-Path $code $script))) { Set-Content (Join-Path $res "$name.started") 'rejected'; continue }
    $dev = if ($j.device -eq 'cpu') { 'cpu' } else { 'gpu' }
    if (($dev -eq 'gpu' -and $gpuN -ge $maxGpu) -or ($dev -eq 'cpu' -and $cpuN -ge $maxCpu)) { continue }
    if ($dev -eq 'cpu') { $env:CUDA_VISIBLE_DEVICES = '-1'; $env:THREADS = '6' } else { Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue; $env:THREADS = '2' }
    foreach ($e in @('HOMEO','DM','NL','BS','SEQ','LR','CKPT','SCANC','TDM','TNL')) { Remove-Item "Env:\$e" -ErrorAction SilentlyContinue }
    if ($script:prevEnv) { foreach ($n in $script:prevEnv) { Remove-Item "Env:\$n" -ErrorAction SilentlyContinue } }   # fix: env vars from the previous job leaked into later jobs
    $script:prevEnv = @(); if ($j.env) { foreach ($pp in $j.env.PSObject.Properties) { if ($pp.Name -match '^[A-Z_]+$') { $script:prevEnv += $pp.Name } } }
    if ($j.env) { foreach ($pp in $j.env.PSObject.Properties) { if ($pp.Name -match '^[A-Z_]+$') { Set-Item "Env:\$($pp.Name)" ([string]$pp.Value) } } }
    $argList = @('-B', $script) + @($j.args | ForEach-Object { [string]$_ })
    $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $code -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput (Join-Path $res "$name.log") -RedirectStandardError (Join-Path $res "$name.err")
    $cap = if ($j.max_minutes) { [double]$j.max_minutes } else { 720 }
    $running[$name] = @{ proc = $p; dev = $dev; start = (Get-Date); cap = $cap }; Set-Content (Join-Path $res "$name.started") (Get-Date -Format o)
    Write-Host "started $name on $dev"
    if ($dev -eq 'gpu') { $gpuN++ } else { $cpuN++ }
  }
  Set-Content (Join-Path $res 'runner_alive.json') (@{ time = (Get-Date -Format o); running = @($running.Keys) } | ConvertTo-Json)
  Start-Sleep -Seconds 10
}
