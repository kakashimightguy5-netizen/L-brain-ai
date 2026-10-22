# Run in an ADMINISTRATOR PowerShell. Stops the runner + the slow L5E jobs, runs the clock-lock power sweep on an idle GPU, restarts the runner.
$r = 'D:\AI\SpikingBrain-study\L-architecture'
Write-Host '1/5 stopping runner and running training jobs...'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'START_L_RUNNER' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_L5E|train_L1|train_m3|eff_bench|serve_bench|power_sweep' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 8
Write-Host '2/5 marking skipped jobs done so the queue continues...'
foreach ($n in '15o_L5E_s1','15p_L5E_s2','1704_cE_brk_s0') { Set-Content -Path "$r\results\$n.done" -Value 'skipped-by-FAST_RESET' }
if (Test-Path "$r\results\power_sweep.json") { Move-Item "$r\results\power_sweep.json" "$r\results\power_sweep_INVALID_no_limit_applied.json" -Force }
Write-Host '3/5 power sweep (clock lock) - about 15 min, GPU is idle...'
$env:LOUT = "$r\results"; $env:LDATA = "$r\data"; $env:LSCAN_DISABLE = '1'; $env:PYTORCH_NVML_BASED_CUDA_CHECK = '1'
Set-Location "$r\code"
& "$r\venv\Scripts\python.exe" -B power_sweep.py
Write-Host '4/5 making sure clocks are back to default...'
nvidia-smi -rgc
Write-Host '5/5 restarting the runner in a new window...'
Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File',"$r\runner\START_L_RUNNER.ps1"
Write-Host 'done'
