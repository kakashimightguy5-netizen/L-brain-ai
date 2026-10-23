# queue_L3.ps1 - runs every remaining L-pure test one after another, hands-free. Safe to re-run: finished runs are skipped.
$ErrorActionPreference = "Continue"
$root = "D:\AI\SpikingBrain-study\L-architecture"; $code = "$root\code"; $res = "$root\results"; $data = "$root\data"
Set-Location $code
$names = 'TEACHER','TEACHER_D','KDW','RAMP','QUANT','STUDENT_D','SPARSE_K','CKPT_EVERY','SEED','LR','NL','K','MIX','D','TAG'
function Reset-Env { foreach ($v in $names) { [Environment]::SetEnvironmentVariable($v, $null, 'Process') }; $env:LDATA = $data; $env:LOUT = $res }
function Wait-Idle { while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'lpure_lm' }) { Start-Sleep -Seconds 60 } }
function Run-Train($tag, $vars) {
  Wait-Idle
  if (Test-Path "$res\$tag.json") { "skip $tag (done)" | Out-File -Append "$res\queue_log.txt"; return }
  Reset-Env; foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $vars[$k], 'Process') }; $env:TAG = $tag
  "=== train $tag $(Get-Date)" | Out-File -Append "$res\queue_log.txt"
  python lpure_lm10.py 12000 *>> "$res\queue_log.txt"
}
function Run-Eval($tag, $vars) {
  if (-not (Test-Path "$res\$tag.pt")) { "no checkpoint for $tag" | Out-File -Append "$res\evalbig_results.txt"; return }
  Reset-Env; foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $vars[$k], 'Process') }
  python evalbig4.py "$res\$tag.pt" 200 *>> "$res\evalbig_results.txt"
}
$k2 = @{MIX='mask'; K='32'; D='256'; QUANT='1'; RAMP='0'; LEVELS='2'; TEACHER='teacher8_d256.pt'; TEACHER_D='256'; KDW='0.5'}
$k3 = @{MIX='mask'; K='32'; D='256'; QUANT='1'; RAMP='0'; LEVELS='3'; TEACHER='teacher8_d256.pt'; TEACHER_D='256'; KDW='0.5'}
Run-Train 'twokd8_mask_K32_d256' $k2
Run-Train 'threekd8_mask_K32_d256' $k3
Run-Eval 'twokd8_mask_K32_d256' $k2
Run-Eval 'threekd8_mask_K32_d256' $k3
"ALL DONE L3 $(Get-Date)" | Out-File -Append "$res\queue_done.txt"
