# queue_L4.ps1 - runs every remaining L-pure test one after another, hands-free. Safe to re-run: finished runs are skipped.
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
$a = @{MIX='mask'; K='64'; D='256'; QUANT='1'; RAMP='0'; LEVELS='2'; TEACHER='teacher8_d256.pt'; TEACHER_D='256'; KDW='0.5'}
$b = @{MIX='mask'; K='64'; D='256'; QUANT='1'; RAMP='0'; LEVELS='3'; TEACHER='teacher8_d256.pt'; TEACHER_D='256'; KDW='0.5'}
$c = @{MIX='mask'; K='32'; D='256'; QUANT='1'; RAMP='0'; LEVELS='3'; TEACHER='teacher8_d256.pt'; TEACHER_D='256'; KDW='0.5'; SEED='1'}
Run-Train 'k64two_kd8_d256' $a
Run-Train 'k64three_kd8_d256' $b
Run-Train 'threekd8_s1_K32_d256' $c
Run-Eval 'k64two_kd8_d256' $a
Run-Eval 'k64three_kd8_d256' $b
Run-Eval 'threekd8_s1_K32_d256' $c
"ALL DONE L4 $(Get-Date)" | Out-File -Append "$res\queue_done.txt"
