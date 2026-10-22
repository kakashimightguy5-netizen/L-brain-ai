# NOMAT sweep: sparse-ternary student distilled from the trained LEdge_s0 teacher. K = inputs per output (dense = 128..384).
cd D:\AI\SpikingBrain-study\L-architecture\code
$env:UNFREEZE="1"; $env:LOUT="..\results"; $env:THREADS="4"
foreach ($K in 32,16,8,4) {
  $env:SPARSE_K="$K"
  ..\venv\Scripts\python.exe -B nomat_ft.py 1.0 1500 all6 nomat_K$K
}
