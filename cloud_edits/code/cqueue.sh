#!/bin/bash
# COT: hop2 with 'think out loud' middle key (HOP_COT=1), training chains 1..4 as baseline, greedy-generation eval. baseline = lim8k (teacher-free single answer)
cd /home/claude/lw
for s in 0 1 2; do
 O=/home/claude/lim_bc/COT; mkdir -p $O /home/claude/lim_bcck/COT
 [ -f $O/lim_L5_hop2_s${s}.json ] && continue
 while [ $(ps -eo args | grep -cE "^python -u (cont3|lim_suite)") -ge 2 ]; do sleep 20; done
 HOP_COT=1 LOUT=$O LCK=/home/claude/lim_bcck/COT THREADS=1 setsid nohup python -u lim_suite.py L5 hop2 $s 8000 >> $O/log_${s}.txt 2>&1 < /dev/null &
 sleep 30
done
