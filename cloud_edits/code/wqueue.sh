#!/bin/bash
# WIDE: train hop2 on 1..16 chains (HOP_MAX=16), plain L5; test same 1..32 chains. baseline = lim8k
cd /home/claude/lw
for s in 0 1 2; do
 O=/home/claude/lim_bc/WIDE; mkdir -p $O /home/claude/lim_bcck/WIDE
 [ -f $O/lim_L5_hop2_s${s}.json ] && continue
 while [ $(ps -eo args | grep -cE "^python -u (cont3|lim_suite)") -ge 2 ]; do sleep 20; done
 HOP_MAX=16 LOUT=$O LCK=/home/claude/lim_bcck/WIDE THREADS=1 setsid nohup python -u lim_suite.py L5 hop2 $s 8000 >> $O/log_${s}.txt 2>&1 < /dev/null &
 sleep 30
done
