#!/bin/bash
# redesign queue: hop2 8000 steps, variants BC (barcode E=2,KF=4), SL (sleep=2), BCSL ; baseline = /home/claude/lim8k (CONV=4)
cd /home/claude/lw
for s in 0 1 2; do for v in BC SL BCSL; do
 O=/home/claude/lim_bc/$v; mkdir -p $O /home/claude/lim_bcck/$v
 [ -f $O/lim_L5_hop2_s${s}.json ] && continue
 while [ $(ps -eo args | grep -cE "^python -u (cont3|lim_suite)") -ge 2 ]; do sleep 20; done
 case $v in BC) E="BARCODE=2 BARCODE_KF=4 SLEEP=1";; SL) E="BARCODE=1 SLEEP=2";; BCSL) E="BARCODE=2 BARCODE_KF=4 SLEEP=2";; esac
 env $E LOUT=$O LCK=/home/claude/lim_bcck/$v THREADS=1 setsid nohup python -u lim_suite.py L5 hop2 $s 8000 >> $O/log_${s}.txt 2>&1 < /dev/null &
 sleep 30
done; done
