#!/bin/bash
cd /home/claude/lw
pgrep -f cqueue.sh >/dev/null || (setsid nohup ./cqueue.sh > /home/claude/cqueue.log 2>&1 < /dev/null &)
sleep 40
pgrep -f wqueue.sh >/dev/null || (setsid nohup ./wqueue.sh > /home/claude/wqueue.log 2>&1 < /dev/null &)
