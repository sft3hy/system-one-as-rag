#!/bin/sh
# Temporary cooker-block watchdog (requested 2026-10-14): while the marker
# /tmp/block_cooker exists, any vllm/LFM server from cooker-deep-dives that
# appears on this machine is killed within 20s. The cooker project's own
# server (vllm serve LiquidAI/LFM2.5-24B --port 8000) hijacked loopback
# traffic from omlx once already and stole GPU from the benchmark run.
#
# UNBLOCK:  rm /tmp/block_cooker  (the sweep's completion flow does this)
# or:       launchctl remove com.homelab.blockcooker
MARKER=/tmp/block_cooker
INTERVAL=20
KILLED=0
while [ -f "$MARKER" ]; do
  PIDS=$(pgrep -f "vllm serve" 2>/dev/null; pgrep -f "LFM2.5" 2>/dev/null)
  for p in $(echo "$PIDS" | sort -u); do
    [ "$p" = "$$" ] && continue
    CMD=$(ps -p "$p" -o command= 2>/dev/null)
    case "$CMD" in
      *cooker*|*lfm2*|*LFM2*|*vllm*)
        echo "$(date '+%H:%M:%S') blocking cooker pid=$p: ${CMD:0:120}"
        kill -9 "$p" 2>/dev/null
        KILLED=$((KILLED + 1)) ;;
    esac
  done
  sleep "$INTERVAL"
done
echo "block marker removed — cooker unblocked (blocked $KILLED launches total)"
