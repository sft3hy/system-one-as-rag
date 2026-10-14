#!/bin/sh
# Sweep-resume payload, run by launchd (com.homelab.sweepresume).
# nohup+& inside harness shells keeps getting process-group-killed;
# launchd-owned processes demonstrably survive (it's how omlx stays up).
cd /Users/samueltownsend/dev/cosmic/system-one-as-rag || exit 3
rm -f results/phase2-resume.done
.venv/bin/python -u -m sufficiency_lab.runner -c configs/sweep-full-resume.yaml \
  > results/phase2-resume.log 2>&1
echo $? > results/phase2-resume.done
