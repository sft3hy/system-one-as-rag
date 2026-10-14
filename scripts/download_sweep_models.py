"""Download the sweep roster into ~/models/<org>/<name> (the layout omlx scans).
Sequential on purpose: predictable disk growth, one failure doesn't poison the rest.
Run: HF_TOKEN=$(cat ~/.config/hf/token) .venv/bin/python scripts/download_sweep_models.py
"""
import os
import pathlib
import sys

from huggingface_hub import snapshot_download

MODELS = [
    # encoders first (small, Tier B)
    ("convaiinnovations/laya", "encoder"),                 # explicitly requested
    ("convaiinnovations/laya-typed-decisions", "encoder"),
    ("wfzyx/von", "encoder"),                              # Tier B pilot
    ("manjunathshiva/opendecider-nano", "encoder"),
    # Tier A, small to large
    ("openjev/OpenJev-Flash-9B-MLX", "decision"),
    ("openjev/openjev-MLX-4bit", "decision"),
    # Tier C structured-decision AR baseline
    ("togethercomputer/Tev1-4B-experimental", "ar-decision"),
    # omni Clef last: big + detection uncertain
    ("mlx-community/clef-omni-4bit", "decision"),
]

DEST = pathlib.Path.home() / "models"

skip_existing = "--skip-existing" in sys.argv
for repo, tier in MODELS:
    local = DEST / repo
    if skip_existing and local.exists() and any(local.glob("*.safetensors")):
        print(f"[skip] {repo} already present", flush=True)
        continue
    print(f"[dl] {repo} ({tier}) -> {local}", flush=True)
    try:
        p = snapshot_download(repo_id=repo, local_dir=str(local),
                              token=os.environ.get("HF_TOKEN") or None,
                              max_workers=4)
        print(f"[ok] {repo} at {p}", flush=True)
    except Exception as e:  # noqa: BLE001 — keep the sweep going, report at end
        print(f"[FAIL] {repo}: {type(e).__name__}: {e}", flush=True)
print("[done] download pass complete", flush=True)
