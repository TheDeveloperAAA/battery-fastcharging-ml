"""Create/update the Hugging Face Space (Docker SDK) and verify it builds.

Pushes ONLY the contents of ``app/`` (Streamlit app, Dockerfile, its own
requirements.txt, and precomputed artifacts) — never raw data or training
code. Token comes from the HF_TOKEN env var (or ~/.sia_hf_token fallback);
nothing is hardcoded.

Usage:
    HF_TOKEN=... python -m src.utils.deploy_space \
        --repo-id rajtheman/battery-fastcharging-dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

from src.utils.config import REPO_ROOT

TERMINAL_BAD = {"BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "DELETING"}


def get_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        fallback = Path.home() / ".sia_hf_token"
        if fallback.exists():
            tok = fallback.read_text().strip()
    if not tok:
        sys.exit("No HF token: set HF_TOKEN")
    return tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--timeout-min", type=float, default=25)
    args = ap.parse_args()

    api = HfApi(token=get_token())
    url = api.create_repo(repo_id=args.repo_id, repo_type="space",
                          space_sdk="docker", exist_ok=True, private=False)
    print(f"space repo: {url}")

    api.upload_folder(
        repo_id=args.repo_id, repo_type="space",
        folder_path=str(REPO_ROOT / "app"),
        commit_message="Deploy dashboard (precomputed artifacts + compact model)",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"])
    print("app/ uploaded; waiting for build...")

    deadline = time.time() + args.timeout_min * 60
    stage = None
    while time.time() < deadline:
        runtime = api.get_space_runtime(args.repo_id)
        if runtime.stage != stage:
            stage = runtime.stage
            print(f"  stage: {stage}", flush=True)
        if stage == "RUNNING":
            user, name = args.repo_id.split("/")
            print(f"LIVE: https://{user}-{name.replace('_', '-')}.hf.space")
            return 0
        if stage in TERMINAL_BAD:
            print("Build failed — fetch logs at "
                  f"https://huggingface.co/spaces/{args.repo_id}")
            return 1
        time.sleep(15)
    print("Timed out waiting for the Space to run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
