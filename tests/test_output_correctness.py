# SPDX-License-Identifier: Apache-2.0
"""Pytest wrapper for the real-vLLM warm-hit output-correctness check.

The actual check requires a working vLLM + Telys runtime and GPU, so it cannot
run in the local CPU-only stubbed environment. This wrapper is skipped unless
``TELYS_OUTPUT_CORRECTNESS_MODEL`` is set; on the pod, set that variable and run
this test first.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    "TELYS_OUTPUT_CORRECTNESS_MODEL" not in os.environ,
    reason=(
        "Set TELYS_OUTPUT_CORRECTNESS_MODEL to a model name/path "
        "(e.g. Qwen/Qwen2.5-1.5B-Instruct) to run the real vLLM output check."
    ),
)
def test_warm_hit_output_matches_cold():
    """Run bench/verify_warm_hit_output.py and require a pass."""
    model = os.environ["TELYS_OUTPUT_CORRECTNESS_MODEL"]
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "bench" / "verify_warm_hit_output.py"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--prefix-len",
        os.environ.get("TELYS_OUTPUT_CORRECTNESS_PREFIX_LEN", "64"),
        "--input-length-range",
        os.environ.get("TELYS_OUTPUT_CORRECTNESS_INPUT_RANGE", "64:128"),
        "--output-len",
        os.environ.get("TELYS_OUTPUT_CORRECTNESS_OUTPUT_LEN", "10"),
        "--seed",
        os.environ.get("TELYS_OUTPUT_CORRECTNESS_SEED", "42"),
        "--reset-telys",
        "--enforce-eager",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, (
        f"warm-hit output-correctness script failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
