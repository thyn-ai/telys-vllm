# SPDX-License-Identifier: Apache-2.0
"""RunPod worker A/B/C environment generation and arm isolation."""

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ARMS_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "integrations"
    / "runpod-worker-vllm"
    / "arms.sh"
)
ENV_KEYS = [
    "MODEL_NAME",
    "ENABLE_PREFIX_CACHING",
    "KV_TRANSFER_CONFIG",
    "TELYS_PATH",
    "TELYS_ADDRESS",
    "TELYS_VLLM_SLAB_PATH",
]
SNAPSHOT_CODE = (
    "import json, os; "
    f"keys = {ENV_KEYS!r}; "
    "print(json.dumps({key: os.environ.get(key) for key in keys}, sort_keys=True))"
)


def run_shell(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
        check=False,
    )


def arm_environment(arm: str, *, prelude: str = "") -> dict[str, str | None]:
    script = f"""
set -euo pipefail
source {shlex.quote(str(ARMS_SCRIPT))}
{prelude}
runpod_worker_arm {shlex.quote(arm)}
python3 -c {shlex.quote(SNAPSHOT_CODE)}
"""
    result = run_shell(script)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def compact_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_all_arms_share_model_and_disable_native_prefix_caching(arm):
    env = arm_environment(arm, prelude="export MODEL_NAME=org/shared-model")
    assert env["MODEL_NAME"] == "org/shared-model"
    assert env["ENABLE_PREFIX_CACHING"] == "false"


def test_arm_a_is_recompute_only():
    env = arm_environment(
        "A",
        prelude=(
            "export KV_TRANSFER_CONFIG=stale TELYS_PATH=/stale "
            "TELYS_ADDRESS=unix:///stale TELYS_VLLM_SLAB_PATH=/stale"
        ),
    )
    assert env == {
        "MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
        "ENABLE_PREFIX_CACHING": "false",
        "KV_TRANSFER_CONFIG": None,
        "TELYS_PATH": None,
        "TELYS_ADDRESS": None,
        "TELYS_VLLM_SLAB_PATH": None,
    }


def test_arm_b_uses_stock_vllm_tiering_with_filesystem_secondary():
    env = arm_environment("B")
    expected = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 1_073_741_824,
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": "/runpod-volume/vllm-filesystem-kv",
                    "n_read_threads": 16,
                    "n_write_threads": 16,
                }
            ],
        },
    }
    assert json.loads(env["KV_TRANSFER_CONFIG"]) == expected
    assert env["KV_TRANSFER_CONFIG"] == compact_json(expected)
    assert env["TELYS_PATH"] is None
    assert env["TELYS_ADDRESS"] is None


def test_arm_c_uses_existing_telys_connector_on_volume():
    env = arm_environment("C")
    expected = {
        "kv_connector": "TelysKVConnector",
        "kv_connector_module_path": "telys_vllm.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "telys_blobstore_no_slab": True,
        },
    }
    assert json.loads(env["KV_TRANSFER_CONFIG"]) == expected
    assert env["KV_TRANSFER_CONFIG"] == compact_json(expected)
    assert env["TELYS_PATH"] == "/runpod-volume/telys-kv"
    assert env["TELYS_ADDRESS"] is None
    assert env["TELYS_VLLM_SLAB_PATH"] is None


def test_switching_arms_replaces_connector_and_clears_telys_state():
    script = f"""
set -euo pipefail
source {shlex.quote(str(ARMS_SCRIPT))}
export TELYS_ADDRESS=unix:///stale TELYS_VLLM_SLAB_PATH=/stale
runpod_worker_arm C
runpod_worker_arm B
python3 -c {shlex.quote(SNAPSHOT_CODE)}
runpod_worker_arm A
python3 -c {shlex.quote(SNAPSHOT_CODE)}
"""
    result = run_shell(script)
    assert result.returncode == 0, result.stderr
    arm_b, arm_a = [json.loads(line) for line in result.stdout.splitlines()]

    assert json.loads(arm_b["KV_TRANSFER_CONFIG"])["kv_connector"] == (
        "OffloadingConnector"
    )
    assert arm_b["TELYS_PATH"] is None
    assert arm_b["TELYS_ADDRESS"] is None
    assert arm_b["TELYS_VLLM_SLAB_PATH"] is None
    assert arm_a["KV_TRANSFER_CONFIG"] is None
    assert arm_a["TELYS_PATH"] is None


def test_volume_override_is_shell_and_json_safe():
    volume = '/tmp/runpod volume "quoted"'
    prelude = f"export RUNPOD_WORKER_VOLUME={shlex.quote(volume)}"
    arm_b = arm_environment("B", prelude=prelude)
    arm_c = arm_environment("C", prelude=prelude)

    fs_config = json.loads(arm_b["KV_TRANSFER_CONFIG"])
    secondary = fs_config["kv_connector_extra_config"]["secondary_tiers"][0]
    assert secondary["root_dir"] == f"{volume}/vllm-filesystem-kv"
    assert arm_c["TELYS_PATH"] == f"{volume}/telys-kv"


def test_cpu_staging_budget_is_configurable_as_an_integer():
    env = arm_environment(
        "B", prelude="export RUNPOD_WORKER_CPU_BYTES_TO_USE=10737418240"
    )
    config = json.loads(env["KV_TRANSFER_CONFIG"])
    assert config["kv_connector_extra_config"]["cpu_bytes_to_use"] == 10_737_418_240


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-a-number"])
def test_arm_b_rejects_invalid_cpu_staging_budget(value):
    script = f"""
source {shlex.quote(str(ARMS_SCRIPT))}
export RUNPOD_WORKER_CPU_BYTES_TO_USE={shlex.quote(value)}
runpod_worker_arm B
"""
    result = run_shell(script)
    assert result.returncode == 64
    assert result.stdout == ""
    assert "positive integer" in result.stderr


def test_relative_volume_is_rejected():
    script = f"""
source {shlex.quote(str(ARMS_SCRIPT))}
export RUNPOD_WORKER_VOLUME=relative/path
runpod_worker_arm C
"""
    result = run_shell(script)
    assert result.returncode == 64
    assert result.stdout == ""
    assert "absolute path" in result.stderr


@pytest.mark.parametrize("invocation", ["runpod_worker_arm", "runpod_worker_arm D"])
def test_invalid_arm_fails_closed(invocation):
    script = f"source {shlex.quote(str(ARMS_SCRIPT))}\n{invocation}"
    result = run_shell(script)
    assert result.returncode == 64
    assert result.stdout == ""


def test_valid_arms_do_not_log_environment_or_secrets():
    script = f"""
set -euo pipefail
source {shlex.quote(str(ARMS_SCRIPT))}
export HF_TOKEN=hf-secret-sentinel TELYS_TOKEN=telys-secret-sentinel
for arm in A B C A; do
    runpod_worker_arm "$arm"
done
"""
    result = run_shell(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert "hf-secret-sentinel" not in result.stdout + result.stderr
    assert "telys-secret-sentinel" not in result.stdout + result.stderr
