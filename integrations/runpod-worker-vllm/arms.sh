#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source this file, then call `runpod_worker_arm A|B|C` before starting the
# RunPod worker. The selected environment is exported to the current shell.
#
# Arm B matches vLLM v0.32.0's OffloadingConnector contract:
# docs/features/kv_offloading_usage.md and
# vllm/v1/kv_offload/tiering/fs/manager.py. The CPU tier is required as the
# staging tier for filesystem-backed KV.

_runpod_worker_arm_json() {
    python3 - "$@" <<'PY'
import json
import sys

kind = sys.argv[1] if len(sys.argv) > 1 else ""
if kind == "filesystem" and len(sys.argv) == 4:
    root_dir = sys.argv[2]
    cpu_bytes_to_use = int(sys.argv[3])
    config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": root_dir,
                    "n_read_threads": 16,
                    "n_write_threads": 16,
                }
            ],
        },
    }
elif kind == "telys" and len(sys.argv) == 2:
    config = {
        "kv_connector": "TelysKVConnector",
        "kv_connector_module_path": "telys_vllm.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "telys_blobstore_no_slab": True,
        },
    }
else:
    raise ValueError("invalid internal KV configuration request")

sys.stdout.write(json.dumps(config, sort_keys=True, separators=(",", ":")))
PY
}

runpod_worker_arm() {
    if [[ $# -ne 1 ]]; then
        printf 'runpod_worker_arm: expected exactly one arm: A, B, or C\n' >&2
        return 64
    fi

    local arm=$1
    case "$arm" in
        A | B | C) ;;
        *)
            printf 'runpod_worker_arm: unknown arm; expected A, B, or C\n' >&2
            return 64
            ;;
    esac

    local volume=${RUNPOD_WORKER_VOLUME:-/runpod-volume}
    if [[ $volume != /* ]]; then
        printf 'runpod_worker_arm: RUNPOD_WORKER_VOLUME must be an absolute path\n' >&2
        return 64
    fi

    export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
    export ENABLE_PREFIX_CACHING=false

    # Do not let a previous arm or a remote Telys endpoint leak into this one.
    unset KV_TRANSFER_CONFIG TELYS_PATH TELYS_ADDRESS TELYS_VLLM_SLAB_PATH

    local kv_config
    case "$arm" in
        A)
            return 0
            ;;
        B)
            local cpu_bytes=${RUNPOD_WORKER_CPU_BYTES_TO_USE:-1073741824}
            if [[ ! $cpu_bytes =~ ^[1-9][0-9]*$ ]]; then
                printf 'runpod_worker_arm: CPU byte budget must be a positive integer\n' >&2
                return 64
            fi
            if ! kv_config=$(_runpod_worker_arm_json filesystem \
                "${volume%/}/vllm-filesystem-kv" "$cpu_bytes"); then
                return 1
            fi
            ;;
        C)
            export TELYS_PATH="${volume%/}/telys-kv"
            if ! kv_config=$(_runpod_worker_arm_json telys); then
                return 1
            fi
            ;;
    esac

    export KV_TRANSFER_CONFIG="$kv_config"
}
