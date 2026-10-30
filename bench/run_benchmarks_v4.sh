#!/usr/bin/env bash
# v4 A/B/C benchmark: slab-backed async writes + batched injection.
# Runs two workloads:
#   1) small  : Qwen2.5-1.5B, 512-token prefix
#   2) payoff : Qwen2.5-7B, 4096-token prefix
#
# Telys stores only metadata; payload bytes live in the local slab file.
# Arm C no longer needs --enforce-eager; A/B keep it for comparability with v2/v3.
set -euo pipefail

source /root/bench-venv/bin/activate

# Embedded Telys engine data directory. The runtime itself is installed under
# /root/.telys/runtime; the engine data and the v4 slab are kept separate.
export TELYS_PATH=/root/.telys/data
# Slab path is derived automatically as ${TELYS_PATH}_kv.slab; export it
# explicitly for clarity and so reset_telys_state can wipe it.
export TELYS_VLLM_SLAB_PATH="${TELYS_PATH}_kv.slab"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v4"

# Base connector config. TelysLocalClient opens the embedded engine at
# TELYS_PATH; the slab path is derived from TELYS_PATH. Write-behind is on by
# default.
KV_CONFIG='{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both","kv_connector_extra_config":{"telys_client_factory":"telys_vllm.local_client:TelysLocalClient"}}'

mkdir -p "$RESULTS_DIR"

run_arm() {
    local workload=$1
    local arm=$2
    shift 2
    local mp=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
    for i in 1 2 3; do
        local out="$RESULTS_DIR/${workload}_${arm}_mp${mp}_run${i}.json"
        echo "=== $workload $arm (MP=$mp) run $i ==="
        python3 "$SCRIPT" \
            "$@" \
            --output-json "$out" \
            2>&1 | tee "$RESULTS_DIR/${workload}_${arm}_mp${mp}_run${i}.log" | tail -20
    done
}

run_once() {
    local workload=$1
    local arm=$2
    shift 2
    local mp=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
    local out="$RESULTS_DIR/${workload}_${arm}_mp${mp}_run1.json"
    echo "=== $workload $arm (MP=$mp) single run ==="
    python3 "$SCRIPT" \
        "$@" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/${workload}_${arm}_mp${mp}_run1.log" | tail -20
}

run_workload() {
    local workload=$1
    local model=$2
    local input_range=$3
    local prefix_len=$4
    local common_args=(
        --model "$model"
        --num-prompts 20
        --repeat-count 5
        --input-length-range "$input_range"
        --output-len 10
        --prefix-len "$prefix_len"
        --seed 42
    )

    echo "=== $workload: ARM A (no prefix caching, eager) ==="
    run_arm "$workload" arm_a --enforce-eager "${common_args[@]}"

    echo "=== $workload: ARM B (native prefix caching, eager) ==="
    run_arm "$workload" arm_b --enable-prefix-caching --enforce-eager "${common_args[@]}"

    echo "=== $workload: ARM C (Telys, cold, MP=0, NO enforce-eager) ==="
    VLLM_ENABLE_V1_MULTIPROCESSING=0 run_arm "$workload" arm_c --enable-prefix-caching --kv-transfer-config "$KV_CONFIG" "${common_args[@]}"

    echo "=== $workload: ARM C warm restart (MP=0, NO enforce-eager) ==="
    VLLM_ENABLE_V1_MULTIPROCESSING=0 run_once "$workload" arm_c_warm_first --enable-prefix-caching --kv-transfer-config "$KV_CONFIG" "${common_args[@]}"
    VLLM_ENABLE_V1_MULTIPROCESSING=0 run_once "$workload" arm_c_warm_second --enable-prefix-caching --kv-transfer-config "$KV_CONFIG" "${common_args[@]}"

    echo "=== $workload: ARM C with multiprocessing (MP=1, NO enforce-eager) ==="
    VLLM_ENABLE_V1_MULTIPROCESSING=1 run_once "$workload" arm_c_mp1 --enable-prefix-caching --kv-transfer-config "$KV_CONFIG" "${common_args[@]}"
}

# Remove any previous embedded engine data and slab so each workload starts cold.
reset_telys_state() {
    echo "Resetting Telys state (engine data + slab)..."
    rm -f "$TELYS_VLLM_SLAB_PATH"
    rm -rf "$TELYS_PATH"
    mkdir -p "$TELYS_PATH"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

mkdir -p "$RESULTS_DIR"

reset_telys_state

# Small model / short prefix.
run_workload small Qwen/Qwen2.5-1.5B-Instruct 512:2048 512

# Reset state between workloads so the payoff workload is cold.
reset_telys_state

# Larger model / long prefix: this is where external prefix caching pays off.
run_workload payoff Qwen/Qwen2.5-7B-Instruct 5120:8192 4096

echo "=== done ==="
