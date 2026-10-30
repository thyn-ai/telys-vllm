#!/usr/bin/env bash
# v5 A/C' benchmark: native prefix caching is DISABLED so Telys is the only cache.
#
# This is the "fair" measurement: v4 showed that with --enable-prefix-caching,
# vLLM computes the shared prefix natively before consulting the connector, giving
# 0% external hits by construction. v5 turns native caching off, so Telys must
# carry the full shared prefix.
#
# Workloads:
#   1) small    : Qwen2.5-1.5B, 512-token prefix (continuity baseline vs v4)
#   2) headline : Qwen2.5-7B, 4096-token prefix, output_len=64
#   3) persist  : Qwen2.5-7B, 16384-token prefix, output_len=64
#                 (native cache cannot hold the whole working set)
set -euo pipefail

source /root/bench-venv/bin/activate

export TELYS_PATH=/root/.telys/data
export TELYS_VLLM_SLAB_PATH="${TELYS_PATH}_kv.slab"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v5"

# Telys connector config. Native prefix caching is NOT passed for C'.
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

reset_telys_state() {
    echo "Resetting Telys state (engine data + slab)..."
    rm -f "$TELYS_VLLM_SLAB_PATH"
    rm -rf "$TELYS_PATH"
    mkdir -p "$TELYS_PATH"
}

# ---------------------------------------------------------------------------
# 1) Small continuity baseline (matches v4 shape but C' has native caching off)
# ---------------------------------------------------------------------------
reset_telys_state

echo "=== small: ARM A (no prefix caching, eager) ==="
run_arm small arm_a \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 512:2048 \
    --output-len 10 --prefix-len 512 --seed 42 \
    --no-enable-prefix-caching --enforce-eager

echo "=== small: ARM C' (Telys, native caching OFF, no enforce-eager) ==="
run_arm small arm_c \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 512:2048 \
    --output-len 10 --prefix-len 512 --seed 42 \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG"

# ---------------------------------------------------------------------------
# 2) Headline: 7B / 4K prefix / output_len=64
# ---------------------------------------------------------------------------
reset_telys_state

echo "=== headline: ARM A (no prefix caching) ==="
run_arm headline arm_a \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 4096:8192 \
    --output-len 64 --prefix-len 4096 --seed 42 \
    --no-enable-prefix-caching

echo "=== headline: ARM C' (Telys, native caching OFF) ==="
run_arm headline arm_c \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 4096:8192 \
    --output-len 64 --prefix-len 4096 --seed 42 \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG"

# ---------------------------------------------------------------------------
# 3) Persistence case: 7B / 16K prefix / output_len=64
#    Native prefix cache cannot hold the whole working set; Telys persists.
# ---------------------------------------------------------------------------
reset_telys_state

echo "=== persist: ARM A (no prefix caching) ==="
run_arm persist arm_a \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 16384:24576 \
    --output-len 64 --prefix-len 16384 --seed 42 \
    --no-enable-prefix-caching

echo "=== persist: ARM C' (Telys, native caching OFF) ==="
run_arm persist arm_c \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 20 --repeat-count 5 \
    --input-length-range 16384:24576 \
    --output-len 64 --prefix-len 16384 --seed 42 \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG"

echo "=== done ==="
