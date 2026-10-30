#!/usr/bin/env bash
# v3 A/B/C benchmark after raw-byte chunk + deferred save + register_kv_caches fix.
# Arm C no longer needs --enforce-eager; A/B keep it for v2 comparability.
set -euo pipefail

source /root/bench-venv/bin/activate
export TELYS_PATH=""
export TELYS_ADDRESS=unix:///root/.telys/telys.sock
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
NUM_PROMPTS=20
REPEAT_COUNT=5
INPUT_RANGE="512:2048"
OUTPUT_LEN=10
PREFIX_LEN=512
SEED=42
BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v3"

# Telys connector config: uses the self-host unix-socket server so KV
# persists across vLLM process lifetimes (warm restart).
KV_CONFIG='{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both","kv_connector_extra_config":{"telys_client_factory":"telys_vllm.local_client:TelysLocalClient"}}'

mkdir -p "$RESULTS_DIR"

run_arm() {
    local arm=$1
    shift
    local mp=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
    for i in 1 2 3; do
        local out="$RESULTS_DIR/${arm}_mp${mp}_run${i}.json"
        echo "=== $arm (MP=$mp) run $i ==="
        python3 "$SCRIPT" \
            --model "$MODEL" \
            --num-prompts "$NUM_PROMPTS" \
            --repeat-count "$REPEAT_COUNT" \
            --input-length-range "$INPUT_RANGE" \
            --output-len "$OUTPUT_LEN" \
            --prefix-len "$PREFIX_LEN" \
            --seed "$SEED" \
            --output-json "$out" \
            "$@" \
            2>&1 | tee "$RESULTS_DIR/${arm}_mp${mp}_run${i}.log" | tail -20
    done
}

run_once() {
    local arm=$1
    shift
    local mp=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
    local out="$RESULTS_DIR/${arm}_mp${mp}_run1.json"
    echo "=== $arm (MP=$mp) single run ==="
    python3 "$SCRIPT" \
        --model "$MODEL" \
        --num-prompts "$NUM_PROMPTS" \
        --repeat-count "$REPEAT_COUNT" \
        --input-length-range "$INPUT_RANGE" \
        --output-len "$OUTPUT_LEN" \
        --prefix-len "$PREFIX_LEN" \
        --seed "$SEED" \
        --output-json "$out" \
        "$@" \
        2>&1 | tee "$RESULTS_DIR/${arm}_mp${mp}_run1.log" | tail -20
}

echo "=== ARM A: no prefix caching (eager, like v2) ==="
run_arm arm_a --enforce-eager

echo "=== ARM B: native prefix caching (eager, like v2) ==="
run_arm arm_b --enable-prefix-caching --enforce-eager

echo "=== ARM C: Telys connector (cold, MP=0, NO enforce-eager) ==="
VLLM_ENABLE_V1_MULTIPROCESSING=0 run_arm arm_c --enable-prefix-caching --kv-transfer-config "$KV_CONFIG"

echo "=== ARM C warm restart (MP=0, NO enforce-eager) ==="
VLLM_ENABLE_V1_MULTIPROCESSING=0 run_once arm_c_warm_first --enable-prefix-caching --kv-transfer-config "$KV_CONFIG"
VLLM_ENABLE_V1_MULTIPROCESSING=0 run_once arm_c_warm_second --enable-prefix-caching --kv-transfer-config "$KV_CONFIG"

echo "=== ARM C: Telys connector with multiprocessing (MP=1, NO enforce-eager) ==="
VLLM_ENABLE_V1_MULTIPROCESSING=1 run_once arm_c_mp1 --enable-prefix-caching --kv-transfer-config "$KV_CONFIG"

echo "=== done ==="
