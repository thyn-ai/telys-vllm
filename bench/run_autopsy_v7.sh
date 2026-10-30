#!/usr/bin/env bash
# RUN 2 — two-replica shared-cache demo (v7)
set -euo pipefail

source /root/bench-venv/bin/activate
export PYTHONPATH=/root/bench/telys-vllm${PYTHONPATH:+:$PYTHONPATH}
export TELYS_PATH=/root/.telys/data
export TELYS_VLLM_SLAB_PATH=""
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v7-replica"
TELEMETRY_DIR="$RESULTS_DIR/telemetry"

mkdir -p "$RESULTS_DIR" "$TELEMETRY_DIR"

kv_config() {
    local telemetry_log=$1
    printf '%s' "{\"kv_connector\":\"TelysKVConnector\",\"kv_connector_module_path\":\"telys_vllm.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"telys_telemetry\":\"1\",\"telys_telemetry_log\":\"$telemetry_log\",\"telys_slab_path\":\"\"}}"
}

reset_telys_state() {
    echo "Resetting Telys state ..."
    rm -f "$TELYS_VLLM_SLAB_PATH"
    rm -rf "${TELYS_PATH}_blobs"
    rm -rf "$TELYS_PATH"
    mkdir -p "$TELYS_PATH"
}

common_args=(
    --model Qwen/Qwen2.5-7B-Instruct
    --num-prompts 20
    --output-len 64
    --seed 42
    --no-enable-prefix-caching
    --gpu-memory-utilization 0.95
    --max-num-seqs 32
)

# Baseline: recompute (Arm A)
run_arm_a() {
    local out="$RESULTS_DIR/arm_a_recompute.json"
    echo "=== v7 ARM A (recompute baseline) ==="
    python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len 4096 \
        --input-length-range 4096:8192 \
        --repeat-count 5 \
        --max-model-len 12288 \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/arm_a_recompute.log" | tail -30
}

# Engine A: cold run, populates shared Telys store, then process exits.
run_engine_a_cold() {
    local out="$RESULTS_DIR/engine_a_cold.json"
    local telemetry_log="$TELEMETRY_DIR/engine_a_cold.jsonl"
    echo "=== v7 Engine A (cold populate, fresh process) ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len 4096 \
        --input-length-range 4096:8192 \
        --repeat-count 1 \
        --max-model-len 12288 \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/engine_a_cold.log" | tail -30
}

# Engine B: completely fresh process, native cache empty, hits shared Telys store.
run_engine_b_warm() {
    local out="$RESULTS_DIR/engine_b_warm.json"
    local telemetry_log="$TELEMETRY_DIR/engine_b_warm.jsonl"
    echo "=== v7 Engine B (warm hit against shared store, fresh process) ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len 4096 \
        --input-length-range 4096:8192 \
        --repeat-count 5 \
        --max-model-len 12288 \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/engine_b_warm.log" | tail -30
}

reset_telys_state
run_arm_a
run_engine_a_cold
run_engine_b_warm

echo "=== v7 two-replica shared-cache demo complete ==="
