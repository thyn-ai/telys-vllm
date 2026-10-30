#!/usr/bin/env bash
# RUN 1 — 16K-prefix stability retest (v6)
set -euo pipefail

source /root/bench-venv/bin/activate
export PYTHONPATH=/root/bench/telys-vllm${PYTHONPATH:+:$PYTHONPATH}
export TELYS_PATH=/root/.telys/data
export TELYS_VLLM_SLAB_PATH=""
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v6-stability"
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

run_arm_a() {
    local prefix_len=$1
    local input_range=$2
    local out="$RESULTS_DIR/arm_a_p${prefix_len}.json"
    echo "=== v6 ARM A (recompute) prefix=$prefix_len input=$input_range ==="
    python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len "$prefix_len" \
        --input-length-range "$input_range" \
        --repeat-count 5 \
        --max-model-len $((prefix_len + 8192 + 512)) \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/arm_a_p${prefix_len}.log" | tail -30
}

run_arm_c_cold() {
    local prefix_len=$1
    local input_range=$2
    local out="$RESULTS_DIR/arm_c_cold_p${prefix_len}.json"
    local telemetry_log="$TELEMETRY_DIR/arm_c_cold_p${prefix_len}.jsonl"
    echo "=== v6 ARM C' cold prefix=$prefix_len input=$input_range ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len "$prefix_len" \
        --input-length-range "$input_range" \
        --repeat-count 1 \
        --max-model-len $((prefix_len + 8192 + 512)) \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/arm_c_cold_p${prefix_len}.log" | tail -30
}

run_arm_c_warm() {
    local prefix_len=$1
    local input_range=$2
    local out="$RESULTS_DIR/arm_c_warm_p${prefix_len}.json"
    local telemetry_log="$TELEMETRY_DIR/arm_c_warm_p${prefix_len}.jsonl"
    echo "=== v6 ARM C' warm prefix=$prefix_len input=$input_range ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" "${common_args[@]}" \
        --prefix-len "$prefix_len" \
        --input-length-range "$input_range" \
        --repeat-count 5 \
        --max-model-len $((prefix_len + 8192 + 512)) \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/arm_c_warm_p${prefix_len}.log" | tail -30
}

run_one_prefix() {
    local prefix_len=$1
    local input_range=$2
    reset_telys_state
    run_arm_a "$prefix_len" "$input_range"
    reset_telys_state
    run_arm_c_cold "$prefix_len" "$input_range"
    run_arm_c_warm "$prefix_len" "$input_range"
}

# 8K-prefix test
run_one_prefix 8192 "8192:12288"

# 16K-prefix test (only if 8K warm finished without crashing)
run_one_prefix 16384 "16384:20480"

echo "=== v6 stability retest complete ==="
