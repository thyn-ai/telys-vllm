#!/usr/bin/env bash
# Telemetry autopsy for the v5 headline workload.
#
# Runs Arm A (recompute) and Arm C' (Telys, native prefix caching OFF) with
# per-phase telemetry enabled.  Captures benchmark JSON, vLLM logs, and
# telemetry JSONL under /root/bench/results-v5-autopsy/.
set -euo pipefail

source /root/bench-venv/bin/activate

export PYTHONPATH=/root/bench/telys-vllm${PYTHONPATH:+:$PYTHONPATH}
export TELYS_PATH=/root/.telys/data
# No slab path: TelysBlobClient stores payloads directly in BlobStore.
export TELYS_VLLM_SLAB_PATH=""
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO

BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/telys-vllm/bench/benchmark_prefix_caching_metrics.py"
RESULTS_DIR="$BENCH_DIR/results-v5-autopsy"
TELEMETRY_DIR="$RESULTS_DIR/telemetry"

mkdir -p "$RESULTS_DIR" "$TELEMETRY_DIR"

# Telys connector config with telemetry enabled.
# No explicit client factory: the connector auto-selects TelysBlobClient when
# telys.BlobStore is available, otherwise falls back to TelysLocalClient.
kv_config() {
    local telemetry_log=$1
    printf '%s' "{\"kv_connector\":\"TelysKVConnector\",\"kv_connector_module_path\":\"telys_vllm.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"telys_telemetry\":\"1\",\"telys_telemetry_log\":\"$telemetry_log\",\"telys_slab_path\":\"\"}}"
}

reset_telys_state() {
    echo "Resetting Telys state (engine data + slab + blobs) ..."
    rm -f "$TELYS_VLLM_SLAB_PATH"
    rm -rf "${TELYS_PATH}_blobs"
    rm -rf "$TELYS_PATH"
    mkdir -p "$TELYS_PATH"
}

run_arm_a() {
    local out="$RESULTS_DIR/headline_arm_a.json"
    echo "=== headline: ARM A (recompute, no prefix caching) ==="
    python3 "$SCRIPT" \
        --model Qwen/Qwen2.5-7B-Instruct \
        --num-prompts 20 --repeat-count 5 \
        --input-length-range 4096:8192 \
        --output-len 64 --prefix-len 4096 --seed 42 \
        --no-enable-prefix-caching \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/headline_arm_a.log" | tail -25
}

run_arm_c_cold() {
    local out="$RESULTS_DIR/headline_arm_c_cold.json"
    local telemetry_log="$TELEMETRY_DIR/headline_arm_c_cold.jsonl"
    echo "=== headline: ARM C' cold (Telys empty, native caching OFF) ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" \
        --model Qwen/Qwen2.5-7B-Instruct \
        --num-prompts 20 --repeat-count 1 \
        --input-length-range 4096:8192 \
        --output-len 64 --prefix-len 4096 --seed 42 \
        --no-enable-prefix-caching \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/headline_arm_c_cold.log" | tail -25
}

run_arm_c_warm() {
    local out="$RESULTS_DIR/headline_arm_c_warm.json"
    local telemetry_log="$TELEMETRY_DIR/headline_arm_c_warm.jsonl"
    echo "=== headline: ARM C' warm (full-prefix hit, native caching OFF) ==="
    TELYS_VLLM_TELEMETRY=1 TELYS_VLLM_TELEMETRY_LOG="$telemetry_log" \
        python3 "$SCRIPT" \
        --model Qwen/Qwen2.5-7B-Instruct \
        --num-prompts 20 --repeat-count 5 \
        --input-length-range 4096:8192 \
        --output-len 64 --prefix-len 4096 --seed 42 \
        --no-enable-prefix-caching \
        --kv-transfer-config "$(kv_config "$telemetry_log")" \
        --output-json "$out" \
        2>&1 | tee "$RESULTS_DIR/headline_arm_c_warm.log" | tail -25
}

# ---------------------------------------------------------------------------
# Arm A: recompute baseline.
# ---------------------------------------------------------------------------
run_arm_a

# ---------------------------------------------------------------------------
# Arm C' cold: Telys starts empty.
# ---------------------------------------------------------------------------
reset_telys_state
run_arm_c_cold

# ---------------------------------------------------------------------------
# Arm C' warm: reuse the state populated by the cold run.
# ---------------------------------------------------------------------------
run_arm_c_warm

# ---------------------------------------------------------------------------
# Summarize telemetry.
# ---------------------------------------------------------------------------
echo "=== telemetry summaries ==="
for tel in "$TELEMETRY_DIR"/*.jsonl; do
    [ -e "$tel" ] || continue
    echo "--- $(basename "$tel") ---"
    python3 "$BENCH_DIR/telys-vllm/bench/summarize_telemetry.py" "$tel" | tail -40
done

echo "=== done ==="
