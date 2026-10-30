#!/usr/bin/env bash
# Run A/B/C benchmark arms on the Runpod pod and emit JSON results.
set -euo pipefail

source /root/bench-venv/bin/activate
export TELYS_API_KEY=$(cat /root/.telys/api_key)
export TELYS_PATH=/root/.telys/runtime
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=INFO

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
NUM_PROMPTS=20
REPEAT_COUNT=5
INPUT_RANGE="512:2048"
OUTPUT_LEN=10
PREFIX_LEN=512
SEED=42
BENCH_DIR=/root/bench
SCRIPT="$BENCH_DIR/benchmark_prefix_caching_metrics.py"

mkdir -p "$BENCH_DIR/results"

run_arm() {
    local arm=$1
    shift
    local extra_args=("$@")
    for i in 1 2 3; do
        local out="$BENCH_DIR/results/${arm}_run${i}.json"
        echo "=== $arm run $i ==="
        python3 "$SCRIPT" \
            --model "$MODEL" \
            --num-prompts "$NUM_PROMPTS" \
            --repeat-count "$REPEAT_COUNT" \
            --input-length-range "$INPUT_RANGE" \
            --output-len "$OUTPUT_LEN" \
            --prefix-len "$PREFIX_LEN" \
            --seed "$SEED" \
            --output-json "$out" \
            "${extra_args[@]}" \
            2>&1 | tee "$BENCH_DIR/results/${arm}_run${i}.log" | tail -20
    done
}

echo "=== ARM A: no prefix caching ==="
run_arm arm_a

echo "=== ARM B: native prefix caching ==="
run_arm arm_b --enable-prefix-caching

echo "=== ARM C: Telys connector (cold) ==="
run_arm arm_c --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both","kv_connector_extra_config":{"telys_client_factory":"telys_vllm.local_client:TelysLocalClient"}}'

echo "=== ARM C warm restart ==="
# First boot: compute + save to Telys.
python3 "$SCRIPT" \
    --model "$MODEL" \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both","kv_connector_extra_config":{"telys_client_factory":"telys_vllm.local_client:TelysLocalClient"}}' \
    --num-prompts "$NUM_PROMPTS" \
    --repeat-count "$REPEAT_COUNT" \
    --input-length-range "$INPUT_RANGE" \
    --output-len "$OUTPUT_LEN" \
    --prefix-len "$PREFIX_LEN" \
    --seed "$SEED" \
    --output-json "$BENCH_DIR/results/arm_c_warm_first.json" \
    2>&1 | tee "$BENCH_DIR/results/arm_c_warm_first.log" | tail -20

# Second boot in same process cannot re-create LLM easily; we run a second
# process.  With the embedded Telys engine this starts a fresh store, so this
# test mainly measures whether the on-disk Telys path persists.
python3 "$SCRIPT" \
    --model "$MODEL" \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both","kv_connector_extra_config":{"telys_client_factory":"telys_vllm.local_client:TelysLocalClient"}}' \
    --num-prompts "$NUM_PROMPTS" \
    --repeat-count "$REPEAT_COUNT" \
    --input-length-range "$INPUT_RANGE" \
    --output-len "$OUTPUT_LEN" \
    --prefix-len "$PREFIX_LEN" \
    --seed "$SEED" \
    --output-json "$BENCH_DIR/results/arm_c_warm_second.json" \
    2>&1 | tee "$BENCH_DIR/results/arm_c_warm_second.log" | tail -20

echo "=== done ==="
