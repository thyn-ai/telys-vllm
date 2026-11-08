#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a fresh-process A/B/C external-prefix-cache benchmark without changing
# the worker's environment-to-CLI translation. Every harness invocation gets
# an explicit --kv-transfer-config.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=integrations/runpod-worker-vllm/arms.sh
ARMS_SCRIPT="$SCRIPT_DIR/arms.sh"
METRICS_SCRIPT=${RUNPOD_BENCHMARK_METRICS_SCRIPT:-$SCRIPT_DIR/../../bench/benchmark_prefix_caching_metrics.py}
CORRECTNESS_SCRIPT=${RUNPOD_BENCHMARK_CORRECTNESS_SCRIPT:-$SCRIPT_DIR/../../bench/verify_warm_hit_output.py}
LIFECYCLE_SCRIPT="$SCRIPT_DIR/../../bench/verify_persisted_chunks.py"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_results.py"
MODEL=Qwen/Qwen2.5-7B-Instruct
OUTPUT_LEN=64
SEED=42

usage() {
    printf 'Usage: %s [--dry-run] [4096|8192|16384] [run-id]\n' "${0##*/}" >&2
}

fail() {
    printf 'benchmark: %s\n' "$*" >&2
    return 64
}

require_file() {
    [[ -f $1 ]] || fail "required file not found: $1"
}

prepare_root() {
    local root=$1
    [[ $root == /* ]] || fail "run root must be an absolute path: $root"
    [[ $root != / ]] || fail "refusing unsafe run root: /"
    if [[ -e $root || -L $root ]]; then
        fail "refusing to reuse run root: $root"
    fi

    if ((BENCHMARK_DRY_RUN)); then
        return 0
    fi

    mkdir -p -- "$(dirname -- "$root")"
    # mkdir is intentionally atomic: a concurrent creator also fails closed.
    mkdir -- "$root"
    mkdir -- "$root/results"
}

print_command() {
    local arm=$1 phase=$2 cache_root=$3
    shift 3
    printf 'arm=%q phase=%q cache_root=%q vllm_batch_invariant=%q' \
        "$arm" "$phase" "$cache_root" "${VLLM_BATCH_INVARIANT-0}"
    if [[ ${TELYS_PATH-} ]]; then
        printf ' telys_path=%q telys_blob_path=%q telys_slab_path=%q' \
            "$TELYS_PATH" "${TELYS_PATH}_blobs" "${TELYS_VLLM_SLAB_PATH}"
    fi
    printf ' command='
    printf '%q ' "$@"
    printf '\n'
}

cache_has_files() {
    local arm=$1 cache_root=$2
    local -a paths=()
    case "$arm" in
        B)
            paths=("$cache_root/vllm-filesystem-kv")
            ;;
        C)
            local telys_path="$cache_root/telys-kv"
            paths=("$telys_path" "${telys_path}_blobs" "${telys_path}_kv.slab")
            ;;
        *)
            fail "internal cache check requested for invalid arm: $arm"
            ;;
    esac

    local path first_file
    for path in "${paths[@]}"; do
        if [[ -f $path ]]; then
            return 0
        fi
        if [[ -d $path ]]; then
            first_file=$(find "$path" -type f -print -quit)
            if [[ $first_file ]]; then
                return 0
            fi
        fi
    done
    printf 'benchmark: cold %s populate produced no cache files under %s\n' \
        "$arm" "$cache_root" >&2
    return 1
}

run_process() {
    local arm=$1 phase=$2 cache_root=$3 num_prompts=$4 range=$5 harness=$6
    local output_json="$BENCHMARK_RESULTS/arm_${arm}_${phase}.json"
    if [[ $phase == run1 || $phase == run2 ]]; then
        output_json="$BENCHMARK_RESULTS/correctness_arm_${arm}_${phase}.json"
    fi
    local output_log="$BENCHMARK_RESULTS/arm_${arm}_${phase}.log"
    local telemetry="$BENCHMARK_RESULTS/arm_${arm}_${phase}.telemetry.jsonl"
    local evidence="$BENCHMARK_RESULTS/arm_${arm}_${phase}.lifecycle.json"
    local manifests="$BENCHMARK_RESULTS/manifests/arm_${arm}_${phase}"

    (
        export RUNPOD_WORKER_VOLUME="$cache_root"
        export MODEL_NAME="$MODEL"
        export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
        export VLLM_LOGGING_LEVEL=INFO VLLM_LOG_STATS_INTERVAL=1
        export VLLM_ENABLE_V1_MULTIPROCESSING \
            HF_HUB_ENABLE_HF_TRANSFER=0
        if [[ $phase == run1 || $phase == run2 ]]; then
            # Batch-invariant kernels prevent numerical batching variation from
            # being mistaken for KV-cache corruption in output comparisons.
            export VLLM_BATCH_INVARIANT=1
        else
            # Never pay this correctness-only cost in performance timing.
            unset VLLM_BATCH_INVARIANT
        fi
        # shellcheck source=integrations/runpod-worker-vllm/arms.sh
        source "$ARMS_SCRIPT"
        runpod_worker_arm "$arm"

        if [[ $arm == C ]]; then
            export TELYS_VLLM_SLAB_PATH="${TELYS_PATH}_kv.slab"
        fi

        local lifecycle=0
        if ((BENCHMARK_LIFECYCLE)) && [[ $arm == C && $harness == "$METRICS_SCRIPT" ]]; then
            lifecycle=1
            export TELYS_VLLM_TELEMETRY=1
            export TELYS_VLLM_TELEMETRY_LOG="$telemetry"
        fi

        local -a command=(
            python3 "$harness"
            --model "$MODEL"
            --output-len "$OUTPUT_LEN"
            --seed "$SEED"
            --no-enable-prefix-caching
            --gpu-memory-utilization "$BENCHMARK_GPU_MEMORY_UTILIZATION"
            --max-num-seqs 32
            --max-model-len "$BENCHMARK_MAX_MODEL_LEN"
        )
        if [[ $arm != A ]]; then
            command+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
        elif [[ $phase != cold && $phase != warm ]]; then
            # Metrics defaults to no connector. Correctness defaults to Telys,
            # so its explicit override must restore the no-config baseline.
            command+=(--disable-kv-connector)
        fi
        command+=(--output-json "$output_json")
        if ((lifecycle)); then
            command+=(
                --manifest-dir "$manifests"
                --manifest-arm "$arm"
                --manifest-phase "$phase"
                --manifest-namespace "${TELYS_VLLM_NAMESPACE:-telys-vllm}"
                --manifest-max-chunk-blocks "${TELYS_VLLM_MAX_CHUNK_BLOCKS:-0}"
            )
        fi
        if [[ $phase == cold || $phase == warm ]]; then
            command+=(
                --num-prompts "$num_prompts"
                --repeat-count 1
                --prefix-len "$BENCHMARK_PREFIX"
                --input-length-range "$range"
            )
        else
            command+=(
                --prefix-len "$BENCHMARK_PREFIX"
                --input-length-range "$range"
            )
        fi

        local -a record_command=(
            python3 "$LIFECYCLE_SCRIPT" record
            --output "$evidence"
            --telemetry "$telemetry"
            --manifest-dir "$manifests"
            --arm "$arm"
            --phase "$phase"
            --exit-status 0
        )
        if ((BENCHMARK_DRY_RUN)); then
            print_command "$arm" "$phase" "$cache_root" "${command[@]}"
            if ((lifecycle)); then
                printf 'lifecycle_record_command='
                printf '%q ' "${record_command[@]}"
                printf '\n'
            fi
            return 0
        fi

        print_command "$arm" "$phase" "$cache_root" "${command[@]}" \
            >>"$BENCHMARK_COMMANDS_LOG"
        if ((lifecycle)); then
            local command_status=0 tee_status=0
            local -a pipeline_status=()
            set +e
            "${command[@]}" 2>&1 | tee "$output_log"
            pipeline_status=("${PIPESTATUS[@]}")
            command_status=${pipeline_status[0]}
            tee_status=${pipeline_status[1]}
            set -e
            record_command[${#record_command[@]}-1]=$command_status
            "${record_command[@]}"
            if ((command_status != 0)); then
                return "$command_status"
            fi
            if ((tee_status != 0)); then
                return "$tee_status"
            fi
        else
            "${command[@]}" 2>&1 | tee "$output_log"
        fi
    )
}

run_arm() {
    local arm=$1 cache_root=$2
    run_process "$arm" cold "$cache_root" 1 \
        "$BENCHMARK_PREFIX:$BENCHMARK_PREFIX" "$METRICS_SCRIPT"
    if ((BENCHMARK_DRY_RUN == 0)) && [[ $arm != A ]]; then
        cache_has_files "$arm" "$cache_root"
    fi
    run_process "$arm" warm "$cache_root" 20 \
        "$BENCHMARK_PREFIX:$BENCHMARK_WARM_MAX" "$METRICS_SCRIPT"
}

run_correctness() {
    local arm run cache_root
    for arm in A B C; do
        for run in run1 run2; do
            if [[ $arm == A && $run == run2 ]]; then
                continue
            fi
            cache_root="$BENCHMARK_ROOT/correctness/arm-$arm"
            run_process "$arm" "$run" "$cache_root" 1 \
                "$BENCHMARK_PREFIX:$BENCHMARK_PREFIX" "$CORRECTNESS_SCRIPT"
            if ((BENCHMARK_DRY_RUN == 0)) && [[ $arm != A && $run == run1 ]]; then
                cache_has_files "$arm" "$cache_root"
            fi
        done
    done
}

main() {
    BENCHMARK_DRY_RUN=0
    if [[ ${1-} == --dry-run ]]; then
        BENCHMARK_DRY_RUN=1
        shift
    fi
    if (($# > 2)); then
        usage
        return 64
    fi

    BENCHMARK_PREFIX=${1:-4096}
    BENCHMARK_LIFECYCLE=0
    local run_id=${2:-run1}
    case "$BENCHMARK_PREFIX" in
        4096)
            BENCHMARK_WARM_MAX=8192
            BENCHMARK_MAX_MODEL_LEN=12288
            ;;
        8192)
            BENCHMARK_WARM_MAX=12288
            BENCHMARK_MAX_MODEL_LEN=16896
            ;;
        16384)
            BENCHMARK_WARM_MAX=20480
            BENCHMARK_MAX_MODEL_LEN=25088
            BENCHMARK_LIFECYCLE=1
            ;;
        *)
            fail "prefix must be 4096, 8192, or 16384"
            ;;
    esac
    [[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid run-id: $run_id"

    BENCHMARK_ROOT=${RUNPOD_BENCHMARK_ROOT:-/runpod-volume/telys-vllm-benchmark/p${BENCHMARK_PREFIX}-seed${SEED}-${run_id}}
    BENCHMARK_RESULTS="$BENCHMARK_ROOT/results"
    BENCHMARK_COMMANDS_LOG="$BENCHMARK_ROOT/commands.log"
    BENCHMARK_GPU_MEMORY_UTILIZATION=${RUNPOD_BENCHMARK_GPU_MEMORY_UTILIZATION:-0.95}
    VLLM_ENABLE_V1_MULTIPROCESSING=${RUNPOD_BENCHMARK_MULTIPROCESSING:-0}
    export VLLM_ENABLE_V1_MULTIPROCESSING

    require_file "$ARMS_SCRIPT"
    require_file "$METRICS_SCRIPT"
    require_file "$CORRECTNESS_SCRIPT"
    require_file "$SUMMARY_SCRIPT"
    if ((BENCHMARK_LIFECYCLE)); then
        require_file "$LIFECYCLE_SCRIPT"
    fi
    prepare_root "$BENCHMARK_ROOT"

    local arm cache_root
    for arm in A B C; do
        cache_root="$BENCHMARK_ROOT/performance/arm-$arm"
        run_arm "$arm" "$cache_root"
    done
    run_correctness

    local -a summary_command=(python3 "$SUMMARY_SCRIPT" "$BENCHMARK_RESULTS")
    if ((BENCHMARK_DRY_RUN)); then
        printf 'summary command='
        printf '%q ' "${summary_command[@]}"
        printf '\n'
    else
        "${summary_command[@]}"
    fi
}

main "$@"
