#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that a warm Telys hit produces bit-identical output to a cold run.

This script is intended to run on real vLLM + Telys hardware (pod). It is not
part of the local CPU-only pytest suite; the matching pytest wrapper skips by
default and only runs when ``TELYS_OUTPUT_CORRECTNESS_MODEL`` is set.

Usage (on the pod):

    export TELYS_PATH=/root/.telys/data
    python3 bench/verify_warm_hit_output.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --prefix-len 64 --input-length-range 64:128 \
        --output-len 10 --seed 42 --reset-telys --enforce-eager

It exits 0 when cold and warm outputs match, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmark_prefix_caching_metrics import sample_requests_from_random

LOGGER = logging.getLogger(__name__)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:  # pragma: no cover - older vLLM layouts
    from backend_request_func import get_tokenizer


DEFAULT_KV_TRANSFER_CONFIG: dict[str, Any] = {
    "kv_connector": "TelysKVConnector",
    "kv_connector_module_path": "telys_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "telys_client_factory": "telys_vllm.local_client:TelysLocalClient",
    },
}


def _ensure_telys_path() -> tuple[str, str]:
    """Return (telys_path, slab_path) to use for this run."""
    telys_path = os.environ.get("TELYS_PATH", "").strip()
    if not telys_path:
        telys_path = tempfile.mkdtemp(prefix="telys-output-correctness-")
        os.environ["TELYS_PATH"] = telys_path
    slab_path = os.environ.get("TELYS_VLLM_SLAB_PATH", "").strip()
    if not slab_path:
        slab_path = os.path.join(telys_path, "kv.slab")
        os.environ["TELYS_VLLM_SLAB_PATH"] = slab_path
    return telys_path, slab_path


def _reset_telys(telys_path: str, slab_path: str) -> None:
    LOGGER.info("Resetting Telys state at %s", telys_path)
    shutil.rmtree(telys_path, ignore_errors=True)
    Path(telys_path).mkdir(parents=True, exist_ok=True)
    Path(slab_path).unlink(missing_ok=True)


def _build_kv_transfer_config(slab_path: str) -> str:
    config = dict(DEFAULT_KV_TRANSFER_CONFIG)
    extra = dict(config["kv_connector_extra_config"])
    extra["telys_slab_path"] = slab_path
    config["kv_connector_extra_config"] = extra
    return json.dumps(config)


def _generate_prompt(args: argparse.Namespace) -> str:
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    random.seed(args.seed)
    min_len, max_len = tuple(map(int, args.input_length_range.split(":")))
    requests = sample_requests_from_random(
        num_requests=1,
        tokenizer=tokenizer,
        input_length_range=(min_len, max_len),
        fixed_output_len=args.output_len,
        prefix_len=args.prefix_len,
    )
    return requests[0].prompt


def _run_once(llm: LLM, prompt: str, output_len: int, seed: int) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        seed=seed,
    )
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    out = outputs[0]
    generated = out.outputs[0]
    metrics = out.metrics
    return {
        "text": generated.text,
        "token_ids": list(generated.token_ids),
        "ttft_s": float(metrics.first_token_latency) if metrics else None,
    }


def main() -> int:
    parser = FlexibleArgumentParser(
        description="Verify warm-hit output equals cold output through Telys."
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--input-length-range", type=str, default="64:128")
    parser.add_argument("--reset-telys", action="store_true")
    parser = EngineArgs.add_cli_args(parser)

    # Disable native prefix caching by default so Telys must carry the full hit.
    parser.set_defaults(enable_prefix_caching=False)

    telys_path, slab_path = _ensure_telys_path()
    # Set a default kv-transfer-config unless the caller provided one.
    parser.set_defaults(kv_transfer_config=_build_kv_transfer_config(slab_path))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.reset_telys:
        _reset_telys(telys_path, slab_path)

    prompt = _generate_prompt(args)
    LOGGER.info("Prompt token length target range: %s", args.input_length_range)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)

    LOGGER.info("Running COLD request...")
    cold = _run_once(llm, prompt, args.output_len, args.seed)
    LOGGER.info("Cold TTFT: %.3f s", cold["ttft_s"])

    LOGGER.info("Running WARM request (same prompt)...")
    warm = _run_once(llm, prompt, args.output_len, args.seed)
    LOGGER.info("Warm TTFT: %.3f s", warm["ttft_s"])

    match_text = cold["text"] == warm["text"]
    match_tokens = cold["token_ids"] == warm["token_ids"]
    passed = match_text and match_tokens

    result = {
        "model": args.model,
        "prompt": prompt,
        "cold": cold,
        "warm": warm,
        "match_text": match_text,
        "match_tokens": match_tokens,
        "passed": passed,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))

    if passed:
        LOGGER.info("PASS: warm-hit output is bit-identical to cold output.")
        return 0
    LOGGER.error("FAIL: warm-hit output differs from cold output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
