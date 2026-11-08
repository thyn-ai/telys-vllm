# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enhanced prefix-caching benchmark that reports TTFT and throughput.

Based on vLLM benchmarks/benchmark_prefix_caching.py at commit
5df63a7203cdb932 (tag v0.32.0).  Adds deterministic JSON output with
per-request TTFT, throughput, and (when available) prefix-cache hit rate.
Optional lifecycle manifests preserve actual token IDs and chunk-hash inputs.
"""

import dataclasses
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)
    # Sort vocabulary to guarantee deterministic token sampling order across
    # runs.  ``get_vocab()`` insertion order can vary with HF cache/state.
    candidates = sorted(v for v in vocab.values() if v not in all_special_ids)
    return random.choices(candidates, k=length)


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
    prefix_len: int,
) -> list[Request]:
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range
    for _ in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert min_len <= prompt_len <= max_len, (
            f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        )
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False
) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _token_ids_digest(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _actual_token_ids(raw_ids: Any, *, field_name: str) -> list[int]:
    token_ids = list(raw_ids)
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(f"{field_name} must contain non-negative integer token IDs")
    return token_ids


def _kv_manifest_context(vllm_config: Any, block_size: int) -> dict[str, Any]:
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    kv_dtype = get_kv_cache_torch_dtype(
        vllm_config.cache_config.cache_dtype, model_config.dtype
    )
    is_full_attention = not bool(
        getattr(model_config, "use_mla", False)
        or getattr(model_config, "is_hybrid", False)
    )
    expected_shape = None
    if is_full_attention:
        expected_shape = [
            block_size,
            int(model_config.get_num_kv_heads(parallel_config)),
            int(model_config.get_head_size()),
        ]
    return {
        "dtype": str(kv_dtype),
        "expected_block_shape": expected_shape,
        "expected_layer_count": int(model_config.get_num_layers(parallel_config)),
        "shape_kind": "full_attention" if is_full_attention else "unresolved",
    }


def build_manifest_context(llm: Any, args: Any) -> dict[str, Any]:
    from vllm.v1.core.kv_cache_utils import get_none_hash_seed

    vllm_config = llm.llm_engine.vllm_config
    cache_config = vllm_config.cache_config
    block_size = int(cache_config.block_size)
    hash_block_size = int(cache_config.hash_block_size or block_size)
    algorithm = str(cache_config.prefix_caching_hash_algo)
    if block_size <= 0 or hash_block_size <= 0:
        raise ValueError("resolved vLLM block sizes must be positive")
    if algorithm not in ("sha256", "sha256_cbor"):
        raise ValueError("lifecycle manifests require a cryptographic block hash")
    if args.manifest_max_chunk_blocks < 0:
        raise ValueError("--manifest-max-chunk-blocks must be non-negative")
    if not args.manifest_namespace.strip():
        raise ValueError("--manifest-namespace must not be empty")
    return {
        "schema_version": 1,
        "kind": "telys_vllm_request_manifest",
        "model": str(vllm_config.model_config.model),
        "run": {"arm": args.manifest_arm, "phase": args.manifest_phase},
        "token_digest": "sha256-json-array-utf8-v1",
        "block_hash": {
            "algorithm": algorithm,
            "none_hash_seed": get_none_hash_seed(),
            "block_size": block_size,
            "hash_block_size": hash_block_size,
            "extra_keys_per_block": None,
        },
        "kv": _kv_manifest_context(vllm_config, block_size),
        "storage": {
            "namespace": args.manifest_namespace,
            "max_chunk_blocks": args.manifest_max_chunk_blocks,
        },
    }


def _request_manifest(out: Any, request_index: int, context: dict[str, Any]) -> dict:
    if len(out.outputs) != 1:
        raise ValueError(
            "lifecycle manifests require exactly one completion per request"
        )
    completion = out.outputs[0]
    prompt_ids = _actual_token_ids(out.prompt_token_ids, field_name="prompt_token_ids")
    generated_ids = _actual_token_ids(
        completion.token_ids, field_name="generated_token_ids"
    )
    all_ids = prompt_ids + generated_ids
    block_size = context["block_hash"]["block_size"]
    persisted_count = max(0, (len(all_ids) - 1) // block_size)
    prefix_count = min(len(prompt_ids) // block_size, persisted_count)
    request_id = str(out.request_id)
    return {
        **context,
        "response": {
            "request_id": request_id,
            "request_index": request_index,
            "completion_index": int(completion.index),
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "text_sha256": hashlib.sha256(completion.text.encode("utf-8")).hexdigest(),
        },
        "prompt": {
            "token_ids": prompt_ids,
            "count": len(prompt_ids),
            "sha256": _token_ids_digest(prompt_ids),
        },
        "generated": {
            "token_ids": generated_ids,
            "count": len(generated_ids),
            "sha256": _token_ids_digest(generated_ids),
        },
        "expected": {
            "all_token_count": len(all_ids),
            "all_token_sha256": _token_ids_digest(all_ids),
            "full_block_count": len(all_ids) // block_size,
            "persisted_block_count": persisted_count,
            "prefix_block_count": prefix_count,
            "suffix_block_count": persisted_count - prefix_count,
            "suffix_start_block": prefix_count,
            "suffix_start_token": prefix_count * block_size,
            "last_token_excluded": True,
        },
    }


def write_request_manifests(
    outputs: list[Any], manifest_dir: str, context: dict[str, Any]
) -> None:
    directory = Path(manifest_dir)
    for request_index, out in enumerate(outputs):
        manifest = _request_manifest(out, request_index, context)
        request_id = manifest["response"]["request_id"]
        identity = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
        path = directory / f"{request_index:06d}-{identity}.json"
        _write_json_atomic(path, manifest)


def run_benchmark(
    llm,
    prompts,
    sampling_params,
    output_json_path: str | None,
    *,
    manifest_dir: str | None = None,
    manifest_context: dict[str, Any] | None = None,
):
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()

    ttfts = []
    total_output_tokens = 0
    for out in outputs:
        m = out.metrics
        if m is None:
            raise RuntimeError(
                "RequestOutput.metrics is None; run with "
                "--disable-log-stats=False or pass disable_log_stats=False "
                "to LLM.from_engine_args."
            )
        # vLLM 0.32.0 V1 engine: first_token_latency is the full TTFT.
        ttfts.append(float(m.first_token_latency))
        total_output_tokens += len(out.outputs[0].token_ids)

    if manifest_dir is not None:
        if manifest_context is None:
            raise ValueError("manifest_context is required when manifest_dir is set")
        write_request_manifests(outputs, manifest_dir, manifest_context)
        print(f"Wrote {len(outputs)} request manifests to {manifest_dir}")

    total_time = end_time - start_time
    throughput = total_output_tokens / total_time if total_time > 0 else 0.0

    result = {
        "total_time_s": total_time,
        "num_prompts": len(prompts),
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "ttft_s": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
            "p50": _percentile(ttfts, 50),
            "p95": _percentile(ttfts, 95),
            "min": min(ttfts) if ttfts else 0.0,
            "max": max(ttfts) if ttfts else 0.0,
        },
        "per_request_ttft_s": ttfts,
    }

    print(json.dumps(result, indent=2))
    if output_json_path:
        Path(output_json_path).write_text(json.dumps(result, indent=2))
        print(f"Wrote results to {output_json_path}")


def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    random.seed(args.seed)

    print(f"Start to sample {args.num_prompts} prompts from random")
    filtered_requests = sample_requests_from_random(
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        input_length_range=input_length_range,
        fixed_output_len=args.output_len,
        prefix_len=args.prefix_len,
    )

    print(f"Sampled {len(filtered_requests)} requests.")
    prompt_lens = [req.prompt_len for req in filtered_requests]
    print(f"Average input length: {sum(prompt_lens) / len(prompt_lens)}")
    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
    print(f"Min Prompt Length: {min(prompt_lens)}")
    print(f"Max Prompt Length: {max(prompt_lens)}")

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    prompts = repeat_and_sort_requests(
        filtered_requests, repeat_count=args.repeat_count, sort=args.sort
    )

    manifest_context = None
    if args.manifest_dir is not None:
        manifest_context = build_manifest_context(llm, args)

    print("------start generating------")
    run_benchmark(
        llm,
        prompts,
        sampling_params,
        args.output_json,
        manifest_dir=args.manifest_dir,
        manifest_context=manifest_context,
    )


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark prefix caching with TTFT/throughput output."
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
    )
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument("--manifest-dir", type=str, default=None)
    parser.add_argument("--manifest-arm", type=str, default=None)
    parser.add_argument("--manifest-phase", type=str, default=None)
    parser.add_argument("--manifest-namespace", type=str, default="telys-vllm")
    parser.add_argument("--manifest-max-chunk-blocks", type=int, default=0)
    parser = EngineArgs.add_cli_args(parser)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
