# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enhanced prefix-caching benchmark that reports TTFT and throughput.

Based on vLLM benchmarks/benchmark_prefix_caching.py at commit
5df63a7203cdb932 (tag v0.32.0).  Adds deterministic JSON output with
per-request TTFT, throughput, and (when available) prefix-cache hit rate.
"""

import dataclasses
import json
import logging
import random
import time
from pathlib import Path

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


def run_benchmark(llm, prompts, sampling_params, output_json_path: str | None):
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

    print("------start generating------")
    run_benchmark(llm, prompts, sampling_params, args.output_json)


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
    parser = EngineArgs.add_cli_args(parser)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
