#!/usr/bin/env python3
"""Parse benchmark JSON + logs and print a markdown results table.

Supports v2/v3 naming (arm_*.json) and v4 naming (<workload>_arm_*.json) via
--prefix.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def external_hit_rate(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    needle = "External prefix cache hit rate:"
    for line in log_path.read_text().splitlines():
        if needle in line:
            # "..., External prefix cache hit rate: 86.7%"
            return line.split(needle)[-1].strip().rstrip("%").strip()
    return None


def summarize(result_paths: list[Path]) -> dict:
    if not result_paths:
        return {}
    times = []
    throughputs = []
    ttft_means = []
    ttft_p50s = []
    ttft_p95s = []
    hit_rates = []
    for path in result_paths:
        r = load_result(path)
        times.append(r["total_time_s"])
        throughputs.append(r["throughput_tokens_per_s"])
        ttft = r["ttft_s"]
        ttft_means.append(ttft["mean"])
        ttft_p50s.append(ttft["p50"])
        ttft_p95s.append(ttft["p95"])
        hr = external_hit_rate(path.with_suffix(".log"))
        if hr is not None:
            try:
                hit_rates.append(float(hr))
            except ValueError:
                pass

    def rng(vals: list[float]) -> str:
        return f"{min(vals):.3f}–{max(vals):.3f}"

    def med(vals: list[float]) -> float:
        return statistics.median(vals)

    return {
        "runs": len(result_paths),
        "total_time_s": f"{med(times):.3f} ({rng(times)})",
        "throughput": f"{med(throughputs):.3f} ({rng(throughputs)})",
        "ttft_mean": f"{med(ttft_means):.3f} ({rng(ttft_means)})",
        "ttft_p50": f"{med(ttft_p50s):.3f} ({rng(ttft_p50s)})",
        "ttft_p95": f"{med(ttft_p95s):.3f} ({rng(ttft_p95s)})",
        "external_hit_rate": f"{med(hit_rates):.1f}% ({min(hit_rates):.1f}–{max(hit_rates):.1f}%)"
        if hit_rates
        else "n/a",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark results into a markdown table."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("results-v2"),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filename prefix, e.g. 'small_' or 'payoff_'",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    prefix = args.prefix
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    arms = [
        ("A (no prefix caching)", f"{prefix}arm_a_mp0_run*.json"),
        ("B (native prefix caching)", f"{prefix}arm_b_mp0_run*.json"),
        ("C (Telys, MP=0, cold)", f"{prefix}arm_c_mp0_run*.json"),
        ("C warm first (MP=0)", f"{prefix}arm_c_warm_first_mp0_run1.json"),
        ("C warm second (MP=0)", f"{prefix}arm_c_warm_second_mp0_run1.json"),
        ("C (Telys, MP=1)", f"{prefix}arm_c_mp1*_run1.json"),
    ]

    print(
        "| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |"
    )
    print(
        "|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|"
    )
    for label, pattern in arms:
        paths = sorted(results_dir.glob(pattern))
        s = summarize(paths)
        if not s:
            print(f"| {label} | — | — | — | — | — | — | — |")
            continue
        print(
            f"| {label} | {s['runs']} | {s['total_time_s']} | {s['throughput']} | "
            f"{s['ttft_mean']} | {s['ttft_p50']} | {s['ttft_p95']} | {s['external_hit_rate']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
