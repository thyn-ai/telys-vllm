#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize fresh-process RunPod worker benchmark results without inventing data."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ARMS = (("A", "no cache"), ("B", "filesystem"), ("C", "Telys"))
PHASES = ("cold", "warm")
HIT_RATE_NEEDLE = "External prefix cache hit rate:"
TELYS_LOOKUP = re.compile(r"hit_blocks=(\d+)(?:.*?hit_tokens=(\d+))?")


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def external_hit_rate(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    found: float | None = None
    for line in log_path.read_text().splitlines():
        if HIT_RATE_NEEDLE not in line:
            continue
        raw = line.split(HIT_RATE_NEEDLE, 1)[1].strip().rstrip("%").strip()
        try:
            parsed = float(raw)
        except ValueError:
            continue
        if math.isfinite(parsed):
            found = parsed
    return found


def last_telys_lookup(log_path: Path) -> dict[str, int] | None:
    if not log_path.exists():
        return None
    found: dict[str, int] | None = None
    for line in log_path.read_text().splitlines():
        if "telys lookup:" not in line:
            continue
        match = TELYS_LOOKUP.search(line)
        if match is None:
            continue
        found = {"hit_blocks": int(match.group(1))}
        if match.group(2) is not None:
            found["hit_tokens"] = int(match.group(2))
    return found


def summarize_run(
    results_dir: Path, arm: str, label: str, phase: str
) -> dict[str, Any]:
    stem = f"arm_{arm}_{phase}"
    result_path = results_dir / f"{stem}.json"
    log_path = results_dir / f"{stem}.log"
    result = load_json(result_path)
    ttft = result.get("ttft_s")
    if not isinstance(ttft, dict):
        ttft = {}
    raw_per_request = result.get("per_request_ttft_s")
    per_request: list[float] = []
    if isinstance(raw_per_request, list):
        per_request = [
            value for item in raw_per_request if (value := number(item)) is not None
        ]

    first_warm = per_request[0] if phase == "warm" and per_request else None
    later_warm = (
        statistics.median(per_request[1:])
        if phase == "warm" and len(per_request) > 1
        else None
    )
    hit_rate = external_hit_rate(log_path)
    lookup = last_telys_lookup(log_path)
    evidence = (
        [f"vLLM external hit rate={hit_rate:.1f}%"]
        if hit_rate is not None
        else ["external hit rate=unreported"]
    )
    if lookup is not None:
        lookup_text = ",".join(f"{key}={value}" for key, value in lookup.items())
        evidence.append(f"Telys last lookup {lookup_text}")

    return {
        "arm": arm,
        "label": label,
        "phase": phase,
        "result_present": bool(result),
        "num_prompts": result.get("num_prompts")
        if isinstance(result.get("num_prompts"), int)
        else None,
        "total_output_tokens": result.get("total_output_tokens")
        if isinstance(result.get("total_output_tokens"), int)
        else None,
        "total_time_s": number(result.get("total_time_s")),
        "throughput_tokens_per_s": number(result.get("throughput_tokens_per_s")),
        "ttft_mean_s": number(ttft.get("mean")),
        "ttft_p50_s": number(ttft.get("p50")),
        "ttft_p95_s": number(ttft.get("p95")),
        "first_warm_ttft_s": first_warm,
        "later_warm_ttft_p50_s": later_warm,
        "external_hit_rate_pct": hit_rate,
        "telys_last_lookup": lookup,
        "external_hit_evidence": "; ".join(evidence),
    }


def output_comparison(actual: Any, reference: dict[str, Any]) -> bool | None:
    if not isinstance(actual, dict):
        return None
    token_ids = actual.get("token_ids")
    text = actual.get("text")
    if not isinstance(token_ids, list) or not isinstance(text, str):
        return None
    return (
        token_ids == reference["cold"]["token_ids"]
        and text == reference["cold"]["text"]
        and len(token_ids) == 64
    )


def summarize_correctness(results_dir: Path) -> list[dict[str, Any]]:
    reference_path = results_dir / "correctness_arm_A_run1.json"
    reference = load_json(reference_path)
    if reference:
        cold = reference.get("cold")
        if (
            not isinstance(cold, dict)
            or not isinstance(cold.get("token_ids"), list)
            or not isinstance(cold.get("text"), str)
        ):
            raise ValueError(f"invalid no-cache reference in {reference_path}")

    summaries: list[dict[str, Any]] = []
    for arm, label in ARMS:
        runs = ("run1",) if arm == "A" else ("run1", "run2")
        comparisons: list[bool | None] = []
        fresh_comparison: bool | None = None
        verifier_passes: list[bool] = []
        for run in runs:
            result = load_json(results_dir / f"correctness_arm_{arm}_{run}.json")
            if not result:
                comparisons.append(None)
                if arm != "A" and run == "run2":
                    fresh_comparison = None
                continue
            verifier_passes.append(result.get("passed") is True)
            for phase in ("cold", "warm"):
                comparison = (
                    output_comparison(result.get(phase), reference)
                    if reference
                    else None
                )
                comparisons.append(comparison)
                if arm != "A" and run == "run2" and phase == "cold":
                    fresh_comparison = comparison

        if not reference:
            status = "unreported"
        elif any(value is False for value in comparisons) or any(
            not passed for passed in verifier_passes
        ):
            status = "fail"
        elif (
            not comparisons
            or any(value is None for value in comparisons)
            or len(verifier_passes) != len(runs)
        ):
            status = "unreported"
        elif all(comparisons) and all(verifier_passes):
            status = "pass"
        else:
            status = "fail"
        summaries.append(
            {
                "arm": arm,
                "label": label,
                "status": status,
                "fresh_warm_matches_reference": fresh_comparison,
            }
        )
    return summaries


def summarize(results_dir: Path) -> dict[str, Any]:
    return {
        "runs": [
            summarize_run(results_dir, arm, label, phase)
            for arm, label in ARMS
            for phase in PHASES
        ],
        "correctness": summarize_correctness(results_dir),
    }


def format_number(value: float | None, digits: int = 3) -> str:
    return "unreported" if value is None else f"{value:.{digits}f}"


def print_markdown(summary: dict[str, Any]) -> None:
    print(
        "| Arm | Phase | Total s | Tok/s | TTFT mean | TTFT p50 | TTFT p95 | First warm TTFT | Later warm p50 | External hit evidence |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for run in summary["runs"]:
        print(
            f"| {run['arm']} ({run['label']}) | {run['phase']} | "
            f"{format_number(run['total_time_s'])} | "
            f"{format_number(run['throughput_tokens_per_s'])} | "
            f"{format_number(run['ttft_mean_s'])} | "
            f"{format_number(run['ttft_p50_s'])} | "
            f"{format_number(run['ttft_p95_s'])} | "
            f"{format_number(run['first_warm_ttft_s'])} | "
            f"{format_number(run['later_warm_ttft_p50_s'])} | "
            f"{run['external_hit_evidence']} |"
        )
    print("\n| Arm | Correctness | Fresh-process warm output vs no-cache reference |")
    print("|---|---|---|")
    for result in summary["correctness"]:
        fresh = result["fresh_warm_matches_reference"]
        fresh_text = (
            "reference"
            if result["arm"] == "A"
            else "unreported"
            if fresh is None
            else "match"
            if fresh
            else "mismatch"
        )
        print(
            f"| {result['arm']} ({result['label']}) | {result['status']} | {fresh_text} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of Markdown"
    )
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.results_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"summarize_results: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_markdown(summary)
    return 1 if any(item["status"] == "fail" for item in summary["correctness"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
