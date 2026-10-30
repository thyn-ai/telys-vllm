# SPDX-License-Identifier: Apache-2.0
"""Summarize a Telys vLLM telemetry JSONL file into a per-phase table.

Usage:
    python3 bench/summarize_telemetry.py <telemetry.jsonl>

Output (markdown):
    - one row per phase with count, median elapsed_us, total elapsed_us,
      mean elapsed_us, p95 elapsed_us
    - top-3 contributors by total time
    - exact file:line of the dominant phase's most recent caller
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict[str, dict]:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_phase[rec.get("phase", "unknown")].append(rec)

    summary: dict[str, dict] = {}
    for phase, items in by_phase.items():
        values = [float(r["elapsed_us"]) for r in items]
        total = sum(values)
        summary[phase] = {
            "count": len(values),
            "total_us": total,
            "median_us": median(values),
            "mean_us": total / len(values),
            "p95_us": sorted(values)[int(len(values) * 0.95)],
            "max_us": max(values),
            "min_us": min(values),
            "dominant_caller": _dominant_caller(items),
        }
    return summary


def _dominant_caller(items: list[dict]) -> str:
    """Return the most common caller_file:caller_line among records."""
    callers: dict[str, int] = defaultdict(int)
    for r in items:
        f = r.get("caller_file") or ""
        line = r.get("caller_line")
        if f and line is not None:
            callers[f"{f}:{line}"] += 1
    if not callers:
        return "n/a"
    return max(callers, key=callers.get)


def format_table(summary: dict[str, dict]) -> str:
    rows = []
    rows.append(
        "| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |"
    )
    rows.append(
        "|-------|-------|------------|-------------|-----------|----------|--------------------|"
    )

    # Sort by total time descending.
    for phase, s in sorted(
        summary.items(), key=lambda kv: kv[1]["total_us"], reverse=True
    ):
        rows.append(
            f"| {phase} | {s['count']} | "
            f"{s['total_us'] / 1000:.3f} | "
            f"{s['median_us'] / 1000:.3f} | "
            f"{s['mean_us'] / 1000:.3f} | "
            f"{s['p95_us'] / 1000:.3f} | "
            f"`{s['dominant_caller']}` |"
        )
    return "\n".join(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <telemetry.jsonl>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    records = load_records(path)
    if not records:
        print("no telemetry records")
        return 0

    summary = summarize(records)
    top3 = sorted(summary.items(), key=lambda kv: kv[1]["total_us"], reverse=True)[:3]

    print(f"## Telemetry summary ({len(records)} records, {len(summary)} phases)")
    print()
    print(format_table(summary))
    print()
    print("### Top 3 contributors by total time")
    for phase, s in top3:
        pct = s["total_us"] / sum(v["total_us"] for v in summary.values()) * 100
        print(f"- **{phase}**: {s['total_us'] / 1000:.3f} ms ({pct:.1f}%)")

    dominant_phase, dominant = top3[0]
    print()
    print("### Dominant phase")
    print(f"- phase: `{dominant_phase}`")
    print(f"- total: {dominant['total_us'] / 1000:.3f} ms")
    print(f"- median: {dominant['median_us'] / 1000:.3f} ms")
    print(f"- dominant caller: `{dominant['dominant_caller']}`")

    return 0


if __name__ == "__main__":
    sys.exit(main())
