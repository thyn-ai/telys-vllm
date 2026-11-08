# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the RunPod worker benchmark driver and summary."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_DIR = REPO_ROOT / "integrations" / "runpod-worker-vllm"
BENCHMARK_SCRIPT = INTEGRATION_DIR / "benchmark.sh"
SUMMARY_SCRIPT = INTEGRATION_DIR / "summarize_results.py"
VERIFIER_SCRIPT = REPO_ROOT / "bench" / "verify_warm_hit_output.py"

spec = importlib.util.spec_from_file_location("runpod_worker_summary", SUMMARY_SCRIPT)
assert spec is not None and spec.loader is not None
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


def load_verifier(monkeypatch: pytest.MonkeyPatch):
    benchmark_module = ModuleType("benchmark_prefix_caching_metrics")
    benchmark_module.sample_requests_from_random = lambda **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "benchmark_prefix_caching_metrics", benchmark_module
    )

    class EngineArgs:
        @classmethod
        def add_cli_args(cls, parser):
            parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--kv-transfer-config", default=None)
            return parser

        @classmethod
        def from_cli_args(cls, args):
            raise AssertionError("engine construction is outside this parser test")

    vllm_module = ModuleType("vllm")
    vllm_module.LLM = object
    vllm_module.SamplingParams = object
    engine_module = ModuleType("vllm.engine")
    engine_module.__path__ = []
    engine_args_module = ModuleType("vllm.engine.arg_utils")
    engine_args_module.EngineArgs = EngineArgs
    engine_module.arg_utils = engine_args_module
    vllm_module.engine = engine_module
    utils_module = ModuleType("vllm.utils")
    utils_module.__path__ = []
    argparse_module = ModuleType("vllm.utils.argparse_utils")
    argparse_module.FlexibleArgumentParser = argparse.ArgumentParser
    utils_module.argparse_utils = argparse_module
    vllm_module.utils = utils_module
    tokenizer_module = ModuleType("vllm.tokenizers")
    tokenizer_module.get_tokenizer = lambda **_kwargs: None

    modules = {
        "vllm": vllm_module,
        "vllm.engine": engine_module,
        "vllm.engine.arg_utils": engine_args_module,
        "vllm.utils": utils_module,
        "vllm.utils.argparse_utils": argparse_module,
        "vllm.tokenizers": tokenizer_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    verifier_spec = importlib.util.spec_from_file_location(
        "runpod_verify_warm_hit_output", VERIFIER_SCRIPT
    )
    assert verifier_spec is not None and verifier_spec.loader is not None
    verifier_module = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier_module)
    return verifier_module


def run_shell(
    arguments: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "--noprofile", "--norc", *arguments],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", **env},
        check=False,
    )


def dry_run(tmp_path: Path, prefix: str = "4096") -> subprocess.CompletedProcess[str]:
    return run_shell(
        [str(BENCHMARK_SCRIPT), "--dry-run", prefix, "unit-test"],
        env={"RUNPOD_BENCHMARK_ROOT": str(tmp_path / "fresh-run")},
    )


def command_lines(result: subprocess.CompletedProcess[str], phase: str) -> list[str]:
    return [line for line in result.stdout.splitlines() if f"phase={phase} " in line]


def test_shell_syntax_is_valid():
    result = run_shell(["-n", str(BENCHMARK_SCRIPT)], env={})
    assert result.returncode == 0, result.stderr
    compile(SUMMARY_SCRIPT.read_text(), str(SUMMARY_SCRIPT), "exec")


def test_verifier_disable_flag_overrides_configs_without_changing_defaults(
    tmp_path, monkeypatch
):
    verifier = load_verifier(monkeypatch)
    slab_path = str(tmp_path / "kv.slab")
    custom_config = '{"kv_connector":"CustomConnector"}'

    disabled = verifier.parse_args(slab_path, ["--disable-kv-connector"])
    assert disabled.kv_transfer_config is None
    assert disabled.disable_kv_connector is True

    overridden = verifier.parse_args(
        slab_path,
        ["--disable-kv-connector", "--kv-transfer-config", custom_config],
    )
    assert overridden.kv_transfer_config is None

    default = verifier.parse_args(slab_path, [])
    default_config = json.loads(default.kv_transfer_config)
    assert default_config["kv_connector"] == "TelysKVConnector"
    assert default_config["kv_connector_extra_config"]["telys_slab_path"] == slab_path

    explicit = verifier.parse_args(slab_path, ["--kv-transfer-config", custom_config])
    assert explicit.kv_transfer_config == custom_config
    assert explicit.disable_kv_connector is False


def test_dry_run_generates_all_commands_without_creating_root(tmp_path):
    result = dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert not (tmp_path / "fresh-run").exists()

    assert result.stdout.count("benchmark_prefix_caching_metrics.py") == 6
    assert result.stdout.count("verify_warm_hit_output.py") == 5
    assert result.stdout.count("--kv-transfer-config") == 8
    assert result.stdout.count("--disable-kv-connector") == 1
    assert result.stdout.count("--no-enable-prefix-caching") == 11
    assert result.stdout.count("--repeat-count 1") == 6
    assert "--enforce-eager" not in result.stdout

    cold = command_lines(result, "cold")
    warm = command_lines(result, "warm")
    assert len(cold) == len(warm) == 3
    for line in cold:
        assert "--num-prompts 1" in line
        assert "--input-length-range 4096:4096" in line
    for line in warm:
        assert "--num-prompts 20" in line
        assert "--input-length-range 4096:8192" in line

    for line in cold + warm:
        if line.startswith("arm=A "):
            assert "--kv-transfer-config" not in line
        else:
            assert "--kv-transfer-config" in line
            assert "--kv-transfer-config null" not in line

    arm_a_correctness = next(
        line for line in command_lines(result, "run1") if line.startswith("arm=A ")
    )
    assert "--disable-kv-connector" in arm_a_correctness
    assert "--kv-transfer-config" not in arm_a_correctness
    assert "--kv-transfer-config null" not in result.stdout
    assert "'{}'" not in result.stdout
    assert r"\{\}" not in result.stdout

    for arm, connector in (("B", "OffloadingConnector"), ("C", "TelysKVConnector")):
        for phase in ("cold", "warm", "run1", "run2"):
            line = next(
                line
                for line in command_lines(result, phase)
                if line.startswith(f"arm={arm} ")
            )
            assert connector in line
            assert "--kv-transfer-config null" not in line
            assert "--disable-kv-connector" not in line


def test_batch_invariant_is_correctness_only_even_when_inherited(tmp_path):
    result = run_shell(
        [str(BENCHMARK_SCRIPT), "--dry-run", "4096", "unit-test"],
        env={
            "RUNPOD_BENCHMARK_ROOT": str(tmp_path / "fresh-run"),
            "VLLM_BATCH_INVARIANT": "1",
        },
    )
    assert result.returncode == 0, result.stderr

    for phase in ("cold", "warm"):
        lines = command_lines(result, phase)
        assert len(lines) == 3
        for line in lines:
            assert "vllm_batch_invariant=0" in line
            assert "vllm_batch_invariant=1" not in line

    for phase, expected_count in (("run1", 3), ("run2", 2)):
        lines = command_lines(result, phase)
        assert len(lines) == expected_count
        for line in lines:
            assert "vllm_batch_invariant=1" in line

    for arm, connector in (("B", "OffloadingConnector"), ("C", "TelysKVConnector")):
        for phase in ("cold", "warm", "run1", "run2"):
            line = next(
                line
                for line in command_lines(result, phase)
                if line.startswith(f"arm={arm} ")
            )
            assert connector in line
            assert "--kv-transfer-config null" not in line
            assert "--disable-kv-connector" not in line


def test_cache_roots_are_fresh_and_isolated(tmp_path):
    result = dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    root = tmp_path / "fresh-run"

    for arm in ("B", "C"):
        cold = next(
            line
            for line in command_lines(result, "cold")
            if line.startswith(f"arm={arm} ")
        )
        warm = next(
            line
            for line in command_lines(result, "warm")
            if line.startswith(f"arm={arm} ")
        )
        assert f"performance/arm-{arm}" in cold
        assert f"performance/arm-{arm}" in warm

    assert f"{root}/performance/arm-B/vllm-filesystem-kv" in result.stdout
    assert f"{root}/performance/arm-C/telys-kv" in result.stdout
    assert f"{root}/performance/arm-C/telys-kv_blobs" in result.stdout
    assert f"{root}/performance/arm-C/telys-kv_kv.slab" in result.stdout
    assert f"{root}/correctness/arm-C/telys-kv" in result.stdout
    assert f"{root}/correctness/arm-B/vllm-filesystem-kv" in result.stdout
    assert "rm -rf" not in BENCHMARK_SCRIPT.read_text()


def test_8192_extension_changes_only_the_workload_shape(tmp_path):
    result = dry_run(tmp_path, prefix="8192")
    assert result.returncode == 0, result.stderr
    for line in command_lines(result, "cold"):
        assert "--input-length-range 8192:8192" in line
    for line in command_lines(result, "warm"):
        assert "--input-length-range 8192:12288" in line
        assert "--max-model-len 16896" in line


def test_existing_root_is_refused_without_modification(tmp_path):
    root = tmp_path / "fresh-run"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("preexisting cache")
    result = dry_run(tmp_path)
    assert result.returncode == 64
    assert "refusing to reuse run root" in result.stderr
    assert sentinel.read_text() == "preexisting cache"
    assert list(root.iterdir()) == [sentinel]


def test_relative_root_is_rejected(tmp_path):
    result = run_shell(
        [str(BENCHMARK_SCRIPT), "--dry-run"],
        env={"RUNPOD_BENCHMARK_ROOT": "relative/cache"},
    )
    assert result.returncode == 64
    assert "absolute path" in result.stderr


def write_performance(
    results_dir: Path, arm: str, phase: str, *, log: str | None = None
):
    result = {
        "total_time_s": 12.5,
        "num_prompts": 20,
        "total_output_tokens": 1280,
        "throughput_tokens_per_s": 102.4,
        "ttft_s": {"mean": 4.0, "p50": 4.0, "p95": 8.0, "min": 1.0, "max": 8.0},
        "per_request_ttft_s": [1.0, 2.0, 4.0, 8.0],
    }
    (results_dir / f"arm_{arm}_{phase}.json").write_text(json.dumps(result))
    if log is not None:
        (results_dir / f"arm_{arm}_{phase}.log").write_text(log)


def test_summary_reports_timing_first_later_and_external_hit_rate(tmp_path):
    write_performance(
        tmp_path,
        "B",
        "warm",
        log="first External prefix cache hit rate: 0.0%\nlast External prefix cache hit rate: 66.7%\n",
    )
    summary = summary_module.summarize(tmp_path)
    run = next(
        item
        for item in summary["runs"]
        if item["arm"] == "B" and item["phase"] == "warm"
    )
    assert run["total_time_s"] == 12.5
    assert run["throughput_tokens_per_s"] == 102.4
    assert run["ttft_mean_s"] == 4.0
    assert run["ttft_p50_s"] == 4.0
    assert run["ttft_p95_s"] == 8.0
    assert run["first_warm_ttft_s"] == 1.0
    assert run["later_warm_ttft_p50_s"] == 4.0
    assert run["external_hit_rate_pct"] == 66.7
    assert "66.7%" in run["external_hit_evidence"]


def test_summary_distinguishes_zero_missing_and_telys_lookup_evidence(tmp_path):
    write_performance(
        tmp_path, "B", "cold", log="External prefix cache hit rate: 0.0%\n"
    )
    write_performance(
        tmp_path,
        "C",
        "warm",
        log="INFO:telys_vllm.connector:telys lookup: hit_blocks=255 hit_tokens=4080\n",
    )
    summary = summary_module.summarize(tmp_path)
    cold_b = next(
        item
        for item in summary["runs"]
        if item["arm"] == "B" and item["phase"] == "cold"
    )
    warm_c = next(
        item
        for item in summary["runs"]
        if item["arm"] == "C" and item["phase"] == "warm"
    )
    warm_a = next(
        item
        for item in summary["runs"]
        if item["arm"] == "A" and item["phase"] == "warm"
    )

    assert cold_b["external_hit_rate_pct"] == 0.0
    assert warm_c["external_hit_rate_pct"] is None
    assert warm_c["telys_last_lookup"] == {"hit_blocks": 255, "hit_tokens": 4080}
    assert "external hit rate=unreported" in warm_c["external_hit_evidence"]
    assert "hit_tokens=4080" in warm_c["external_hit_evidence"]
    assert warm_a["result_present"] is False
    assert warm_a["total_time_s"] is None
    assert warm_a["external_hit_rate_pct"] is None
    assert warm_a["first_warm_ttft_s"] is None


def correctness_result(token_ids: list[int], *, passed: bool = True) -> dict:
    output = {"token_ids": token_ids, "text": "reference text", "ttft_s": 1.0}
    return {"cold": dict(output), "warm": dict(output), "passed": passed}


def write_correctness(
    results_dir: Path, arm: str, run: str, token_ids: list[int], *, passed: bool = True
):
    (results_dir / f"correctness_arm_{arm}_{run}.json").write_text(
        json.dumps(correctness_result(token_ids, passed=passed))
    )


def populate_correctness(results_dir: Path, reference: list[int]):
    write_correctness(results_dir, "A", "run1", reference)
    for arm in ("B", "C"):
        for run in ("run1", "run2"):
            write_correctness(results_dir, arm, run, reference)


def test_correctness_passes_only_when_all_arms_match_reference(tmp_path):
    reference = list(range(64))
    populate_correctness(tmp_path, reference)
    summary = summary_module.summarize(tmp_path)
    assert [item["status"] for item in summary["correctness"]] == [
        "pass",
        "pass",
        "pass",
    ]
    assert summary["correctness"][1]["fresh_warm_matches_reference"] is True
    assert summary["correctness"][2]["fresh_warm_matches_reference"] is True
    assert summary_module.main([str(tmp_path)]) == 0


def test_fresh_warm_correctness_mismatch_fails_closed(tmp_path, capsys):
    reference = list(range(64))
    populate_correctness(tmp_path, reference)
    mismatched = reference.copy()
    mismatched[0] = -1
    write_correctness(tmp_path, "C", "run2", mismatched, passed=False)
    summary = summary_module.summarize(tmp_path)
    assert summary["correctness"][2]["status"] == "fail"
    assert summary["correctness"][2]["fresh_warm_matches_reference"] is False
    assert summary_module.main([str(tmp_path)]) == 1
    assert "mismatch" in capsys.readouterr().out


def test_missing_correctness_and_metrics_are_unreported_not_zero(tmp_path, capsys):
    summary = summary_module.summarize(tmp_path)
    assert all(item["status"] == "unreported" for item in summary["correctness"])
    assert all(item["total_time_s"] is None for item in summary["runs"])
    assert summary_module.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "unreported" in output
    assert "0.000" not in output


def test_json_summary_is_deterministic_and_uses_null_for_missing(tmp_path, capsys):
    assert summary_module.main([str(tmp_path), "--json"]) == 0
    first = capsys.readouterr().out
    assert summary_module.main([str(tmp_path), "--json"]) == 0
    second = capsys.readouterr().out
    assert first == second
    decoded = json.loads(first)
    assert decoded["runs"][0]["total_time_s"] is None
    assert decoded["correctness"][0]["status"] == "unreported"


@pytest.mark.parametrize("prefix", ["1024", "not-a-number"])
def test_invalid_prefix_fails_closed(tmp_path, prefix):
    result = dry_run(tmp_path, prefix=prefix)
    assert result.returncode == 64
    assert "prefix must be 4096 or 8192" in result.stderr
    assert not (tmp_path / "fresh-run").exists()
