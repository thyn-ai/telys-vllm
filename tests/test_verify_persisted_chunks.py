# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lifecycle manifests and persisted-chunk verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from telys_vllm.config import TelysConnectorConfig
from telys_vllm.gpu_ops import serialize_kv_tensor
from telys_vllm.store import TelysBlockStore
from tests.test_store import FakeBlobStoreClient

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIER_PATH = REPO_ROOT / "bench" / "verify_persisted_chunks.py"
METRICS_PATH = REPO_ROOT / "bench" / "benchmark_prefix_caching_metrics.py"

spec = importlib.util.spec_from_file_location("persisted_chunk_verifier", VERIFIER_PATH)
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verifier
spec.loader.exec_module(verifier)


def _load_metrics(monkeypatch: pytest.MonkeyPatch):
    transformers = ModuleType("transformers")
    transformers.PreTrainedTokenizerBase = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    vllm = ModuleType("vllm")
    vllm.LLM = object
    vllm.SamplingParams = object
    engine = ModuleType("vllm.engine")
    engine.__path__ = []
    engine_args = ModuleType("vllm.engine.arg_utils")
    engine_args.EngineArgs = object
    engine.arg_utils = engine_args
    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    argparse_utils = ModuleType("vllm.utils.argparse_utils")
    argparse_utils.FlexibleArgumentParser = argparse.ArgumentParser
    utils.argparse_utils = argparse_utils
    tokenizers = ModuleType("vllm.tokenizers")
    tokenizers.get_tokenizer = lambda **_kwargs: None
    modules = {
        "vllm": vllm,
        "vllm.engine": engine,
        "vllm.engine.arg_utils": engine_args,
        "vllm.utils": utils,
        "vllm.utils.argparse_utils": argparse_utils,
        "vllm.tokenizers": tokenizers,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    metrics_spec = importlib.util.spec_from_file_location(
        "lifecycle_benchmark_metrics", METRICS_PATH
    )
    assert metrics_spec is not None and metrics_spec.loader is not None
    metrics = importlib.util.module_from_spec(metrics_spec)
    monkeypatch.setitem(sys.modules, metrics_spec.name, metrics)
    metrics_spec.loader.exec_module(metrics)
    return metrics


def _context(*, block_size: int = 4, dtype: str = "torch.bfloat16") -> dict:
    return {
        "schema_version": 1,
        "kind": "telys_vllm_request_manifest",
        "model": "model",
        "run": {"arm": "C", "phase": "cold"},
        "token_digest": "sha256-json-array-utf8-v1",
        "block_hash": {
            "algorithm": "sha256",
            "none_hash_seed": "seed",
            "block_size": block_size,
            "hash_block_size": block_size,
            "extra_keys_per_block": None,
        },
        "kv": {
            "dtype": dtype,
            "expected_block_shape": [block_size, 2, 3],
            "expected_layer_count": 1,
            "shape_kind": "full_attention",
        },
        "storage": {"namespace": "test-ns", "max_chunk_blocks": 0},
    }


def _write_manifest(
    path: Path,
    *,
    prompt_ids: list[int],
    generated_ids: list[int],
    context: dict | None = None,
) -> None:
    context = context or _context()
    block_size = context["block_hash"]["block_size"]
    all_ids = prompt_ids + generated_ids
    persisted_count = max(0, (len(all_ids) - 1) // block_size)
    prefix_count = min(len(prompt_ids) // block_size, persisted_count)
    manifest = {
        **context,
        "response": {
            "request_id": "request-0",
            "request_index": 0,
            "completion_index": 0,
            "finish_reason": "length",
            "stop_reason": None,
            "text_sha256": hashlib.sha256(b"response").hexdigest(),
        },
        "prompt": {
            "token_ids": prompt_ids,
            "count": len(prompt_ids),
            "sha256": verifier._token_ids_digest(prompt_ids),
        },
        "generated": {
            "token_ids": generated_ids,
            "count": len(generated_ids),
            "sha256": verifier._token_ids_digest(generated_ids),
        },
        "expected": {
            "all_token_count": len(all_ids),
            "all_token_sha256": verifier._token_ids_digest(all_ids),
            "full_block_count": len(all_ids) // block_size,
            "persisted_block_count": persisted_count,
            "prefix_block_count": prefix_count,
            "suffix_block_count": persisted_count - prefix_count,
            "suffix_start_block": prefix_count,
            "suffix_start_token": prefix_count * block_size,
            "last_token_excluded": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _fake_hash(value) -> bytes:
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).digest()


def _fake_hash_block_tokens(hash_function, parent, token_ids, extra_keys) -> bytes:
    return hash_function((parent, tuple(token_ids), extra_keys))


def _backend() -> verifier.HashBackend:
    return verifier.HashBackend("fake-vllm", _fake_hash, _fake_hash_block_tokens)


def _expected_hashes(prompt_ids: list[int], generated_ids: list[int]) -> list[bytes]:
    all_ids = prompt_ids + generated_ids
    persisted_count = max(0, (len(all_ids) - 1) // 4)
    parent = _fake_hash("seed")
    hashes = []
    for index in range(persisted_count):
        parent = _fake_hash_block_tokens(
            _fake_hash, parent, all_ids[index * 4 : (index + 1) * 4], None
        )
        hashes.append(parent)
    return hashes


def _layer_tensors(block_count: int) -> dict[str, torch.Tensor]:
    values = torch.arange(block_count * 4 * 2 * 3, dtype=torch.bfloat16).reshape(
        block_count, 4, 2, 3
    )
    return {"layer.0": values}


def _save_run(store: TelysBlockStore, hashes: list[bytes], chunk_format: str) -> None:
    if chunk_format == "fast":
        store.save_chunk_fast(hashes[0], hashes, _layer_tensors(len(hashes)))
        return
    blobs = [
        serialize_kv_tensor(tensor)
        for tensor in _layer_tensors(len(hashes))["layer.0"].unbind()
    ]
    store.save_chunk(hashes[0], hashes, {"layer.0": blobs})


@pytest.fixture(params=["fast", "legacy"])
def persisted_case(request, tmp_path: Path):
    prompt_ids = list(range(8))
    generated_ids = list(range(100, 106))
    manifest_dir = tmp_path / "manifests"
    manifest_path = manifest_dir / "request.json"
    _write_manifest(manifest_path, prompt_ids=prompt_ids, generated_ids=generated_ids)
    hashes = _expected_hashes(prompt_ids, generated_ids)
    client = FakeBlobStoreClient()
    config = TelysConnectorConfig(
        namespace="test-ns",
        slab_path=None,
        blobstore_no_slab=True,
        write_behind=False,
    )
    store = TelysBlockStore(
        client,
        config,
        model_id="model",
        engine_id="engine",
        block_size=4,
    )
    _save_run(store, hashes[:2], request.param)
    _save_run(store, hashes[2:], request.param)
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "phase": "wait_for_save",
                "pid": 123,
                "context": {"pending_chunks": 2, "saved_chunks": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def record(exit_status: int = 0) -> Path:
        output = tmp_path / "evidence.json"
        verifier.record_evidence(
            argparse.Namespace(
                output=str(output),
                telemetry=str(telemetry),
                manifest_dir=str(manifest_dir),
                arm="C",
                phase="cold",
                exit_status=exit_status,
            )
        )
        return output

    yield SimpleNamespace(
        client=client,
        store=store,
        hashes=hashes,
        telemetry=telemetry,
        record=record,
        manifest_path=manifest_path,
        chunk_format=request.param,
    )
    store.close()


def _verify(case, evidence_path: Path) -> dict:
    return verifier.verify_evidence(
        evidence_path,
        slab_path="",
        client=case.client,
        hash_backend=_backend(),
    )


def test_manifest_writer_records_actual_response_tokens_and_identity(
    tmp_path, monkeypatch, capsys
):
    metrics = _load_metrics(monkeypatch)
    completion = SimpleNamespace(
        index=0,
        token_ids=[20, 21, 22, 23, 24, 25],
        text="actual response",
        finish_reason="length",
        stop_reason=None,
    )
    output = SimpleNamespace(
        request_id="actual-request-id",
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        outputs=[completion],
        metrics=SimpleNamespace(first_token_latency=0.25),
    )
    llm = SimpleNamespace(generate=lambda *_args, **_kwargs: [output])
    manifest_dir = tmp_path / "manifests"
    metrics.run_benchmark(
        llm,
        ["decoded prompt is not the token source"],
        object(),
        None,
        manifest_dir=str(manifest_dir),
        manifest_context=_context(),
    )
    capsys.readouterr()
    files = list(manifest_dir.glob("*.json"))
    assert len(files) == 1
    manifest = json.loads(files[0].read_text())
    assert manifest["prompt"]["token_ids"] == output.prompt_token_ids
    assert manifest["generated"]["token_ids"] == completion.token_ids
    assert manifest["prompt"]["count"] == 8
    assert manifest["generated"]["count"] == 6
    assert manifest["response"]["request_id"] == "actual-request-id"
    assert manifest["expected"]["persisted_block_count"] == 3
    assert manifest["expected"]["prefix_block_count"] == 2
    assert manifest["expected"]["suffix_block_count"] == 1
    assert "environment" not in files[0].read_text().lower()
    assert "headers" not in files[0].read_text().lower()


def test_verifier_recomputes_dynamic_prefix_and_suffix_coverage(persisted_case):
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "pass"
    assert result["passed"] is True
    assert result["summary"]["expected_blocks"] == 3
    assert result["summary"]["expected_prefix_blocks"] == 2
    assert result["summary"]["expected_suffix_blocks"] == 1
    assert result["summary"]["prefix_chunks"] == 1
    assert result["summary"]["suffix_chunks"] == 1
    assert result["requests"][0]["verified"]["blocks"] == 3
    assert all(
        len(chunk["payload_sha256"]) == 64 for chunk in result["requests"][0]["chunks"]
    )


def test_complete_chunks_without_pending_write_observation_are_inconclusive(
    persisted_case,
):
    persisted_case.telemetry.write_text(
        json.dumps({"phase": "wait_for_save", "context": {"pending_chunks": 0}}) + "\n",
        encoding="utf-8",
    )
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "inconclusive"
    assert result["passed"] is False
    assert result["summary"]["missing_blocks"] == 0
    assert result["inconclusive_reasons"][0]["code"] == "pending_writes_not_observed"


def test_abrupt_exit_evidence_is_supported_but_missing_suffix_fails(persisted_case):
    suffix_key = persisted_case.store.chunk_key(persisted_case.hashes[2], 1)
    persisted_case.client.delete(suffix_key)
    result = _verify(persisted_case, persisted_case.record(exit_status=137))
    assert result["lifecycle"]["exit_kind"] == "abrupt-exit"
    assert result["status"] == "fail"
    assert result["requests"][0]["missing_block_indices"] == [2]
    assert result["requests"][0]["errors"][0]["code"] == "missing_expected_chunks"


def test_abrupt_exit_with_complete_chunks_is_inconclusive(persisted_case):
    result = _verify(persisted_case, persisted_case.record(exit_status=137))
    assert result["lifecycle"]["exit_kind"] == "abrupt-exit"
    assert result["status"] == "inconclusive"
    assert result["passed"] is False
    assert result["summary"]["missing_blocks"] == 0
    assert any(
        reason["code"] == "quiescence_unconfirmed"
        for reason in result["inconclusive_reasons"]
    )


def test_corrupt_payload_fails_decoding(persisted_case):
    prefix_key = persisted_case.store.chunk_key(persisted_case.hashes[0], 2)
    payload = bytearray(persisted_case.client._values[prefix_key])
    payload[0] ^= 0xFF
    persisted_case.client._values[prefix_key] = bytes(payload)
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    assert any(
        error["code"] == "invalid_chunk" for error in result["requests"][0]["errors"]
    )


def test_empty_inline_payload_fails_in_chunk_decoder(persisted_case):
    prefix_key = persisted_case.store.chunk_key(persisted_case.hashes[0], 2)
    persisted_case.client._values[prefix_key] = b""
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    request = result["requests"][0]
    chunk = request["chunks"][0]
    assert chunk["payload_bytes"] == 0
    assert chunk["payload_sha256"] == hashlib.sha256(b"").hexdigest()
    expected_message = (
        "truncated fast chunk header"
        if persisted_case.chunk_format == "fast"
        else "truncated chunk header"
    )
    assert request["errors"][0] == {
        "code": "invalid_chunk",
        "message": expected_message,
        "details": {
            "key": prefix_key,
            "exception_type": "PayloadCodecError",
        },
    }


def test_false_chunk_length_is_invalid(persisted_case, monkeypatch):
    monkeypatch.setattr(persisted_case.client, "chunk_length", lambda _key: False)
    get_count = persisted_case.client.get_count
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    request = result["requests"][0]
    start_key = f"test-ns/chunk/{persisted_case.hashes[0].hex()}"
    assert request["errors"][0] == {
        "code": "invalid_chunk_length",
        "message": "client returned an invalid chunk length",
        "details": {"key": start_key},
    }
    assert request["missing_block_indices"] == [0, 1, 2]
    assert persisted_case.client.get_count == get_count


def test_wrong_expected_dtype_fails(persisted_case):
    context = _context(dtype="torch.float16")
    _write_manifest(
        persisted_case.manifest_path,
        prompt_ids=list(range(8)),
        generated_ids=list(range(100, 106)),
        context=context,
    )
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    assert any("dtype" in error["message"] for error in result["requests"][0]["errors"])


def test_wrong_expected_shape_fails(persisted_case):
    context = _context()
    context["kv"]["expected_block_shape"] = [4, 2, 4]
    _write_manifest(
        persisted_case.manifest_path,
        prompt_ids=list(range(8)),
        generated_ids=list(range(100, 106)),
        context=context,
    )
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    assert any("shape" in error["message"] for error in result["requests"][0]["errors"])


def test_unresolved_model_shape_is_inconclusive_not_a_pass(persisted_case):
    context = _context()
    context["kv"]["expected_block_shape"] = None
    context["kv"]["shape_kind"] = "unresolved"
    _write_manifest(
        persisted_case.manifest_path,
        prompt_ids=list(range(8)),
        generated_ids=list(range(100, 106)),
        context=context,
    )
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "inconclusive"
    assert result["passed"] is False
    assert result["requests"][0]["inconclusive_reasons"][0]["code"] == (
        "unresolved_expected_shape"
    )


def test_changed_generated_tokens_require_a_new_suffix_chunk(persisted_case):
    _write_manifest(
        persisted_case.manifest_path,
        prompt_ids=list(range(8)),
        generated_ids=[100, 101, 102, 999, 104, 105],
    )
    result = _verify(persisted_case, persisted_case.record())
    assert result["status"] == "fail"
    assert result["requests"][0]["verified"]["blocks"] == 2
    assert result["requests"][0]["missing_block_indices"] == [2]


def test_manifest_mutation_after_recording_fails_evidence_integrity(persisted_case):
    evidence_path = persisted_case.record()
    manifest = json.loads(persisted_case.manifest_path.read_text())
    manifest["generated"]["token_ids"][0] = 999
    persisted_case.manifest_path.write_text(json.dumps(manifest))
    result = _verify(persisted_case, evidence_path)
    assert result["status"] == "fail"
    assert any(error["code"] == "invalid_manifest" for error in result["errors"])


def test_verify_cli_emits_structured_json_and_inconclusive_exit(
    tmp_path, monkeypatch, capsys
):
    result = {
        "schema_version": 1,
        "kind": "telys_vllm_persisted_chunk_verification",
        "status": "inconclusive",
        "passed": False,
    }
    monkeypatch.setattr(verifier, "verify_evidence", lambda *_args, **_kwargs: result)
    output_path = tmp_path / "verification.json"
    exit_code = verifier.main(
        [
            "verify",
            "--evidence",
            "unused.json",
            "--output-json",
            str(output_path),
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 2
    assert json.loads(stdout)["status"] == "inconclusive"
    assert json.loads(output_path.read_text()) == result
