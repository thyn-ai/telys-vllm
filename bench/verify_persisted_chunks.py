#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record and independently verify Telys persisted-chunk lifecycle evidence.

The 16384-token benchmark profile writes one request manifest per response and
calls ``record`` after the harness exits. Verification requires a quiesced Telys
store, the deployed vLLM 0.32.0 installation, and the Telys SDK. Connection
settings are read by the existing client factory; secrets are never copied into
manifests, evidence, or verification output.

Example after a lifecycle benchmark run::

    TELYS_PATH=/runpod-volume/telys-vllm-benchmark/p16384-seed42-run1/performance/arm-C/telys-kv \
    TELYS_VLLM_SLAB_PATH=/runpod-volume/telys-vllm-benchmark/p16384-seed42-run1/performance/arm-C/telys-kv_kv.slab \
    python3 bench/verify_persisted_chunks.py verify \
      --evidence /runpod-volume/telys-vllm-benchmark/p16384-seed42-run1/results/arm_C_cold.lifecycle.json \
      --output-json /runpod-volume/telys-vllm-benchmark/p16384-seed42-run1/results/arm_C_cold.verification.json

Exit codes are 0 for pass, 1 for fail, and 2 for inconclusive. A run without
observed pending-write telemetry is always inconclusive unless another check
already failed. An abrupt harness exit is also inconclusive when every other
check passes. Verification covers block identity, expected coverage, codec
integrity, dtype, shape, and payload hashes; it does not compare KV values with
a second GPU extraction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from telys_vllm.client import TelysClient, TelysRecord  # noqa: E402, RUF100
from telys_vllm.codec import TelysChunkCodec, TelysFastChunkCodec  # noqa: E402, RUF100
from telys_vllm.config import TelysConnectorConfig  # noqa: E402, RUF100
from telys_vllm.gpu_ops import deserialize_kv_tensor  # noqa: E402, RUF100
from telys_vllm.slab import KVSlab  # noqa: E402, RUF100

SCHEMA_VERSION = 1
SUPPORTED_VLLM_VERSION = "0.32.0"
PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"
EXIT_CODES = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}


class EvidenceError(ValueError):
    """Raised when lifecycle evidence or a manifest violates its schema."""


@dataclass(frozen=True)
class HashBackend:
    version: str
    hash_function: Any
    hash_block_tokens: Any


@dataclass(frozen=True)
class RequestExpectation:
    path: Path
    manifest: Mapping[str, Any]
    request_id: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    block_size: int
    prefix_count: int
    persisted_count: int
    namespace: str
    model: str
    kv_dtype: str
    expected_shape: tuple[int, ...] | None
    expected_layer_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_ids_digest(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"expected a JSON object at {path}")
    return payload


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _exit_kind(exit_status: int) -> str:
    if exit_status == 0:
        return "completed"
    if 129 <= exit_status <= 192:
        return "abrupt-exit"
    return "failed"


def _telemetry_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path.resolve()),
        "present": path.is_file(),
        "sha256": None,
        "pending_write_spans": [],
        "persisted_chunk_spans": [],
        "malformed_lines": [],
    }
    if not path.is_file():
        return snapshot
    snapshot["sha256"] = _sha256_file(path)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TypeError("telemetry event is not an object")
        except (json.JSONDecodeError, TypeError, ValueError):
            snapshot["malformed_lines"].append(line_number)
            continue
        context = event.get("context")
        if not isinstance(context, dict):
            context = {}
        if event.get("phase") == "wait_for_save":
            pending = context.get("pending_chunks", 0)
            if (
                isinstance(pending, int)
                and not isinstance(pending, bool)
                and pending > 0
            ):
                snapshot["pending_write_spans"].append(
                    {
                        "line": line_number,
                        "pid": event.get("pid"),
                        "pending_chunks": pending,
                        "saved_chunks": context.get("saved_chunks"),
                    }
                )
        elif event.get("phase") == "store_persist":
            snapshot["persisted_chunk_spans"].append(
                {
                    "line": line_number,
                    "pid": event.get("pid"),
                    "key": context.get("key"),
                    "block_count": context.get("block_count"),
                    "payload_bytes": context.get("payload_bytes"),
                }
            )
    return snapshot


def record_evidence(args: argparse.Namespace) -> dict[str, Any]:
    manifest_dir = Path(args.manifest_dir).resolve()
    manifest_files = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for path in sorted(manifest_dir.glob("*.json"))
        if path.is_file()
    ]
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": "telys_vllm_lifecycle_evidence",
        "run": {"arm": args.arm, "phase": args.phase},
        "process": {
            "exit_status": args.exit_status,
            "exit_kind": _exit_kind(args.exit_status),
        },
        "require_pending_writes": True,
        "telemetry": _telemetry_snapshot(Path(args.telemetry).resolve()),
        "manifests": {
            "directory": str(manifest_dir),
            "files": manifest_files,
        },
    }
    output = Path(args.output).resolve()
    _write_json_atomic(output, evidence)
    return evidence


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{key} must be an object")
    return value


def _required_int(parent: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{key} must be an integer >= {minimum}")
    return value


def _validated_ids(group: Mapping[str, Any], name: str) -> tuple[int, ...]:
    raw_ids = group.get("token_ids")
    if not isinstance(raw_ids, list):
        raise EvidenceError(f"{name}.token_ids must be a list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_ids
    ):
        raise EvidenceError(f"{name}.token_ids must contain non-negative integers")
    token_ids = tuple(raw_ids)
    if _required_int(group, "count") != len(token_ids):
        raise EvidenceError(f"{name}.count does not match its token IDs")
    if group.get("sha256") != _token_ids_digest(token_ids):
        raise EvidenceError(f"{name}.sha256 does not match its token IDs")
    return token_ids


def _validated_shape(kv: Mapping[str, Any]) -> tuple[int, ...] | None:
    shape = kv.get("expected_block_shape")
    if shape is None:
        return None
    if not isinstance(shape, list) or not shape:
        raise EvidenceError("kv.expected_block_shape must be null or a non-empty list")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
    ):
        raise EvidenceError("kv.expected_block_shape dimensions must be positive")
    return tuple(shape)


def _validate_manifest(path: Path, manifest: Mapping[str, Any]) -> RequestExpectation:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("unsupported manifest schema_version")
    if manifest.get("kind") != "telys_vllm_request_manifest":
        raise EvidenceError("unexpected manifest kind")
    if manifest.get("token_digest") != "sha256-json-array-utf8-v1":
        raise EvidenceError("unsupported token digest")
    response = _required_mapping(manifest, "response")
    request_id = response.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise EvidenceError("response.request_id must be a non-empty string")
    prompt_ids = _validated_ids(_required_mapping(manifest, "prompt"), "prompt")
    generated_ids = _validated_ids(
        _required_mapping(manifest, "generated"), "generated"
    )
    block_hash = _required_mapping(manifest, "block_hash")
    algorithm = block_hash.get("algorithm")
    if algorithm not in ("sha256", "sha256_cbor"):
        raise EvidenceError("only cryptographic vLLM block hashes are supported")
    if not isinstance(block_hash.get("none_hash_seed"), str):
        raise EvidenceError("block_hash.none_hash_seed must be a string")
    block_size = _required_int(block_hash, "block_size", minimum=1)
    if _required_int(block_hash, "hash_block_size", minimum=1) != block_size:
        raise EvidenceError("hash_block_size differing from block_size is unsupported")
    if block_hash.get("extra_keys_per_block") is not None:
        raise EvidenceError("non-null block-hash extra keys are unsupported")

    all_ids = prompt_ids + generated_ids
    persisted_count = max(0, (len(all_ids) - 1) // block_size)
    prefix_count = min(len(prompt_ids) // block_size, persisted_count)
    expected = _required_mapping(manifest, "expected")
    expected_values = {
        "all_token_count": len(all_ids),
        "full_block_count": len(all_ids) // block_size,
        "persisted_block_count": persisted_count,
        "prefix_block_count": prefix_count,
        "suffix_block_count": persisted_count - prefix_count,
        "suffix_start_block": prefix_count,
        "suffix_start_token": prefix_count * block_size,
    }
    for key, actual in expected_values.items():
        if _required_int(expected, key) != actual:
            raise EvidenceError(f"expected.{key} does not match the token IDs")
    if expected.get("all_token_sha256") != _token_ids_digest(all_ids):
        raise EvidenceError("expected.all_token_sha256 does not match the token IDs")
    if expected.get("last_token_excluded") is not True:
        raise EvidenceError("expected.last_token_excluded must be true")

    kv = _required_mapping(manifest, "kv")
    kv_dtype = kv.get("dtype")
    if not isinstance(kv_dtype, str) or not kv_dtype.startswith("torch."):
        raise EvidenceError("kv.dtype must be a torch dtype string")
    storage = _required_mapping(manifest, "storage")
    namespace = storage.get("namespace")
    if not isinstance(namespace, str) or not namespace.strip():
        raise EvidenceError("storage.namespace must be a non-empty string")
    model = manifest.get("model")
    if not isinstance(model, str) or not model:
        raise EvidenceError("model must be a non-empty string")
    return RequestExpectation(
        path=path,
        manifest=manifest,
        request_id=request_id,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        block_size=block_size,
        prefix_count=prefix_count,
        persisted_count=persisted_count,
        namespace=namespace,
        model=model,
        kv_dtype=kv_dtype,
        expected_shape=_validated_shape(kv),
        expected_layer_count=_required_int(kv, "expected_layer_count", minimum=1),
    )


def _override_manifest_paths(raw_paths: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            paths.extend(sorted(item for item in path.glob("*.json") if item.is_file()))
        else:
            paths.append(path)
    return paths


def _load_expectations(
    evidence: Mapping[str, Any], overrides: Sequence[str]
) -> tuple[list[RequestExpectation], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    expected_hashes: dict[Path, str] = {}
    if overrides:
        paths = _override_manifest_paths(overrides)
    else:
        manifests = _required_mapping(evidence, "manifests")
        files = manifests.get("files")
        if not isinstance(files, list):
            raise EvidenceError("manifests.files must be a list")
        paths = []
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise EvidenceError("each manifest evidence entry must contain a path")
            path = Path(entry["path"]).resolve()
            paths.append(path)
            if not isinstance(entry.get("sha256"), str):
                raise EvidenceError("each manifest evidence entry must contain sha256")
            expected_hashes[path] = entry["sha256"]
    expectations: list[RequestExpectation] = []
    seen_request_ids: set[str] = set()
    for path in paths:
        try:
            if path in expected_hashes and _sha256_file(path) != expected_hashes[path]:
                raise EvidenceError("manifest hash differs from lifecycle evidence")
            expectation = _validate_manifest(path, _read_json_object(path))
            if expectation.request_id in seen_request_ids:
                raise EvidenceError(f"duplicate request_id {expectation.request_id!r}")
            seen_request_ids.add(expectation.request_id)
            expectations.append(expectation)
        except (EvidenceError, OSError) as exc:
            errors.append(_error("invalid_manifest", str(exc), manifest_path=str(path)))
    if not paths:
        errors.append(_error("no_manifests", "no request manifests were supplied"))
    return expectations, errors


def _load_hash_backend(algorithm: str) -> HashBackend:
    version = importlib.metadata.version("vllm")
    if version != SUPPORTED_VLLM_VERSION:
        raise EvidenceError(
            f"vLLM {SUPPORTED_VLLM_VERSION} is required, found {version}"
        )
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import hash_block_tokens

    return HashBackend(version, get_hash_fn_by_name(algorithm), hash_block_tokens)


def _recompute_hashes(
    expectation: RequestExpectation, backend: HashBackend
) -> list[bytes]:
    block_hash = expectation.manifest["block_hash"]
    hash_function = backend.hash_function
    parent = bytes(hash_function(block_hash["none_hash_seed"]))
    all_ids = expectation.prompt_ids + expectation.generated_ids
    hashes: list[bytes] = []
    for start in range(
        0, expectation.persisted_count * expectation.block_size, expectation.block_size
    ):
        parent = bytes(
            backend.hash_block_tokens(
                hash_function,
                parent,
                all_ids[start : start + expectation.block_size],
                None,
            )
        )
        hashes.append(parent)
    return hashes


def _chunk_payload(record: TelysRecord, slab: KVSlab | None) -> bytes:
    if record.value is not None:
        return bytes(record.value)
    metadata = record.metadata
    try:
        offset = int(metadata["vllm.slab_offset"])
        length = int(metadata["vllm.slab_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(
            "chunk has neither an inline payload nor a slab pointer"
        ) from exc
    if slab is None:
        raise EvidenceError("chunk requires a slab but no slab path was configured")
    return slab.read(offset, length)


def _validate_tensor(
    tensor: Any,
    expectation: RequestExpectation,
    *,
    block_count: int | None,
) -> tuple[list[int], str]:
    shape = list(tensor.shape)
    if block_count is not None and (not shape or shape[0] != block_count):
        raise EvidenceError("stacked tensor block count does not match chunk metadata")
    block_shape = shape[1:] if block_count is not None else shape
    if not block_shape or block_shape[0] != expectation.block_size:
        raise EvidenceError("tensor token dimension does not match block_size")
    if (
        expectation.expected_shape is not None
        and tuple(block_shape) != expectation.expected_shape
    ):
        raise EvidenceError(
            f"tensor block shape {block_shape} does not match expected "
            f"{list(expectation.expected_shape)}"
        )
    dtype = str(tensor.dtype)
    if dtype != expectation.kv_dtype:
        raise EvidenceError(
            f"tensor dtype {dtype} does not match expected {expectation.kv_dtype}"
        )
    return block_shape, dtype


def _decode_fast_chunk(
    payload: bytes, expectation: RequestExpectation, block_count: int
) -> dict[str, Any]:
    tensors = TelysFastChunkCodec().decode_chunk(payload)
    if len(tensors) != expectation.expected_layer_count:
        raise EvidenceError("fast chunk layer count does not match the manifest")
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    for tensor in tensors.values():
        shape, dtype = _validate_tensor(tensor, expectation, block_count=block_count)
        shapes.add(tuple(shape))
        dtypes.add(dtype)
    return {
        "format": "fast",
        "layer_count": len(tensors),
        "block_shapes": sorted(shapes),
        "dtypes": sorted(dtypes),
    }


def _decode_legacy_chunk(
    payload: bytes, expectation: RequestExpectation, block_count: int
) -> dict[str, Any]:
    blocks = TelysChunkCodec().decode_chunk(payload)
    if len(blocks) != block_count:
        raise EvidenceError("legacy chunk block count does not match its key")
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    layer_names: set[str] | None = None
    for block in blocks:
        if layer_names is None:
            layer_names = set(block)
            if len(layer_names) != expectation.expected_layer_count:
                raise EvidenceError("legacy chunk layer count does not match manifest")
        elif set(block) != layer_names:
            raise EvidenceError("legacy chunk layers differ between blocks")
        for blob in block.values():
            tensor = deserialize_kv_tensor(blob)
            shape, dtype = _validate_tensor(tensor, expectation, block_count=None)
            shapes.add(tuple(shape))
            dtypes.add(dtype)
    return {
        "format": "legacy",
        "layer_count": len(layer_names or ()),
        "block_shapes": sorted(shapes),
        "dtypes": sorted(dtypes),
    }


def _verify_chunk(
    client: TelysClient,
    slab: KVSlab | None,
    expectation: RequestExpectation,
    *,
    key: str,
    start_hash: bytes,
    block_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunk: dict[str, Any] = {
        "key": key,
        "block_count": block_count,
        "payload_bytes": None,
        "payload_sha256": None,
    }
    try:
        record = client.get(key)
        if record is None:
            raise EvidenceError("chunk index returned a length but no record")
        metadata = record.metadata
        expected_metadata = {
            "vllm.block_start_hash": start_hash.hex(),
            "vllm.block_count": str(block_count),
            "vllm.block_size": str(expectation.block_size),
            "vllm.model": expectation.model,
        }
        for field, expected_value in expected_metadata.items():
            if metadata.get(field) != expected_value:
                raise EvidenceError(
                    f"chunk metadata {field} does not match the recomputed request"
                )
        payload = _chunk_payload(record, slab)
        chunk["payload_bytes"] = len(payload)
        chunk["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        if metadata.get("vllm.chunk_format") == "fast":
            decoded = _decode_fast_chunk(payload, expectation, block_count)
        else:
            decoded = _decode_legacy_chunk(payload, expectation, block_count)
        chunk.update(decoded)
        return chunk, []
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return chunk, [
            _error(
                "invalid_chunk",
                str(exc)
                if isinstance(exc, (EvidenceError, ValueError, OSError))
                else type(exc).__name__,
                key=key,
                exception_type=type(exc).__name__,
            )
        ]


def _verify_request(
    expectation: RequestExpectation,
    client: TelysClient,
    slab: KVSlab | None,
    backend: HashBackend,
    chunk_cache: dict[tuple[Any, ...], tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    covered: set[int] = set()
    hashes = _recompute_hashes(expectation, backend)
    index = 0
    while index < expectation.persisted_count:
        start_hash = hashes[index]
        start_key = f"{expectation.namespace}/chunk/{start_hash.hex()}"
        block_count = client.chunk_length(start_key)
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count < 0
        ):
            errors.append(
                _error(
                    "invalid_chunk_length",
                    "client returned an invalid chunk length",
                    key=start_key,
                )
            )
            break
        if block_count == 0:
            index += 1
            continue
        key = f"{start_key}/{block_count}"
        cache_key = (
            key,
            expectation.model,
            expectation.block_size,
            expectation.kv_dtype,
            expectation.expected_shape,
            expectation.expected_layer_count,
        )
        if cache_key not in chunk_cache:
            chunk_cache[cache_key] = _verify_chunk(
                client,
                slab,
                expectation,
                key=key,
                start_hash=start_hash,
                block_count=block_count,
            )
        chunk, chunk_errors = copy.deepcopy(chunk_cache[cache_key])
        end_index = index + block_count
        chunk.update(
            {
                "start_block": index,
                "origin": "prefix"
                if end_index <= expectation.prefix_count
                else "suffix",
                "crosses_prompt_generation_boundary": (
                    index < expectation.prefix_count < end_index
                ),
                "token_start": index * expectation.block_size,
                "token_end_exclusive": end_index * expectation.block_size,
            }
        )
        chunks.append(chunk)
        errors.extend(chunk_errors)
        covered.update(range(index, min(end_index, expectation.persisted_count)))
        index = end_index

    missing = sorted(set(range(expectation.persisted_count)) - covered)
    if missing:
        errors.append(
            _error(
                "missing_expected_chunks",
                "one or more recomputed blocks have no persisted chunk",
                block_indices=missing,
            )
        )
    prefix_chunks = sum(chunk["origin"] == "prefix" for chunk in chunks)
    suffix_chunks = len(chunks) - prefix_chunks
    status = FAIL if errors else PASS
    inconclusive_reasons: list[dict[str, Any]] = []
    if expectation.expected_shape is None and not errors:
        status = INCONCLUSIVE
        inconclusive_reasons.append(
            _error(
                "unresolved_expected_shape",
                "manifest does not contain an exact full-attention KV block shape",
            )
        )
    return {
        "request_id": expectation.request_id,
        "manifest_path": str(expectation.path),
        "status": status,
        "expected": {
            "prefix_blocks": expectation.prefix_count,
            "suffix_blocks": expectation.persisted_count - expectation.prefix_count,
            "total_blocks": expectation.persisted_count,
        },
        "verified": {
            "blocks": len(covered),
            "prefix_chunks": prefix_chunks,
            "suffix_chunks": suffix_chunks,
            "boundary_chunks": sum(
                chunk["crosses_prompt_generation_boundary"] for chunk in chunks
            ),
        },
        "chunks": chunks,
        "missing_block_indices": missing,
        "errors": errors,
        "inconclusive_reasons": inconclusive_reasons,
    }


def _create_client(namespace: str, slab_path: str | None) -> TelysClient:
    extra: dict[str, Any] = {
        "telys_namespace": namespace,
        "telys_load_async": False,
    }
    if slab_path is not None:
        extra["telys_slab_path"] = slab_path
    config = TelysConnectorConfig.from_sources(extra=extra)
    from telys_vllm.connector import _auto_client

    return _auto_client(config)


def _evidence_errors(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if evidence.get("schema_version") != SCHEMA_VERSION:
        errors.append(_error("invalid_evidence", "unsupported evidence schema_version"))
    if evidence.get("kind") != "telys_vllm_lifecycle_evidence":
        errors.append(_error("invalid_evidence", "unexpected evidence kind"))
    process = evidence.get("process")
    if not isinstance(process, dict) or process.get("exit_kind") not in (
        "completed",
        "abrupt-exit",
        "failed",
    ):
        errors.append(_error("invalid_evidence", "invalid process exit evidence"))
    elif process["exit_kind"] == "failed":
        errors.append(
            _error(
                "process_failed",
                "the harness exited with a non-signal failure",
                exit_status=process.get("exit_status"),
            )
        )
    recorded_telemetry = evidence.get("telemetry")
    if not isinstance(recorded_telemetry, dict) or not isinstance(
        recorded_telemetry.get("path"), str
    ):
        errors.append(_error("invalid_evidence", "telemetry.path is required"))
        return errors
    current_telemetry = _telemetry_snapshot(Path(recorded_telemetry["path"]))
    if current_telemetry != recorded_telemetry:
        errors.append(
            _error(
                "telemetry_changed",
                "telemetry contents differ from the recorded lifecycle evidence",
            )
        )
    if current_telemetry["malformed_lines"]:
        errors.append(
            _error(
                "malformed_telemetry",
                "telemetry contains malformed JSON lines",
                lines=current_telemetry["malformed_lines"],
            )
        )
    return errors


def verify_evidence(
    evidence_path: str | os.PathLike[str],
    *,
    manifest_paths: Sequence[str] = (),
    slab_path: str | None = None,
    client: TelysClient | None = None,
    hash_backend: HashBackend | None = None,
) -> dict[str, Any]:
    path = Path(evidence_path).resolve()
    evidence = _read_json_object(path)
    errors = _evidence_errors(evidence)
    expectations, manifest_errors = _load_expectations(evidence, manifest_paths)
    errors.extend(manifest_errors)
    requests: list[dict[str, Any]] = []
    clients: dict[str, TelysClient] = {}
    slabs: dict[str, KVSlab] = {}
    chunk_cache: dict[tuple[Any, ...], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    backends: dict[str, HashBackend] = {}
    try:
        for expectation in expectations:
            try:
                if client is None and expectation.namespace not in clients:
                    clients[expectation.namespace] = _create_client(
                        expectation.namespace, slab_path
                    )
                request_client = client or clients[expectation.namespace]
                request_slab = None
                resolved_slab_path = slab_path
                if resolved_slab_path is None:
                    config = TelysConnectorConfig.from_sources(
                        extra={
                            "telys_namespace": expectation.namespace,
                            "telys_load_async": False,
                        }
                    )
                    resolved_slab_path = config.slab_path
                if (
                    resolved_slab_path is not None
                    and resolved_slab_path not in slabs
                    and Path(resolved_slab_path).is_file()
                ):
                    slabs[resolved_slab_path] = KVSlab(resolved_slab_path)
                request_slab = slabs.get(resolved_slab_path or "")
                algorithm = expectation.manifest["block_hash"]["algorithm"]
                if hash_backend is not None:
                    backend = hash_backend
                else:
                    if algorithm not in backends:
                        backends[algorithm] = _load_hash_backend(algorithm)
                    backend = backends[algorithm]
                requests.append(
                    _verify_request(
                        expectation,
                        request_client,
                        request_slab,
                        backend,
                        chunk_cache,
                    )
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                message = (
                    str(exc)
                    if isinstance(exc, (EvidenceError, ValueError, OSError))
                    else type(exc).__name__
                )
                errors.append(
                    _error(
                        "request_verification_error",
                        message,
                        request_id=expectation.request_id,
                        exception_type=type(exc).__name__,
                    )
                )
    finally:
        for owned_client in clients.values():
            try:
                owned_client.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(_error("client_close_failed", type(exc).__name__))
        for slab in slabs.values():
            slab.close()

    if any(request["status"] == FAIL for request in requests) or errors:
        status = FAIL
    elif any(request["status"] == INCONCLUSIVE for request in requests):
        status = INCONCLUSIVE
    else:
        status = PASS
    telemetry = evidence.get("telemetry", {})
    process = evidence.get("process", {})
    exit_kind = process.get("exit_kind")
    pending_spans = telemetry.get("pending_write_spans", [])
    inconclusive_reasons: list[dict[str, Any]] = []
    if exit_kind == "abrupt-exit":
        inconclusive_reasons.append(
            _error(
                "quiescence_unconfirmed",
                "the harness exited abruptly; clean write draining was not confirmed",
                exit_status=process.get("exit_status"),
            )
        )
        if status == PASS:
            status = INCONCLUSIVE
    if not pending_spans:
        inconclusive_reasons.append(
            _error(
                "pending_writes_not_observed",
                "no wait_for_save telemetry span observed pending writes",
            )
        )
        if status == PASS:
            status = INCONCLUSIVE
    for request in requests:
        inconclusive_reasons.extend(request["inconclusive_reasons"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "telys_vllm_persisted_chunk_verification",
        "status": status,
        "passed": status == PASS,
        "evidence_path": str(path),
        "lifecycle": {
            "exit_kind": evidence.get("process", {}).get("exit_kind"),
            "exit_status": evidence.get("process", {}).get("exit_status"),
            "pending_write_spans": len(pending_spans),
            "pending_chunks_observed": sum(
                span.get("pending_chunks", 0) for span in pending_spans
            ),
            "persisted_chunk_spans": len(telemetry.get("persisted_chunk_spans", [])),
        },
        "summary": {
            "requests": len(expectations),
            "expected_prefix_blocks": sum(
                expectation.prefix_count for expectation in expectations
            ),
            "expected_suffix_blocks": sum(
                expectation.persisted_count - expectation.prefix_count
                for expectation in expectations
            ),
            "expected_blocks": sum(
                expectation.persisted_count for expectation in expectations
            ),
            "verified_blocks": sum(
                request["verified"]["blocks"] for request in requests
            ),
            "missing_blocks": sum(
                len(request["missing_block_indices"]) for request in requests
            ),
            "prefix_chunks": sum(
                request["verified"]["prefix_chunks"] for request in requests
            ),
            "suffix_chunks": sum(
                request["verified"]["suffix_chunks"] for request in requests
            ),
        },
        "vllm_version": (
            hash_backend.version
            if hash_backend is not None
            else next(iter(backends.values())).version
            if backends
            else None
        ),
        "requests": requests,
        "errors": errors,
        "inconclusive_reasons": inconclusive_reasons,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser(
        "record", help="record post-process lifecycle evidence"
    )
    record.add_argument("--output", required=True)
    record.add_argument("--telemetry", required=True)
    record.add_argument("--manifest-dir", required=True)
    record.add_argument("--arm", required=True)
    record.add_argument("--phase", required=True)
    record.add_argument("--exit-status", required=True, type=int)
    verify = subparsers.add_parser("verify", help="verify persisted chunks")
    verify.add_argument("--evidence", required=True)
    verify.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest file or directory; overrides evidence manifest entries",
    )
    verify.add_argument("--slab-path")
    verify.add_argument("--output-json")
    return parser


def _unexpected_failure(command: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "telys_vllm_persisted_chunk_verification",
        "status": FAIL,
        "passed": False,
        "errors": [
            _error(
                "verification_error",
                f"{command} failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
            )
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.command == "record":
            result = record_evidence(args)
            status = PASS
        else:
            result = verify_evidence(
                args.evidence,
                manifest_paths=args.manifest,
                slab_path=args.slab_path,
            )
            status = result["status"]
            if args.output_json:
                _write_json_atomic(Path(args.output_json).resolve(), result)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        result = _unexpected_failure(args.command, exc)
        status = FAIL
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(encoded)
    return EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
