# SPDX-License-Identifier: Apache-2.0
"""Pinned staging cache bounds, content identity, and H2D lifetime tests."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.codec import TelysFastChunkCodec
from telys_vllm.connector import (
    _CHUNK_CONTENT_SHA256_KEY,
    TelysConnectorMetadata,
    TelysKVConnector,
    TelysLoadSpec,
    _ChunkContentValidator,
    _decoded_chunk_identity,
    _encoded_chunk_identity,
    _pinned_staging_key,
    _PinnedStagingCache,
)
from telys_vllm.gpu_ops import serialize_kv_tensor
from telys_vllm.store import TelysStoreError
from tests.fakes import make_hashes, make_kv_cache_config, make_vllm_config
from tests.test_store import FakeBlobStoreClient


class _FakeEvent:
    def __init__(self, timeline: list[tuple[str, int]] | None = None) -> None:
        self.completed = False
        self.synchronize_calls = 0
        self._timeline = timeline
        self._id = id(self)

    def record(self, stream) -> None:
        self.completed = False
        if self._timeline is not None:
            self._timeline.append(("record", self._id))

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.completed = True
        if self._timeline is not None:
            self._timeline.append(("synchronize", self._id))


class _CopyCountingTensor(torch.Tensor):
    copy_calls = 0

    def copy_(self, source, *args, **kwargs):
        _CopyCountingTensor.copy_calls += 1
        return super().copy_(source, *args, **kwargs)


_REAL_EMPTY = torch.empty


def _fake_empty(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _REAL_EMPTY(*args, **kwargs).as_subclass(_CopyCountingTensor)


def make_inline_client(config) -> FakeBlobStoreClient:
    return FakeBlobStoreClient()


def _make_worker(extra=None) -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(extra), KVConnectorRole.WORKER, make_kv_cache_config()
    )


def _make_inline_worker(extra=None) -> TelysKVConnector:
    return _make_worker(
        {
            "telys_client_factory": (
                "tests.test_pinned_load_cache:make_inline_client"
            ),
            "telys_slab_path": None,
            "telys_blobstore_no_slab": True,
            **(extra or {}),
        }
    )


def _load_fixture() -> tuple[TelysLoadSpec, SimpleNamespace]:
    block_hash = make_hashes(1)[0]
    spec = TelysLoadSpec(
        request_id="req",
        block_hashes=[block_hash],
        num_tokens=16,
        block_ids=[0],
    )
    forward_context = SimpleNamespace(
        attn_metadata={"layer.0": object()},
        no_compile_layers={
            "layer.0": SimpleNamespace(
                kv_cache=torch.zeros(1, 2, 16, 4, dtype=torch.bfloat16)
            )
        },
    )
    return spec, forward_context


def _source(value: int = 0) -> torch.Tensor:
    return (torch.arange(16 * 8, dtype=torch.bfloat16) + value).reshape(1, 16, 2, 4)


def _inject_fast(worker, spec, forward_context):
    injected = []
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(worker, "_get_load_stream", return_value=None),
        patch.object(torch, "empty", side_effect=_fake_empty),
        patch(
            "telys_vllm.connector.inject_kv_blocks_contiguous",
            side_effect=lambda *args: injected.append(args[-1].clone()),
        ),
    ):
        staged = worker._fast_inject_spec(spec, forward_context)
    return staged, injected


@pytest.mark.parametrize(
    ("max_entries", "max_bytes"), [(2, 1024), (8, 16)], ids=["entries", "bytes"]
)
def test_pinned_cache_is_bounded_and_waits_before_eviction(
    max_entries: int, max_bytes: int
):
    cache = _PinnedStagingCache(max_entries=max_entries, max_bytes=max_bytes)
    first = torch.zeros(4, dtype=torch.bfloat16)
    second = torch.ones(4, dtype=torch.bfloat16)
    third = torch.full((4,), 2, dtype=torch.bfloat16)
    first_entry = cache.put(b"first", first)
    second_entry = cache.put(b"second", second)
    assert first_entry is not None and second_entry is not None
    first_entry.event = _FakeEvent()
    second_entry.event = _FakeEvent()
    first_wait = first_entry.event
    second_wait = second_entry.event

    # Refresh first so second is the LRU victim in the entry-bound case.
    assert cache.get(b"first") is first_entry
    assert first_wait.synchronize_calls == 1
    assert first_entry.event is None
    third_entry = cache.put(b"third", third)

    assert third_entry is not None
    assert len(cache) == 2
    assert cache.total_bytes == 16
    assert second_wait.synchronize_calls == 1
    assert second_entry.event is None
    assert cache.get(b"second") is None
    assert cache.get(b"first").tensor is first

    oversized = torch.zeros(max_bytes // 2 + 1, dtype=torch.bfloat16)
    assert cache.put(b"oversized", oversized) is None
    assert len(cache) == 2
    assert cache.total_bytes == 16

    first_entry.event = _FakeEvent()
    third_entry.event = _FakeEvent()
    first_wait = first_entry.event
    third_wait = third_entry.event
    cache.clear()
    assert first_wait.synchronize_calls == 1
    assert third_wait.synchronize_calls == 1
    assert len(cache) == 0
    assert cache.total_bytes == 0


def test_identical_identity_skips_copy_but_changed_identity_does_not():
    worker = _make_worker()
    identity = _encoded_chunk_identity(b"first")
    changed_identity = _encoded_chunk_identity(b"changed")
    source = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16)
    changed = source.clone()
    changed[0, 0] = -1
    empty_calls = []
    _CopyCountingTensor.copy_calls = 0

    def counted_empty(*args, **kwargs):
        tensor = _fake_empty(*args, **kwargs)
        empty_calls.append(tensor)
        return tensor

    try:
        with patch.object(torch, "empty", side_effect=counted_empty):
            first, _ = worker._stage_pinned_tensor(identity, "layer.0", source)
            second, _ = worker._stage_pinned_tensor(identity, "layer.0", source)
            third, _ = worker._stage_pinned_tensor(
                changed_identity, "layer.0", changed
            )

        assert first is second
        assert third is not first
        assert len(empty_calls) == 2
        assert _CopyCountingTensor.copy_calls == 2
        assert torch.equal(first, source)
        assert torch.equal(third, changed)
        assert len(worker._pinned_staging_cache) == 2
    finally:
        worker.shutdown()


def test_concurrent_loads_reuse_one_buffer_only_after_h2d_completion():
    worker = _make_worker()
    block_hash = b"\x02" * 32
    block_size = 16
    spec = TelysLoadSpec(
        request_id="req",
        block_hashes=[block_hash],
        num_tokens=block_size,
        block_ids=[0],
    )
    worker.bind_connector_metadata(TelysConnectorMetadata(loads=[spec]))
    source = torch.arange(block_size * 8, dtype=torch.bfloat16).reshape(
        1, block_size, 2, 4
    )
    chunk = SimpleNamespace(
        block_hashes=(block_hash,), layer_tensors={"layer.0": source}
    )
    worker._store.load_decoded_blocks_fast = lambda hashes: [chunk]
    kv_cache = torch.zeros(1, 2, block_size, 4, dtype=torch.bfloat16)
    forward_context = SimpleNamespace(
        attn_metadata={"layer.0": object()},
        no_compile_layers={"layer.0": SimpleNamespace(kv_cache=kv_cache)},
    )

    timeline: list[tuple[str, int]] = []
    events: list[_FakeEvent] = []

    class FakeStream:
        def synchronize(self) -> None:
            timeline.append(("stream_synchronize", 0))

    class FakeDefaultStream:
        def wait_event(self, event) -> None:
            timeline.append(("default_wait", id(event)))

    load_stream = FakeStream()
    default_stream = FakeDefaultStream()

    def make_event():
        event = _FakeEvent(timeline)
        events.append(event)
        return event

    state = {"active": 0, "max_active": 0, "injections": 0}
    state_lock = threading.Lock()
    staged_ptrs: list[int] = []

    def fake_inject(kv_layer, block_ids, size, attn_metadata, gpu_tensor):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["injections"] += 1
            injection_number = state["injections"]
            staged_ptrs.append(gpu_tensor.data_ptr())
        timeline.append(("inject", injection_number))
        time.sleep(0.02)
        with state_lock:
            state["active"] -= 1

    empty_calls = []
    _CopyCountingTensor.copy_calls = 0

    def counted_empty(*args, **kwargs):
        tensor = _fake_empty(*args, **kwargs)
        empty_calls.append(tensor)
        return tensor

    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def load() -> None:
        try:
            barrier.wait(timeout=5)
            worker.start_load_kv(forward_context)
        except Exception as exc:  # noqa: BLE001 - collect worker thread failures for diagnostics
            errors.append(exc)

    try:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "Stream", return_value=load_stream),
            patch.object(torch.cuda, "Event", side_effect=make_event),
            patch.object(torch.cuda, "stream", return_value=contextlib.nullcontext()),
            patch.object(torch.cuda, "current_stream", return_value=default_stream),
            patch.object(torch, "empty", side_effect=counted_empty),
            patch(
                "telys_vllm.connector.inject_kv_blocks_contiguous",
                side_effect=fake_inject,
            ),
        ):
            threads = [threading.Thread(target=load) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=10)
            assert all(not thread.is_alive() for thread in threads)

        assert not errors
        assert state["max_active"] == 1
        assert state["injections"] == 2
        assert len(empty_calls) == 1
        assert _CopyCountingTensor.copy_calls == 1
        assert len(set(staged_ptrs)) == 1
        assert len(events) == 4
        assert events[0].synchronize_calls >= 1
        assert events[1].synchronize_calls >= 1
        second_inject = timeline.index(("inject", 2))
        assert ("synchronize", id(events[0])) in timeline[:second_inject]
        assert len(worker._pinned_staging_cache) == 1

        worker.reset_cache()
        assert len(worker._pinned_staging_cache) == 0
        assert worker._pinned_staging_cache.total_bytes == 0
    finally:
        worker.shutdown()


def _save_source(worker, spec, source) -> str:
    return worker.store.save_chunk_fast(
        spec.block_hashes[0], spec.block_hashes, {"layer.0": source}
    )


@pytest.mark.parametrize(
    "worker_factory", [_make_worker, _make_inline_worker], ids=["slab", "inline"]
)
def test_storage_version_is_authenticated_once(worker_factory):
    worker = worker_factory()
    spec, _ = _load_fixture()
    _save_source(worker, spec, _source())
    try:
        with patch(
            "telys_vllm.connector._encoded_chunk_identity",
            wraps=_encoded_chunk_identity,
        ) as identity_hash:
            first = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]
            second = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]

        assert identity_hash.call_count == 1
        assert first.content_identity == second.content_identity
        assert len(worker.store._content_validator) == 1
    finally:
        worker.shutdown()


def test_overwrite_with_same_structure_injects_new_bytes_not_stale_reuse():
    worker = _make_worker()
    spec, forward_context = _load_fixture()
    source = _source()
    changed = _source(1000)
    _save_source(worker, spec, source)
    _CopyCountingTensor.copy_calls = 0
    try:
        first_staged, first_injected = _inject_fast(worker, spec, forward_context)
        second_staged, second_injected = _inject_fast(worker, spec, forward_context)

        assert first_staged[0] is second_staged[0]
        assert torch.equal(first_injected[0], source)
        assert torch.equal(second_injected[0], source)
        assert _CopyCountingTensor.copy_calls == 1

        _save_source(worker, spec, changed)
        third_staged, third_injected = _inject_fast(worker, spec, forward_context)

        assert third_staged[0] is not first_staged[0]
        assert torch.equal(third_injected[0], changed)
        assert not torch.equal(third_injected[0], source)
        assert _CopyCountingTensor.copy_calls == 2
        assert len(worker._pinned_staging_cache) == 2
    finally:
        worker.shutdown()


@pytest.mark.parametrize(
    "worker_factory", [_make_worker, _make_inline_worker], ids=["slab", "inline"]
)
def test_identical_repeated_put_reuses_authenticated_content(worker_factory):
    worker = worker_factory()
    spec, _ = _load_fixture()
    source = _source()
    _save_source(worker, spec, source)
    _CopyCountingTensor.copy_calls = 0
    try:
        first_chunk = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]
        with patch.object(torch, "empty", side_effect=_fake_empty):
            first, _ = worker._stage_pinned_tensor(
                first_chunk.content_identity, "layer.0", source
            )
            _save_source(worker, spec, source)
            second_chunk = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]
            second, _ = worker._stage_pinned_tensor(
                second_chunk.content_identity, "layer.0", source
            )

        assert first_chunk.content_identity == second_chunk.content_identity
        assert first is second
        assert _CopyCountingTensor.copy_calls == 1
        assert len(worker._pinned_staging_cache) == 1
    finally:
        worker.shutdown()


def test_sync_persistence_adds_digest_and_rejects_forged_metadata():
    worker = _make_inline_worker()
    spec, _ = _load_fixture()
    client = worker.store._client._delegate
    try:
        key = _save_source(worker, spec, _source())
        record = client.get(key)
        assert record.metadata[_CHUNK_CONTENT_SHA256_KEY] == hashlib.sha256(
            record.value
        ).hexdigest()
        put_count = client.put_count

        with pytest.raises(TelysStoreError, match="content digest mismatch"):
            worker.store.save_chunk_fast(
                spec.block_hashes[0],
                spec.block_hashes,
                {"layer.0": _source(1)},
                extra_metadata={_CHUNK_CONTENT_SHA256_KEY: "00" * 32},
            )
        assert client.put_count == put_count
    finally:
        worker.shutdown()


def test_write_behind_hashes_final_encoded_payload_on_write_worker():
    worker = _make_inline_worker({"telys_write_behind": True})
    spec, _ = _load_fixture()
    caller_thread = threading.get_ident()
    hash_threads = []
    try:
        with patch(
            "telys_vllm.connector._encoded_chunk_identity",
            side_effect=lambda payload: (
                hash_threads.append(threading.get_ident()),
                _encoded_chunk_identity(payload),
            )[1],
        ):
            key = _save_source(worker, spec, _source())
            worker.store.drain()

        assert len(hash_threads) == 1
        assert hash_threads[0] != caller_thread
        record = worker.store._client._delegate.get(key)
        assert record.metadata[_CHUNK_CONTENT_SHA256_KEY] == hashlib.sha256(
            record.value
        ).hexdigest()
    finally:
        worker.shutdown()


def _rewrite_record(worker, key, *, payload=None, digest=None) -> None:
    client = worker.store._client._delegate
    record = client.get(key)
    metadata = dict(record.metadata)
    if digest is None:
        metadata.pop(_CHUNK_CONTENT_SHA256_KEY, None)
    else:
        metadata[_CHUNK_CONTENT_SHA256_KEY] = digest
    client.put(key, record.value if payload is None else payload, metadata)


@pytest.mark.parametrize(
    "worker_factory", [_make_worker, _make_inline_worker], ids=["slab", "inline"]
)
def test_missing_digest_legacy_record_uses_full_hash_once(worker_factory):
    worker = worker_factory()
    spec, _ = _load_fixture()
    source = _source()
    key = _save_source(worker, spec, source)
    _rewrite_record(worker, key)
    try:
        with patch(
            "telys_vllm.connector._encoded_chunk_identity",
            wraps=_encoded_chunk_identity,
        ) as identity_hash:
            first = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]
            second = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]

        assert identity_hash.call_count == 1
        assert first.content_identity == second.content_identity
        assert torch.equal(first.layer_tensors["layer.0"], source)
    finally:
        worker.shutdown()


def test_legacy_codec_without_digest_loads_through_explicit_full_hash():
    worker = _make_worker()
    spec, _ = _load_fixture()
    source = _source()
    key = worker.store.save_chunk(
        spec.block_hashes[0],
        spec.block_hashes,
        {"layer.0": [serialize_kv_tensor(source[0])]},
    )
    _rewrite_record(worker, key)
    try:
        with patch(
            "telys_vllm.connector._encoded_chunk_identity",
            wraps=_encoded_chunk_identity,
        ) as identity_hash:
            loaded = worker.store.load_decoded_blocks_fast(spec.block_hashes)[0]

        assert identity_hash.call_count == 1
        assert torch.equal(loaded.layer_tensors["layer.0"], source)
    finally:
        worker.shutdown()


@pytest.mark.parametrize("prevalidated", [False, True], ids=["new", "cached"])
@pytest.mark.parametrize(
    "bad_digest", ["00" * 32, "not-a-digest", "AA" * 32],
    ids=["wrong", "malformed", "noncanonical"],
)
def test_malicious_or_malformed_digest_fails_closed(prevalidated, bad_digest):
    worker = _make_worker()
    spec, _ = _load_fixture()
    key = _save_source(worker, spec, _source())
    try:
        if prevalidated:
            worker.store.load_decoded_blocks_fast(spec.block_hashes)
        _rewrite_record(worker, key, digest=bad_digest)

        with pytest.raises(TelysStoreError, match="content digest"):
            worker.store.load_decoded_blocks_fast(spec.block_hashes)
    finally:
        worker.shutdown()


def test_stale_digest_on_changed_inline_payload_cannot_reuse_pinned_entry():
    worker = _make_inline_worker()
    spec, forward_context = _load_fixture()
    key = _save_source(worker, spec, _source())
    try:
        first_staged, _ = _inject_fast(worker, spec, forward_context)
        client = worker.store._client._delegate
        stale_metadata = dict(client.get(key).metadata)
        malicious_payload = TelysFastChunkCodec().encode_chunk(
            {"layer.0": _source(2000)}
        )
        client.put(key, malicious_payload, stale_metadata)

        with pytest.raises(TelysStoreError, match="content digest mismatch"):
            _inject_fast(worker, spec, forward_context)
        assert len(worker._pinned_staging_cache) == 1
        assert worker._pinned_staging_cache.get(
            next(iter(worker._pinned_staging_cache._entries))
        ).tensor is first_staged[0]
    finally:
        worker.shutdown()


def test_opaque_decoded_chunks_hash_all_content_and_do_not_trust_structure():
    worker = _make_worker()
    source = _source()
    changed = _source(1)
    identity = _decoded_chunk_identity({"layer.0": source})
    identical_identity = _decoded_chunk_identity({"layer.0": source.clone()})
    changed_identity = _decoded_chunk_identity({"layer.0": changed})
    _CopyCountingTensor.copy_calls = 0
    try:
        with patch.object(torch, "empty", side_effect=_fake_empty):
            first, _ = worker._stage_pinned_tensor(identity, "layer.0", source)
            second, _ = worker._stage_pinned_tensor(
                identical_identity, "layer.0", source
            )
            third, _ = worker._stage_pinned_tensor(
                changed_identity, "layer.0", changed
            )

        assert identity == identical_identity
        assert identity != changed_identity
        assert first is second
        assert third is not first
        assert torch.equal(third, changed)
        assert _CopyCountingTensor.copy_calls == 2
    finally:
        worker.shutdown()


def test_validation_cache_entry_bound_is_deterministic():
    validator = _ChunkContentValidator(max_entries=2, max_bytes=1024)
    first = validator.validate("first", b"aa", {}, None)
    second = validator.validate("second", b"bb", {}, None)
    third = validator.validate("third", b"cc", {}, None)

    assert len(validator) == 2
    assert validator.retained_bytes == 4
    assert validator.validate("second", b"bb", {}, None) == second
    assert validator.validate("third", b"cc", {}, None) == third
    assert validator.validate("first", b"aa", {}, None) == first


def test_validation_cache_byte_bound_and_oversized_bypass():
    validator = _ChunkContentValidator(max_entries=8, max_bytes=3)
    validator.validate("first", b"aa", {}, None)
    second = validator.validate("second", b"bb", {}, None)

    assert len(validator) == 1
    assert validator.retained_bytes == 2
    assert validator.validate("second", b"bb", {}, None) == second
    validator.validate("oversized", b"too-large", {}, None)
    assert len(validator) == 1
    assert validator.retained_bytes == 2


def test_concurrent_validation_of_one_immutable_payload_hashes_once():
    validator = _ChunkContentValidator(max_entries=8, max_bytes=1024)
    payload = b"immutable encoded payload"
    barrier = threading.Barrier(5)
    identities = []
    errors = []

    def validate() -> None:
        try:
            barrier.wait(timeout=5)
            identities.append(validator.validate("key", payload, {}, None))
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)

    with patch(
        "telys_vllm.connector._encoded_chunk_identity",
        wraps=_encoded_chunk_identity,
    ) as identity_hash:
        threads = [threading.Thread(target=validate) for _ in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert identity_hash.call_count == 1
    assert identities == [identities[0]] * 4


def test_reset_and_shutdown_clear_validation_state():
    worker = _make_worker()
    spec, _ = _load_fixture()
    _save_source(worker, spec, _source())
    worker.store.load_decoded_blocks_fast(spec.block_hashes)
    assert len(worker.store._content_validator) == 1

    worker.reset_cache()
    assert len(worker.store._content_validator) == 0
    assert len(worker._pinned_staging_cache) == 0

    _save_source(worker, spec, _source(1))
    worker.store.load_decoded_blocks_fast(spec.block_hashes)
    assert len(worker.store._content_validator) == 1
    worker.shutdown()
    assert len(worker.store._content_validator) == 0


def test_pinned_key_never_reads_tensor_bytes():
    class NoNumpyTensor(torch.Tensor):
        def numpy(self, *args, **kwargs):
            raise AssertionError("pinned key attempted to read tensor bytes")

    identity = _encoded_chunk_identity(b"encoded chunk")
    source = _source().as_subclass(NoNumpyTensor)
    key = _pinned_staging_key(identity, "layer.0", source)

    assert key == _pinned_staging_key(identity, "layer.0", source)
    assert key != _pinned_staging_key(identity, "layer.1", source)
    assert key != _pinned_staging_key(
        _encoded_chunk_identity(b"different chunk"), "layer.0", source
    )
    assert key != _pinned_staging_key(
        identity, "layer.0", source.reshape(1, 16, 8)
    )


def test_non_pinned_legacy_load_does_not_pay_for_staging_authentication():
    worker = _make_worker()
    spec, _ = _load_fixture()
    _save_source(worker, spec, _source())
    try:
        with patch(
            "telys_vllm.connector._encoded_chunk_identity",
            wraps=_encoded_chunk_identity,
        ) as identity_hash:
            blocks = worker.store.load_decoded_blocks(spec.block_hashes)

        assert len(blocks) == 1
        assert identity_hash.call_count == 0
        assert len(worker.store._content_validator) == 0
    finally:
        worker.shutdown()
