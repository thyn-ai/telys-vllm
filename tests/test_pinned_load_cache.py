# SPDX-License-Identifier: Apache-2.0
"""Pinned staging cache bounds, content identity, and H2D lifetime tests."""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import (
    TelysConnectorMetadata,
    TelysKVConnector,
    TelysLoadSpec,
    _PinnedStagingCache,
)
from tests.fakes import make_kv_cache_config, make_vllm_config


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


def _make_worker() -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(), KVConnectorRole.WORKER, make_kv_cache_config()
    )


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


def test_identical_contents_skip_copy_but_reused_key_with_new_bytes_does_not():
    worker = _make_worker()
    block_hashes = (b"\x01" * 32,)
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
            first, _ = worker._stage_pinned_tensor(block_hashes, "layer.0", source)
            second, _ = worker._stage_pinned_tensor(block_hashes, "layer.0", source)
            third, _ = worker._stage_pinned_tensor(block_hashes, "layer.0", changed)

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
