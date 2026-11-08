# SPDX-License-Identifier: Apache-2.0
"""Batched async KV extraction stream, parity, and fallback tests."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import pytest
import torch

from telys_vllm.gpu_ops import AsyncKVExtractionBatch
from tests.test_gpu_ops_layout import (
    BLOCK_SIZE,
    LAYER,
    LAYOUTS,
    NUM_LAYERS,
    _expected_block,
    _make_logical_view,
    _raw_bytes,
)


class _FakeCudaTensor(torch.Tensor):
    cpu_calls = 0

    @property
    def is_cuda(self):
        return True

    def cpu(self, *args, **kwargs):
        _FakeCudaTensor.cpu_calls += 1
        return self.as_subclass(torch.Tensor)


class _PinnedTestTensor(torch.Tensor):
    copy_calls: ClassVar[list[bool]] = []
    fail_copy = False

    def is_pinned(self):
        return True

    def copy_(self, source, *args, **kwargs):
        non_blocking = bool(kwargs.get("non_blocking", False))
        _PinnedTestTensor.copy_calls.append(non_blocking)
        if _PinnedTestTensor.fail_copy:
            raise RuntimeError("injected nonblocking copy failure")
        return super().copy_(source.as_subclass(torch.Tensor), *args, **kwargs)


class _FakeEvent:
    def __init__(self, synchronize_error: Exception | None = None) -> None:
        self.record_calls = 0
        self.synchronize_calls = 0
        self.stream = None
        self.synchronize_error = synchronize_error

    def record(self, stream) -> None:
        self.record_calls += 1
        self.stream = stream

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.synchronize_error is not None:
            raise self.synchronize_error


_REAL_EMPTY = torch.empty


def _pinned_empty(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _REAL_EMPTY(*args, **kwargs).as_subclass(_PinnedTestTensor)


def _fake_cuda_layers():
    logical_5d, _ = _make_logical_view(LAYOUTS["LBNHC"])
    return [
        logical_5d[layer].to(torch.bfloat16).as_subclass(_FakeCudaTensor)
        for layer in range(NUM_LAYERS)
    ]


def _expected_stacked(kv_layer, block_ids):
    base_layer = kv_layer.as_subclass(torch.Tensor)
    return torch.stack(
        [_expected_block(base_layer, block_id) for block_id in block_ids]
    )


def test_async_batch_transfers_each_layer_once_and_synchronizes_once():
    layers = _fake_cuda_layers()
    block_ids = [3, 1, 2]
    expected = [_expected_stacked(layer, block_ids) for layer in layers]
    event = _FakeEvent()
    stream = object()
    _FakeCudaTensor.cpu_calls = 0
    _PinnedTestTensor.copy_calls = []
    _PinnedTestTensor.fail_copy = False

    with (
        patch.object(torch, "empty", side_effect=_pinned_empty),
        patch.object(torch.cuda, "current_stream", return_value=stream),
        patch.object(torch.cuda, "Event", return_value=event),
        AsyncKVExtractionBatch() as batch,
    ):
        extracted = [
            batch.extract(layer, block_ids, BLOCK_SIZE, None) for layer in layers
        ]
        assert event.record_calls == 0
        assert event.synchronize_calls == 0

    assert _PinnedTestTensor.copy_calls == [True, True]
    assert _FakeCudaTensor.cpu_calls == 0
    assert event.record_calls == 1
    assert event.synchronize_calls == 1
    assert event.stream is stream
    for actual, expected_tensor in zip(extracted, expected):
        assert actual.dtype == torch.bfloat16
        assert actual.shape == expected_tensor.shape
        assert _raw_bytes(actual) == _raw_bytes(expected_tensor)


def test_async_batch_falls_back_to_synchronous_cpu_without_pinned_memory():
    layers = _fake_cuda_layers()
    block_ids = [3, 1, 2]
    expected = [_expected_stacked(layer, block_ids) for layer in layers]
    _FakeCudaTensor.cpu_calls = 0
    _PinnedTestTensor.copy_calls = []

    with (
        patch.object(
            torch, "empty", side_effect=RuntimeError("pin_memory is unavailable")
        ),
        patch.object(
            torch.cuda,
            "current_stream",
            side_effect=AssertionError("CUDA stream must not be used"),
        ),
        patch.object(
            torch.cuda,
            "Event",
            side_effect=AssertionError("CUDA event must not be created"),
        ),
        AsyncKVExtractionBatch() as batch,
    ):
        extracted = [
            batch.extract(layer, block_ids, BLOCK_SIZE, None) for layer in layers
        ]

    assert _FakeCudaTensor.cpu_calls == NUM_LAYERS
    assert _PinnedTestTensor.copy_calls == []
    for actual, expected_tensor in zip(extracted, expected):
        assert actual.dtype == torch.bfloat16
        assert actual.shape == expected_tensor.shape
        assert _raw_bytes(actual) == _raw_bytes(expected_tensor)


def test_async_batch_does_not_hide_unrelated_pinned_allocation_errors():
    layer = _fake_cuda_layers()[LAYER]
    _FakeCudaTensor.cpu_calls = 0

    with (
        patch.object(torch, "empty", side_effect=RuntimeError("CUDA out of memory")),
        patch.object(
            torch.cuda,
            "Event",
            side_effect=AssertionError("no copy should be synchronized"),
        ),
        pytest.raises(RuntimeError, match="CUDA out of memory"),
        AsyncKVExtractionBatch() as batch,
    ):
        batch.extract(layer, [1, 2], BLOCK_SIZE, None)

    assert _FakeCudaTensor.cpu_calls == 0


def test_async_batch_fails_fast_on_nonblocking_copy_error_and_still_drains():
    layer = _fake_cuda_layers()[LAYER]
    event = _FakeEvent()
    stream = object()
    _FakeCudaTensor.cpu_calls = 0
    _PinnedTestTensor.copy_calls = []
    _PinnedTestTensor.fail_copy = True

    try:
        with (
            patch.object(torch, "empty", side_effect=_pinned_empty),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "Event", return_value=event),
            pytest.raises(
                RuntimeError, match="nonblocking pinned KV extraction failed"
            ),
            AsyncKVExtractionBatch() as batch,
        ):
            batch.extract(layer, [1, 2], BLOCK_SIZE, None)
    finally:
        _PinnedTestTensor.fail_copy = False

    assert _PinnedTestTensor.copy_calls == [True]
    assert _FakeCudaTensor.cpu_calls == 0
    assert event.record_calls == 1
    assert event.synchronize_calls == 1


def test_async_batch_rejects_stream_change_and_drains_pending_copy():
    layers = _fake_cuda_layers()
    event = _FakeEvent()
    streams = [object(), object()]
    _FakeCudaTensor.cpu_calls = 0
    _PinnedTestTensor.copy_calls = []
    _PinnedTestTensor.fail_copy = False

    with (
        patch.object(torch, "empty", side_effect=_pinned_empty),
        patch.object(torch.cuda, "current_stream", side_effect=streams),
        patch.object(torch.cuda, "Event", return_value=event),
        pytest.raises(RuntimeError, match="one CUDA stream per save batch"),
        AsyncKVExtractionBatch() as batch,
    ):
        batch.extract(layers[0], [1, 2], BLOCK_SIZE, None)
        batch.extract(layers[1], [1, 2], BLOCK_SIZE, None)

    assert _PinnedTestTensor.copy_calls == [True]
    assert _FakeCudaTensor.cpu_calls == 0
    assert event.record_calls == 1
    assert event.synchronize_calls == 1


def test_async_batch_preserves_copy_error_when_synchronization_also_fails():
    layer = _fake_cuda_layers()[LAYER]
    event = _FakeEvent(RuntimeError("injected event synchronization failure"))
    stream = object()
    _PinnedTestTensor.copy_calls = []
    _PinnedTestTensor.fail_copy = True

    try:
        with pytest.raises(
            RuntimeError, match="nonblocking pinned KV extraction failed"
        ) as exc_info:
            with (
                patch.object(torch, "empty", side_effect=_pinned_empty),
                patch.object(torch.cuda, "current_stream", return_value=stream),
                patch.object(torch.cuda, "Event", return_value=event),
                AsyncKVExtractionBatch() as batch,
            ):
                batch.extract(layer, [1, 2], BLOCK_SIZE, None)
    finally:
        _PinnedTestTensor.fail_copy = False

    assert event.record_calls == 1
    assert event.synchronize_calls == 1
    assert batch._closed
    assert any(
        "async KV extraction synchronization failure" in note
        and "injected event synchronization failure" in note
        for note in exc_info.value.__notes__
    )


def test_async_batch_normal_exit_raises_synchronization_error_and_closes():
    layer = _fake_cuda_layers()[LAYER]
    event = _FakeEvent(RuntimeError("injected event synchronization failure"))
    stream = object()
    _PinnedTestTensor.fail_copy = False

    with pytest.raises(
        RuntimeError, match="failed to synchronize async KV extraction"
    ) as exc_info:
        with (
            patch.object(torch, "empty", side_effect=_pinned_empty),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "Event", return_value=event),
            AsyncKVExtractionBatch() as batch,
        ):
            batch.extract(layer, [1, 2], BLOCK_SIZE, None)

    assert str(exc_info.value.__cause__) == "injected event synchronization failure"
    assert event.synchronize_calls == 1
    assert batch._closed
    with pytest.raises(RuntimeError, match="async KV extraction batch is closed"):
        with batch:
            pass


def test_async_batch_owns_pinned_storage_until_copy_drains():
    import gc
    import weakref

    layer = _fake_cuda_layers()[LAYER]
    stream = object()
    pinned_refs = []

    class LifetimeCheckingEvent(_FakeEvent):
        def synchronize(self):
            assert pinned_refs[0]() is not None
            super().synchronize()

    event = LifetimeCheckingEvent()
    _PinnedTestTensor.fail_copy = False
    with (
        patch.object(torch, "empty", side_effect=_pinned_empty),
        patch.object(torch.cuda, "current_stream", return_value=stream),
        patch.object(torch.cuda, "Event", return_value=event),
        AsyncKVExtractionBatch() as batch,
    ):
        pinned_refs.append(weakref.ref(batch.extract(layer, [1, 2], BLOCK_SIZE, None)))
        gc.collect()

    gc.collect()
    assert pinned_refs[0]() is None
    assert event.synchronize_calls == 1
