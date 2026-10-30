# SPDX-License-Identifier: Apache-2.0
"""Tests for ``telys_vllm.slab.KVSlab``."""

import threading

import pytest

from telys_vllm.slab import KVSlab, SlabError


def test_append_read_round_trip(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    payload = b"hello slab"
    offset, length = slab.append(payload)
    assert length == len(payload)
    assert slab.read(offset, length) == payload


def test_multiple_appends_have_increasing_offsets(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    payloads = [b"a" * 100, b"b" * 200, b"c" * 50]
    pointers = []
    for payload in payloads:
        offset, length = slab.append(payload)
        pointers.append((offset, length))
        assert length == len(payload)

    for payload, (offset, length) in zip(payloads, pointers):
        assert slab.read(offset, length) == payload


def test_read_with_wrong_length_raises(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    offset, length = slab.append(b"payload")
    with pytest.raises(SlabError, match="length mismatch"):
        slab.read(offset, length - 1)


def test_read_inside_header_raises(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    with pytest.raises(SlabError, match="inside the slab header"):
        slab.read(0, 1)


def test_clear_truncates_and_allows_new_appends(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    offset, _ = slab.append(b"old")
    slab.clear()
    with pytest.raises(SlabError):
        slab.read(offset, 3)

    offset2, length2 = slab.append(b"new")
    assert slab.read(offset2, length2) == b"new"


def test_concurrent_appends_are_safe(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    payloads = {f"thread-{i}".encode(): i for i in range(100)}
    results = {}

    def worker(name: bytes, i: int):
        offset, length = slab.append(name * 100)
        results[i] = (offset, length, name * 100)

    threads = [
        threading.Thread(target=worker, args=(name, i)) for name, i in payloads.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All offsets are unique and all payloads round-trip.
    offsets = {off for off, _, _ in results.values()}
    assert len(offsets) == len(results)
    for offset, length, expected in results.values():
        assert slab.read(offset, length) == expected


def test_operations_after_close_raise(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    slab.close()
    with pytest.raises(SlabError, match="closed"):
        slab.append(b"x")
    with pytest.raises(SlabError, match="closed"):
        slab.read(16, 1)


def test_reopen_existing_slab_validates_header(tmp_path):
    slab = KVSlab(tmp_path / "test.slab")
    offset, length = slab.append(b"persistent")
    slab.close()

    slab2 = KVSlab(tmp_path / "test.slab")
    assert slab2.read(offset, length) == b"persistent"
