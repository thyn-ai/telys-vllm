# SPDX-License-Identifier: Apache-2.0
"""Write-behind ownership and persistence tests for fast tensor chunks."""

from __future__ import annotations

import gc
import threading
import weakref
from unittest.mock import patch

import pytest
import torch

from telys_vllm.codec import PayloadCodecError, TelysChunkCodec, TelysFastChunkCodec
from telys_vllm.gpu_ops import deserialize_kv_tensor, serialize_kv_tensor
from tests.fakes import make_hashes
from tests.test_store import FakeBlobStoreClient, make_store


def _make_fast_store(tmp_path, client, *, write_behind=True, **config_kwargs):
    return make_store(
        tmp_path,
        client,
        slab_path=None,
        blobstore_no_slab=True,
        write_behind=write_behind,
        **config_kwargs,
    )


def _layer_tensors() -> dict[str, torch.Tensor]:
    first = torch.arange(2 * 4 * 2 * 8, dtype=torch.bfloat16).reshape(2, 4, 2, 8)
    second = torch.arange(2 * 4 * 2 * 8, dtype=torch.bfloat16).reshape(2, 4, 2, 8)
    return {"layer.1": second + 100, "layer.0": first}


def test_fast_encoding_runs_on_write_worker_and_preserves_dedup_and_queue_bound(
    tmp_path,
):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client, max_pending_writes=1)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    caller_thread = threading.get_ident()
    encode_threads: list[int] = []
    original_encode = store._fast_chunk_codec.encode_chunk

    def record_thread(layer_tensors):
        encode_threads.append(threading.get_ident())
        return original_encode(layer_tensors)

    try:
        with patch.object(
            store._fast_chunk_codec, "encode_chunk", side_effect=record_thread
        ):
            store.save_chunk_fast(hashes[0], hashes, tensors)
            store.save_chunk_fast(hashes[0], hashes, tensors)
            store.drain()

        assert len(encode_threads) == 1
        assert encode_threads[0] != caller_thread
        assert client.put_count == 1
        assert store._write_queue.maxsize == 1
    finally:
        store.close()


def test_fast_write_behind_matches_synchronous_bytes_metadata_and_provenance(tmp_path):
    sync_client = FakeBlobStoreClient()
    async_client = FakeBlobStoreClient()
    sync_store = _make_fast_store(tmp_path, sync_client, write_behind=False)
    async_store = _make_fast_store(tmp_path, async_client, write_behind=True)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    extra_metadata = {"vllm.test": "parity", "request-kind": "continuation"}
    expected = TelysFastChunkCodec().encode_chunk(tensors)

    try:
        sync_key = sync_store.save_chunk_fast(
            hashes[0], hashes, tensors, extra_metadata=extra_metadata
        )
        sync_record = sync_client.get(sync_key)
        assert sync_record is not None

        async_key = async_store.save_chunk_fast(
            hashes[0], hashes, tensors, extra_metadata=extra_metadata
        )
        async_store.drain()
        async_record = async_client.get(async_key)
        assert async_record is not None

        assert async_key == sync_key
        assert async_record.value == sync_record.value == expected
        assert async_record.metadata == sync_record.metadata
        assert async_record.provenance == sync_record.provenance
        loaded = async_store.load_decoded_blocks_fast(hashes)
        assert len(loaded) == 1
        assert loaded[0].provenance == sync_record.provenance
        for layer_name, expected_tensor in tensors.items():
            assert torch.equal(loaded[0].layer_tensors[layer_name], expected_tensor)
    finally:
        async_store.close()
        sync_store.close()


def test_pending_fast_write_owns_tensors_until_worker_encoding_finishes(tmp_path):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    tensor_ref = weakref.ref(tensors["layer.0"])
    expected = TelysFastChunkCodec().encode_chunk(tensors)
    encode_started = threading.Event()
    finish_encode = threading.Event()
    original_encode = store._fast_chunk_codec.encode_chunk

    def gated_encode(layer_tensors):
        encode_started.set()
        if not finish_encode.wait(timeout=5):
            raise RuntimeError("timed out waiting to finish test encoding")
        return original_encode(layer_tensors)

    try:
        with patch.object(
            store._fast_chunk_codec, "encode_chunk", side_effect=gated_encode
        ) as encode_mock:
            key = store.save_chunk_fast(hashes[0], hashes, tensors)
            assert encode_started.wait(timeout=5)
            tensors.clear()
            gc.collect()
            assert tensor_ref() is not None
            finish_encode.set()
            store.drain()
            encode_mock.reset_mock()

        gc.collect()
        assert tensor_ref() is None
        record = client.get(key)
        assert record is not None
        assert record.value == expected
    finally:
        finish_encode.set()
        store.close()


def test_fast_worker_encoding_failure_surfaces_on_drain_and_allows_retry(tmp_path):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client)
    hashes = make_hashes(2)
    tensors = _layer_tensors()

    try:
        with patch.object(
            store._fast_chunk_codec,
            "encode_chunk",
            side_effect=RuntimeError("injected fast encode failure"),
        ):
            key = store.save_chunk_fast(hashes[0], hashes, tensors)
            with pytest.raises(RuntimeError, match="injected fast encode failure"):
                store.drain()

        assert store._pending_count == 0
        assert not store.contains(key)
        store.drain()

        store.save_chunk_fast(hashes[0], hashes, tensors)
        store.drain()
        assert client.put_count == 1
    finally:
        store.close()


def _put_metadata_free_chunk(store, client, hashes, payload):
    key = store.chunk_key(hashes[0], len(hashes), hashes[-1])
    client.put(key, payload, {})
    return key


def _legacy_payload(tensors):
    return TelysChunkCodec().encode_chunk(
        {
            name: [serialize_kv_tensor(block) for block in tensor]
            for name, tensor in tensors.items()
        }
    )


def _assert_fast_chunks_equal(loaded, tensors):
    assert len(loaded) == 1
    for name, expected in tensors.items():
        assert torch.equal(loaded[0].layer_tensors[name], expected)


def test_two_block_metadata_free_legacy_chunk_loads_as_fast_tensors(tmp_path):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client, write_behind=False)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    _put_metadata_free_chunk(store, client, hashes, _legacy_payload(tensors))

    loaded = store.load_decoded_blocks_fast(hashes)

    _assert_fast_chunks_equal(loaded, tensors)


def test_metadata_free_fast_chunk_loads_in_both_directions(tmp_path):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client, write_behind=False)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    payload = TelysFastChunkCodec().encode_chunk(tensors)
    _put_metadata_free_chunk(store, client, hashes, payload)

    fast_loaded = store.load_decoded_blocks_fast(hashes)
    legacy_loaded = store.load_decoded_blocks(hashes)

    _assert_fast_chunks_equal(fast_loaded, tensors)
    for block_idx, block in enumerate(legacy_loaded):
        for name, expected in tensors.items():
            actual = deserialize_kv_tensor(block.blobs[name])
            assert torch.equal(actual, expected[block_idx])


@pytest.mark.parametrize(
    "loader_name", ["load_decoded_blocks", "load_decoded_blocks_fast"]
)
def test_metadata_free_corrupt_chunk_is_rejected_in_both_directions(
    tmp_path, loader_name
):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client, write_behind=False)
    hashes = make_hashes(2)
    payload = TelysFastChunkCodec().encode_chunk(_layer_tensors()) + b"trailing"
    _put_metadata_free_chunk(store, client, hashes, payload)

    with pytest.raises(PayloadCodecError):
        getattr(store, loader_name)(hashes)


@pytest.mark.parametrize(
    "loader_name", ["load_decoded_blocks", "load_decoded_blocks_fast"]
)
def test_metadata_free_ambiguous_chunk_is_rejected(tmp_path, loader_name):
    client = FakeBlobStoreClient()
    store = _make_fast_store(tmp_path, client, write_behind=False)
    hashes = make_hashes(2)
    tensors = _layer_tensors()
    _put_metadata_free_chunk(store, client, hashes, b"\x02ambiguous")
    legacy_blocks = [
        {name: serialize_kv_tensor(tensor[idx]) for name, tensor in tensors.items()}
        for idx in range(len(hashes))
    ]

    with (
        patch.object(store._chunk_codec, "decode_chunk", return_value=legacy_blocks),
        patch.object(store._fast_chunk_codec, "decode_chunk", return_value=tensors),
        pytest.raises(PayloadCodecError, match="ambiguous"),
    ):
        getattr(store, loader_name)(hashes)
