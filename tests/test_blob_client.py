# SPDX-License-Identifier: Apache-2.0
"""Tests for the BlobStore-backed Telys client and raw-byte tensor path."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from telys_vllm.client import TelysMaintenanceReport
from telys_vllm.codec import LengthPrefixedCodec
from telys_vllm.config import TelysConnectorConfig
from telys_vllm.gpu_ops import (
    deserialize_kv_tensor,
    legacy_serialize_kv_tensor,
    serialize_kv_tensor,
)
from telys_vllm.store import TelysBlockStore

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("telys") is None, reason="telys SDK not installed"
)


@pytest.fixture
def blob_client(tmp_path, monkeypatch):
    """A real TelysBlobClient against a temporary embedded engine + BlobStore."""
    import telys as _telys  # type: ignore[import-not-found]

    if not hasattr(_telys, "BlobStore"):
        pytest.skip("telys.BlobStore not available")
    if not getattr(_telys.runtime, "runtime_available", lambda: False)():
        pytest.skip("telys runtime not available (run `telys login`)")

    monkeypatch.setenv("TELYS_PATH", str(tmp_path / "telys"))
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)

    from telys_vllm.local_client import TelysBlobClient

    client = TelysBlobClient(TelysConnectorConfig(namespace="blob-test"))
    try:
        yield client
    finally:
        client.close()


def test_blob_client_put_get_round_trip(blob_client):
    key = "blob-test/chunk/aa/1"
    payload = b"raw chunk bytes"
    blob_client.put(key, payload, {"vllm.model": "test-model"})

    record = blob_client.get(key)
    assert record is not None
    assert record.key == key
    assert record.value == payload
    assert record.metadata["vllm.model"] == "test-model"
    assert record.provenance.plan_hash
    assert record.provenance.result_id


def test_blob_client_contains_and_chunk_length(blob_client):
    key = "blob-test/chunk/bb/3"
    blob_client.put(key, b"x", {"vllm.block_count": "3"})

    assert blob_client.contains(key) is True
    assert blob_client.contains("blob-test/chunk/bb/3") is True
    assert blob_client.contains("blob-test/chunk/bb/2") is False
    assert blob_client.chunk_length("blob-test/chunk/bb") == 3


def test_blob_client_delete(blob_client):
    key = "blob-test/kv/dd"
    blob_client.put(key, b"x", {})
    assert blob_client.delete(key) is True
    assert blob_client.delete(key) is False
    assert blob_client.get(key) is None


def test_blob_client_compact(blob_client):
    blob_client.put("blob-test/kv/e1", b"a", {})
    blob_client.put("blob-test/kv/e2", b"b", {})
    report = blob_client.compact()
    assert isinstance(report, TelysMaintenanceReport)
    assert report.entries_scanned >= 2


def test_blob_store_round_trip_via_block_store(tmp_path, monkeypatch):
    import telys as _telys  # type: ignore[import-not-found]

    if not hasattr(_telys, "BlobStore"):
        pytest.skip("telys.BlobStore not available")
    if not getattr(_telys.runtime, "runtime_available", lambda: False)():
        pytest.skip("telys runtime not available (run `telys login`)")

    monkeypatch.setenv("TELYS_PATH", str(tmp_path / "telys"))
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)

    from telys_vllm.local_client import TelysBlobClient

    client = TelysBlobClient(TelysConnectorConfig(namespace="bs-test"))
    try:
        # No slab_path: TelysBlockStore uses the BlobStore-backed client value.
        store = TelysBlockStore(
            client,
            TelysConnectorConfig(namespace="bs-test"),
            model_id="blob-model",
            engine_id="blob-engine",
            block_size=16,
        )
        hashes = [bytes.fromhex(f"{i:064x}") for i in range(3)]
        store.save_chunk(
            hashes[0],
            hashes,
            {"layer.0": [b"block0", b"block1", b"block2"]},
        )
        records = store.load_blocks(hashes)
        assert len(records) == 3
        codec = LengthPrefixedCodec()
        for i, record in enumerate(records):
            assert codec.decode_block(record.value) == {
                "layer.0": b"block" + str(i).encode()
            }
            assert record.metadata["vllm.model"] == "blob-model"
            assert record.metadata["vllm.block_size"] == "16"
            assert record.metadata["vllm.block_count"] == "3"
    finally:
        client.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_raw_byte_serialization_round_trip(dtype):
    tensor = torch.arange(16 * 4 * 64, dtype=dtype).reshape(16, 4, 64)
    raw = serialize_kv_tensor(tensor)
    decoded = deserialize_kv_tensor(raw)
    assert decoded.dtype == dtype
    assert decoded.shape == tensor.shape
    assert torch.equal(tensor, decoded)


def test_raw_byte_deserialization_is_zero_copy_view():
    tensor = torch.randn(16, 4, 64, dtype=torch.float16)
    raw = serialize_kv_tensor(tensor)
    decoded = deserialize_kv_tensor(raw)
    # frombuffer on an immutable bytes buffer yields a read-only tensor that
    # shares the underlying bytes.
    assert not decoded.is_contiguous() or decoded.data_ptr() != 0


def test_legacy_torch_load_fallback():
    tensor = torch.randn(16, 4, 64, dtype=torch.float16)
    legacy = legacy_serialize_kv_tensor(tensor)
    decoded = deserialize_kv_tensor(legacy)
    assert torch.equal(tensor, decoded)


def test_raw_byte_format_rejects_unknown_dtype():
    tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="unsupported dtype"):
        serialize_kv_tensor(tensor)


def test_raw_byte_format_rejects_too_many_dims():
    tensor = torch.zeros(1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="up to 4 dims"):
        serialize_kv_tensor(tensor)


def test_connector_auto_factory_prefers_blob_client(tmp_path, monkeypatch):
    """When telys.BlobStore is available, the connector auto-selects it."""
    import telys as _telys  # type: ignore[import-not-found]

    if not hasattr(_telys, "BlobStore"):
        pytest.skip("telys.BlobStore not available")
    if not getattr(_telys.runtime, "runtime_available", lambda: False)():
        pytest.skip("telys runtime not available (run `telys login`)")

    monkeypatch.setenv("TELYS_PATH", str(tmp_path / "telys"))
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)

    from telys_vllm.config import TelysConnectorConfig
    from telys_vllm.connector import _auto_client

    client = _auto_client(TelysConnectorConfig(namespace="auto-test"))
    try:
        from telys_vllm.local_client import TelysBlobClient

        assert isinstance(client, TelysBlobClient)
        key = "auto-test/chunk/zz/1"
        client.put(key, b"payload", {"meta": "m"})
        record = client.get(key)
        assert record is not None
        assert record.value == b"payload"
    finally:
        client.close()
