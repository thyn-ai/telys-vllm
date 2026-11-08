# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`~telys_vllm.local_client.TelysLocalClient`.

The real Telys runtime is never loaded here.  Instead we stub ``sys.modules``
with a minimal in-memory Telys SDK and exercise the Protocol mapping:
get/put/delete/compact, bytes payload encoding, metadata passthrough, and the
login-missing error path.
"""

import base64
import hashlib
import importlib
import importlib.util
import json
import sys
from types import ModuleType
from typing import Any, ClassVar

import pytest

from telys_vllm.client import TelysMaintenanceReport, TelysRecord
from telys_vllm.config import TelysConnectorConfig

# ---------------------------------------------------------------------------
# Minimal stub Telys SDK
# ---------------------------------------------------------------------------


class _StubCollection:
    """In-memory collection that mirrors the subset of ``telys.engine.Collection``
    that ``TelysLocalClient`` calls."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.search_calls = 0
        self.ids_calls = 0
        self.full_ids_calls = 0

    def upsert(self, vectors, ids, metadata):
        for key, row in zip(ids, metadata):
            self._rows[key] = dict(row)

    def search(
        self,
        vector,
        top_k: int = 10,
        where=None,
        explain: bool = False,
        with_metadata: bool = False,
    ):
        self.search_calls += 1
        key = (where or {}).get("__key")
        row = self._rows.get(key)
        if row is None:
            return {"ids": [], "scores": []}
        return {"ids": [key], "scores": [1.0], "metadata": [dict(row)]}

    def ids(self, where=None):
        self.ids_calls += 1
        if where is None:
            self.full_ids_calls += 1
            return sorted(self._rows)
        key = where.get("__key")
        return [key] if key in self._rows else []

    def delete(self, ids):
        for key in ids:
            self._rows.pop(key, None)

    def compact(self):
        return {"entries_scanned": len(self._rows), "entries_dropped": 0}

    def stats(self):
        return {"rows": len(self._rows)}


class _StubEngine:
    def __init__(self, path: str) -> None:
        self.path = path
        self._cols: dict[str, _StubCollection] = {}

    def collections(self):
        return sorted(self._cols)

    def create_collection(
        self, name, dim, partition_by, *, filter_columns=(), **kwargs
    ):
        self.last_filter_columns = filter_columns
        col = _StubCollection()
        self._cols[name] = col
        return col

    def open_collection(self, name):
        return self._cols[name]


class _StubRemoteEngine(_StubEngine):
    def __init__(self, address: str, token: str | None = None) -> None:
        super().__init__(address)
        self.address = address
        self.token = token
        self.closed = False

    def close(self):
        self.closed = True


class _RuntimeNotInstalled(RuntimeError):
    pass


def _install_telys_stubs(monkeypatch):
    """Install a stub ``telys`` package into ``sys.modules`` and reload
    ``telys_vllm.local_client`` so it binds to the stub."""
    telys_pkg = ModuleType("telys")
    telys_pkg.__path__ = []  # make it a package so submodules bind as attributes
    telys_pkg.Telys = _StubEngine
    telys_pkg.runtime = ModuleType("telys.runtime")
    telys_pkg.runtime.RuntimeNotInstalled = _RuntimeNotInstalled

    client_mod = ModuleType("telys.client")
    client_mod.connect = lambda address, token=None: _StubRemoteEngine(address, token)

    # Explicitly attach the submodules so attribute access works even when the
    # import machinery's parent-module binding is skipped.
    telys_pkg.runtime = telys_pkg.runtime
    telys_pkg.client = client_mod

    monkeypatch.setitem(sys.modules, "telys", telys_pkg)
    monkeypatch.setitem(sys.modules, "telys.runtime", telys_pkg.runtime)
    monkeypatch.setitem(sys.modules, "telys.client", client_mod)

    import telys_vllm.local_client as lc

    importlib.reload(lc)
    return lc


@pytest.fixture
def lc(monkeypatch):
    """Reloaded ``telys_vllm.local_client`` bound to the stub SDK."""
    return _install_telys_stubs(monkeypatch)


def _make_client(lc, monkeypatch, **env):
    """Create a ``TelysLocalClient`` against the stub SDK."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = TelysConnectorConfig(namespace="test-ns")
    return lc.TelysLocalClient(config)


# ---------------------------------------------------------------------------
# Mapping tests
# ---------------------------------------------------------------------------


def test_put_and_get_round_trip(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/chunk/aa/1", b"", {"vllm.block_hash": "aa"})

    record = client.get("ns/chunk/aa/1")
    assert isinstance(record, TelysRecord)
    assert record.key == "ns/chunk/aa/1"
    # v4: payload lives in the slab; Telys record value is empty bytes.
    assert record.value == b""
    assert record.metadata["vllm.block_hash"] == "aa"
    # Internal bookkeeping columns stay hidden from the connector.
    assert "__key" not in record.metadata
    assert "vllm.payload" not in record.metadata


def test_get_returns_none_for_missing_key(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    assert client.get("ns/chunk/missing/1") is None


def test_metadata_only_record(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put(
        "ns/chunk/bb/1",
        b"ignored-by-v4",
        {"vllm.slab_offset": "1234", "vllm.slab_length": "56", "meta": "x"},
    )

    col = client._collection
    row = col._rows["ns/chunk/bb/1"]
    # v4: no base64 payload column; metadata is stored directly.
    assert "vllm.payload" not in row
    assert row["vllm.slab_offset"] == "1234"
    assert row["vllm.slab_length"] == "56"
    assert row["meta"] == "x"

    client._record_cache.clear()
    record = client.get("ns/chunk/bb/1")
    assert record.value == b""
    assert record.metadata["vllm.slab_offset"] == "1234"
    assert record.metadata["meta"] == "x"


def test_get_decodes_v2_meta_blob(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    key = "ns/chunk/v2/1"
    metadata = {"vllm.chunk_format": "legacy", "meta": "v2"}
    raw = json.dumps(
        {
            "payload": base64.b64encode(b"v2 payload").decode("ascii"),
            "metadata": metadata,
        }
    )
    client._collection._rows[key] = {
        "__key": key,
        client._legacy_meta_blob_col: raw,
    }

    record = client.get(key)

    assert record is not None
    assert record.value == b"v2 payload"
    assert record.metadata == metadata


def test_chunk_length_tracks_longest_chunk(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/chunk/cc/1", b"v1", {})
    assert client.chunk_length("ns/chunk/cc") == 1
    client.put("ns/chunk/cc/3", b"v3", {})
    assert client.chunk_length("ns/chunk/cc") == 3
    client.delete("ns/chunk/cc/3")
    assert client.chunk_length("ns/chunk/cc") == 1
    client.delete("ns/chunk/cc/1")
    assert client.chunk_length("ns/chunk/cc") == 0


def test_put_is_idempotent(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/kv/cc", b"v1", {})
    client.put("ns/kv/cc", b"v2", {"meta": "second"})
    assert client.get("ns/kv/cc").metadata["meta"] == "second"


def test_delete_existing_and_missing(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/kv/dd", b"x", {})
    assert client.delete("ns/kv/dd") is True
    assert client.delete("ns/kv/dd") is False
    assert client.get("ns/kv/dd") is None


def test_local_keys_returns_snapshot_and_tracks_delete(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/kv/one", b"", {})
    client.put("ns/kv/two", b"", {})

    snapshot = client.keys()
    assert snapshot == {"ns/kv/one", "ns/kv/two"}
    snapshot.clear()
    assert client.keys() == {"ns/kv/one", "ns/kv/two"}

    assert client.delete("ns/kv/one") is True
    assert client.keys() == {"ns/kv/two"}


def test_compact_parses_report(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/kv/e1", b"a", {})
    client.put("ns/kv/e2", b"b", {})
    report = client.compact()
    assert report == TelysMaintenanceReport(entries_scanned=2, entries_dropped=0)


def test_shared_engine_makes_writes_visible_across_clients(lc, monkeypatch):
    """Two connector roles in the same process must share one Telys collection
    and hot cache; otherwise scheduler lookups see 0% hits (the v1 bench bug)."""
    client1 = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client1.put("ns/kv/shared", b"payload", {"meta": "x"})

    # A second client for the same path/namespace must reuse the shared engine.
    client2 = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    assert client2._engine is client1._engine
    assert client2._collection is client1._collection

    # Visibility through the hot cache, with no Telys search needed.
    assert client2.contains("ns/kv/shared") is True
    record = client2.get("ns/kv/shared")
    assert record is not None
    assert record.value == b""
    assert record.metadata["meta"] == "x"


def test_close_is_idempotent_and_closes_remote(lc, monkeypatch):
    client = _make_client(
        lc, monkeypatch, TELYS_ADDRESS="/tmp/stub.sock", TELYS_TOKEN="tok"
    )
    assert isinstance(client._engine, _StubRemoteEngine)
    client.close()
    assert client._engine.closed is True
    assert client._closed is True


def test_operations_after_close_raise(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.close()
    with pytest.raises(lc.TelysLocalClientError, match="closed"):
        client.get("ns/kv/x")


def test_provenance_is_derived_from_key(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    key = "ns/kv/prov"
    client.put(key, b"v", {})
    record = client.get(key)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    assert record.provenance.plan_hash == digest[:32]
    assert record.provenance.result_id == digest[32:56]


def test_collection_name_derived_from_namespace(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    assert client._collection_name == "test_ns_kv"


def test_collection_name_override(lc, monkeypatch):
    monkeypatch.setenv("TELYS_COLLECTION", "my_kv")
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    assert client._collection_name == "my_kv"


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------


def test_missing_connection_raises(lc, monkeypatch):
    monkeypatch.delenv("TELYS_PATH", raising=False)
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)
    with pytest.raises(lc.TelysLocalClientError, match="TELYS_PATH"):
        lc.TelysLocalClient(TelysConnectorConfig())


def test_invalid_kv_dim_raises(lc, monkeypatch):
    with pytest.raises(lc.TelysLocalClientError, match="TELYS_KV_DIM"):
        _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys", TELYS_KV_DIM="abc")


def test_non_positive_kv_dim_raises(lc, monkeypatch):
    with pytest.raises(lc.TelysLocalClientError, match="positive"):
        _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys", TELYS_KV_DIM="0")


def test_runtime_not_installed_gives_login_message(lc, monkeypatch):
    """When Telys raises RuntimeNotInstalled, the adapter must surface the
    exact ``telys login`` guidance."""
    telys_pkg = ModuleType("telys")
    telys_pkg.__path__ = []
    telys_pkg.Telys = lambda path: (_ for _ in ()).throw(
        _RuntimeNotInstalled("no runtime")
    )
    runtime_mod = ModuleType("telys.runtime")
    runtime_mod.RuntimeNotInstalled = _RuntimeNotInstalled
    telys_pkg.runtime = runtime_mod

    client_mod = ModuleType("telys.client")
    client_mod.connect = lambda address, token=None: (_ for _ in ()).throw(
        _RuntimeNotInstalled("no runtime")
    )
    telys_pkg.client = client_mod

    monkeypatch.setitem(sys.modules, "telys", telys_pkg)
    monkeypatch.setitem(sys.modules, "telys.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "telys.client", client_mod)

    import telys_vllm.local_client as lc2

    importlib.reload(lc2)
    monkeypatch.setenv("TELYS_PATH", "/tmp/stub-telys")
    with pytest.raises(lc2.TelysLocalClientError, match="telys login"):
        lc2.TelysLocalClient(TelysConnectorConfig())


# ---------------------------------------------------------------------------
# Integration smoke test (skipped unless real Telys + login are present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("telys") is None, reason="telys SDK not installed"
)
def test_integration_round_trip_with_real_telys(tmp_path, monkeypatch):
    import telys as _telys  # type: ignore[import-not-found]

    if not getattr(_telys.runtime, "runtime_available", lambda: False)():
        pytest.skip("telys runtime not available (run `telys login`)")

    monkeypatch.setenv("TELYS_PATH", str(tmp_path / "real-telys"))
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)

    # Import the real adapter against the real SDK.
    import telys_vllm.local_client as lc

    importlib.reload(lc)

    client = lc.TelysLocalClient(TelysConnectorConfig(namespace="integ"))
    key = "integ/kv/01"
    client.put(key, b"ignored", {"vllm.model": "integ-model"})

    record = client.get(key)
    assert record is not None
    # v4: payload lives in the slab; Telys stores only metadata.
    assert record.value == b""
    assert record.metadata["vllm.model"] == "integ-model"
    assert record.provenance.plan_hash
    assert record.provenance.result_id

    assert client.delete(key) is True
    assert client.get(key) is None

    report = client.compact()
    assert isinstance(report, TelysMaintenanceReport)

    client.close()


class _StubBlobStore:
    """Path-scoped in-memory BlobStore with durable payload and metadata."""

    _stores: ClassVar[dict[str, dict[str, tuple[bytes, dict[str, Any]]]]] = {}

    def __init__(self, path: str) -> None:
        self.records = self._stores.setdefault(path, {})
        self.last_sync = False
        self.last_delete_sync = False
        self.sync_calls = 0
        self.get_calls = 0
        self.keys_calls = 0
        self.closed = False

    def put(self, key, payload, metadata=None, *, sync=False):
        self.last_sync = sync
        self.records[key] = (bytes(payload), dict(metadata or {}))

    def get(self, key):
        self.get_calls += 1
        try:
            payload, metadata = self.records[key]
        except KeyError:
            raise KeyError(key) from None
        return payload, dict(metadata)

    def keys(self):
        self.keys_calls += 1
        return set(self.records)

    def contains(self, key):
        return key in self.records

    def delete(self, key, *, sync=False):
        self.last_delete_sync = sync
        self.records.pop(key, None)

    def sync(self):
        self.sync_calls += 1

    def close(self):
        self.closed = True


def _make_blob_client(lc, monkeypatch, path):
    monkeypatch.setenv("TELYS_PATH", str(path))
    monkeypatch.delenv("TELYS_ADDRESS", raising=False)
    sys.modules["telys"].BlobStore = _StubBlobStore
    return lc.TelysBlobClient(TelysConnectorConfig(namespace="test-ns"))


def test_blob_clients_share_store_across_overlapping_lifetimes(
    lc, monkeypatch, tmp_path
):
    first = _make_blob_client(lc, monkeypatch, tmp_path / "shared")
    second = _make_blob_client(lc, monkeypatch, tmp_path / "shared")
    shared_store = first._blob_store
    first_key = "test-ns/kv/first"
    second_key = "test-ns/kv/second"

    try:
        assert second._blob_store is shared_store
        first.put(first_key, b"first payload", {"meta": "first"})
        first.close()
        assert shared_store.closed is False

        record = second.get(first_key)
        assert record is not None
        assert record.value == b"first payload"
        assert record.metadata["meta"] == "first"

        second.put(second_key, b"second payload", {"meta": "second"})
        assert shared_store.get(second_key) == (
            b"second payload",
            {"meta": "second"},
        )
    finally:
        if not first._closed:
            first.close()
        second.close()

    assert shared_store.closed is True
    assert shared_store.sync_calls == 1


def test_blob_delete_invalidates_overlapping_client_caches(lc, monkeypatch, tmp_path):
    first = _make_blob_client(lc, monkeypatch, tmp_path / "overlap-delete")
    second = _make_blob_client(lc, monkeypatch, tmp_path / "overlap-delete")
    key = "test-ns/chunk/overlap/2"

    try:
        first.put(key, b"payload", {"meta": "first"})
        assert first.get(key) is not None
        assert first.contains(key) is True

        assert second.delete(key) is True

        assert first.get(key) is None
        assert first.contains(key) is False
        assert key not in first._key_set
        assert key not in first._record_cache
        assert first.chunk_length("test-ns/chunk/overlap") == 0
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("blob_store_enabled", [True, False])
def test_blob_get_decodes_v2_meta_blob(lc, monkeypatch, tmp_path, blob_store_enabled):
    if blob_store_enabled:
        client = _make_blob_client(lc, monkeypatch, tmp_path / "v2")
    else:
        monkeypatch.setenv("TELYS_ADDRESS", "/tmp/v2-no-blob-store.sock")
        monkeypatch.delenv("TELYS_PATH", raising=False)
        sys.modules["telys"].BlobStore = _StubBlobStore
        client = lc.TelysBlobClient(TelysConnectorConfig(namespace="test-ns"))
    key = "test-ns/chunk/v2/1"
    metadata = {"vllm.chunk_format": "legacy", "meta": "v2"}
    raw = json.dumps(
        {
            "payload": base64.b64encode(b"v2 payload").decode("ascii"),
            "metadata": metadata,
        }
    )
    client._collection._rows[key] = {
        "__key": key,
        "vllm.meta_blob": raw,
    }

    try:
        if client._blob_store is not None:
            assert client._blob_store.contains(key) is False
        record = client.get(key)

        assert record is not None
        assert record.value == b"v2 payload"
        assert record.metadata == metadata
        assert lc._LEGACY_META_BLOB_COLUMN in client._engine.last_filter_columns
        assert lc._PAYLOAD_COLUMN in client._engine.last_filter_columns
    finally:
        client.close()


def test_blob_get_populates_overlapping_client_caches(lc, monkeypatch, tmp_path):
    first = _make_blob_client(lc, monkeypatch, tmp_path / "shared-get")
    second = _make_blob_client(lc, monkeypatch, tmp_path / "shared-get")
    key = "test-ns/chunk/shared-get/1"

    try:
        first.put(key, b"payload", {"meta": "x"})
        second._record_cache.clear()
        assert key not in second._record_cache

        assert first.get(key) is not None
        assert second._record_cache[key].value == b"payload"
    finally:
        first.close()
        second.close()


def test_blob_close_prunes_empty_client_registry(lc, monkeypatch, tmp_path):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "registry-cleanup")
    shared_key = client._shared_key
    assert shared_key in lc._BLOB_CLIENT_REGISTRY

    client.close()

    assert shared_key not in lc._BLOB_CLIENT_REGISTRY


def test_blob_delete_removes_metadata_blob_and_cached_keys(lc, monkeypatch, tmp_path):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "delete")
    key = "test-ns/chunk/delete/1"
    client.put(key, b"payload", {"meta": "x"})
    assert client.get(key) is not None

    snapshot = client.keys()
    assert snapshot == {key}
    snapshot.clear()
    assert client.keys() == {key}

    assert client.delete(key) is True
    assert client._blob_store.last_delete_sync is True
    assert key not in client._blob_store.records
    assert key not in client._collection._rows
    assert client.keys() == set()
    assert client.get(key) is None
    assert client.delete(key) is False
    client.close()


def test_blob_delete_handles_blob_only_record_without_metadata_collection(
    lc, monkeypatch, tmp_path
):
    path = tmp_path / "blob-only"
    key = "test-ns/chunk/blob-only/1"
    blob_store = _StubBlobStore(f"{path}_blobs")
    blob_store.put(key, b"blob only", {"meta": "blob"})
    client = _make_blob_client(lc, monkeypatch, path)
    blob_store = client._blob_store

    def missing_metadata_ids(where=None):
        if where is not None:
            raise RuntimeError("unknown filter column: __key")
        return []

    monkeypatch.setattr(client._collection, "ids", missing_metadata_ids)

    assert client.delete(key) is True
    assert blob_store.last_delete_sync is True
    assert key not in blob_store.records
    assert client.keys() == set()
    assert client.get(key) is None
    client.close()


def test_blob_delete_propagates_failure_and_clears_caches(lc, monkeypatch, tmp_path):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "delete-failure")
    key = "test-ns/chunk/failure/1"
    client.put(key, b"payload", {"meta": "x"})
    assert client.get(key) is not None
    blob_store = client._blob_store

    def fail_delete(delete_key, *, sync=False):
        blob_store.last_delete_sync = sync
        raise RuntimeError("injected blob delete failure")

    monkeypatch.setattr(blob_store, "delete", fail_delete)

    with pytest.raises(RuntimeError, match="injected blob delete failure"):
        client.delete(key)
    assert blob_store.last_delete_sync is True
    assert key in blob_store.records
    assert key not in client._key_set
    assert key not in client._record_cache
    assert client._chunk_lengths == {}
    client.close()


def test_blob_restart_uses_durable_keys_and_metadata(lc, monkeypatch):
    monkeypatch.setenv("TELYS_PATH", "/tmp/blob-restart-metadata")
    sys.modules["telys"].BlobStore = _StubBlobStore
    config = TelysConnectorConfig(namespace="test-ns")
    key = "test-ns/chunk/aa/2"

    first = lc.TelysBlobClient(config)
    first.put(
        key,
        b"payload",
        {"vllm.chunk_format": "fast", "vllm.block_count": "2"},
    )
    assert first._blob_store.last_sync is True
    first.close()

    # The fresh collection has no rows, matching a worker whose embedded engine
    # metadata is unavailable even though its BlobStore records are durable.
    second = lc.TelysBlobClient(config)
    assert second.chunk_length("test-ns/chunk/aa") == 2
    record = second.get(key)
    assert record is not None
    assert record.value == b"payload"
    assert record.metadata["vllm.chunk_format"] == "fast"
    assert record.metadata["vllm.block_count"] == "2"
    second.close()


def test_blob_restart_decodes_legacy_chunk_without_metadata(lc, monkeypatch):
    import torch

    from telys_vllm.codec import TelysFastChunkCodec
    from telys_vllm.store import TelysBlockStore

    monkeypatch.setenv("TELYS_PATH", "/tmp/blob-restart-legacy")
    sys.modules["telys"].BlobStore = _StubBlobStore
    block_hash = b"\xaa" * 32
    key = f"test-ns/chunk/{block_hash.hex()}/{block_hash.hex()}/1"
    tensor = torch.arange(16 * 4 * 256, dtype=torch.bfloat16).reshape(1, 16, 4, 256)
    payload = TelysFastChunkCodec().encode_chunk({"layer.0": tensor})
    _StubBlobStore("/tmp/blob-restart-legacy_blobs").put(key, payload)

    client = lc.TelysBlobClient(TelysConnectorConfig(namespace="test-ns"))
    store = TelysBlockStore(
        client,
        TelysConnectorConfig(namespace="test-ns"),
        model_id="model",
        engine_id="engine",
        block_size=16,
    )
    assert store.lookup_prefix([block_hash]) == 1
    chunks = store.load_decoded_blocks_fast([block_hash])
    assert len(chunks) == 1
    assert torch.equal(chunks[0].layer_tensors["layer.0"], tensor)
    client.close()


class _IterationCountingSet(set[str]):
    def __init__(self, values) -> None:
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _complete_metadata(**overrides):
    metadata = {
        "vllm.block_start_hash": "aa",
        "vllm.block_count": "2",
        "vllm.block_size": "16",
        "vllm.model": "model",
        "vllm.engine_id": "engine",
    }
    metadata.update(overrides)
    return metadata


def _chunk_key(key_format: str, length: int = 2, end_hash: str = "bb") -> str:
    start_key = "test-ns/chunk/aa"
    if key_format == "chain":
        return f"{start_key}/{end_hash}/{length}"
    return f"{start_key}/{length}"


def _make_index_client(lc, monkeypatch, tmp_path, client_kind: str, remote=False):
    if client_kind == "blob":
        return _make_blob_client(lc, monkeypatch, tmp_path / "blob")
    if remote:
        monkeypatch.setenv("TELYS_ADDRESS", str(tmp_path / "remote.sock"))
        monkeypatch.delenv("TELYS_PATH", raising=False)
        return lc.TelysLocalClient(TelysConnectorConfig(namespace="test-ns"))
    return _make_client(lc, monkeypatch, TELYS_PATH=str(tmp_path / "local"))


@pytest.mark.parametrize("client_kind", ["local", "blob"])
def test_content_digest_survives_close_reopen(lc, monkeypatch, tmp_path, client_kind):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    engine = client._engine
    key = _chunk_key("chain")
    digest = hashlib.sha256(b"payload").hexdigest()
    metadata = {"vllm.content_sha256": digest}
    if client_kind == "blob":
        metadata = _complete_metadata(**metadata)

    try:
        client.put(key, b"payload", metadata)
        assert "vllm.content_sha256" in engine.last_filter_columns
        client.close()

        monkeypatch.setattr(sys.modules["telys"], "Telys", lambda _: engine)
        client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
        client._collection.search_calls = 0
        record = client.get(key)

        assert record is not None
        assert record.metadata["vllm.content_sha256"] == digest
        if client_kind == "blob":
            assert record.metadata == metadata
            assert client._collection.search_calls == 0
    finally:
        if not client._closed:
            client.close()


@pytest.mark.parametrize(
    "digest_override",
    [
        pytest.param({}, id="absent"),
        pytest.param({"vllm.content_sha256": "not-a-sha256"}, id="malformed"),
    ],
)
def test_blob_read_preserves_digest_state_for_connector(
    lc, monkeypatch, tmp_path, digest_override
):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "digest-state")
    key = _chunk_key("chain")
    metadata = _complete_metadata(**digest_override)
    client._blob_store.put(key, b"payload", metadata)

    try:
        record = client.get(key)

        assert record is not None
        assert record.metadata == metadata
        assert record.metadata.get("vllm.content_sha256") == digest_override.get(
            "vllm.content_sha256"
        )
    finally:
        client.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
@pytest.mark.parametrize("key_format", ["chain", "legacy"])
def test_chunk_candidates_indexed_hit_avoids_reseed_and_key_scan(
    lc, monkeypatch, tmp_path, client_kind, key_format
):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    key = _chunk_key(key_format)
    try:
        client.put(key, b"payload", {"meta": "first"})
        client.put(_chunk_key(key_format, end_hash="cc").replace("/aa/", "/ab/"), b"", {})
        counting_keys = _IterationCountingSet(client._key_set)
        client._key_set = counting_keys
        full_ids_calls = client._collection.full_ids_calls
        blob_keys_calls = client._blob_store.keys_calls if client_kind == "blob" else 0

        assert client.chunk_candidates("test-ns/chunk/aa") == {2: key}
        assert counting_keys.iterations == 0
        assert client._collection.full_ids_calls == full_ids_calls
        if client_kind == "blob":
            assert client._blob_store.keys_calls == blob_keys_calls
    finally:
        client.close()


@pytest.mark.parametrize("remote", [False, True])
def test_local_second_client_uses_coherent_index_without_reseed(
    lc, monkeypatch, tmp_path, remote
):
    first = _make_index_client(lc, monkeypatch, tmp_path, "local", remote=remote)
    second = _make_index_client(lc, monkeypatch, tmp_path, "local", remote=remote)
    key = _chunk_key("chain")
    try:
        first.put(key, b"payload", {"meta": "shared"})
        assert second.chunk_length("test-ns/chunk/aa") == 2
        full_ids_calls = second._collection.full_ids_calls

        assert second.chunk_candidates("test-ns/chunk/aa") == {2: key}
        assert second._collection.full_ids_calls == full_ids_calls
        assert second.get(key).metadata["meta"] == "shared"
        first.put(key, b"overwritten", {"meta": "overwritten"})
        assert second.get(key).metadata["meta"] == "overwritten"
        assert second._collection.search_calls == 0
        assert first.delete(key) is True
        assert second.chunk_length("test-ns/chunk/aa") == 0
        assert second.chunk_candidates("test-ns/chunk/aa") == {}

        first.put(key, b"reinserted", {"meta": "reinserted"})
        assert second.chunk_length("test-ns/chunk/aa") == 2
        full_ids_calls = second._collection.full_ids_calls
        assert second.chunk_candidates("test-ns/chunk/aa") == {2: key}
        assert second._collection.full_ids_calls == full_ids_calls
    finally:
        first.close()
        second.close()


def test_blob_second_client_uses_index_and_complete_metadata_without_search(
    lc, monkeypatch, tmp_path
):
    first = _make_blob_client(lc, monkeypatch, tmp_path / "shared-fast-path")
    second = _make_blob_client(lc, monkeypatch, tmp_path / "shared-fast-path")
    key = _chunk_key("chain")
    metadata = _complete_metadata(meta="complete")
    try:
        first.put(key, b"payload", metadata)
        assert second.chunk_length("test-ns/chunk/aa") == 2
        full_ids_calls = second._collection.full_ids_calls
        blob_keys_calls = second._blob_store.keys_calls
        assert second.chunk_candidates("test-ns/chunk/aa") == {2: key}
        assert second._collection.full_ids_calls == full_ids_calls
        assert second._blob_store.keys_calls == blob_keys_calls

        second._record_cache.clear()
        second._collection.search_calls = 0
        second._blob_store.get_calls = 0
        record = second.get(key)

        assert record is not None
        assert record.value == b"payload"
        assert record.metadata == metadata
        assert second._blob_store.get_calls == 1
        assert second._collection.search_calls == 0
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert record.provenance.plan_hash == digest[:32]
        assert record.provenance.result_id == digest[32:56]
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
@pytest.mark.parametrize("key_format", ["chain", "legacy"])
def test_chunk_index_delete_reinsert_and_overwrite(
    lc, monkeypatch, tmp_path, client_kind, key_format
):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    key = _chunk_key(key_format)
    longer_key = _chunk_key(key_format, length=4, end_hash="cc")
    start_key = "test-ns/chunk/aa"
    try:
        client.put(key, b"first", {"meta": "first"})
        client.put(key, b"second", {"meta": "second"})
        client.put(longer_key, b"longer", {"meta": "longer"})
        assert client.chunk_candidates(start_key) == {2: key, 4: longer_key}
        assert client.get(key).metadata["meta"] == "second"

        assert client.delete(longer_key) is True
        assert client.chunk_candidates(start_key) == {2: key}
        assert client.chunk_length(start_key) == 2
        assert client.delete(key) is True
        assert client.chunk_candidates(start_key) == {}
        assert client.chunk_length(start_key) == 0

        client.put(key, b"reinserted", {"meta": "reinserted"})
        assert client.chunk_candidates(start_key) == {2: key}
        record = client.get(key)
        assert record is not None
        assert record.metadata["meta"] == "reinserted"
        assert record.value == (b"reinserted" if client_kind == "blob" else b"")
    finally:
        client.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
def test_chunk_index_prefers_chain_key_and_restores_legacy_after_delete(
    lc, monkeypatch, tmp_path, client_kind
):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    legacy_key = _chunk_key("legacy")
    chain_key = _chunk_key("chain")
    start_key = "test-ns/chunk/aa"
    try:
        client.put(legacy_key, b"legacy", {})
        client.put(chain_key, b"chain", {})
        assert client.chunk_candidates(start_key) == {2: chain_key}

        assert client.delete(chain_key) is True
        assert client.chunk_candidates(start_key) == {2: legacy_key}
        client.put(chain_key, b"reinserted", {})
        assert client.chunk_candidates(start_key) == {2: chain_key}
    finally:
        client.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
def test_chunk_candidates_rebuilds_stale_index(
    lc, monkeypatch, tmp_path, client_kind
):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    key = _chunk_key("chain")
    start_key = "test-ns/chunk/aa"
    try:
        client.put(key, b"payload", {})
        client._chunk_key_index[start_key] = {99: "stale/chunk/key/99"}
        client._chunk_lengths[start_key] = 99
        assert client.chunk_candidates(start_key) == {2: key}

        client._chunk_key_index.clear()
        client._chunk_lengths.clear()
        assert client.chunk_candidates(start_key) == {2: key}
        assert client.chunk_length(start_key) == 2
    finally:
        client.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
def test_chunk_candidates_propagates_seed_backend_failure(
    lc, monkeypatch, tmp_path, client_kind
):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    try:
        def fail_ids(where=None):
            raise RuntimeError("injected ids failure")

        monkeypatch.setattr(client._collection, "ids", fail_ids)
        with pytest.raises(lc.TelysLocalClientError, match="Failed to seed") as exc_info:
            client.chunk_candidates("test-ns/chunk/missing")
        assert isinstance(exc_info.value.__cause__, RuntimeError)
    finally:
        client.close()


def test_blob_get_propagates_blob_store_failure(lc, monkeypatch, tmp_path):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "blob-get-failure")
    key = _chunk_key("chain")
    client.put(key, b"payload", _complete_metadata())
    client._record_cache.clear()

    def fail_get(failed_key):
        raise RuntimeError("injected blob get failure")

    monkeypatch.setattr(client._blob_store, "get", fail_get)
    with pytest.raises(RuntimeError, match="injected blob get failure"):
        client.get(key)
    assert client._collection.search_calls == 0
    client.close()


def test_blob_get_incomplete_metadata_propagates_search_failure(
    lc, monkeypatch, tmp_path
):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "search-failure")
    key = _chunk_key("chain")
    client._blob_store.put(key, b"payload", {})

    def fail_search(*args, **kwargs):
        raise RuntimeError("injected collection search failure")

    monkeypatch.setattr(client._collection, "search", fail_search)
    with pytest.raises(RuntimeError, match="injected collection search failure"):
        client.get(key)
    client.close()


@pytest.mark.parametrize("metadata_case", ["empty", "partial", "zero", "wrong_type"])
def test_blob_get_incomplete_or_invalid_metadata_uses_collection_fallback(
    lc, monkeypatch, tmp_path, metadata_case
):
    client = _make_blob_client(lc, monkeypatch, tmp_path / metadata_case)
    key = _chunk_key("chain")
    metadata = {
        "empty": {},
        "partial": {"meta": "partial"},
        "zero": _complete_metadata(**{"vllm.block_count": "0"}),
        "wrong_type": _complete_metadata(**{"vllm.model": None}),
    }[metadata_case]
    client._blob_store.records[key] = (b"payload", metadata)
    client._collection.search_calls = 0

    record = client.get(key)

    assert record is not None
    assert record.value == b"payload"
    assert record.metadata == metadata
    assert client._collection.search_calls == 1
    client.close()


def test_blob_get_malformed_metadata_does_not_bypass_fallback_errors(
    lc, monkeypatch, tmp_path
):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "malformed")
    key = _chunk_key("chain")
    monkeypatch.setattr(client._blob_store, "get", lambda _: (b"payload", None))
    client._collection.search_calls = 0

    with pytest.raises(TypeError):
        client.get(key)
    assert client._collection.search_calls == 1
    client.close()


def test_blob_get_legacy_collection_record_wins_when_blob_has_legacy_metadata(
    lc, monkeypatch, tmp_path
):
    client = _make_blob_client(lc, monkeypatch, tmp_path / "legacy-fast-path-guard")
    key = _chunk_key("chain")
    legacy_metadata = {"vllm.chunk_format": "legacy", "meta": "v2"}
    raw = json.dumps(
        {
            "payload": base64.b64encode(b"v2 payload").decode("ascii"),
            "metadata": legacy_metadata,
        }
    )
    blob_metadata = _complete_metadata(**{lc._LEGACY_META_BLOB_COLUMN: raw})
    client._blob_store.put(key, b"blob payload", blob_metadata)
    client._collection._rows[key] = {
        "__key": key,
        lc._LEGACY_META_BLOB_COLUMN: raw,
    }
    client._collection.search_calls = 0

    record = client.get(key)

    assert record is not None
    assert record.value == b"v2 payload"
    assert record.metadata == legacy_metadata
    assert client._collection.search_calls == 1
    client.close()


@pytest.mark.parametrize("client_kind", ["local", "blob"])
def test_close_resets_chunk_index(lc, monkeypatch, tmp_path, client_kind):
    client = _make_index_client(lc, monkeypatch, tmp_path, client_kind)
    client.put(_chunk_key("chain"), b"payload", {})
    assert client._chunk_key_index

    client.close()

    assert client._chunk_key_index == {}
    assert client._chunk_lengths == {}
