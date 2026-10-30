# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`~telys_vllm.local_client.TelysLocalClient`.

The real Telys runtime is never loaded here.  Instead we stub ``sys.modules``
with a minimal in-memory Telys SDK and exercise the Protocol mapping:
get/put/delete/compact, bytes payload encoding, metadata passthrough, and the
login-missing error path.
"""

import hashlib
import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Any

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
        key = (where or {}).get("__key")
        row = self._rows.get(key)
        if row is None:
            return {"ids": [], "scores": []}
        return {"ids": [key], "scores": [1.0], "metadata": [dict(row)]}

    def ids(self, where=None):
        key = (where or {}).get("__key")
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

    record = client.get("ns/chunk/bb/1")
    assert record.value == b""
    assert record.metadata["vllm.slab_offset"] == "1234"
    assert record.metadata["meta"] == "x"


def test_chunk_length_tracks_longest_chunk(lc, monkeypatch):
    client = _make_client(lc, monkeypatch, TELYS_PATH="/tmp/stub-telys")
    client.put("ns/chunk/cc/1", b"v1", {})
    assert client.chunk_length("ns/chunk/cc") == 1
    client.put("ns/chunk/cc/3", b"v3", {})
    assert client.chunk_length("ns/chunk/cc") == 3
    client.delete("ns/chunk/cc/3")
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
