# SPDX-License-Identifier: Apache-2.0
"""Real Telys SDK binding for the :class:`~telys_vllm.client.TelysClient` Protocol.

Telys is a vector-memory engine, not a native key/value store, so this adapter
stores each contiguous run of KV blocks as one collection row keyed by the
run's start block hash.  The row's vector is a fixed zero vector (dimension is
configurable via ``TELYS_KV_DIM``); the run payload is stored as raw bytes in a
metadata column (``vllm.payload``).

To make the connector practical, this client keeps a process-local write-through
hot cache:

* A shared ``Telys`` engine / ``Collection`` instance is reused across all
  connector roles in the same process.  This is required because vLLM constructs
  separate scheduler-side and worker-side connector instances; without sharing,
  each instance would see a different in-memory collection view and external
  cache hits would read as 0%.
* A ``set`` of known chunk keys, a ``dict`` of cached records, and a
  ``chunk_length`` map let prefix lookups and loads run at in-process dict speed
  after the first write/read.

Configuration is read from environment variables:

* ``TELYS_PATH``     — local on-disk engine directory (embedded ``Telys(path)``).
* ``TELYS_ADDRESS``  — remote self-host server (``telys.client.connect``).
* ``TELYS_TOKEN``    — access token for a TCP self-host server.
* ``TELYS_KV_DIM``   — zero-vector dimension for the backing collection (default 1).
* ``TELYS_COLLECTION`` — override the collection name derived from the namespace.

At least one of ``TELYS_PATH`` or ``TELYS_ADDRESS`` must be set.  Creating or
opening the collection loads the Telys runtime, which requires the one-time
free ``telys login`` device authorization.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from telys_vllm.client import (
    TelysClient,
    TelysMaintenanceReport,
    TelysProvenance,
    TelysRecord,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from telys_vllm.config import TelysConnectorConfig


class TelysLocalClientError(RuntimeError):
    """Raised when the local Telys client cannot be created or used."""


_KEY_COLUMN = "__key"
_PAYLOAD_COLUMN = "vllm.payload"
_LEGACY_META_BLOB_COLUMN = "vllm.meta_blob"
_DEFAULT_KV_DIM = 1

# Process-global shared engines/collections.  vLLM constructs separate scheduler
# and worker connectors; sharing one collection object inside a process makes
# writes visible to reads immediately and keeps the hot cache coherent.
# Map: (path_or_empty, address_or_empty, collection_name) -> (engine, collection, ref_count, remote)
_SHARED: dict[tuple[str, str, str], tuple[Any, Any, int, bool]] = {}
_SHARED_LOCK = threading.Lock()


def _derived_provenance(key: str) -> TelysProvenance:
    """Deterministic provenance for rows Telys does not annotate itself.

    The real Telys SDK does not expose ``plan_hash`` / ``result_id`` on reads,
    so the adapter derives a stable pair from the key.  This keeps the
    connector's audit/observability contract intact without inventing
    untraceable values.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return TelysProvenance(plan_hash=digest[:32], result_id=digest[32:56])


def _zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def _collection_name(namespace: str, override: str | None) -> str:
    if override:
        return override
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", namespace).strip("_")
    return f"{safe or 'telys_vllm'}_kv"


def _decode_legacy_blob(key: str, raw: str) -> TelysRecord:
    """Decode a v2-era base64-JSON metadata blob (kept for migration)."""
    blob = json.loads(raw)
    payload_b64 = blob["payload"]
    connector_metadata = dict(blob.get("metadata") or {})
    value = base64.b64decode(payload_b64, validate=True)
    return TelysRecord(
        key=key,
        value=value,
        provenance=_derived_provenance(key),
        metadata=connector_metadata,
    )


def _chunk_start_key(chunk_key: str) -> str:
    """Strip the trailing block-count segment from a chunk key."""
    return chunk_key.rsplit("/", 1)[0]


class TelysLocalClient(TelysClient):
    """Adapter from the connector's :class:`TelysClient` Protocol to Telys.

    The backing Telys collection is created or opened during construction.
    That is the point at which the signed on-device runtime is loaded; if
    ``telys login`` has not been run, construction fails fast.

    The SDK is imported inside ``__init__`` so the adapter class can be
    imported and inspected without the runtime being installed.
    """

    def __init__(self, config: TelysConnectorConfig) -> None:
        path = os.environ.get("TELYS_PATH", "").strip() or ""
        address = os.environ.get("TELYS_ADDRESS", "").strip() or ""
        token = os.environ.get("TELYS_TOKEN") or None

        if not path and not address:
            raise TelysLocalClientError(
                "TelysLocalClient needs a connection. Set TELYS_PATH "
                "(embedded engine directory) or TELYS_ADDRESS (self-host server). "
                "Run `telys login` once, it's free, device authorization, "
                "to make the runtime available."
            )

        dim_raw = os.environ.get("TELYS_KV_DIM", str(_DEFAULT_KV_DIM)).strip()
        try:
            self._dim = int(dim_raw)
        except ValueError as exc:
            raise TelysLocalClientError(
                f"TELYS_KV_DIM must be an integer, got {dim_raw!r}"
            ) from exc
        if self._dim <= 0:
            raise TelysLocalClientError(
                f"TELYS_KV_DIM must be positive, got {self._dim}"
            )

        collection_override = os.environ.get("TELYS_COLLECTION") or None
        self._collection_name = _collection_name(config.namespace, collection_override)
        self._key_col = _KEY_COLUMN
        self._payload_col = _PAYLOAD_COLUMN
        self._legacy_meta_blob_col = _LEGACY_META_BLOB_COLUMN
        self._shared_key = (path, address, self._collection_name)
        self._closed = False
        self._remote = False
        self._engine: Any = None
        self._collection: Any = None

        import telys
        import telys.client
        import telys.runtime

        with _SHARED_LOCK:
            shared = _SHARED.get(self._shared_key)
            if shared is not None:
                self._engine, self._collection, refcount, self._remote = shared
                _SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    refcount + 1,
                    self._remote,
                )
            else:
                try:
                    if address:
                        self._engine = telys.client.connect(address, token)
                        self._remote = True
                    else:
                        self._engine = telys.Telys(path)
                        self._remote = False
                    self._collection = self._open_or_create_collection()
                except telys.runtime.RuntimeNotInstalled as exc:
                    raise TelysLocalClientError(
                        "Telys runtime is not available. Run `telys login` once, "
                        "it's free, device authorization, to fetch the signed "
                        "on-device runtime."
                    ) from exc
                except Exception as exc:
                    raise TelysLocalClientError(
                        f"Failed to connect to Telys: {exc}"
                    ) from exc
                _SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    1,
                    self._remote,
                )

        # Hot cache: known chunk keys, decoded records, and chunk-length map.
        # Seeding from the persisted collection lets warm restarts find existing
        # chunks without scanning Telys per lookup.
        self._lock = threading.Lock()
        self._key_set: set[str] = set()
        self._record_cache: dict[str, TelysRecord] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._seed_key_set()

    def _open_or_create_collection(self) -> Any:
        """Open an existing collection or create the KV backing collection."""
        names = (
            self._engine.collections() if hasattr(self._engine, "collections") else []
        )
        if self._collection_name in names:
            return self._engine.open_collection(self._collection_name)
        # Persist all connector metadata columns so a reopened collection can
        # resolve slab pointers and block provenance after a process restart.
        filter_columns = (
            self._key_col,
            self._payload_col,
            "vllm.slab_offset",
            "vllm.slab_length",
            "vllm.chunk_format",
            "vllm.block_start_hash",
            "vllm.block_count",
            "vllm.block_size",
            "vllm.model",
            "vllm.engine_id",
        )
        return self._engine.create_collection(
            self._collection_name,
            dim=self._dim,
            partition_by=self._key_col,
            filter_columns=filter_columns,
            dtype="f32",
        )

    def _seed_key_set(self) -> None:
        """Load existing chunk keys and chunk-length map from Telys."""
        try:
            ids = self._collection.ids()
            for key in ids:
                self._key_set.add(key)
                self._track_chunk_key(key)
            logger.info(
                "telys seeded key set with %d keys (%d chunk starts)",
                len(self._key_set),
                len(self._chunk_lengths),
            )
        except Exception as exc:
            raise TelysLocalClientError(
                f"Failed to seed key set from Telys: {exc}"
            ) from exc

    def _track_chunk_key(self, key: str) -> None:
        """Update the chunk-length map from a chunk key."""
        if "/chunk/" not in key:
            return
        start_key = _chunk_start_key(key)
        try:
            length = int(key.rsplit("/", 1)[1])
        except (ValueError, IndexError):
            return
        self._chunk_lengths[start_key] = max(
            self._chunk_lengths.get(start_key, 0), length
        )

    def _check_open(self) -> None:
        if self._closed:
            raise TelysLocalClientError("TelysLocalClient is closed")

    def contains(self, key: str) -> bool:
        """Fast in-process membership check; falls back to Telys on cache miss."""
        self._check_open()
        with self._lock:
            if key in self._key_set:
                return True
        return self.get(key) is not None

    def chunk_length(self, start_key: str) -> int:
        """Return the number of blocks in the stored chunk starting at ``start_key``."""
        self._check_open()
        with self._lock:
            length = self._chunk_lengths.get(start_key, 0)
        if length:
            return length
        # Local cache miss: refresh from the shared Telys store.  This is needed
        # when the scheduler/worker connectors are separate client objects (vLLM
        # V1 with multiprocessing, or even same-process role-split connectors
        # created before the first save) and the chunk was written by another
        # client.  Embedded-engine paths may still return 0 until a save/flush.
        self._seed_key_set()
        with self._lock:
            return self._chunk_lengths.get(start_key, 0)

    def get(self, key: str) -> TelysRecord | None:
        self._check_open()
        with self._lock:
            record = self._record_cache.get(key)
            if record is not None:
                return record

        try:
            result = self._collection.search(
                _zero_vector(self._dim),
                top_k=1,
                where={self._key_col: key},
                with_metadata=True,
            )
        except Exception as exc:
            if "unknown filter column" in str(exc):
                return None
            raise
        ids = result.get("ids") or []
        if not ids:
            return None
        metadata_list = result.get("metadata") or []
        if not metadata_list:
            raise TelysLocalClientError(
                f"Telys returned a hit for {key!r} but no metadata"
            )
        raw_metadata = dict(metadata_list[0])

        # v4: payload lives in the slab; Telys stores only the index metadata.
        # value is empty bytes.  Legacy v3 records carry a base64 string in
        # vllm.payload; v2 records carry a base64-JSON blob in vllm.meta_blob.
        payload = raw_metadata.pop(self._payload_col, None)
        if isinstance(payload, str):
            try:
                value = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError):
                # Fall back to legacy v2 base64-JSON metadata blob.
                record = _decode_legacy_blob(key, payload)
                with self._lock:
                    self._key_set.add(key)
                    self._record_cache[key] = record
                    self._track_chunk_key(key)
                return record
            # v3 inline payload: keep it readable but the connector prefers slab.
            record = TelysRecord(
                key=key,
                value=value,
                provenance=_derived_provenance(key),
                metadata={},  # connector metadata is carried in the chunk payload header
            )
        elif isinstance(payload, bytes):
            record = TelysRecord(
                key=key,
                value=payload,
                provenance=_derived_provenance(key),
                metadata={},
            )
        else:
            # v4 metadata-only record.
            record = TelysRecord(
                key=key,
                value=b"",
                provenance=_derived_provenance(key),
                metadata=dict(raw_metadata),
            )
        with self._lock:
            self._key_set.add(key)
            self._record_cache[key] = record
            self._track_chunk_key(key)
        return record

    def get_batch(self, keys: list[str]) -> list[TelysRecord | None]:
        """Return records for ``keys`` in order, using the hot cache when possible."""
        return [self.get(key) for key in keys]

    def put(self, key: str, value: bytes, metadata: Mapping[str, str]) -> None:
        self._check_open()
        # v4: Telys stores only the index metadata.  The payload bytes (if any)
        # are ignored here; the connector keeps them in the KVSlab.
        row_metadata = {
            self._key_col: key,
            **metadata,
        }
        self._collection.upsert(
            vectors=[_zero_vector(self._dim)],
            ids=[key],
            metadata=[row_metadata],
        )
        record = TelysRecord(
            key=key,
            value=b"",
            provenance=_derived_provenance(key),
            metadata=dict(metadata),
        )
        with self._lock:
            self._key_set.add(key)
            self._record_cache[key] = record
            self._track_chunk_key(key)

    def delete(self, key: str) -> bool:
        self._check_open()
        try:
            present = bool(self._collection.ids(where={self._key_col: key}))
        except Exception as exc:
            if "unknown filter column" in str(exc):
                present = False
            else:
                raise
        if not present:
            with self._lock:
                self._key_set.discard(key)
                self._record_cache.pop(key, None)
                self._chunk_lengths.pop(_chunk_start_key(key), None)
            return False
        self._collection.delete([key])
        with self._lock:
            self._key_set.discard(key)
            self._record_cache.pop(key, None)
            self._chunk_lengths.pop(_chunk_start_key(key), None)
        return True

    def compact(self) -> TelysMaintenanceReport:
        self._check_open()
        report = self._collection.compact()
        if isinstance(report, Mapping):
            scanned = report.get("entries_scanned")
            dropped = report.get("entries_dropped")
            if scanned is not None and dropped is not None:
                return TelysMaintenanceReport(
                    entries_scanned=int(scanned), entries_dropped=int(dropped)
                )
        before = self._collection.stats() if hasattr(self._collection, "stats") else {}
        after = self._collection.stats() if hasattr(self._collection, "stats") else {}
        rows_before = before.get(
            "live_rows", before.get("rows", before.get("external_ids", 0))
        )
        rows_after = after.get(
            "live_rows", after.get("rows", after.get("external_ids", 0))
        )
        return TelysMaintenanceReport(
            entries_scanned=int(rows_before),
            entries_dropped=max(0, int(rows_before) - int(rows_after)),
        )

    def close(self) -> None:
        self._closed = True
        with _SHARED_LOCK:
            shared = _SHARED.get(self._shared_key)
            if shared is None:
                return
            engine, collection, refcount, remote = shared
            refcount -= 1
            if refcount > 0:
                _SHARED[self._shared_key] = (engine, collection, refcount, remote)
                return
            _SHARED.pop(self._shared_key, None)
        # Final client in this process: save and compact so the next process
        # can see the persisted chunks. Embedded Telys collections are not
        # durable until ``save()`` is called.
        with contextlib.suppress(Exception):
            collection.save()
        with contextlib.suppress(Exception):
            collection.compact()
        with contextlib.suppress(Exception):
            engine.close()


# Shared engine/collection state reused by both TelysLocalClient and
# TelysBlobClient. vLLM constructs separate scheduler and worker connectors
# inside one process; sharing one engine/collection keeps writes visible
# immediately and the hot cache coherent.
_BLOB_SHARED: dict[tuple[str, str, str], tuple[Any, Any, int, bool]] = {}
_BLOB_SHARED_LOCK = threading.Lock()


class TelysBlobClient(TelysClient):
    """BlobStore-backed Telys client.

    Payload bytes live in the native SDK ``telys.BlobStore``; the Telys
    collection stores only small JSON metadata (provenance, block hashes,
    model/engine ids). This is the fast path when the installed SDK provides
    ``BlobStore``; older SDKs can keep using :class:`TelysLocalClient`.

    The same shared-engine/collection machinery as :class:`TelysLocalClient`
    is used so scheduler and worker connectors in one process see the same
    state.
    """

    def __init__(self, config: TelysConnectorConfig) -> None:
        path = os.environ.get("TELYS_PATH", "").strip() or ""
        address = os.environ.get("TELYS_ADDRESS", "").strip() or ""
        token = os.environ.get("TELYS_TOKEN") or None

        if not path and not address:
            raise TelysLocalClientError(
                "TelysBlobClient needs a connection. Set TELYS_PATH "
                "(embedded engine directory) or TELYS_ADDRESS (self-host server). "
                "Run `telys login` once, it's free, device authorization, "
                "to make the runtime available."
            )

        collection_override = os.environ.get("TELYS_COLLECTION") or None
        self._collection_name = _collection_name(config.namespace, collection_override)
        self._shared_key = (path, address, self._collection_name)
        self._closed = False
        self._remote = False
        self._engine: Any = None
        self._collection: Any = None
        self._blob_store: Any = None
        self._blob_path = path + "_blobs" if path else ""

        import telys
        import telys.client
        import telys.runtime

        if not hasattr(telys, "BlobStore"):
            raise TelysLocalClientError(
                "telys.BlobStore is not available; install a newer telys SDK "
                "or use TelysLocalClient instead."
            )

        with _BLOB_SHARED_LOCK:
            shared = _BLOB_SHARED.get(self._shared_key)
            if shared is not None:
                self._engine, self._collection, refcount, self._remote = shared
                _BLOB_SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    refcount + 1,
                    self._remote,
                )
            else:
                try:
                    if address:
                        self._engine = telys.client.connect(address, token)
                        self._remote = True
                    else:
                        self._engine = telys.Telys(path)
                        self._remote = False
                    self._collection = self._open_or_create_collection()
                    if self._blob_path:
                        self._blob_store = telys.BlobStore(self._blob_path)
                except telys.runtime.RuntimeNotInstalled as exc:
                    raise TelysLocalClientError(
                        "Telys runtime is not available. Run `telys login` once, "
                        "it's free, device authorization, to fetch the signed "
                        "on-device runtime."
                    ) from exc
                except Exception as exc:
                    raise TelysLocalClientError(
                        f"Failed to connect to Telys BlobStore: {exc}"
                    ) from exc
                _BLOB_SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    1,
                    self._remote,
                )

        self._lock = threading.Lock()
        self._key_set: set[str] = set()
        self._record_cache: dict[str, TelysRecord] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._seed_key_set()

    def _open_or_create_collection(self) -> Any:
        """Open an existing collection or create the metadata backing collection."""
        names = (
            self._engine.collections() if hasattr(self._engine, "collections") else []
        )
        if self._collection_name in names:
            return self._engine.open_collection(self._collection_name)
        # Only the columns needed for metadata lookups.
        filter_columns = (
            _KEY_COLUMN,
            "vllm.chunk_format",
            "vllm.block_start_hash",
            "vllm.block_count",
            "vllm.block_size",
            "vllm.model",
            "vllm.engine_id",
        )
        return self._engine.create_collection(
            self._collection_name,
            dim=1,
            partition_by=_KEY_COLUMN,
            filter_columns=filter_columns,
            dtype="f32",
        )

    def _seed_key_set(self) -> None:
        """Load existing chunk keys and chunk-length map from Telys."""
        try:
            ids = self._collection.ids()
            for key in ids:
                self._key_set.add(key)
                self._track_chunk_key(key)
            logger.info(
                "telys blob client seeded key set with %d keys (%d chunk starts)",
                len(self._key_set),
                len(self._chunk_lengths),
            )
        except Exception as exc:
            raise TelysLocalClientError(
                f"Failed to seed key set from Telys: {exc}"
            ) from exc

    def _track_chunk_key(self, key: str) -> None:
        """Update the chunk-length map from a chunk key."""
        if "/chunk/" not in key:
            return
        start_key = _chunk_start_key(key)
        try:
            length = int(key.rsplit("/", 1)[1])
        except (ValueError, IndexError):
            return
        self._chunk_lengths[start_key] = max(
            self._chunk_lengths.get(start_key, 0), length
        )

    def _check_open(self) -> None:
        if self._closed:
            raise TelysLocalClientError("TelysBlobClient is closed")

    def contains(self, key: str) -> bool:
        """Fast in-process membership check; falls back to BlobStore/Telys on miss."""
        self._check_open()
        with self._lock:
            if key in self._key_set:
                return True
        if self._blob_store is not None and self._blob_store.contains(key):
            return True
        return self.get(key) is not None

    def chunk_length(self, start_key: str) -> int:
        """Return the number of blocks in the stored chunk starting at ``start_key``."""
        self._check_open()
        with self._lock:
            length = self._chunk_lengths.get(start_key, 0)
        if length:
            return length
        # Local cache miss: refresh from the shared Telys store.  This is needed
        # when the scheduler/worker connectors are separate client objects (vLLM
        # V1 with multiprocessing, or even same-process role-split connectors
        # created before the first save) and the chunk was written by another
        # client.  Embedded-engine paths may still return 0 until a save/flush.
        self._seed_key_set()
        with self._lock:
            return self._chunk_lengths.get(start_key, 0)

    def get(self, key: str) -> TelysRecord | None:
        self._check_open()
        with self._lock:
            record = self._record_cache.get(key)
            if record is not None:
                return record

        try:
            result = self._collection.search(
                _zero_vector(1),
                top_k=1,
                where={_KEY_COLUMN: key},
                with_metadata=True,
            )
        except Exception as exc:
            if "unknown filter column" in str(exc):
                return None
            raise
        ids = result.get("ids") or []
        if not ids:
            return None
        metadata_list = result.get("metadata") or []
        if not metadata_list:
            raise TelysLocalClientError(
                f"Telys returned a hit for {key!r} but no metadata"
            )
        raw_metadata = dict(metadata_list[0])
        raw_metadata.pop(_KEY_COLUMN, None)

        if self._blob_store is not None:
            try:
                payload, _ = self._blob_store.get(key)
            except KeyError:
                raise TelysLocalClientError(
                    f"metadata exists for {key!r} but BlobStore payload is missing"
                ) from None
        else:
            payload = b""

        record = TelysRecord(
            key=key,
            value=payload,
            provenance=_derived_provenance(key),
            metadata=raw_metadata,
        )
        with self._lock:
            self._key_set.add(key)
            self._record_cache[key] = record
            self._track_chunk_key(key)
        return record

    def get_batch(self, keys: list[str]) -> list[TelysRecord | None]:
        """Return records for ``keys`` in order, using the hot cache when possible."""
        return [self.get(key) for key in keys]

    def put(self, key: str, value: bytes, metadata: Mapping[str, str]) -> None:
        self._check_open()
        if self._blob_store is not None:
            self._blob_store.put(key, value)
        row_metadata = {
            _KEY_COLUMN: key,
            **metadata,
        }
        self._collection.upsert(
            vectors=[_zero_vector(1)],
            ids=[key],
            metadata=[row_metadata],
        )
        record = TelysRecord(
            key=key,
            value=value,
            provenance=_derived_provenance(key),
            metadata=dict(metadata),
        )
        with self._lock:
            self._key_set.add(key)
            self._record_cache[key] = record
            self._track_chunk_key(key)

    def delete(self, key: str) -> bool:
        self._check_open()
        try:
            present = bool(self._collection.ids(where={_KEY_COLUMN: key}))
        except Exception as exc:
            if "unknown filter column" in str(exc):
                present = False
            else:
                raise
        if not present:
            with self._lock:
                self._key_set.discard(key)
                self._record_cache.pop(key, None)
                self._chunk_lengths.pop(_chunk_start_key(key), None)
            return False
        self._collection.delete([key])
        if self._blob_store is not None:
            with contextlib.suppress(Exception):
                self._blob_store.delete(key)
        with self._lock:
            self._key_set.discard(key)
            self._record_cache.pop(key, None)
            self._chunk_lengths.pop(_chunk_start_key(key), None)
        return True

    def compact(self) -> TelysMaintenanceReport:
        self._check_open()
        report = self._collection.compact()
        if isinstance(report, Mapping):
            scanned = report.get("entries_scanned")
            dropped = report.get("entries_dropped")
            if scanned is not None and dropped is not None:
                return TelysMaintenanceReport(
                    entries_scanned=int(scanned), entries_dropped=int(dropped)
                )
        before = self._collection.stats() if hasattr(self._collection, "stats") else {}
        after = self._collection.stats() if hasattr(self._collection, "stats") else {}
        rows_before = before.get(
            "live_rows", before.get("rows", before.get("external_ids", 0))
        )
        rows_after = after.get(
            "live_rows", after.get("rows", after.get("external_ids", 0))
        )
        return TelysMaintenanceReport(
            entries_scanned=int(rows_before),
            entries_dropped=max(0, int(rows_before) - int(rows_after)),
        )

    def close(self) -> None:
        self._closed = True
        with _BLOB_SHARED_LOCK:
            shared = _BLOB_SHARED.get(self._shared_key)
            if shared is None:
                return
            engine, collection, refcount, remote = shared
            refcount -= 1
            if refcount > 0:
                _BLOB_SHARED[self._shared_key] = (engine, collection, refcount, remote)
                return
            _BLOB_SHARED.pop(self._shared_key, None)
        # Final client in this process: flush BlobStore and close Telys.
        if self._blob_store is not None:
            with contextlib.suppress(Exception):
                self._blob_store.sync()
                self._blob_store.close()
        with contextlib.suppress(Exception):
            collection.save()
        with contextlib.suppress(Exception):
            collection.compact()
        with contextlib.suppress(Exception):
            engine.close()
