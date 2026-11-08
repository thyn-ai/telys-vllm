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
import weakref
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
_CONTENT_SHA256_COLUMN = "vllm.content_sha256"
_MAX_CHUNK_LENGTH_DIGITS = 10
_DEFAULT_KV_DIM = 1

# Process-global shared engines/collections.  vLLM constructs separate scheduler
# and worker connectors; sharing one collection object inside a process makes
# writes visible to reads immediately and keeps the hot cache coherent.
# Map: (path_or_empty, address_or_empty, collection_name) -> shared state.
# The mutation lock covers each backend mutation and its registry propagation.
_SHARED: dict[
    tuple[str, str, str], tuple[Any, Any, int, bool, threading.Lock]
] = {}
_SHARED_LOCK = threading.Lock()
_LOCAL_CLIENT_REGISTRY: dict[
    tuple[str, str, str], weakref.WeakSet[TelysLocalClient]
] = {}
_LOCAL_CLIENT_REGISTRY_LOCK = threading.Lock()


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


def _encode_legacy_blob(value: bytes, metadata: Mapping[str, str]) -> str:
    """Encode complete metadata in the filter column available to old schemas."""
    return json.dumps(
        {
            "payload": base64.b64encode(value).decode("ascii"),
            "metadata": dict(metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


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
    """Extract the first hash from chain-bound or legacy chunk keys."""
    prefix, marker, hashes = chunk_key.rpartition("/chunk/")
    if not marker:
        return chunk_key.rsplit("/", 1)[0]
    start_hash = hashes.split("/", 1)[0]
    return f"{prefix}{marker}{start_hash}"


def _parse_chunk_length(raw: str) -> int | None:
    if (
        not raw
        or len(raw) > _MAX_CHUNK_LENGTH_DIGITS
        or not raw.isascii()
        or not raw.isdigit()
    ):
        return None
    return int(raw)


def _parse_chunk_key(key: str) -> tuple[str, int, bool] | None:
    """Return the start key, length, and chain-bound flag for a valid chunk key."""
    if "/chunk/" not in key:
        return None
    start_key = _chunk_start_key(key)
    if not key.startswith(f"{start_key}/"):
        return None
    parts = key[len(start_key) + 1 :].split("/")
    if len(parts) == 2:
        length = _parse_chunk_length(parts[1])
        if length is not None:
            return start_key, length, True
    elif len(parts) == 1:
        length = _parse_chunk_length(parts[0])
        if length is not None:
            return start_key, length, False
    return None


_REQUIRED_BLOB_METADATA_FIELDS = (
    "vllm.block_start_hash",
    "vllm.block_count",
    "vllm.block_size",
    "vllm.model",
    "vllm.engine_id",
)


def _well_formed_content_digest(metadata: Any) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(_CONTENT_SHA256_COLUMN)
    if not isinstance(raw, str) or len(raw) != 64:
        return None
    try:
        digest = bytes.fromhex(raw)
    except ValueError:
        return None
    return raw if raw == digest.hex() else None


def _complete_blob_metadata(metadata: Any) -> dict[str, Any] | None:
    """Validate metadata that makes a BlobStore read self-contained."""
    if not isinstance(metadata, Mapping):
        return None
    if _LEGACY_META_BLOB_COLUMN in metadata or _PAYLOAD_COLUMN in metadata:
        return None
    if any(field not in metadata for field in _REQUIRED_BLOB_METADATA_FIELDS):
        return None
    if any(not isinstance(metadata[field], str) for field in _REQUIRED_BLOB_METADATA_FIELDS):
        return None
    if not metadata["vllm.block_start_hash"]:
        return None
    if _well_formed_content_digest(metadata) is None:
        return None
    for field in ("vllm.block_count", "vllm.block_size"):
        value = metadata[field]
        if not value.isdigit() or int(value) <= 0:
            return None
    return dict(metadata)


def _recover_collection_metadata(
    key: str, blob_metadata: Any, collection_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep collection metadata authoritative when recovering a blob read."""
    blob_digest = _well_formed_content_digest(blob_metadata)
    collection_digest = _well_formed_content_digest(collection_metadata)
    if (
        blob_digest is not None
        and collection_digest is not None
        and blob_digest != collection_digest
    ):
        raise TelysLocalClientError(f"conflicting content digests for {key!r}")
    recovered = dict(collection_metadata)
    if _CONTENT_SHA256_COLUMN not in recovered and blob_digest is not None:
        recovered[_CONTENT_SHA256_COLUMN] = blob_digest
    return recovered


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

        self._mutation_lock = threading.Lock()
        with _SHARED_LOCK:
            shared = _SHARED.get(self._shared_key)
            if shared is not None:
                (
                    self._engine,
                    self._collection,
                    refcount,
                    self._remote,
                    self._mutation_lock,
                ) = shared
                _SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    refcount + 1,
                    self._remote,
                    self._mutation_lock,
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
                    self._mutation_lock,
                )

        filter_columns = getattr(self._collection, "col_names", None)
        self._use_legacy_digest_sidecar = (
            filter_columns is None or _CONTENT_SHA256_COLUMN not in filter_columns
        )
        self._legacy_digest_sidecar_available = (
            filter_columns is None or _LEGACY_META_BLOB_COLUMN in filter_columns
        )

        # Hot cache: known chunk keys, decoded records, and chunk-length map.
        # Seeding from the persisted collection lets warm restarts find existing
        # chunks without scanning Telys per lookup.
        self._lock = threading.Lock()
        self._key_set: set[str] = set()
        self._record_cache: dict[str, TelysRecord] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._chunk_key_index: dict[str, dict[int, str]] = {}
        self._chunk_keys_by_start: dict[str, set[str]] = {}
        self._seed_key_set()
        with _LOCAL_CLIENT_REGISTRY_LOCK:
            _LOCAL_CLIENT_REGISTRY.setdefault(
                self._shared_key, weakref.WeakSet()
            ).add(self)

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
            self._legacy_meta_blob_col,
            "vllm.slab_offset",
            "vllm.slab_length",
            "vllm.chunk_format",
            "vllm.content_sha256",
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
        """Load existing chunk keys and rebuild the exact chunk index."""
        try:
            ids = self._collection.ids()
            with self._lock:
                self._key_set.update(ids)
                self._chunk_lengths.clear()
                self._chunk_key_index.clear()
                self._chunk_keys_by_start.clear()
                for key in sorted(self._key_set):
                    self._track_chunk_key(key)
                key_count = len(self._key_set)
                chunk_start_count = len(self._chunk_lengths)
            logger.info(
                "telys seeded key set with %d keys (%d chunk starts)",
                key_count,
                chunk_start_count,
            )
        except Exception as exc:
            raise TelysLocalClientError(
                f"Failed to seed key set from Telys: {exc}"
            ) from exc

    def _track_chunk_key(self, key: str) -> None:
        """Update the exact candidate index and chunk-length map."""
        parsed = _parse_chunk_key(key)
        if parsed is None:
            return
        start_key, length, chain_bound = parsed
        self._chunk_keys_by_start.setdefault(start_key, set()).add(key)
        candidates = self._chunk_key_index.setdefault(start_key, {})
        existing = candidates.get(length)
        if existing is not None:
            existing_chain_bound = "/" in existing[len(start_key) + 1 :]
            if chain_bound != existing_chain_bound:
                if not chain_bound:
                    return
            elif key >= existing:
                return
        candidates[length] = key
        self._chunk_lengths[start_key] = max(candidates)

    def _untrack_chunk_key(self, key: str) -> None:
        """Remove one key, scanning only its start for a selected replacement."""
        parsed = _parse_chunk_key(key)
        if parsed is None:
            return
        start_key, length, _ = parsed
        start_keys = self._chunk_keys_by_start.get(start_key)
        if start_keys is not None:
            start_keys.discard(key)
            if not start_keys:
                self._chunk_keys_by_start.pop(start_key, None)
        candidates = self._chunk_key_index.get(start_key)
        if not candidates or candidates.get(length) != key:
            return
        del candidates[length]
        for candidate_key in tuple(start_keys or ()):
            candidate = _parse_chunk_key(candidate_key)
            if candidate is not None and candidate[1] == length:
                self._track_chunk_key(candidate_key)
        if candidates:
            self._chunk_lengths[start_key] = max(candidates)
        else:
            self._chunk_key_index.pop(start_key, None)
            self._chunk_lengths.pop(start_key, None)

    def _cache_record_for_overlapping_clients(
        self, key: str, record: TelysRecord
    ) -> None:
        with _LOCAL_CLIENT_REGISTRY_LOCK:
            clients = tuple(_LOCAL_CLIENT_REGISTRY.get(self._shared_key, ()))
        for client in clients:
            with client._lock:
                client._key_set.add(key)
                client._record_cache[key] = record
                client._track_chunk_key(key)

    def _evict_key_from_overlapping_clients(self, key: str) -> None:
        with _LOCAL_CLIENT_REGISTRY_LOCK:
            clients = tuple(_LOCAL_CLIENT_REGISTRY.get(self._shared_key, ()))
        for client in clients:
            with client._lock:
                client._key_set.discard(key)
                client._record_cache.pop(key, None)
                client._untrack_chunk_key(key)

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

    def keys(self) -> set[str]:
        """Return a snapshot of keys in the local cache and Telys collection."""
        self._check_open()
        self._seed_key_set()
        with self._lock:
            return set(self._key_set)

    def chunk_candidates(self, start_key: str) -> dict[int, str]:
        """Return exact chunk lengths and keys without I/O on an indexed hit."""
        self._check_open()
        with self._lock:
            candidates = self._chunk_key_index.get(start_key)
            if candidates and all(
                key in self._key_set for key in candidates.values()
            ):
                return dict(candidates)
        self._seed_key_set()
        with self._lock:
            return dict(self._chunk_key_index.get(start_key, {}))

    def chunk_length(self, start_key: str) -> int:
        """Return the number of blocks in the stored chunk starting at ``start_key``."""
        self._check_open()
        with self._lock:
            length = self._chunk_lengths.get(start_key, 0)
        if length:
            return length
        # Local cache miss: refresh from the shared Telys store before treating
        # the prefix position as absent.
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
        legacy_blob = raw_metadata.pop(self._legacy_meta_blob_col, None)
        payload = raw_metadata.pop(self._payload_col, None)
        if isinstance(legacy_blob, str):
            record = _decode_legacy_blob(key, legacy_blob)
        elif isinstance(payload, str):
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
        if (
            _CONTENT_SHA256_COLUMN in metadata
            and self._use_legacy_digest_sidecar
        ):
            if not self._legacy_digest_sidecar_available:
                raise TelysLocalClientError(
                    "the existing collection schema cannot persist vllm.content_sha256 "
                    "or the legacy metadata compatibility column"
                )
            # The SDK has no schema-migration API. Preserve the complete metadata
            # in the already-declared v2 column instead of dropping the digest.
            row_metadata[self._legacy_meta_blob_col] = _encode_legacy_blob(b"", metadata)
        record = TelysRecord(
            key=key,
            value=b"",
            provenance=_derived_provenance(key),
            metadata=dict(metadata),
        )
        with self._mutation_lock:
            self._collection.upsert(
                vectors=[_zero_vector(self._dim)],
                ids=[key],
                metadata=[row_metadata],
            )
            self._cache_record_for_overlapping_clients(key, record)

    def delete(self, key: str) -> bool:
        self._check_open()
        with self._mutation_lock:
            try:
                present = bool(self._collection.ids(where={self._key_col: key}))
            except Exception as exc:
                if "unknown filter column" in str(exc):
                    present = False
                else:
                    raise
            if present:
                self._collection.delete([key])
            self._evict_key_from_overlapping_clients(key)
            return present

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
        with self._lock:
            self._chunk_lengths.clear()
            self._chunk_key_index.clear()
            self._chunk_keys_by_start.clear()
        with _LOCAL_CLIENT_REGISTRY_LOCK:
            clients = _LOCAL_CLIENT_REGISTRY.get(self._shared_key)
            if clients is not None:
                clients.discard(self)
                if not clients:
                    _LOCAL_CLIENT_REGISTRY.pop(self._shared_key, None)
        with _SHARED_LOCK:
            shared = _SHARED.get(self._shared_key)
            if shared is None:
                return
            engine, collection, refcount, remote, mutation_lock = shared
            refcount -= 1
            if refcount > 0:
                _SHARED[self._shared_key] = (
                    engine,
                    collection,
                    refcount,
                    remote,
                    mutation_lock,
                )
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


# Shared engine/collection/BlobStore state reused by both TelysLocalClient and
# TelysBlobClient. vLLM constructs separate scheduler and worker connectors
# inside one process; sharing one engine/collection keeps writes visible
# immediately and the hot cache coherent.
_BLOB_SHARED: dict[
    tuple[str, str, str], tuple[Any, Any, Any, int, bool, threading.Lock]
] = {}
_BLOB_SHARED_LOCK = threading.Lock()
_BLOB_CLIENT_REGISTRY: dict[tuple[str, str, str], weakref.WeakSet[TelysBlobClient]] = {}
_BLOB_CLIENT_REGISTRY_LOCK = threading.Lock()


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
        self._collection_created = False

        import telys
        import telys.client
        import telys.runtime

        if not hasattr(telys, "BlobStore"):
            raise TelysLocalClientError(
                "telys.BlobStore is not available; install a newer telys SDK "
                "or use TelysLocalClient instead."
            )

        self._mutation_lock = threading.Lock()
        with _BLOB_SHARED_LOCK:
            shared = _BLOB_SHARED.get(self._shared_key)
            if shared is not None:
                (
                    self._engine,
                    self._collection,
                    self._blob_store,
                    refcount,
                    self._remote,
                    self._mutation_lock,
                ) = shared
                _BLOB_SHARED[self._shared_key] = (
                    self._engine,
                    self._collection,
                    self._blob_store,
                    refcount + 1,
                    self._remote,
                    self._mutation_lock,
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
                    self._blob_store,
                    1,
                    self._remote,
                    self._mutation_lock,
                )

        filter_columns = getattr(self._collection, "col_names", None)
        self._use_legacy_digest_sidecar = (
            filter_columns is None or _CONTENT_SHA256_COLUMN not in filter_columns
        )
        self._legacy_digest_sidecar_available = (
            filter_columns is None or _LEGACY_META_BLOB_COLUMN in filter_columns
        )

        self._lock = threading.Lock()
        self._key_set: set[str] = set()
        self._record_cache: dict[str, TelysRecord] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._chunk_key_index: dict[str, dict[int, str]] = {}
        self._chunk_keys_by_start: dict[str, set[str]] = {}
        self._legacy_migration_candidates: set[str] = set()
        self._legacy_migration_pending = self._collection_created
        self._seed_key_set()
        with _BLOB_CLIENT_REGISTRY_LOCK:
            _BLOB_CLIENT_REGISTRY.setdefault(self._shared_key, weakref.WeakSet()).add(
                self
            )

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
            _PAYLOAD_COLUMN,
            _LEGACY_META_BLOB_COLUMN,
            "vllm.chunk_format",
            "vllm.content_sha256",
            "vllm.block_start_hash",
            "vllm.block_count",
            "vllm.block_size",
            "vllm.model",
            "vllm.engine_id",
        )
        collection = self._engine.create_collection(
            self._collection_name,
            dim=1,
            partition_by=_KEY_COLUMN,
            filter_columns=filter_columns,
            dtype="f32",
        )
        self._collection_created = True
        return collection

    def _seed_key_set(self) -> None:
        """Load authoritative collection keys and rebuild the exact chunk index."""
        try:
            with self._mutation_lock:
                ids = set(self._collection.ids())
                if self._legacy_migration_pending:
                    if self._blob_store is not None:
                        self._legacy_migration_candidates.update(
                            set(self._blob_store.keys()) - ids
                        )
                    self._legacy_migration_pending = False
                ids.update(self._legacy_migration_candidates)
                with self._lock:
                    self._key_set = ids
                    self._record_cache = {
                        key: record
                        for key, record in self._record_cache.items()
                        if key in ids
                    }
                    self._chunk_lengths.clear()
                    self._chunk_key_index.clear()
                    self._chunk_keys_by_start.clear()
                    for key in sorted(self._key_set):
                        self._track_chunk_key(key)
                    key_count = len(self._key_set)
                    chunk_start_count = len(self._chunk_lengths)
            logger.info(
                "telys blob client seeded key set with %d keys (%d chunk starts)",
                key_count,
                chunk_start_count,
            )
        except Exception as exc:
            raise TelysLocalClientError(
                f"Failed to seed key set from Telys: {exc}"
            ) from exc

    def _track_chunk_key(self, key: str) -> None:
        """Update the exact candidate index and chunk-length map."""
        parsed = _parse_chunk_key(key)
        if parsed is None:
            return
        start_key, length, chain_bound = parsed
        self._chunk_keys_by_start.setdefault(start_key, set()).add(key)
        candidates = self._chunk_key_index.setdefault(start_key, {})
        existing = candidates.get(length)
        if existing is not None:
            existing_chain_bound = "/" in existing[len(start_key) + 1 :]
            if chain_bound != existing_chain_bound:
                if not chain_bound:
                    return
            elif key >= existing:
                return
        candidates[length] = key
        self._chunk_lengths[start_key] = max(candidates)

    def _untrack_chunk_key(self, key: str) -> None:
        """Remove one key, scanning only its start for a selected replacement."""
        parsed = _parse_chunk_key(key)
        if parsed is None:
            return
        start_key, length, _ = parsed
        start_keys = self._chunk_keys_by_start.get(start_key)
        if start_keys is not None:
            start_keys.discard(key)
            if not start_keys:
                self._chunk_keys_by_start.pop(start_key, None)
        candidates = self._chunk_key_index.get(start_key)
        if not candidates or candidates.get(length) != key:
            return
        del candidates[length]
        for candidate_key in tuple(start_keys or ()):
            candidate = _parse_chunk_key(candidate_key)
            if candidate is not None and candidate[1] == length:
                self._track_chunk_key(candidate_key)
        if candidates:
            self._chunk_lengths[start_key] = max(candidates)
        else:
            self._chunk_key_index.pop(start_key, None)
            self._chunk_lengths.pop(start_key, None)

    def _cache_record_for_overlapping_clients(
        self, key: str, record: TelysRecord
    ) -> None:
        with _BLOB_CLIENT_REGISTRY_LOCK:
            clients = tuple(_BLOB_CLIENT_REGISTRY.get(self._shared_key, ()))
        for client in clients:
            with client._lock:
                client._legacy_migration_candidates.discard(key)
                client._key_set.add(key)
                client._record_cache[key] = record
                client._track_chunk_key(key)

    def _evict_key_from_overlapping_clients(self, key: str) -> None:
        with _BLOB_CLIENT_REGISTRY_LOCK:
            clients = tuple(_BLOB_CLIENT_REGISTRY.get(self._shared_key, ()))
        for client in clients:
            with client._lock:
                client._legacy_migration_candidates.discard(key)
                client._key_set.discard(key)
                client._record_cache.pop(key, None)
                client._untrack_chunk_key(key)

    def _check_open(self) -> None:
        if self._closed:
            raise TelysLocalClientError("TelysBlobClient is closed")

    @property
    def stores_payload_values(self) -> bool:
        """Whether this client persists ``put`` values in an active BlobStore."""
        return self._blob_store is not None

    def contains(self, key: str) -> bool:
        """Check authoritative membership, falling back to a validated read."""
        self._check_open()
        with self._lock:
            if key in self._key_set:
                return True
        return self.get(key) is not None

    def keys(self) -> set[str]:
        """Return collection keys plus explicitly migratable legacy blobs."""
        self._check_open()
        self._seed_key_set()
        with self._lock:
            return set(self._key_set)

    def chunk_candidates(self, start_key: str) -> dict[int, str]:
        """Return exact chunk lengths and keys without I/O on an indexed hit."""
        self._check_open()
        with self._lock:
            candidates = self._chunk_key_index.get(start_key)
            if candidates and all(
                key in self._key_set for key in candidates.values()
            ):
                return dict(candidates)
        self._seed_key_set()
        with self._lock:
            return dict(self._chunk_key_index.get(start_key, {}))

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

    def _has_collection_membership(self, key: str) -> bool:
        with self._lock:
            return (
                key in self._key_set
                and key not in self._legacy_migration_candidates
            )

    def get(self, key: str) -> TelysRecord | None:
        self._check_open()
        with self._lock:
            record = self._record_cache.get(key)
        if record is not None and self._has_collection_membership(key):
            self._cache_record_for_overlapping_clients(key, record)
            return record

        blob_found = False
        payload = b""
        blob_metadata: Any = {}
        legacy_blob: str | None = None
        if self._blob_store is not None:
            try:
                payload, blob_metadata = self._blob_store.get(key)
                blob_found = True
            except KeyError:
                pass

        if blob_found:
            complete_metadata = _complete_blob_metadata(blob_metadata)
            if complete_metadata is not None and self._has_collection_membership(key):
                complete_metadata.pop(_KEY_COLUMN, None)
                record = TelysRecord(
                    key=key,
                    value=payload,
                    provenance=_derived_provenance(key),
                    metadata=complete_metadata,
                )
                self._cache_record_for_overlapping_clients(key, record)
                return record

        try:
            result = self._collection.search(
                _zero_vector(1),
                top_k=1,
                where={_KEY_COLUMN: key},
                with_metadata=True,
            )
        except Exception as exc:
            if "unknown filter column" not in str(exc):
                raise
            if not blob_found:
                return None
            result = {"ids": []}
        ids = result.get("ids") or []
        migrate_legacy = False
        if ids:
            metadata_list = result.get("metadata") or []
            if not metadata_list:
                raise TelysLocalClientError(
                    f"Telys returned a hit for {key!r} but no metadata"
                )
            raw_metadata = dict(metadata_list[0])
            raw_metadata.pop(_KEY_COLUMN, None)
            legacy_blob = raw_metadata.pop(_LEGACY_META_BLOB_COLUMN, None)
            if (
                not isinstance(legacy_blob, str)
                and self._blob_store is not None
                and not blob_found
            ):
                raise TelysLocalClientError(
                    f"metadata exists for {key!r} but BlobStore payload is missing"
                )
            if not isinstance(legacy_blob, str):
                raw_metadata = _recover_collection_metadata(
                    key, blob_metadata, raw_metadata
                )
        elif blob_found:
            raw_metadata = dict(blob_metadata)
            with self._lock:
                migrate_legacy = key in self._legacy_migration_candidates
                if not migrate_legacy:
                    self._key_set.discard(key)
                    self._record_cache.pop(key, None)
                    self._untrack_chunk_key(key)
                    return None
        else:
            return None

        if isinstance(legacy_blob, str):
            record = _decode_legacy_blob(key, legacy_blob)
            if blob_found and not record.value:
                record = TelysRecord(
                    key=key,
                    value=payload,
                    provenance=record.provenance,
                    metadata=_recover_collection_metadata(
                        key, blob_metadata, record.metadata
                    ),
                )
        else:
            record = TelysRecord(
                key=key,
                value=payload,
                provenance=_derived_provenance(key),
                metadata=raw_metadata,
            )
        if migrate_legacy:
            self.put(key, record.value, record.metadata)
        else:
            self._cache_record_for_overlapping_clients(key, record)
        return record

    def get_batch(self, keys: list[str]) -> list[TelysRecord | None]:
        """Return records for ``keys`` in order, using the hot cache when possible."""
        return [self.get(key) for key in keys]

    def put(self, key: str, value: bytes, metadata: Mapping[str, str]) -> None:
        self._check_open()
        row_metadata = {
            _KEY_COLUMN: key,
            **metadata,
        }
        if (
            _CONTENT_SHA256_COLUMN in metadata
            and self._use_legacy_digest_sidecar
        ):
            if not self._legacy_digest_sidecar_available:
                raise TelysLocalClientError(
                    "the existing collection schema cannot persist vllm.content_sha256 "
                    "or the legacy metadata compatibility column"
                )
            row_metadata[_LEGACY_META_BLOB_COLUMN] = _encode_legacy_blob(b"", metadata)
        record = TelysRecord(
            key=key,
            value=value,
            provenance=_derived_provenance(key),
            metadata=dict(metadata),
        )
        with self._mutation_lock:
            if self._blob_store is not None:
                # Make the payload and index durable before acknowledging the save;
                # buffered writes are not reliably visible to other NFS workers.
                self._blob_store.put(key, value, metadata=dict(metadata), sync=True)
            self._collection.upsert(
                vectors=[_zero_vector(1)],
                ids=[key],
                metadata=[row_metadata],
            )
            self._cache_record_for_overlapping_clients(key, record)

    def delete(self, key: str) -> bool:
        self._check_open()
        with self._mutation_lock:
            try:
                present = bool(self._collection.ids(where={_KEY_COLUMN: key}))
            except Exception as exc:
                if "unknown filter column" in str(exc):
                    present = False
                else:
                    raise
            blob_present = self._blob_store is not None and self._blob_store.contains(
                key
            )
            try:
                if present:
                    self._collection.delete([key])
                if blob_present:
                    self._blob_store.delete(key, sync=True)
            finally:
                self._evict_key_from_overlapping_clients(key)
            return present or blob_present

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
        with self._lock:
            self._chunk_lengths.clear()
            self._chunk_key_index.clear()
            self._chunk_keys_by_start.clear()
        with _BLOB_CLIENT_REGISTRY_LOCK:
            clients = _BLOB_CLIENT_REGISTRY.get(self._shared_key)
            if clients is not None:
                clients.discard(self)
                if not clients:
                    _BLOB_CLIENT_REGISTRY.pop(self._shared_key, None)
        with _BLOB_SHARED_LOCK:
            shared = _BLOB_SHARED.get(self._shared_key)
            if shared is None:
                return
            engine, collection, blob_store, refcount, remote, mutation_lock = shared
            refcount -= 1
            if refcount > 0:
                _BLOB_SHARED[self._shared_key] = (
                    engine,
                    collection,
                    blob_store,
                    refcount,
                    remote,
                    mutation_lock,
                )
                return
            _BLOB_SHARED.pop(self._shared_key, None)
        # Final client in this process: flush BlobStore and close Telys.
        if blob_store is not None:
            with contextlib.suppress(Exception):
                blob_store.sync()
                blob_store.close()
        with contextlib.suppress(Exception):
            collection.save()
        with contextlib.suppress(Exception):
            collection.compact()
        with contextlib.suppress(Exception):
            engine.close()
