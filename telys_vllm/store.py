# SPDX-License-Identifier: Apache-2.0
"""Pure-Python core: vLLM block hashes mapped onto chunks in a TelysClient.

This module has no vLLM imports and is the unit the CPU tests exercise
directly: chunk key mapping, contiguous-prefix lookup, save/load with provenance,
and the compaction hook.

Telys stores the chunk index and provenance metadata. Encoded chunk bytes live
either in an append-only ``KVSlab`` file or directly in a BlobStore-backed
client. Saves can be asynchronous: ``save_chunk_async`` returns immediately
after enqueueing; a background worker persists the payload and metadata record.
"""

import atexit
import logging
import queue
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only dev/test environments
    torch = None  # type: ignore[assignment]

from telys_vllm.client import (
    TelysClient,
    TelysMaintenanceReport,
    TelysProvenance,
    TelysRecord,
)
from telys_vllm.codec import (
    LengthPrefixedCodec,
    PayloadCodecError,
    TelysChunkCodec,
    TelysFastChunkCodec,
)
from telys_vllm.config import TelysConnectorConfig
from telys_vllm.slab import KVSlab
from telys_vllm.telemetry import NOOP, Telemetry, get_telemetry

logger = logging.getLogger(__name__)


class TelysStoreError(RuntimeError):
    """Raised when the store cannot honor a planned operation."""


@dataclass(frozen=True)
class DecodedBlock:
    """One block's decoded layer blobs plus read provenance."""

    key: str
    provenance: TelysProvenance
    metadata: Mapping[str, str]
    blobs: dict[str, bytes]


@dataclass(frozen=True)
class FastDecodedChunk:
    """One chunk decoded to stacked layer tensors for the zero-copy fast path."""

    provenance: TelysProvenance
    metadata: Mapping[str, str]
    block_hashes: tuple[bytes, ...]
    layer_tensors: dict[str, "torch.Tensor"]  # {layer_name: [N, block_size, *]}


@dataclass(frozen=True)
class _PendingWrite:
    """One asynchronous save operation queued for the background worker."""

    key: str
    start_hash: bytes
    block_hashes: tuple[bytes, ...]
    encoded_chunk: bytes
    extra_metadata: Mapping[str, str]


@dataclass(frozen=True)
class _PendingFastWrite:
    """A fast chunk whose tensors are encoded by the background worker."""

    key: str
    start_hash: bytes
    block_hashes: tuple[bytes, ...]
    layer_tensors: Mapping[str, "torch.Tensor"]
    extra_metadata: Mapping[str, str]


class TelysBlockStore:
    """Chunk-granular KV store: Telys index plus BlobStore or slab payloads.

    A *chunk* is one contiguous run of full blocks.  The Telys key is
    ``{namespace}/chunk/{start_hash}/{end_hash}/{block_count}``. Slab-backed
    records include an offset/length pointer; BlobStore records carry the payload
    directly as the client value.
    """

    _SLAB_OFFSET_KEY = "vllm.slab_offset"
    _SLAB_LENGTH_KEY = "vllm.slab_length"
    _CHUNK_FORMAT_KEY = "vllm.chunk_format"
    _CHUNK_FORMAT_FAST = "fast"

    def __init__(
        self,
        client: TelysClient,
        config: TelysConnectorConfig,
        *,
        model_id: str,
        engine_id: str,
        block_size: int,
        slab: KVSlab | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        if block_size <= 0:
            raise TelysStoreError(f"block_size must be positive, got {block_size}")
        self._client = client
        self._config = config
        self._model_id = model_id
        self._engine_id = engine_id
        self._block_size = block_size
        if telemetry is not None:
            self._telemetry = telemetry
        elif config.telemetry:
            self._telemetry = get_telemetry(
                enabled=config.telemetry,
                log_path=config.telemetry_log,
            )
        else:
            self._telemetry = NOOP
        self._known_keys: set[str] = set()
        self._pending_keys: set[str] = set()
        self._block_codec = LengthPrefixedCodec()
        self._chunk_codec = TelysChunkCodec()
        self._fast_chunk_codec = TelysFastChunkCodec()

        self._slab = slab if slab is not None else self._open_slab_optional()

        # Async write machinery.
        self._write_queue: queue.Queue[_PendingWrite | _PendingFastWrite | None] = (
            queue.Queue(maxsize=max(1, config.max_pending_writes))
        )
        self._pending_count = 0
        self._write_errors: list[tuple[str, Exception]] = []
        self._lock = threading.Lock()
        self._drain_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._closed = False
        if config.write_behind:
            self._worker = threading.Thread(target=self._write_worker, daemon=True)
            self._worker.start()
        # Drain any pending writes when the interpreter shuts down so a vLLM
        # process that exits without calling shutdown() does not lose saves that
        # warm-restart runs depend on. The registration is idempotent per store
        # instance; close() unregisters to avoid a no-op drain after a clean
        # shutdown.
        atexit.register(self.drain)

    def _open_slab_optional(self) -> KVSlab | None:
        """Open the slab unless the client is the configured direct payload store."""
        if self._config.blobstore_no_slab and getattr(
            self._client, "stores_payload_values", False
        ):
            return None
        path = self._config.slab_path
        if path is None:
            return None
        return KVSlab(path)

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def slab(self) -> KVSlab:
        return self._slab

    def block_key(self, block_hash: bytes) -> str:
        """Map one vLLM block hash to a synthetic per-block key (used for records)."""
        return f"{self._config.namespace}/kv/{bytes(block_hash).hex()}"

    def chunk_start_key(self, block_hash: bytes) -> str:
        """Key prefix for the chunk (if any) that starts at ``block_hash``."""
        return f"{self._config.namespace}/chunk/{bytes(block_hash).hex()}"

    def chunk_key(self, start_hash: bytes, block_count: int, end_hash: bytes) -> str:
        """Full chain-bound Telys key for a chunk of ``block_count`` blocks."""
        return (
            f"{self.chunk_start_key(start_hash)}/{bytes(end_hash).hex()}/{block_count}"
        )

    def _exact_chunk(
        self,
        block_hashes: Sequence[bytes],
        idx: int,
        chunk_length: int,
    ) -> tuple[int, str]:
        """Return one exact chain-bound chunk without reading payload data."""
        start_key = self.chunk_start_key(block_hashes[idx])
        get_candidates = getattr(self._client, "chunk_candidates", None)
        candidates = get_candidates(start_key) if callable(get_candidates) else {}
        remaining = len(block_hashes) - idx
        length = min(chunk_length, remaining)
        if length > 0:
            key = self.chunk_key(
                block_hashes[idx], length, block_hashes[idx + length - 1]
            )
            if candidates:
                if candidates.get(length) == key:
                    return length, key
            elif self._client.contains(key):
                return length, key
        for length in sorted(candidates, reverse=True):
            if length > remaining:
                continue
            key = self.chunk_key(
                block_hashes[idx], length, block_hashes[idx + length - 1]
            )
            if candidates[length] == key:
                return length, key
        return 0, ""

    def contains(self, key: str) -> bool:
        """Fast membership check backed by the in-memory write tracking.

        Returns True for both confirmed and pending (write-behind) writes so the
        connector can deduplicate saves without waiting for the background worker.
        """
        with self._lock:
            if key in self._known_keys or key in self._pending_keys:
                return True
        return self._client.contains(key)

    def lookup_prefix(self, block_hashes: Sequence[bytes]) -> int:
        """Count consecutive stored blocks from the start of ``block_hashes``.

        Stops at the first missing chunk start, at the end of ``block_hashes``,
        and at ``config.max_lookup_blocks`` when non-zero. Never raises on miss.

        Only confirmed writes are counted; pending write-behind chunks are not
        offered as hits until they have been durably indexed in Telys.
        """
        with self._telemetry.span("store_lookup_prefix") as ctx:
            limit = self._config.max_lookup_blocks
            hits = 0
            idx = 0
            total = len(block_hashes)
            lookups = 0
            while idx < total:
                if limit and hits >= limit:
                    break
                start_key = self.chunk_start_key(block_hashes[idx])
                stored_length = self._client.chunk_length(start_key)
                if idx == 0:
                    ctx["first_chunk_start_key"] = start_key
                    ctx["first_chunk_length"] = stored_length
                lookups += 1
                length, _ = self._exact_chunk(block_hashes, idx, stored_length)
                if length == 0:
                    break
                hits += length
                idx += length
            ctx["total_blocks"] = total
            ctx["hits"] = hits
            ctx["lookups"] = lookups
        return hits

    def load_blocks(self, block_hashes: Sequence[bytes]) -> list[TelysRecord]:
        """Fetch blocks planned by a prior lookup, with provenance.

        Convenience wrapper around :meth:`load_decoded_blocks` that re-encodes
        each block into the legacy per-block payload format.
        """
        return [
            TelysRecord(
                key=decoded.key,
                value=self._block_codec.encode_block(decoded.blobs),
                provenance=decoded.provenance,
                metadata=decoded.metadata,
            )
            for decoded in self.load_decoded_blocks(block_hashes)
        ]

    def load_decoded_blocks(self, block_hashes: Sequence[bytes]) -> list[DecodedBlock]:
        """Fetch blocks as decoded layer blobs, with provenance.

        The prefix is reconstructed from stored chunks.  A chunk that vanished
        between lookup and load is a hard error, never a silent partial load.
        If the chunk is still queued in the write-behind path, the store drains
        until it is persisted before reading.
        """
        blocks: list[DecodedBlock] = []
        idx = 0
        total = len(block_hashes)
        while idx < total:
            start_hash = block_hashes[idx]
            start_key = self.chunk_start_key(start_hash)
            stored_length = self._client.chunk_length(start_key)
            chunk_len, full_key = self._exact_chunk(block_hashes, idx, stored_length)
            if chunk_len == 0:
                # The chunk may be pending in the write-behind queue.
                key = self.chunk_key(start_hash, total - idx, block_hashes[-1])
                if self._is_pending(key):
                    self.drain()
                    stored_length = self._client.chunk_length(start_key)
                    chunk_len, full_key = self._exact_chunk(
                        block_hashes, idx, stored_length
                    )
                if chunk_len == 0:
                    raise TelysStoreError(
                        f"planned load failed: block {start_key!r} is no longer present "
                        f"(evicted between lookup and load)"
                    )
            needed = min(chunk_len, total - idx)
            chunk_record = self._client.get(full_key)
            if chunk_record is None:
                raise TelysStoreError(
                    f"planned load failed: chunk {full_key!r} is no longer present "
                    f"(evicted between lookup and load)"
                )
            block_blobs = self._read_chunk_from_slab(full_key, chunk_record)
            if len(block_blobs) != chunk_len:
                raise TelysStoreError(
                    f"chunk {full_key!r} decoded {len(block_blobs)} blocks, "
                    f"expected {chunk_len}"
                )
            for offset in range(needed):
                block_hash = block_hashes[idx + offset]
                blocks.append(
                    DecodedBlock(
                        key=self.block_key(block_hash),
                        blobs=block_blobs[offset],
                        provenance=chunk_record.provenance,
                        metadata={
                            **chunk_record.metadata,
                            "vllm.block_hash": block_hash.hex(),
                        },
                    )
                )
            idx += needed
        return blocks

    def _is_pending(self, key: str) -> bool:
        with self._lock:
            return key in self._pending_keys

    def _read_chunk_bytes(
        self, key: str, record: TelysRecord
    ) -> tuple[bytes, Mapping[str, str]]:
        """Return the raw encoded chunk bytes plus metadata (format, provenance, etc.)."""
        if record.value:
            return record.value, record.metadata

        metadata = dict(record.metadata)
        offset_raw = metadata.pop(self._SLAB_OFFSET_KEY, None)
        length_raw = metadata.pop(self._SLAB_LENGTH_KEY, None)
        if offset_raw is None or length_raw is None:
            raise TelysStoreError(
                f"chunk {key!r} has no payload value, slab pointer, or inline payload"
            )
        if self._slab is None:
            raise TelysStoreError(
                f"chunk {key!r} has slab pointer but no slab is configured"
            )
        try:
            offset = int(offset_raw)
            length = int(length_raw)
        except ValueError as exc:
            raise TelysStoreError(
                f"chunk {key!r} has invalid slab pointer ({offset_raw!r}, {length_raw!r})"
            ) from exc
        return self._slab.read(offset, length), metadata

    def _read_chunk_from_slab(
        self, key: str, record: TelysRecord
    ) -> list[dict[str, bytes]]:
        """Read and decode a chunk into the legacy per-block representation."""
        with self._telemetry.span("store_read_chunk") as read_ctx:
            payload, metadata = self._read_chunk_bytes(key, record)
            prefer_fast = (
                metadata.get(self._CHUNK_FORMAT_KEY) == self._CHUNK_FORMAT_FAST
            )
            with self._telemetry.span("store_decode_chunk") as dec_ctx:
                is_fast, decoded = self._decode_chunk_payload(
                    payload, metadata, prefer_fast=prefer_fast
                )
                dec_ctx["source"] = "fast" if is_fast else "blob"
                dec_ctx["payload_bytes"] = len(payload)
            if is_fast:
                if torch is None:
                    raise TelysStoreError("torch is required to decode fast chunks")
                result = self._tensors_to_block_blobs(decoded)
            else:
                result = decoded
            read_ctx["decoded_blocks"] = len(result)
        return result

    def _decode_chunk_payload(
        self,
        payload: bytes,
        metadata: Mapping[str, str],
        *,
        prefer_fast: bool,
    ) -> tuple[bool, list[dict[str, bytes]] | dict[str, "torch.Tensor"]]:
        """Decode with the intended codec, falling back without magic-byte guesses."""
        intended_decode = (
            self._fast_chunk_codec.decode_chunk
            if prefer_fast
            else self._chunk_codec.decode_chunk
        )
        other_decode = (
            self._chunk_codec.decode_chunk
            if prefer_fast
            else self._fast_chunk_codec.decode_chunk
        )
        try:
            decoded = intended_decode(payload)
        except (PayloadCodecError, UnicodeError) as intended_error:
            try:
                decoded = other_decode(payload)
            except (PayloadCodecError, UnicodeError):
                raise intended_error
            return not prefer_fast, decoded

        if self._CHUNK_FORMAT_KEY not in metadata:
            try:
                other_decode(payload)
            except (PayloadCodecError, UnicodeError):
                pass
            else:
                raise PayloadCodecError(
                    "ambiguous chunk payload: valid as both legacy and fast formats"
                )
        return prefer_fast, decoded

    def load_decoded_blocks_fast(
        self, block_hashes: Sequence[bytes]
    ) -> list[FastDecodedChunk]:
        """Fetch blocks as stacked layer tensors for the fast zero-decode path.

        Returns one :class:`FastDecodedChunk` per stored chunk (a prefix may span
        multiple chunks).  Each chunk's ``layer_tensors`` is a dict mapping layer
        name to a CPU tensor of shape ``[num_blocks, block_size, *]``.
        """
        if torch is None:
            raise TelysStoreError("torch is required for load_decoded_blocks_fast")
        chunks: list[FastDecodedChunk] = []
        idx = 0
        total = len(block_hashes)
        while idx < total:
            start_hash = block_hashes[idx]
            start_key = self.chunk_start_key(start_hash)
            stored_length = self._client.chunk_length(start_key)
            chunk_len, full_key = self._exact_chunk(block_hashes, idx, stored_length)
            if chunk_len == 0:
                key = self.chunk_key(start_hash, total - idx, block_hashes[-1])
                if self._is_pending(key):
                    self.drain()
                    stored_length = self._client.chunk_length(start_key)
                    chunk_len, full_key = self._exact_chunk(
                        block_hashes, idx, stored_length
                    )
                if chunk_len == 0:
                    raise TelysStoreError(
                        f"planned load failed: block {start_key!r} is no longer present "
                        f"(evicted between lookup and load)"
                    )
            needed = min(chunk_len, total - idx)
            chunk_record = self._client.get(full_key)
            if chunk_record is None:
                raise TelysStoreError(
                    f"planned load failed: chunk {full_key!r} is no longer present "
                    f"(evicted between lookup and load)"
                )
            payload, metadata = self._read_chunk_bytes(full_key, chunk_record)
            prefer_fast = (
                self._CHUNK_FORMAT_KEY not in metadata
                or metadata.get(self._CHUNK_FORMAT_KEY) == self._CHUNK_FORMAT_FAST
            )

            with self._telemetry.span("store_read_chunk") as read_ctx:
                is_fast, decoded = self._decode_chunk_payload(
                    payload, metadata, prefer_fast=prefer_fast
                )
                layer_tensors = (
                    decoded if is_fast else self._block_blobs_to_tensors(decoded)
                )
                read_ctx["fast_path"] = is_fast
                read_ctx["payload_bytes"] = len(payload)
                if not layer_tensors:
                    raise TelysStoreError(f"chunk {full_key!r} has no layers")
                decoded_blocks = len(layer_tensors.get(next(iter(layer_tensors)), []))
                read_ctx["decoded_blocks"] = decoded_blocks
            if decoded_blocks != chunk_len:
                raise TelysStoreError(
                    f"chunk {full_key!r} decoded {decoded_blocks} blocks, "
                    f"expected {chunk_len}"
                )

            chunks.append(
                FastDecodedChunk(
                    provenance=chunk_record.provenance,
                    metadata={
                        **chunk_record.metadata,
                        "vllm.block_hashes": " ".join(
                            block_hashes[idx + offset].hex() for offset in range(needed)
                        ),
                    },
                    block_hashes=tuple(block_hashes[idx : idx + needed]),
                    layer_tensors=layer_tensors,
                )
            )
            idx += needed
        return chunks

    @staticmethod
    def _tensors_to_block_blobs(
        layer_tensors: Mapping[str, "torch.Tensor"],
    ) -> list[dict[str, bytes]]:
        """Split stacked layer tensors into per-block serialized bytes."""
        from telys_vllm.gpu_ops import serialize_kv_tensor

        if not layer_tensors:
            return []
        items = sorted(layer_tensors.items())
        num_blocks = items[0][1].shape[0]
        blocks: list[dict[str, bytes]] = []
        for block_idx in range(num_blocks):
            block_blobs: dict[str, bytes] = {}
            for name, tensor in items:
                block_tensor = tensor[block_idx]
                block_blobs[name] = serialize_kv_tensor(block_tensor)
            blocks.append(block_blobs)
        return blocks

    @staticmethod
    def _block_blobs_to_tensors(
        block_blobs: Sequence[Mapping[str, bytes]],
    ) -> dict[str, "torch.Tensor"]:
        """Convert legacy serialized blocks to stacked tensors for fast loads."""
        from telys_vllm.gpu_ops import deserialize_kv_tensor

        if not block_blobs:
            raise PayloadCodecError("legacy chunk contained no blocks")
        layer_names = set(block_blobs[0])
        if not layer_names or any(set(block) != layer_names for block in block_blobs):
            raise PayloadCodecError("legacy chunk layers differ between blocks")

        layer_tensors: dict[str, torch.Tensor] = {}
        for name in sorted(layer_names):
            try:
                tensors = [deserialize_kv_tensor(block[name]) for block in block_blobs]
                first = tensors[0]
                if any(
                    tensor.shape != first.shape or tensor.dtype != first.dtype
                    for tensor in tensors[1:]
                ):
                    raise PayloadCodecError(
                        f"legacy chunk tensor shape or dtype differs in layer {name!r}"
                    )
                layer_tensors[name] = torch.stack(tensors)
            except PayloadCodecError:
                raise
            except Exception as exc:
                raise PayloadCodecError(
                    f"invalid serialized tensor in legacy chunk layer {name!r}"
                ) from exc
        return layer_tensors

    def save_chunk(
        self,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        layer_blobs: Mapping[str, Sequence[bytes]],
        extra_metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Persist one contiguous run of blocks.

        When write-behind is enabled, this returns immediately after enqueueing
        the work.  Call ``drain()`` before relying on the write being readable.
        """
        encoded_chunk = self._encode_chunk(start_hash, block_hashes, layer_blobs)
        key = self.chunk_key(start_hash, len(block_hashes), block_hashes[-1])
        if self._config.write_behind:
            self._enqueue(key, start_hash, block_hashes, encoded_chunk, extra_metadata)
            return key
        self._persist_now(key, start_hash, block_hashes, encoded_chunk, extra_metadata)
        return key

    def save_chunk_fast(
        self,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        layer_tensors: Mapping[str, "torch.Tensor"],
        extra_metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Persist one contiguous run of blocks using the raw-byte fast format.

        ``layer_tensors`` maps layer name to a stacked CPU tensor of shape
        ``[len(block_hashes), block_size, *]``.  The encoded bytes are the raw
        tensor data and are decoded with ``torch.frombuffer`` on the hot path.
        With write-behind enabled, the pending write owns the tensors, which the
        caller must treat as immutable until :meth:`drain` returns.
        """
        if torch is None:
            raise TelysStoreError("torch is required for save_chunk_fast")
        if not block_hashes:
            raise TelysStoreError("save_chunk_fast requires at least one block")
        counts = {tensor.shape[0] for tensor in layer_tensors.values()}
        if len(counts) != 1 or counts.pop() != len(block_hashes):
            raise TelysStoreError(
                "layer_tensors must contain one entry per block in the chunk"
            )
        key = self.chunk_key(start_hash, len(block_hashes), block_hashes[-1])
        fast_metadata = {"vllm.chunk_format": self._CHUNK_FORMAT_FAST}
        if extra_metadata:
            fast_metadata.update(extra_metadata)
        if self._config.write_behind:
            self._enqueue_fast(
                key, start_hash, block_hashes, layer_tensors, fast_metadata
            )
            return key
        encoded_chunk = self._fast_chunk_codec.encode_chunk(dict(layer_tensors))
        self._persist_now(key, start_hash, block_hashes, encoded_chunk, fast_metadata)
        return key

    def save_chunk_async(
        self,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        encoded_chunk: bytes,
        extra_metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Enqueue an already-encoded chunk for background persistence.

        The connector uses this from ``wait_for_save`` so the expensive work of
        tensor extraction/serialization is done in the caller thread, while the
        background worker only appends to the slab and writes the tiny Telys
        metadata record.

        When write-behind is disabled, the chunk is persisted synchronously so
        callers can rely on it being readable immediately.
        """
        key = self.chunk_key(start_hash, len(block_hashes), block_hashes[-1])
        if self._config.write_behind:
            self._enqueue(key, start_hash, block_hashes, encoded_chunk, extra_metadata)
            return key
        self._persist_now(key, start_hash, block_hashes, encoded_chunk, extra_metadata)
        return key

    def _encode_chunk(
        self,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        layer_blobs: Mapping[str, Sequence[bytes]],
    ) -> bytes:
        if not block_hashes:
            raise TelysStoreError("save_chunk requires at least one block")
        counts = {len(blobs) for blobs in layer_blobs.values()}
        if len(counts) != 1 or counts.pop() != len(block_hashes):
            raise TelysStoreError(
                "layer_blobs must contain one entry per block in the chunk"
            )
        return self._chunk_codec.encode_chunk(layer_blobs)

    def _build_metadata(
        self,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        extra_metadata: Mapping[str, str] | None,
        offset: int | None,
        length: int | None,
        chunk_format: str | None = None,
    ) -> dict[str, str]:
        metadata: dict[str, str] = {
            "vllm.block_start_hash": bytes(start_hash).hex(),
            "vllm.block_count": str(len(block_hashes)),
            "vllm.block_size": str(self._block_size),
            "vllm.model": self._model_id,
            "vllm.engine_id": self._engine_id,
        }
        if offset is not None and length is not None:
            metadata[self._SLAB_OFFSET_KEY] = str(offset)
            metadata[self._SLAB_LENGTH_KEY] = str(length)
        if chunk_format is not None:
            metadata[self._CHUNK_FORMAT_KEY] = chunk_format
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _enqueue(
        self,
        key: str,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        encoded_chunk: bytes,
        extra_metadata: Mapping[str, str] | None,
    ) -> None:
        item = _PendingWrite(
            key=key,
            start_hash=start_hash,
            block_hashes=tuple(block_hashes),
            encoded_chunk=encoded_chunk,
            extra_metadata=dict(extra_metadata or {}),
        )
        self._enqueue_item(item)

    def _enqueue_fast(
        self,
        key: str,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        layer_tensors: Mapping[str, "torch.Tensor"],
        extra_metadata: Mapping[str, str] | None,
    ) -> None:
        item = _PendingFastWrite(
            key=key,
            start_hash=start_hash,
            block_hashes=tuple(block_hashes),
            layer_tensors=dict(layer_tensors),
            extra_metadata=dict(extra_metadata or {}),
        )
        self._enqueue_item(item)

    def _enqueue_item(self, item: _PendingWrite | _PendingFastWrite) -> None:
        """Add a write to the background queue, marking it pending for dedup."""
        with self._lock:
            if item.key in self._known_keys or item.key in self._pending_keys:
                return
            self._pending_keys.add(item.key)
            self._pending_count += 1
        try:
            self._write_queue.put(item, block=True)
        except Exception:
            # If enqueue fails, roll back the pending marker.
            with self._lock:
                self._pending_keys.discard(item.key)
                self._pending_count -= 1
            raise

    def _persist_now(
        self,
        key: str,
        start_hash: bytes,
        block_hashes: Sequence[bytes],
        encoded_chunk: bytes,
        extra_metadata: Mapping[str, str] | None,
    ) -> None:
        """Synchronous persistence path used when write-behind is disabled."""
        with self._telemetry.span("store_persist") as ctx:
            ctx["key"] = key
            ctx["block_count"] = len(block_hashes)
            ctx["payload_bytes"] = len(encoded_chunk)
            if self._slab is not None:
                offset, length = self._slab.append(encoded_chunk)
            else:
                offset, length = None, None
            metadata = self._build_metadata(
                start_hash, block_hashes, extra_metadata, offset, length
            )
            # Pass the encoded chunk as the client value. BlobStore-backed
            # clients store it; legacy slab clients ignore the value and use
            # the slab pointer in metadata.
            self._client.put(key, encoded_chunk, metadata)
            self._track_write_completed(key)

    def _write_worker(self) -> None:
        """Background thread: encode fast chunks and persist queued writes."""
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            try:
                self._persist_item(item)
            except Exception as exc:
                with self._lock:
                    self._write_errors.append((item.key, exc))
                logger.exception("telys async write failed for key=%s", item.key)
            finally:
                with self._lock:
                    self._pending_count -= 1
                    if self._pending_count == 0:
                        self._drain_event.set()
            del item

    def _persist_item(self, item: _PendingWrite | _PendingFastWrite) -> None:
        with self._telemetry.span("store_persist") as ctx:
            if isinstance(item, _PendingFastWrite):
                encoded_chunk = self._fast_chunk_codec.encode_chunk(item.layer_tensors)
            else:
                encoded_chunk = item.encoded_chunk
            ctx["key"] = item.key
            ctx["block_count"] = len(item.block_hashes)
            ctx["payload_bytes"] = len(encoded_chunk)
            if self._slab is not None:
                offset, length = self._slab.append(encoded_chunk)
            else:
                offset, length = None, None
            metadata = self._build_metadata(
                item.start_hash,
                item.block_hashes,
                item.extra_metadata,
                offset,
                length,
            )
            self._client.put(item.key, encoded_chunk, metadata)
            self._track_write_completed(item.key)

    def _track_write_completed(self, key: str) -> None:
        """Update in-memory indexes after a successful write."""
        with self._lock:
            self._known_keys.add(key)
            self._pending_keys.discard(key)

    def drain(self) -> None:
        """Wait for queued writes, then raise the oldest worker error, if any."""
        if not self._config.write_behind or self._closed:
            return
        while True:
            with self._lock:
                if self._pending_count == 0 and self._write_queue.empty():
                    if self._write_errors:
                        key, error = self._write_errors.pop(0)
                        self._pending_keys.discard(key)
                        raise error
                    return
            self._drain_event.clear()
            self._drain_event.wait(timeout=1.0)

    def compact(self) -> TelysMaintenanceReport:
        """Ask Telys to run compaction / index maintenance (eviction hook)."""
        return self._client.compact()

    def reset(self) -> None:
        """Delete every persisted connector key, clear the slab, then compact."""
        self.drain()
        for key in sorted(self._client.keys()):
            self._client.delete(key)
        self._known_keys.clear()
        with self._lock:
            self._pending_keys.clear()
            self._pending_count = 0
        if self._slab is not None:
            self._slab.clear()
        self.compact()

    def close(self) -> None:
        """Drain pending writes, stop the worker, and close the slab."""
        if self._closed:
            return
        self._closed = True
        # Unregister the atexit drain first; we are about to drain explicitly.
        atexit.unregister(self.drain)
        self.drain()
        if self._worker is not None and self._worker.is_alive():
            try:
                self._write_queue.put_nowait(None)
                self._worker.join(timeout=5.0)
            except Exception:
                logger.exception("error stopping async write worker")
        if self._slab is not None:
            self._slab.close()
        self._client.close()
