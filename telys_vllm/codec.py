# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-block payload framing.

A Telys entry holds every layer's KV bytes for one block.
``LengthPrefixedCodec`` frames them as::

    [u32 name_len][name bytes][u64 blob_len][blob bytes] * num_layers

with layers sorted by name, so the same inputs always produce byte-identical
payloads (and identical block hashes therefore always map to identical
Telys values). The GPU tensor -> per-block-bytes extraction is a separate,
later implementation of the same ``KVPayloadCodec`` seam (DESIGN.md
section 5.1).
"""

import struct
import warnings
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only dev/test environments
    torch = None  # type: ignore[assignment]

_BLOCK_COUNT = struct.Struct("<I")
_BLOCK_LEN = struct.Struct("<Q")
_NAME_LEN = struct.Struct("<I")
_BLOB_LEN = struct.Struct("<Q")


class PayloadCodecError(ValueError):
    """Raised on malformed payloads or unsupported blob types."""


@runtime_checkable
class KVPayloadCodec(Protocol):
    """Framing contract between per-layer blobs and one block payload."""

    def encode_block(self, layer_blobs: Mapping[str, bytes]) -> bytes:
        """Frame all layers' blobs for one block into a single payload."""
        ...

    def decode_block(self, payload: bytes) -> dict[str, bytes]:
        """Invert :meth:`encode_block`. Raises PayloadCodecError if malformed."""
        ...


class TelysChunkCodec:
    """Framing for a contiguous run of blocks in a single Telys record.

    Format::

        [u32 block_count]
        [u64 block_payload_len][block_payload_bytes] * block_count

    Each block payload is a standard :class:`LengthPrefixedCodec` block, so
    the same inputs always produce byte-identical chunks.
    """

    def encode_chunk(self, layer_blobs: Mapping[str, Sequence[bytes]]) -> bytes:
        """Frame a contiguous run of blocks.

        ``layer_blobs`` maps layer name -> list of per-block bytes, where every
        list has the same length (the number of blocks in the run).
        """
        if not layer_blobs:
            raise PayloadCodecError("encode_chunk requires at least one layer")
        counts = {len(blobs) for blobs in layer_blobs.values()}
        if len(counts) != 1:
            raise PayloadCodecError(
                f"all layers must have the same block count, got {counts}"
            )
        block_count = counts.pop()
        if block_count == 0:
            raise PayloadCodecError("encode_chunk requires at least one block")

        block_codec = LengthPrefixedCodec()
        parts: list[bytes] = [_BLOCK_COUNT.pack(block_count)]
        for block_idx in range(block_count):
            block_payload = block_codec.encode_block(
                {name: blobs[block_idx] for name, blobs in layer_blobs.items()}
            )
            parts.append(_BLOCK_LEN.pack(len(block_payload)))
            parts.append(block_payload)
        return b"".join(parts)

    def decode_chunk(self, payload: bytes) -> list[dict[str, bytes]]:
        """Invert :meth:`encode_chunk`. Returns one per-block blob dict per block."""
        if len(payload) < _BLOCK_COUNT.size:
            raise PayloadCodecError("truncated chunk header")
        (block_count,) = _BLOCK_COUNT.unpack_from(payload, 0)
        offset = _BLOCK_COUNT.size
        block_codec = LengthPrefixedCodec()
        blocks: list[dict[str, bytes]] = []
        for _ in range(block_count):
            if offset + _BLOCK_LEN.size > len(payload):
                raise PayloadCodecError("truncated block length")
            (block_len,) = _BLOCK_LEN.unpack_from(payload, offset)
            offset += _BLOCK_LEN.size
            if offset + block_len > len(payload):
                raise PayloadCodecError("truncated block payload")
            blocks.append(
                block_codec.decode_block(payload[offset : offset + block_len])
            )
            offset += block_len
        if offset != len(payload):
            raise PayloadCodecError("trailing bytes after chunk")
        return blocks


class TelysFastChunkCodec:
    """Flat raw-byte chunk framing for zero-decode loads.

    Stores one contiguous raw-byte tensor per layer (all blocks of the chunk),
    so decoding is a single ``torch.frombuffer`` view per layer.  Layers are
    sorted by name to keep the encoding deterministic.

    Format (v2)::

        [1 byte magic=0x02]
        [1 byte flags]
        [2 bytes num_blocks big-endian uint16]
        [2 bytes num_layers big-endian uint16]
        [4 bytes block_size big-endian uint32]
        For each layer (sorted by name):
            [2 bytes name_len]
            [name_len bytes name]
            [1 byte dtype_code]
            [1 byte ndim]
            [ndim * 2 bytes shape big-endian uint16]
        For each layer (sorted by name):
            [num_blocks * prod(shape) * dtype_itemsize bytes raw data]
    """

    _MAGIC = 0x02
    _HEADER_SIZE = 8  # magic + flags + num_blocks + num_layers + block_size

    def __init__(self, max_layers: int = 256, max_blocks: int = 65535) -> None:
        self._max_layers = max_layers
        self._max_blocks = max_blocks

    def encode_chunk(self, layer_tensors: Mapping[str, "torch.Tensor"]) -> bytes:
        """Encode stacked layer tensors (shape [num_blocks, *block_shape]) to bytes."""
        if torch is None:
            raise PayloadCodecError("torch is required for TelysFastChunkCodec")
        if not layer_tensors:
            raise PayloadCodecError("encode_chunk requires at least one layer")

        items = sorted(layer_tensors.items())
        num_layers = len(items)
        if num_layers > self._max_layers:
            raise PayloadCodecError(
                f"too many layers: {num_layers} > {self._max_layers}"
            )

        block_counts = {tensor.shape[0] for tensor in layer_tensors.values()}
        if len(block_counts) != 1:
            raise PayloadCodecError(
                f"all layer tensors must have the same block count, got {block_counts}"
            )
        num_blocks = block_counts.pop()
        if num_blocks == 0:
            raise PayloadCodecError("encode_chunk requires at least one block")
        if num_blocks > self._max_blocks:
            raise PayloadCodecError(
                f"too many blocks: {num_blocks} > {self._max_blocks}"
            )

        # All layers share the same block_size (the first dimension of the per-block shape).
        per_block_shape = items[0][1].shape[1:]
        block_size = per_block_shape[0] if per_block_shape else 1

        parts: list[bytes] = [
            struct.pack(
                ">BBHHH",
                self._MAGIC,
                0,  # flags
                num_blocks,
                num_layers,
                block_size,
            )
        ]
        layer_headers: list[bytes] = []
        data_parts: list[bytes] = []
        for name, tensor in items:
            tensor = tensor.detach().cpu().contiguous()
            if tensor.ndim - 1 > 4:
                raise PayloadCodecError(
                    f"per-block tensor supports up to 4 dims, got {tensor.ndim - 1}"
                )
            for dim in tensor.shape:
                if dim >= 65536:
                    raise PayloadCodecError(f"tensor dimension too large: {dim}")

            dtype_code = self._dtype_to_code(tensor.dtype)
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > 65535:
                raise PayloadCodecError(f"layer name too long: {name!r}")

            header = struct.pack(
                ">HB",
                len(name_bytes),
                dtype_code,
            )
            header += struct.pack(">B", tensor.ndim - 1)
            for dim in tensor.shape[1:]:
                header += struct.pack(">H", int(dim))
            layer_headers.append(header + name_bytes)

            data = tensor.view(torch.uint8).numpy().tobytes()
            data_parts.append(data)

        parts.extend(layer_headers)
        parts.extend(data_parts)
        return b"".join(parts)

    def decode_chunk(self, payload: bytes) -> dict[str, "torch.Tensor"]:
        """Decode bytes to stacked CPU tensors via zero-copy ``torch.frombuffer``."""
        if torch is None:
            raise PayloadCodecError("torch is required for TelysFastChunkCodec")
        if len(payload) < self._HEADER_SIZE:
            raise PayloadCodecError("truncated fast chunk header")

        magic, flags, num_blocks, num_layers, _block_size = struct.unpack(
            ">BBHHH", payload[: self._HEADER_SIZE]
        )
        if magic != self._MAGIC:
            raise PayloadCodecError(f"unknown fast chunk magic: {magic}")
        if flags != 0:
            raise PayloadCodecError(f"unknown fast chunk flags: {flags}")
        if num_layers == 0:
            raise PayloadCodecError("fast chunk has no layers")

        offset = self._HEADER_SIZE
        layer_infos: list[tuple[str, int, tuple[int, ...], int]] = []
        total_data_size = 0
        for _ in range(num_layers):
            if offset + 4 > len(payload):
                raise PayloadCodecError("truncated fast chunk layer header")
            name_len, dtype_code = struct.unpack(">HB", payload[offset : offset + 3])
            offset += 3
            ndim = payload[offset]
            offset += 1
            if offset + ndim * 2 > len(payload):
                raise PayloadCodecError("truncated fast chunk layer shape")
            shape = tuple(
                struct.unpack(">H", payload[offset + i * 2 : offset + i * 2 + 2])[0]
                for i in range(ndim)
            )
            offset += ndim * 2
            if offset + name_len > len(payload):
                raise PayloadCodecError("truncated fast chunk layer name")
            name = payload[offset : offset + name_len].decode("utf-8")
            offset += name_len

            dtype = self._code_to_dtype(dtype_code)
            block_nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
            layer_nbytes = num_blocks * block_nbytes
            layer_infos.append((name, layer_nbytes, shape, dtype_code))
            total_data_size += layer_nbytes

        if len(payload) - offset != total_data_size:
            raise PayloadCodecError(
                f"fast chunk data size mismatch: expected {total_data_size} bytes, "
                f"got {len(payload) - offset}"
            )

        result: dict[str, torch.Tensor] = {}
        for name, layer_nbytes, shape, dtype_code in layer_infos:
            dtype = self._code_to_dtype(dtype_code)
            itemsize = torch.empty((), dtype=dtype).element_size()
            count = layer_nbytes // itemsize
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tensor = torch.frombuffer(
                    payload, dtype=dtype, offset=offset, count=count
                ).reshape((num_blocks,) + shape)
            result[name] = tensor
            offset += layer_nbytes

        return result

    def decode_chunk_to_pinned(self, payload: bytes) -> dict[str, "torch.Tensor"]:
        """Decode to stacked tensors in pinned CPU memory.

        Returns writable tensors suitable as the source of an async H2D copy.
        """
        if torch is None:
            raise PayloadCodecError("torch is required for TelysFastChunkCodec")
        decoded = self.decode_chunk(payload)
        pinned: dict[str, torch.Tensor] = {}
        for name, tensor in decoded.items():
            # frombuffer on a bytes object gives a read-only view; copy into a
            # pinned writable buffer so non_blocking H2D copies are valid.
            if tensor.numel() == 0:
                pinned[name] = tensor.clone()
                continue
            pinned_buf = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            pinned_buf.copy_(tensor)
            pinned[name] = pinned_buf
        return pinned

    @staticmethod
    def _dtype_to_code(dtype: "torch.dtype") -> int:
        from telys_vllm.gpu_ops import _dtype_to_code as gpu_dtype_to_code

        return gpu_dtype_to_code(dtype)

    @staticmethod
    def _code_to_dtype(code: int) -> "torch.dtype":
        from telys_vllm.gpu_ops import _code_to_dtype as gpu_code_to_dtype

        return gpu_code_to_dtype(code)


class LengthPrefixedCodec:
    """Length-prefixed, layer-name-sorted deterministic framing."""

    def encode_block(self, layer_blobs: Mapping[str, bytes]) -> bytes:
        if not layer_blobs:
            raise PayloadCodecError("encode_block requires at least one layer blob")
        parts: list[bytes] = []
        for name in sorted(layer_blobs):
            blob = layer_blobs[name]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise PayloadCodecError(
                    f"layer {name!r}: expected a bytes-like blob, "
                    f"got {type(blob).__name__}"
                )
            name_bytes = name.encode("utf-8")
            blob_bytes = bytes(blob)
            parts.append(_NAME_LEN.pack(len(name_bytes)))
            parts.append(name_bytes)
            parts.append(_BLOB_LEN.pack(len(blob_bytes)))
            parts.append(blob_bytes)
        return b"".join(parts)

    def decode_block(self, payload: bytes) -> dict[str, bytes]:
        layers: dict[str, bytes] = {}
        offset = 0
        while offset < len(payload):
            name, offset = self._read_str(payload, offset)
            blob, offset = self._read_blob(payload, offset)
            if name in layers:
                raise PayloadCodecError(f"duplicate layer {name!r} in payload")
            layers[name] = blob
        if not layers:
            raise PayloadCodecError("payload contained no layer blobs")
        return layers

    @staticmethod
    def _read_str(payload: bytes, offset: int) -> tuple[str, int]:
        if offset + _NAME_LEN.size > len(payload):
            raise PayloadCodecError("truncated layer-name length")
        (name_len,) = _NAME_LEN.unpack_from(payload, offset)
        offset += _NAME_LEN.size
        if offset + name_len > len(payload):
            raise PayloadCodecError("truncated layer name")
        name = payload[offset : offset + name_len].decode("utf-8")
        return name, offset + name_len

    @staticmethod
    def _read_blob(payload: bytes, offset: int) -> tuple[bytes, int]:
        if offset + _BLOB_LEN.size > len(payload):
            raise PayloadCodecError("truncated blob length")
        (blob_len,) = _BLOB_LEN.unpack_from(payload, offset)
        offset += _BLOB_LEN.size
        if offset + blob_len > len(payload):
            raise PayloadCodecError("truncated blob")
        return payload[offset : offset + blob_len], offset + blob_len
