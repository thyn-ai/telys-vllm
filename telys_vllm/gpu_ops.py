# SPDX-License-Identifier: Apache-2.0
"""GPU tensor extraction/injection helpers for the Telys KV connector.

Per-block KV slices are moved to CPU, serialized to deterministic raw bytes
(avoiding pickle/torch.save on the hot path), and framed by
:class:`~telys_vllm.codec.LengthPrefixedCodec`. Legacy ``torch.save`` blobs
are still decoded for backward compatibility.
"""

from __future__ import annotations

import io
import struct
import warnings
from typing import TYPE_CHECKING

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only dev/test environments
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for GPU KV extraction/injection")


def _slot_mapping(
    block_ids: list[int], block_size: int, device: torch.device
) -> torch.Tensor:
    """Flattened slot indices for the given block ids."""
    block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=device)
    offsets = torch.arange(block_size, dtype=torch.long, device=device)
    return (block_ids_t.unsqueeze(1) * block_size + offsets).flatten()


def _is_mla(attn_metadata: AttentionMetadata | None) -> bool:
    """True when the current layer uses MLA layout."""
    if attn_metadata is None:
        return False
    # Local import to avoid a hard dependency on model-executor internals.
    try:
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadata,
        )
    except ImportError:
        return False
    return isinstance(attn_metadata, MLACommonMetadata)


def extract_kv_blocks(
    kv_layer: torch.Tensor,
    block_ids: list[int],
    block_size: int,
    attn_metadata: AttentionMetadata | None,
) -> list[torch.Tensor]:
    """Extract one CPU tensor per block from a paged KV layer.

    Returns a list aligned with ``block_ids``.  Serialization is deferred to
    the caller so the connector can group blocks into chunks before paying for
    ``torch.save`` / codec framing.
    """
    _require_torch()
    if not block_ids:
        return []

    slot_mapping = _slot_mapping(block_ids, block_size, kv_layer.device)

    if _is_mla(attn_metadata):
        # [B, 1, N, C] -> [B * N, C]; slot_mapping indexes B * N slots.
        extracted = kv_layer.reshape(-1, kv_layer.shape[-1])[slot_mapping, ...]
        extracted = extracted.view(len(block_ids), block_size, -1)
    else:
        # [B, H, N, C]; index by block and offset within the block.
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        extracted = kv_layer[block_idxs, :, offsets]
        extracted = extracted.view(
            len(block_ids), block_size, extracted.shape[1], extracted.shape[2]
        )

    return [extracted[i].detach().cpu() for i in range(len(block_ids))]


def serialize_kv_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a CPU KV block tensor to deterministic raw bytes.

    Format (v1)::

        [1 byte magic=0x01]
        [1 byte dtype_code]
        [1 byte ndim]
        [ndim * 2 bytes shape as big-endian uint16]
        [tensor.nbytes of raw data]

    This avoids pickle/torch.save overhead and makes deserialization a
    zero-copy ``torch.frombuffer`` view.
    """
    _require_torch()
    if tensor.ndim > 4:
        raise ValueError(
            f"serialize_kv_tensor supports up to 4 dims, got {tensor.ndim}"
        )
    for dim in tensor.shape:
        if dim >= 65536:
            raise ValueError(f"serialize_kv_tensor dimension too large: {dim}")

    dtype_code = _dtype_to_code(tensor.dtype)
    header = struct.pack(
        ">BBB",
        _RAW_FORMAT_VERSION,
        dtype_code,
        tensor.ndim,
    )
    shape_bytes = b"".join(struct.pack(">H", int(dim)) for dim in tensor.shape)
    # Use a uint8 view + numpy().tobytes() so views serialize only their own
    # data and dtypes unsupported by numpy (e.g. bfloat16) are handled uniformly.
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return header + shape_bytes + data


# Raw-byte format constants.
_RAW_FORMAT_VERSION = 0x01
_MAX_RAW_DIMS = 4

# Pickle protocol 2 marker; legacy blobs start with this.
_PICKLE_PROTOCOL_MARKER = 0x80

# torch.dtype -> code mapping for the raw-byte header.
_DTYPE_TO_CODE: dict[torch.dtype, int] = {}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {}


def _register_dtype(code: int, dtype: torch.dtype) -> None:
    _DTYPE_TO_CODE[dtype] = code
    _CODE_TO_DTYPE[code] = dtype


if torch is not None:
    _register_dtype(0, torch.float32)
    _register_dtype(1, torch.float16)
    _register_dtype(2, torch.bfloat16)


def _dtype_to_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype for raw-byte serialization: {dtype}"
        ) from exc


def _code_to_dtype(code: int) -> torch.dtype:
    if torch is None:
        raise RuntimeError("torch is required for dtype code lookup")
    try:
        return _CODE_TO_DTYPE[code]
    except KeyError as exc:
        raise ValueError(f"unknown dtype code: {code}") from exc


def deserialize_kv_tensor(buf: bytes) -> torch.Tensor:
    """Deserialize a KV block tensor from raw bytes or a legacy torch.save blob.

    Raw-byte blobs are decoded with ``torch.frombuffer`` (zero-copy). Legacy
    blobs produced by older versions with ``torch.save`` are detected by the
    pickle protocol marker and decoded with ``torch.load``.
    """
    _require_torch()
    if (
        len(buf) < 3
        or buf[0] != _RAW_FORMAT_VERSION
        or buf[0] == _PICKLE_PROTOCOL_MARKER
    ):
        # Legacy path: pickle/torch.save format.
        return torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)

    version, dtype_code, ndim = struct.unpack(">BBB", buf[:3])
    if version != _RAW_FORMAT_VERSION:
        raise ValueError(f"unknown raw-byte format version: {version}")
    if ndim > _MAX_RAW_DIMS:
        raise ValueError(f"raw-byte tensor has too many dims: {ndim}")

    header_end = 3 + ndim * 2
    if len(buf) < header_end:
        raise ValueError("raw-byte tensor header truncated")
    shape = tuple(
        struct.unpack(">H", buf[3 + i * 2 : 5 + i * 2])[0] for i in range(ndim)
    )

    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"unknown raw-byte dtype code: {dtype_code}")

    expected_nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
    if len(buf) - header_end != expected_nbytes:
        raise ValueError(
            f"raw-byte tensor size mismatch: expected {expected_nbytes} bytes, "
            f"got {len(buf) - header_end}"
        )

    # frombuffer returns a 1-D view; reshape to the stored shape.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.frombuffer(buf, dtype=dtype, offset=header_end).reshape(shape)


def legacy_serialize_kv_tensor(tensor: torch.Tensor) -> bytes:
    """Legacy pickle-based serializer kept for migration tests."""
    _require_torch()
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def inject_kv_block(
    kv_layer: torch.Tensor,
    block_id: int,
    block_size: int,
    attn_metadata: AttentionMetadata | None,
    block_tensor: torch.Tensor,
) -> None:
    """Inject one block's KV tensor into the paged KV layer."""
    inject_kv_blocks(kv_layer, [block_id], block_size, attn_metadata, [block_tensor])


def inject_kv_blocks(
    kv_layer: torch.Tensor,
    block_ids: list[int],
    block_size: int,
    attn_metadata: AttentionMetadata | None,
    block_tensors: list[torch.Tensor],
) -> None:
    """Inject multiple blocks' KV tensors into the paged KV layer in one pass.

    ``block_tensors`` must align with ``block_ids`` and each tensor must have the
    shape of a single block: ``[block_size, *, head_dim]``.
    """
    _require_torch()
    if not block_ids:
        return
    if len(block_ids) != len(block_tensors):
        raise ValueError(
            f"block_ids ({len(block_ids)}) and block_tensors ({len(block_tensors)}) "
            "must have the same length"
        )

    device = kv_layer.device
    slot_mapping = _slot_mapping(block_ids, block_size, device)
    stacked = torch.stack(block_tensors).to(device)
    _inject_stacked(kv_layer, slot_mapping, block_size, attn_metadata, stacked)


def _inject_stacked(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    attn_metadata: AttentionMetadata | None,
    stacked: torch.Tensor,
) -> None:
    """Inject an already-stacked ``[num_blocks, block_size, ...]`` tensor."""
    _require_torch()
    if _is_mla(attn_metadata):
        dst = kv_layer.reshape(-1, kv_layer.shape[-1])
        dst[slot_mapping, ...] = stacked.view(-1, stacked.shape[-1])
    else:
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        kv_layer[block_idxs, :, offsets] = stacked.view(
            len(slot_mapping), stacked.shape[-2], stacked.shape[-1]
        )


def inject_kv_blocks_contiguous(
    kv_layer: torch.Tensor,
    block_ids: list[int],
    block_size: int,
    attn_metadata: AttentionMetadata | None,
    contiguous_gpu_tensor: torch.Tensor,
) -> None:
    """Inject a contiguous ``[num_blocks, block_size, ...]`` GPU tensor.

    This avoids per-block Python iteration and the ``torch.stack`` that the
    list-based path requires.  ``contiguous_gpu_tensor`` must already be on the
    same device as ``kv_layer`` and have shape ``[len(block_ids), block_size, *]``.
    """
    _require_torch()
    if not block_ids:
        return
    device = kv_layer.device
    if contiguous_gpu_tensor.device != device:
        raise ValueError(
            f"contiguous_gpu_tensor is on {contiguous_gpu_tensor.device}, "
            f"expected {device}"
        )
    if contiguous_gpu_tensor.shape[0] != len(block_ids):
        raise ValueError(
            f"contiguous_gpu_tensor has {contiguous_gpu_tensor.shape[0]} blocks, "
            f"but {len(block_ids)} block ids were provided"
        )
    slot_mapping = _slot_mapping(block_ids, block_size, device)
    _inject_stacked(
        kv_layer, slot_mapping, block_size, attn_metadata, contiguous_gpu_tensor
    )
