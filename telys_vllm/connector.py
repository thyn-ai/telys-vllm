# SPDX-License-Identifier: Apache-2.0
"""TelysKVConnector: Telys as an external KV/prefix-cache store for vLLM.

Implements ``KVConnectorBase_V1`` (+ ``SupportsHMA``) against the abstract
:class:`~telys_vllm.client.TelysClient` boundary. Structure mirrors
``OffloadingConnector`` (vllm
``kv_connector/v1/offloading_connector.py``): a thin role-split adapter over a
pure core (:class:`~telys_vllm.store.TelysBlockStore`).
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger

from telys_vllm.client import TelysClient, TelysRecord
from telys_vllm.codec import (
    LengthPrefixedCodec,
    TelysChunkCodec,
    TelysFastChunkCodec,
)
from telys_vllm.config import (
    EXTRA_CLIENT_FACTORY,
    TelysConfigError,
    TelysConnectorConfig,
)
from telys_vllm.gpu_ops import (
    extract_kv_blocks,
    inject_kv_blocks_contiguous,
    serialize_kv_tensor,
)
from telys_vllm.store import TelysBlockStore
from telys_vllm.telemetry import Telemetry, get_telemetry


def _auto_client(config: TelysConnectorConfig) -> TelysClient:
    """Return the best available local Telys client.

    Prefers :class:`~telys_vllm.local_client.TelysBlobClient` when the SDK
    exposes ``telys.BlobStore``; otherwise falls back to the legacy
    slab-backed :class:`~telys_vllm.local_client.TelysLocalClient`.
    """
    try:
        import telys
    except ImportError as exc:
        raise TelysConfigError(
            "telys SDK is not installed; install it or set "
            f"kv_connector_extra_config[{EXTRA_CLIENT_FACTORY!r}]"
        ) from exc

    if hasattr(telys, "BlobStore"):
        from telys_vllm.local_client import TelysBlobClient

        return TelysBlobClient(config)
    from telys_vllm.local_client import TelysLocalClient

    return TelysLocalClient(config)


if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class TelysConnectorError(RuntimeError):
    """Raised for connector-side failures (never silently skipped)."""


@dataclass
class TelysLoadSpec:
    """One request's planned load: Telys keys plus destination GPU blocks."""

    request_id: str
    block_hashes: list[bytes]
    num_tokens: int
    # Logical block index where the hit prefix starts; used to align the
    # allocated physical block ids with the loaded block hashes.
    start_block: int = 0
    # Group-0 destination block ids, filled in by build_connector_meta.
    block_ids: list[int] = field(default_factory=list)


@dataclass
class TelysSaveSpec:
    """One request's newly completed full blocks to persist this step."""

    request_id: str
    block_hashes: list[bytes]
    block_ids: list[int] = field(default_factory=list)


@dataclass
class TelysConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker metadata for one engine step. Must stay
    msgpack-serializable (bytes/str/int containers only)."""

    loads: list[TelysLoadSpec] = field(default_factory=list)
    saves: list[TelysSaveSpec] = field(default_factory=list)


def _load_client_factory(
    dotted_path: str,
) -> "Any":
    """Resolve ``module:attr`` to a client factory callable.

    Same lazy-import idiom as vLLM's own KVConnectorFactory
    (``factory.py:36-38``, ``106-110``).
    """
    module_name, sep, attr = dotted_path.partition(":")
    if not sep or not module_name or not attr:
        raise TelysConfigError(
            f"kv_connector_extra_config[{EXTRA_CLIENT_FACTORY!r}] must be a "
            f"'module:callable' dotted path, got {dotted_path!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TelysConfigError(
            f"cannot import client-factory module {module_name!r}: {exc}"
        ) from exc
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise TelysConfigError(
            f"client factory {dotted_path!r} is not a callable in {module_name!r}"
        )
    return factory


class TelysKVConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM KV connector backed by a Telys store."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size: int = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._config = TelysConnectorConfig.from_sources(extra=extra)
        self._telemetry: Telemetry = get_telemetry(
            enabled=self._config.telemetry,
            log_path=self._config.telemetry_log,
        )
        factory_path = extra.get(EXTRA_CLIENT_FACTORY)
        with self._telemetry.span("telys_startup") as ctx:
            ctx["kind"] = "client_factory"
            if factory_path is None:
                # Auto-select BlobStore path when available, otherwise legacy
                # slab-backed local client.
                client = _auto_client(self._config)
            else:
                client = _load_client_factory(str(factory_path))(self._config)
            logger.info(
                "telys client created: role=%s remote=%s kind=%s",
                role,
                getattr(client, "_remote", None),
                type(client).__name__,
            )
            ctx["remote"] = getattr(client, "_remote", None)
            ctx["client_kind"] = type(client).__name__
        with self._telemetry.span("connector_store_init") as ctx:
            ctx["model_id"] = str(vllm_config.model_config.model)
            ctx["engine_id"] = str(self._kv_transfer_config.engine_id or "")
            self._store = TelysBlockStore(
                client,
                self._config,
                model_id=str(vllm_config.model_config.model),
                engine_id=str(self._kv_transfer_config.engine_id or ""),
                block_size=self._block_size,
                telemetry=self._telemetry,
            )
        self._codec: LengthPrefixedCodec = LengthPrefixedCodec()
        self._chunk_codec: TelysChunkCodec = TelysChunkCodec()
        self._fast_chunk_codec: TelysFastChunkCodec = TelysFastChunkCodec()
        self._fast_path: bool = self._config.fast_path
        # Dedicated CUDA stream for async H2D copies.  Created lazily so CPU-only
        # tests never pay for it.
        self._load_stream_obj: torch.cuda.Stream | None = None

        # Scheduler-side state.
        self._reqs: dict[str, Request] = {}
        self._load_plans: dict[str, TelysLoadSpec] = {}
        self._save_cursors: dict[str, int] = {}
        # Idempotent observation of each request's last lookup boundary.
        self._lookup_starts: dict[str, int] = {}

        # Worker-side state.
        # GPU path: KV cache tensors registered once at model load.
        self._kv_caches: dict[str, torch.Tensor] | None = None
        # CPU/test path: layer_name -> {block_hash: per-layer bytes}.
        self._layer_buffers: dict[str, dict[bytes, bytes]] = {}
        # request_id -> records fetched by start_load_kv, awaiting injection.
        self._fetched: dict[str, list[TelysRecord]] = {}
        # Forward context captured during start_load_kv; used by wait_for_save
        # when save_kv_layer is skipped under torch.compile.
        self._forward_context: ForwardContext | None = None
        self._step_attn_metadata: (
            AttentionMetadata | Mapping[str, AttentionMetadata] | None
        ) = None
        # Pinned CPU tensors retained until async H2D copies complete.
        self._last_staged: list[torch.Tensor] = []

    # ==============================
    # Scheduler-side methods
    # ==============================

    def on_new_request(self, request: "Request") -> None:
        self._reqs[request.request_id] = request

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Longest contiguous Telys-held prefix beyond the local hit.

        Miss is exactly ``(0, False)``; we never defer (sync Telys lookups,
        so no ``None``). The only state touched is an idempotent note of the
        lookup boundary (same inputs -> same value), matching the reference
        offloading connector (offloading/scheduler.py:1156).
        """
        block_hashes = request.block_hashes
        if request.skip_reading_prefix_cache:
            return 0, False
        start_block = num_computed_tokens // self._block_size
        # Never offer the trailing partial block: it has no block hash and its
        # last token must be computed to produce output.
        max_block = (request.num_tokens - 1) // self._block_size
        if start_block >= max_block:
            return 0, False
        with self._telemetry.span("lookup") as ctx:
            ctx["request_id"] = request.request_id
            ctx["start_block"] = start_block
            ctx["max_block"] = max_block
            hit_blocks = self._store.lookup_prefix(block_hashes[start_block:max_block])
            hit_tokens = hit_blocks * self._block_size
            ctx["hit_blocks"] = hit_blocks
            ctx["hit_tokens"] = hit_tokens
        logger.info(
            "telys lookup: req=%s num_tokens=%d num_computed=%d start_block=%d max_block=%d hit_blocks=%d hit_tokens=%d",
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
            start_block,
            max_block,
            hit_blocks,
            hit_tokens,
        )
        if hit_tokens == 0:
            return 0, False
        # Idempotent bookkeeping (same inputs -> same value), used by
        # update_state_after_alloc to locate the hit region.
        self._lookup_starts[request.request_id] = num_computed_tokens
        return hit_tokens, self._config.load_async

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Record the load plan; skip re-saving blocks Telys already served."""
        self._reqs[request.request_id] = request
        if num_external_tokens <= 0:
            return
        # The external hit starts at the block-aligned local-hit boundary,
        # observed idempotently in get_num_new_matched_tokens (the same
        # bookkeeping the reference connector does at
        # offloading/scheduler.py:1156).
        start_block = self._lookup_starts.get(request.request_id, 0) // self._block_size
        num_blocks = num_external_tokens // self._block_size
        self._load_plans[request.request_id] = TelysLoadSpec(
            request_id=request.request_id,
            block_hashes=list(
                request.block_hashes[start_block : start_block + num_blocks]
            ),
            num_tokens=num_external_tokens,
            start_block=start_block,
        )
        self._save_cursors[request.request_id] = max(
            self._save_cursors.get(request.request_id, 0),
            start_block + num_blocks,
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> TelysConnectorMetadata:
        """Emit this step's load/save specs, then reset per-step state."""
        meta = TelysConnectorMetadata()
        num_scheduled = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            total_tokens = new_req.num_computed_tokens + num_scheduled[new_req.req_id]
            spec = self._load_plans.pop(new_req.req_id, None)
            if spec is not None:
                # Group-0 destination blocks (multi-group: DESIGN.md 5.6).
                # Slice to the region that corresponds to the loaded hashes.
                end = spec.start_block + len(spec.block_hashes)
                spec.block_ids = list(new_req.block_ids[0][spec.start_block : end])
                meta.loads.append(spec)
            self._advance_save_cursor(
                new_req.req_id, total_tokens, list(new_req.block_ids[0]), meta
            )

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            total_tokens = cached.num_computed_tokens[i] + num_scheduled[req_id]
            # Only new requests carry the block-id list we need for saves; the
            # benchmark workload is one-shot so this is sufficient for now.
            self._advance_save_cursor(req_id, total_tokens, [], meta)

        return meta

    def _advance_save_cursor(
        self,
        req_id: str,
        total_tokens: int,
        block_ids: list[int] | None,
        meta: TelysConnectorMetadata,
    ) -> None:
        request = self._reqs.get(req_id)
        if request is None:
            return
        max_block = (request.num_tokens - 1) // self._block_size
        boundary = min(total_tokens // self._block_size, max_block)
        cursor = self._save_cursors.get(req_id, 0)
        if boundary <= cursor:
            return
        # block_ids are the allocated physical block ids for this request,
        # ordered by block index.  CPU/test paths may not have them; the worker
        # can still supply bytes via the ``block_payloads`` kwarg.  The GPU
        # path skips specs with empty block_ids inside ``wait_for_save``.
        block_ids = block_ids or []
        meta.saves.append(
            TelysSaveSpec(
                request_id=req_id,
                block_hashes=list(request.block_hashes[cursor:boundary]),
                block_ids=list(block_ids[cursor:boundary]),
            )
        )
        self._save_cursors[req_id] = boundary

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._finish_request(request)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._finish_request(request)

    def _finish_request(self, request: "Request") -> tuple[bool, dict[str, Any] | None]:
        # Best-effort cache: blocks free immediately, nothing is delayed.
        self._reqs.pop(request.request_id, None)
        self._load_plans.pop(request.request_id, None)
        self._save_cursors.pop(request.request_id, None)
        self._lookup_starts.pop(request.request_id, None)
        return False, None

    @property
    def requires_kv_delivery(self) -> bool:
        # Best-effort cache: a dropped save is just a future cache miss
        # (pattern: offloading_connector.py:73-77).
        return False

    def reset_cache(self) -> bool | None:
        self._store.drain()
        self._store.reset()
        self._reqs.clear()
        self._load_plans.clear()
        self._save_cursors.clear()
        self._lookup_starts.clear()
        return True

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        """Capture per-layer KV cache tensors for deferred saving.

        This runs once at model load, outside the compiled forward region, so
        it is safe even when ``save_kv_layer`` is skipped by ``torch.compile``.
        """
        self._kv_caches = dict(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Fetch every planned chunk from Telys and inject into paged KV buffers."""
        self._forward_context = forward_context
        if forward_context is not None:
            self._step_attn_metadata = getattr(forward_context, "attn_metadata", None)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, TelysConnectorMetadata)
        if not metadata.loads:
            return

        with self._telemetry.span("start_load_kv") as outer_ctx:
            total_blocks = sum(len(spec.block_hashes) for spec in metadata.loads)
            outer_ctx["total_specs"] = len(metadata.loads)
            outer_ctx["total_blocks"] = total_blocks

            # Fast path is only safe when we have a real forward context with GPU
            # layers to inject into.  CPU/test paths keep the legacy per-block
            # decode so ``fetched_records`` stays populated for assertions.
            use_fast_path = self._fast_path and forward_context is not None

            if use_fast_path:
                import torch

                pinned_staged_all: list[torch.Tensor] = []
                for spec in metadata.loads:
                    pinned_staged_all.extend(
                        self._fast_inject_spec(spec, forward_context)
                    )
                # Synchronize the dedicated H2D stream with the default stream so
                # the subsequent forward pass sees the injected KV data.  This is
                # the only sync point on the warm load path.
                stream = self._get_load_stream()
                if stream is not None:
                    event = torch.cuda.Event()
                    event.record(stream)
                    torch.cuda.current_stream().wait_event(event)
                # Keep references until after the sync so pinned memory is not
                # freed before the async copies complete.
                self._last_staged = pinned_staged_all
                return

            for spec in metadata.loads:
                # Load decoded layer blobs directly; the slab is read once per chunk
                # and the chunk codec decodes all blocks in one pass.
                with self._telemetry.span("load_decoded_blocks") as load_ctx:
                    load_ctx["request_id"] = spec.request_id
                    load_ctx["block_count"] = len(spec.block_hashes)
                    decoded_blocks = self._store.load_decoded_blocks(spec.block_hashes)
                for block in decoded_blocks:
                    logger.debug(
                        "telys hit: req=%s key=%s plan_hash=%s result_id=%s",
                        spec.request_id,
                        block.key,
                        block.provenance.plan_hash,
                        block.provenance.result_id,
                    )
                # Preserve the legacy TelysRecord view for tests/inspection.
                self._fetched[spec.request_id] = [
                    TelysRecord(
                        key=block.key,
                        value=self._codec.encode_block(block.blobs),
                        provenance=block.provenance,
                        metadata=block.metadata,
                    )
                    for block in decoded_blocks
                ]

                # CPU/test path: no forward_context means there is nowhere to inject.
                if forward_context is None:
                    continue

                attn_metadata = forward_context.attn_metadata
                if attn_metadata is None:
                    logger.warning("start_load_kv called without attention metadata")
                    continue

                # Inject all loaded blocks into every attention layer's KV cache.
                # Process one layer at a time so the per-block GPU transfers and
                # indexing are batched into a single kernel launch instead of one
                # per block.
                no_compile_layers = getattr(forward_context, "no_compile_layers", None)
                if no_compile_layers is None:
                    continue
                block_list = list(decoded_blocks)
                block_ids = list(spec.block_ids)
                if len(block_ids) != len(block_list):
                    raise TelysConnectorError(
                        f"load plan mismatch for {spec.request_id}: "
                        f"{len(block_list)} decoded blocks but {len(block_ids)} block ids"
                    )
                layer_names = sorted(block_list[0].blobs) if block_list else []
                for layer_name in layer_names:
                    layer = no_compile_layers.get(layer_name)
                    if layer is None:
                        continue
                    kv_cache_layer = getattr(layer, "kv_cache", None)
                    if kv_cache_layer is None:
                        continue
                    layer_attn = (
                        attn_metadata.get(layer_name)
                        if isinstance(attn_metadata, dict)
                        else attn_metadata
                    )
                    if layer_attn is None:
                        continue
                    with self._telemetry.span("inject_block") as block_ctx:
                        block_ctx["request_id"] = spec.request_id
                        block_ctx["layer_name"] = layer_name
                        block_ctx["block_count"] = len(block_list)
                        block_tensors: list[torch.Tensor] = []
                        for block, block_id in zip(block_list, block_ids):
                            with self._telemetry.span("tensor_from_blob") as tfb_ctx:
                                tfb_ctx["layer_name"] = layer_name
                                tfb_ctx["block_id"] = block_id
                                block_tensors.append(
                                    self._tensor_from_blob(block.blobs[layer_name])
                                )
                        with self._telemetry.span("inject_kv_block") as inj_ctx:
                            inj_ctx["layer_name"] = layer_name
                            inj_ctx["block_count"] = len(block_ids)
                            self._inject_kv_blocks(
                                kv_cache_layer,
                                block_ids,
                                layer_name,
                                layer_attn,
                                block_tensors,
                            )

    @staticmethod
    def _tensor_from_blob(blob: bytes) -> "torch.Tensor":
        from telys_vllm.gpu_ops import deserialize_kv_tensor

        return deserialize_kv_tensor(blob)

    def _inject_kv_block(
        self,
        kv_layer: "torch.Tensor",
        block_id: int,
        layer_name: str,
        attn_metadata: "AttentionMetadata",
        block_tensor: "torch.Tensor",
    ) -> None:
        from telys_vllm.gpu_ops import inject_kv_block

        inject_kv_block(
            kv_layer, block_id, self._block_size, attn_metadata, block_tensor
        )

    def _inject_kv_blocks(
        self,
        kv_layer: "torch.Tensor",
        block_ids: list[int],
        layer_name: str,
        attn_metadata: "AttentionMetadata",
        block_tensors: list["torch.Tensor"],
    ) -> None:
        from telys_vllm.gpu_ops import inject_kv_blocks

        inject_kv_blocks(
            kv_layer, block_ids, self._block_size, attn_metadata, block_tensors
        )

    def _get_load_stream(self) -> "torch.cuda.Stream | None":
        """Return a dedicated CUDA stream for async H2D copies, or None if CUDA is unavailable."""
        import torch

        if self._load_stream_obj is None and torch.cuda.is_available():
            self._load_stream_obj = torch.cuda.Stream(priority=-1)
        return self._load_stream_obj

    def _fast_inject_spec(
        self,
        spec: TelysLoadSpec,
        forward_context: "ForwardContext",
    ) -> list["torch.Tensor"]:
        """Zero-decode load+inject for one spec using the fast chunk codec.

        Returns the list of pinned CPU tensors staged for async H2D copies.
        The caller records CUDA events and waits on them after all specs are
        submitted.  When CUDA is unavailable the list is empty.
        """
        import torch

        attn_metadata = forward_context.attn_metadata
        no_compile_layers = getattr(forward_context, "no_compile_layers", None)
        if no_compile_layers is None:
            return []

        chunks = self._store.load_decoded_blocks_fast(spec.block_hashes)
        if not chunks:
            return []

        stream = self._get_load_stream()
        pinned_staged: list[torch.Tensor] = []

        block_id_iter = iter(spec.block_ids)
        for chunk in chunks:
            layer_names = sorted(chunk.layer_tensors)
            chunk_block_ids = [
                next(block_id_iter) for _ in range(len(chunk.block_hashes))
            ]
            for layer_name in layer_names:
                layer = no_compile_layers.get(layer_name)
                if layer is None:
                    continue
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                layer_attn = (
                    attn_metadata.get(layer_name)
                    if isinstance(attn_metadata, dict)
                    else attn_metadata
                )
                if layer_attn is None:
                    continue

                with self._telemetry.span("inject_block") as block_ctx:
                    block_ctx["request_id"] = spec.request_id
                    block_ctx["layer_name"] = layer_name
                    block_ctx["block_count"] = len(chunk.block_hashes)
                    block_ctx["fast_path"] = True

                    stacked_cpu = chunk.layer_tensors[layer_name][
                        : len(chunk_block_ids)
                    ]
                    device = kv_cache_layer.device
                    if torch.cuda.is_available():
                        pinned = torch.empty(
                            stacked_cpu.shape,
                            dtype=stacked_cpu.dtype,
                            pin_memory=True,
                        )
                        pinned.copy_(stacked_cpu)
                        pinned_staged.append(pinned)
                        if stream is not None:
                            with torch.cuda.stream(stream):
                                gpu_tensor = pinned.to(device, non_blocking=True)
                                inject_kv_blocks_contiguous(
                                    kv_cache_layer,
                                    chunk_block_ids,
                                    self._block_size,
                                    layer_attn,
                                    gpu_tensor,
                                )
                        else:
                            gpu_tensor = pinned.to(device)
                            inject_kv_blocks_contiguous(
                                kv_cache_layer,
                                chunk_block_ids,
                                self._block_size,
                                layer_attn,
                                gpu_tensor,
                            )
                    else:
                        gpu_tensor = stacked_cpu.to(device)
                        inject_kv_blocks_contiguous(
                            kv_cache_layer,
                            chunk_block_ids,
                            self._block_size,
                            layer_attn,
                            gpu_tensor,
                        )

        return pinned_staged

    def _collect_layer_stacked_tensors_from_kv_caches(
        self, block_hash_to_id: dict[bytes, int]
    ) -> tuple[list[bytes], dict[str, "torch.Tensor"]] | None:
        """Extract and stack blocks from registered KV caches into contiguous CPU tensors.

        Returns ``(ordered_hashes, layer_tensors)`` where ``ordered_hashes`` is the
        list of block hashes in the same order as the stacked tensors.  This order
        matches the chunk runs produced by ``wait_for_save``.

        Returns ``None`` when extraction does not yield tensors (e.g. some test
        doubles return bytes), signalling the caller to fall back to the legacy
        per-block serialization path.
        """
        import torch

        assert self._kv_caches is not None
        attn_metadata = self._step_attn_metadata
        if attn_metadata is None and self._forward_context is not None:
            attn_metadata = getattr(self._forward_context, "attn_metadata", None)

        if not block_hash_to_id:
            return [], {}

        # Preserve the insertion order of block_hash_to_id; it matches the runs.
        ordered_hashes = list(block_hash_to_id.keys())
        block_ids = [block_hash_to_id[h] for h in ordered_hashes]
        layer_stacked: dict[str, torch.Tensor] = {}
        for layer_name, kv_cache in self._kv_caches.items():
            layer_attn = (
                attn_metadata.get(layer_name)
                if isinstance(attn_metadata, dict)
                else attn_metadata
            )
            tensors = extract_kv_blocks(
                kv_cache, block_ids, self._block_size, layer_attn
            )
            if not tensors or not isinstance(tensors[0], torch.Tensor):
                return None
            stacked = torch.stack(tensors)  # [N, block_size, *]
            layer_stacked[layer_name] = stacked
        return ordered_hashes, layer_stacked

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Loads complete synchronously in start_load_kv; layer-by-layer
        # pipelining is bench-week work.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Buffer layer data for saving; serialization is deferred to wait_for_save.

        In the GPU path the registered KV caches are used instead, so this hook
        can be skipped by ``torch.compile`` without losing saves.  For CPU tests
        and fallback paths, ``block_payloads`` are buffered here.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, TelysConnectorMetadata)
        buffer = self._layer_buffers.setdefault(layer_name, {})

        block_payloads = kwargs.get("block_payloads")
        if block_payloads is not None:
            for block_hash, blob in block_payloads.items():
                if not isinstance(blob, (bytes, bytearray, memoryview)):
                    raise TelysConnectorError(
                        f"layer {layer_name!r}: blob for block "
                        f"{bytes(block_hash).hex()} must be bytes-like, got "
                        f"{type(blob).__name__}"
                    )
                buffer[bytes(block_hash)] = bytes(blob)
            return

        # If KV caches were registered, extraction happens in wait_for_save so
        # that torch.compile can elide this hook entirely.
        if self._kv_caches is not None:
            return

        # No registered caches and no CPU payloads: this environment cannot
        # produce the bytes we need to save.
        raise TelysConnectorError(
            "GPU tensor extraction is not available in this environment; "
            f"pass block_payloads={{block_hash: bytes}} for layer {layer_name!r} "
            "(see DESIGN.md section 5.1)"
        )

    def wait_for_save(self):
        """Serialize buffered layer data into chunks and persist to Telys.

        Once per step; does nothing when there is no save work
        (contract: v1/base.py:389-391).  Redundant chunk keys are deduplicated
        within a step so the shared prefix is extracted and persisted only once.
        """
        import time

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, TelysConnectorMetadata)
        with self._telemetry.span("wait_for_save") as outer_ctx:
            try:
                specs = [spec for spec in metadata.saves if spec.block_hashes]
                if not specs:
                    return

                # Deduplicate chunk keys across specs before paying for extraction.
                # vLLM may give more block_hashes than block_ids when some prefix
                # blocks are already saved; the saveable suffix is aligned with
                # block_ids at the end of block_hashes.
                pending_runs: list[tuple[bytes, list[bytes]]] = []
                seen_keys: set[str] = set()
                block_hash_to_id: dict[bytes, int] = {}
                for spec in specs:
                    num_ids = len(spec.block_ids)
                    if num_ids == 0:
                        continue
                    save_hashes = spec.block_hashes[-num_ids:]
                    for block_hash, block_id in zip(save_hashes, spec.block_ids):
                        block_hash_to_id.setdefault(bytes(block_hash), block_id)
                    for run in self._split_into_runs(save_hashes):
                        key = self._store.chunk_key(run[0], len(run))
                        if key in seen_keys or self._store.contains(key):
                            continue
                        seen_keys.add(key)
                        pending_runs.append((run[0], run))
                if not pending_runs:
                    return

                outer_ctx["pending_chunks"] = len(pending_runs)
                outer_ctx["unique_blocks"] = len(block_hash_to_id)

                with self._telemetry.span("save_collect") as collect_ctx:
                    collect_ctx["block_count"] = len(block_hash_to_id)
                    t0 = time.perf_counter()
                    use_fast_path = False
                    if self._fast_path and self._kv_caches is not None:
                        # Fast path: extract stacked contiguous CPU tensors per layer.
                        stacked_result = (
                            self._collect_layer_stacked_tensors_from_kv_caches(
                                block_hash_to_id
                            )
                        )
                        if stacked_result is not None:
                            use_fast_path = True
                            ordered_hashes, layer_tensors = stacked_result
                            hash_to_idx = {h: i for i, h in enumerate(ordered_hashes)}
                            layer_blobs = {}
                    if not use_fast_path:
                        # Legacy path: per-block serialized bytes.
                        layer_blobs = self._collect_layer_blobs(block_hash_to_id)
                        layer_tensors = {}
                    t1 = time.perf_counter()
                if not layer_blobs and not layer_tensors:
                    logger.info("wait_for_save: no layer data collected")
                    return

                saved = 0
                use_fast_path = bool(layer_tensors)
                required_layers = set(layer_tensors if use_fast_path else layer_blobs)
                with self._telemetry.span("save_encode_persist") as persist_ctx:
                    persist_ctx["pending_chunks"] = len(pending_runs)
                    for start_hash, run in pending_runs:
                        if use_fast_path:
                            if not all(h in hash_to_idx for h in run):
                                continue
                            run_tensors = {
                                layer: layer_tensors[layer][
                                    [hash_to_idx[h] for h in run]
                                ]
                                for layer in required_layers
                            }
                            with self._telemetry.span("save_encode_chunk") as enc_ctx:
                                enc_ctx["block_count"] = len(run)
                                enc_ctx["layer_count"] = len(required_layers)
                                enc_ctx["fast_path"] = True
                                self._store.save_chunk_fast(
                                    start_hash, run, run_tensors
                                )
                            saved += 1
                        else:
                            if not all(
                                layer_blobs[layer].get(block_hash) is not None
                                for layer in required_layers
                                for block_hash in run
                            ):
                                continue
                            run_layer_blobs = {
                                layer: [
                                    layer_blobs[layer][block_hash] for block_hash in run
                                ]
                                for layer in required_layers
                            }
                            with self._telemetry.span("save_encode_chunk") as enc_ctx:
                                enc_ctx["block_count"] = len(run)
                                enc_ctx["layer_count"] = len(required_layers)
                                encoded_chunk = self._chunk_codec.encode_chunk(
                                    run_layer_blobs
                                )
                            with self._telemetry.span("save_chunk_async") as async_ctx:
                                async_ctx["block_count"] = len(run)
                                self._store.save_chunk_async(
                                    start_hash, run, encoded_chunk
                                )
                            saved += 1
                    t2 = time.perf_counter()
                logger.info(
                    "wait_for_save: collect=%.3fs save=%.3fs chunks=%d/%d deduped",
                    t1 - t0,
                    t2 - t1,
                    saved,
                    len(pending_runs),
                )
                outer_ctx["saved_chunks"] = saved
            finally:
                self._layer_buffers.clear()
                self._forward_context = None
                self._step_attn_metadata = None

    def _collect_layer_blobs(
        self, block_hash_to_id: dict[bytes, int]
    ) -> dict[str, dict[bytes, bytes]]:
        """Return {layer_name: {block_hash: bytes}} for ``block_hash_to_id``."""
        if self._kv_caches is not None:
            return self._collect_layer_blobs_from_kv_caches(block_hash_to_id)
        return self._collect_layer_blobs_from_buffers(block_hash_to_id)

    def _collect_layer_blobs_from_kv_caches(
        self, block_hash_to_id: dict[bytes, int]
    ) -> dict[str, dict[bytes, bytes]]:
        """Extract and serialize the needed blocks from registered KV caches."""
        assert self._kv_caches is not None
        attn_metadata = self._step_attn_metadata
        if attn_metadata is None and self._forward_context is not None:
            attn_metadata = getattr(self._forward_context, "attn_metadata", None)

        if not block_hash_to_id:
            return {}

        layer_blobs: dict[str, dict[bytes, bytes]] = {}
        block_ids = sorted(block_hash_to_id.values())
        for layer_name, kv_cache in self._kv_caches.items():
            # Attention metadata is unavailable when torch.compile elides the
            # per-layer save_kv_layer decorator. Extraction only needs it for
            # MLA layout detection; non-MLA models work without it.
            layer_attn = (
                attn_metadata.get(layer_name)
                if isinstance(attn_metadata, dict)
                else attn_metadata
            )
            tensors = extract_kv_blocks(
                kv_cache, block_ids, self._block_size, layer_attn
            )
            blobs = [serialize_kv_tensor(t) for t in tensors]
            layer_blobs[layer_name] = {
                block_hash: blobs[block_ids.index(block_id)]
                for block_hash, block_id in block_hash_to_id.items()
            }
        return layer_blobs

    def _collect_layer_blobs_from_buffers(
        self, block_hash_to_id: dict[bytes, int]
    ) -> dict[str, dict[bytes, bytes]]:
        """Use bytes supplied via ``block_payloads`` in save_kv_layer."""
        if not self._layer_buffers:
            raise TelysConnectorError(
                "save specs exist but no layer data was buffered this step"
            )
        # Only save blocks for which every buffered layer provided a payload.
        present: set[bytes] = set.intersection(
            *(set(buffer) for buffer in self._layer_buffers.values())
        )
        required = set(block_hash_to_id)
        missing = required - present
        if missing:
            raise TelysConnectorError(
                f"{len(missing)} planned block(s) are missing from one or more layers"
            )
        return {name: dict(buffer) for name, buffer in self._layer_buffers.items()}

    def _split_into_runs(self, block_hashes: list[bytes]) -> list[list[bytes]]:
        """Split a contiguous block-hash list into chunk-sized runs."""
        max_size = self._config.max_chunk_blocks
        if max_size == 0 or len(block_hashes) <= max_size:
            return [block_hashes]
        return [
            block_hashes[i : i + max_size]
            for i in range(0, len(block_hashes), max_size)
        ]

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Synchronous transfers only; nothing completes out of band.
        return None, None

    def shutdown(self):
        self._store.drain()
        self._store.close()

    # ==============================
    # Test/inspection surface
    # ==============================

    @property
    def store(self) -> TelysBlockStore:
        return self._store

    @property
    def fetched_records(self) -> Mapping[str, list[TelysRecord]]:
        """Records fetched by start_load_kv, keyed by request id."""
        return self._fetched
