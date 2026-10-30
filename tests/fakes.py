# SPDX-License-Identifier: Apache-2.0
"""Test doubles: FakeTelysClient, fake vLLM request/scheduler-output objects.

All fakes are deterministic: provenance is derived from the key with sha256,
block hashes form a deterministic chain, and the shared backing store is
reset by an autouse fixture between tests.
"""

import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm.config.kv_transfer import KVTransferConfig

from telys_vllm.client import (
    TelysClient,
    TelysMaintenanceReport,
    TelysProvenance,
    TelysRecord,
)

BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# FakeTelysClient
# ---------------------------------------------------------------------------

# One shared backing store: all clients created in a test see the same
# "on-device" Telys index.  In v4 Telys stores only metadata; payload bytes live
# in a KVSlab managed by TelysBlockStore.
_SHARED_STORE: dict[str, dict[str, str]] = {}

#: Map from chunk start key (without the trailing /count) to max block count.
_CHUNK_LENGTHS: dict[str, int] = {}

#: Every client handed out via make_fake_client, oldest first.
CREATED_CLIENTS: list["FakeTelysClient"] = []


def reset_fakes() -> None:
    _SHARED_STORE.clear()
    _CHUNK_LENGTHS.clear()
    CREATED_CLIENTS.clear()


def _chunk_start_key(key: str) -> str:
    return key.rsplit("/", 1)[0]


def track_chunk_key(key: str) -> None:
    """Update the shared chunk-length map from a chunk key."""
    if "/chunk/" not in key:
        return
    try:
        length = int(key.rsplit("/", 1)[1])
    except (ValueError, IndexError):
        return
    start_key = _chunk_start_key(key)
    _CHUNK_LENGTHS[start_key] = max(_CHUNK_LENGTHS.get(start_key, 0), length)


def untrack_chunk_key(key: str) -> None:
    """Remove a chunk key from the shared chunk-length map."""
    start_key = _chunk_start_key(key)
    _CHUNK_LENGTHS.pop(start_key, None)


def _provenance(key: str) -> TelysProvenance:
    return TelysProvenance(
        plan_hash=hashlib.sha256(f"telys-plan:{key}".encode()).hexdigest()[:16],
        result_id=hashlib.sha256(f"telys-result:{key}".encode()).hexdigest()[:12],
    )


class FakeTelysClient(TelysClient):
    """In-memory TelysClient honoring the Protocol contract.

    v4: stores only metadata.  The actual chunk payload bytes are owned by the
    TelysBlockStore's KVSlab.
    """

    def __init__(self) -> None:
        self.get_count = 0
        self.put_count = 0
        self.compact_calls = 0
        self.deleted: list[str] = []
        self.closed = False

    def contains(self, key: str) -> bool:
        return key in _SHARED_STORE

    def chunk_length(self, start_key: str) -> int:
        return _CHUNK_LENGTHS.get(start_key, 0)

    def get(self, key: str) -> TelysRecord | None:
        self.get_count += 1
        metadata = _SHARED_STORE.get(key)
        if metadata is None:
            return None
        return TelysRecord(
            key=key,
            value=b"",
            provenance=_provenance(key),
            metadata=dict(metadata),
        )

    def get_batch(self, keys: list[str]) -> list[TelysRecord | None]:
        return [self.get(key) for key in keys]

    def put(self, key: str, value: bytes, metadata: Mapping[str, str]) -> None:
        self.put_count += 1
        _SHARED_STORE[key] = dict(metadata)
        track_chunk_key(key)

    def delete(self, key: str) -> bool:
        self.deleted.append(key)
        present = _SHARED_STORE.pop(key, None) is not None
        if present:
            untrack_chunk_key(key)
        return present

    def compact(self) -> TelysMaintenanceReport:
        self.compact_calls += 1
        return TelysMaintenanceReport(
            entries_scanned=len(_SHARED_STORE), entries_dropped=0
        )

    def close(self) -> None:
        if not self.closed:
            self.compact()
        self.closed = True


def make_fake_client(config) -> FakeTelysClient:
    """Client factory for the dotted-path config ("tests.fakes:make_fake_client")."""
    client = FakeTelysClient()
    CREATED_CLIENTS.append(client)
    return client


def make_fake_store(tmp_path, *, write_behind: bool = False, **config_kwargs):
    """Create a TelysBlockStore backed by a FakeTelysClient and a temp KVSlab."""
    from telys_vllm.config import TelysConnectorConfig
    from telys_vllm.store import TelysBlockStore

    config_kwargs.setdefault("namespace", "telys-vllm")
    config_kwargs.setdefault("slab_path", str(tmp_path / "kv.slab"))
    config_kwargs.setdefault("write_behind", write_behind)
    config = TelysConnectorConfig(**config_kwargs)
    client = FakeTelysClient()
    return TelysBlockStore(
        client,
        config,
        model_id="fake-model-v0",
        engine_id="test-engine",
        block_size=BLOCK_SIZE,
    )


class LegacyTwoArgConnector:
    """Deliberately outdated 2-arg constructor for the factory rejection test
    (mirrors the supports_kw check at factory.py:116-124)."""

    def __init__(self, vllm_config, role):
        self._vllm_config = vllm_config
        self._role = role


# ---------------------------------------------------------------------------
# Fake vLLM objects
# ---------------------------------------------------------------------------


def make_hashes(count: int) -> list[bytes]:
    """Deterministic chained block hashes, like vLLM's hash_block_tokens."""
    hashes: list[bytes] = []
    parent = b"\x00" * 32
    for i in range(count):
        parent = hashlib.sha256(parent + i.to_bytes(4, "big")).digest()
        hashes.append(parent)
    return hashes


class FakeRequest:
    """Minimal stand-in for vllm.v1.request.Request."""

    def __init__(
        self,
        request_id: str,
        num_tokens: int,
        block_hashes: list[bytes] | None = None,
        skip_reading_prefix_cache: bool = False,
    ) -> None:
        self.request_id = request_id
        self._num_tokens = num_tokens
        self.prompt_token_ids = list(range(num_tokens))
        full_blocks = num_tokens // BLOCK_SIZE
        self.block_hashes: list[bytes] = (
            make_hashes(full_blocks) if block_hashes is None else block_hashes
        )
        self.skip_reading_prefix_cache = skip_reading_prefix_cache
        self.num_computed_tokens = 0

    @property
    def num_tokens(self) -> int:
        return self._num_tokens


#: Temporary directories created when callers don't supply a pytest tmp_path.
_FALLBACK_TMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _default_slab_path(tmp_path: Path | None) -> str:
    if tmp_path is None:
        td = tempfile.TemporaryDirectory(prefix="telys-vllm-test-")
        _FALLBACK_TMP_DIRS.append(td)
        tmp_path = Path(td.name)
    return str(tmp_path / "kv.slab")


def make_kv_transfer_config(
    extra: dict[str, Any] | None = None,
    tmp_path: Path | None = None,
    **kwargs: Any,
) -> KVTransferConfig:
    extra = {
        "telys_client_factory": "tests.fakes:make_fake_client",
        "telys_write_behind": False,
        **(extra or {}),
    }
    if "telys_slab_path" not in extra:
        extra["telys_slab_path"] = _default_slab_path(tmp_path)
    return KVTransferConfig(
        kv_connector="TelysKVConnector",
        kv_role="kv_both",
        engine_id="test-engine",
        kv_connector_extra_config=extra,
        **kwargs,
    )


def make_vllm_config(
    extra: dict[str, Any] | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    tmp_path: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        kv_transfer_config=kv_transfer_config
        or make_kv_transfer_config(extra, tmp_path=tmp_path),
        model_config=SimpleNamespace(model="fake-model-v0"),
    )


def make_kv_cache_config() -> SimpleNamespace:
    return SimpleNamespace(prefix_cacheable_group_ids=(0,))


@dataclass
class FakeNewReq:
    req_id: str
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int = 0


@dataclass
class FakeCachedReqs:
    req_ids: list[str] = field(default_factory=list)
    resumed_req_ids: set[str] = field(default_factory=set)
    num_computed_tokens: list[int] = field(default_factory=list)
    new_block_ids: list[tuple[list[int], ...] | None] = field(default_factory=list)


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list[FakeNewReq] = field(default_factory=list)
    scheduled_cached_reqs: FakeCachedReqs = field(default_factory=FakeCachedReqs)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
