# SPDX-License-Identifier: Apache-2.0
"""Connector configuration: env vars + ``kv_connector_extra_config``.

Precedence: ``kv_connector_extra_config`` > environment > defaults.
All values are validated at construction; invalid input raises
``TelysConfigError`` before any connector state exists.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ENV_NAMESPACE = "TELYS_VLLM_NAMESPACE"
ENV_LOAD_ASYNC = "TELYS_VLLM_LOAD_ASYNC"
ENV_MAX_LOOKUP_BLOCKS = "TELYS_VLLM_MAX_LOOKUP_BLOCKS"
ENV_MAX_CHUNK_BLOCKS = "TELYS_VLLM_MAX_CHUNK_BLOCKS"
ENV_SLAB_PATH = "TELYS_VLLM_SLAB_PATH"
ENV_WRITE_BEHIND = "TELYS_VLLM_WRITE_BEHIND"
ENV_MAX_PENDING_WRITES = "TELYS_VLLM_MAX_PENDING_WRITES"
ENV_TELEMETRY = "TELYS_VLLM_TELEMETRY"
ENV_TELEMETRY_LOG = "TELYS_VLLM_TELEMETRY_LOG"
ENV_FAST_PATH = "TELYS_VLLM_FAST_PATH"

EXTRA_NAMESPACE = "telys_namespace"
EXTRA_LOAD_ASYNC = "telys_load_async"
EXTRA_MAX_LOOKUP_BLOCKS = "telys_max_lookup_blocks"
EXTRA_MAX_CHUNK_BLOCKS = "telys_max_chunk_blocks"
EXTRA_SLAB_PATH = "telys_slab_path"
EXTRA_WRITE_BEHIND = "telys_write_behind"
EXTRA_MAX_PENDING_WRITES = "telys_max_pending_writes"
EXTRA_TELEMETRY = "telys_telemetry"
EXTRA_TELEMETRY_LOG = "telys_telemetry_log"
EXTRA_FAST_PATH = "telys_fast_path"

#: extra_config key holding the dotted ``module:attr`` path of the
#: ``Callable[[TelysConnectorConfig], TelysClient]`` used to build the client.
EXTRA_CLIENT_FACTORY = "telys_client_factory"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class TelysConfigError(ValueError):
    """Raised for any invalid connector configuration."""


def _parse_bool(raw: object, *, source: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    raise TelysConfigError(
        f"{source}: expected a boolean ({sorted(_TRUE_VALUES | _FALSE_VALUES)}), "
        f"got {raw!r}"
    )


def _parse_positive_int(raw: object, *, source: str, allow_zero: bool) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise TelysConfigError(f"{source}: expected an integer, got {raw!r}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise TelysConfigError(f"{source}: expected an integer, got {raw!r}") from exc
    floor = 0 if allow_zero else 1
    if value < floor:
        raise TelysConfigError(f"{source}: expected an integer >= {floor}, got {value}")
    return value


def _parse_namespace(raw: object, *, source: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or raw.strip() != raw:
        raise TelysConfigError(
            f"{source}: expected a non-empty string without leading/trailing "
            f"whitespace, got {raw!r}"
        )
    return raw


def _parse_slab_path(raw: object, *, source: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TelysConfigError(f"{source}: expected a string path, got {raw!r}")
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped


@dataclass(frozen=True)
class TelysConnectorConfig:
    """Validated, immutable connector configuration."""

    namespace: str = "telys-vllm"
    load_async: bool = False
    max_lookup_blocks: int = 0  # 0 = unlimited
    max_chunk_blocks: int = 0  # 0 = unlimited
    slab_path: str | None = None  # None = derive from TELYS_PATH
    write_behind: bool = True
    max_pending_writes: int = 64
    telemetry: bool = False
    telemetry_log: str | None = None
    fast_path: bool = True  # zero-decode + pinned staging + contiguous injection

    @classmethod
    def from_sources(
        cls,
        env: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> "TelysConnectorConfig":
        """Build a config from environment variables and extra config.

        Args:
            env: environment mapping (defaults to ``os.environ``).
            extra: ``KVTransferConfig.kv_connector_extra_config``; values here
                take precedence over the environment.
        """
        env = os.environ if env is None else env
        extra = {} if extra is None else extra

        def pick(env_key: str, extra_key: str) -> tuple[object, str] | None:
            if extra_key in extra:
                return extra[extra_key], f"kv_connector_extra_config[{extra_key!r}]"
            if env_key in env:
                return env[env_key], f"env {env_key}"
            return None

        values: dict[str, Any] = {}
        if picked := pick(ENV_NAMESPACE, EXTRA_NAMESPACE):
            raw, source = picked
            values["namespace"] = _parse_namespace(raw, source=source)
        if picked := pick(ENV_LOAD_ASYNC, EXTRA_LOAD_ASYNC):
            raw, source = picked
            values["load_async"] = _parse_bool(raw, source=source)
        if picked := pick(ENV_MAX_LOOKUP_BLOCKS, EXTRA_MAX_LOOKUP_BLOCKS):
            raw, source = picked
            values["max_lookup_blocks"] = _parse_positive_int(
                raw, source=source, allow_zero=True
            )
        if picked := pick(ENV_MAX_CHUNK_BLOCKS, EXTRA_MAX_CHUNK_BLOCKS):
            raw, source = picked
            values["max_chunk_blocks"] = _parse_positive_int(
                raw, source=source, allow_zero=True
            )
        if picked := pick(ENV_SLAB_PATH, EXTRA_SLAB_PATH):
            raw, source = picked
            values["slab_path"] = _parse_slab_path(raw, source=source)
        if picked := pick(ENV_WRITE_BEHIND, EXTRA_WRITE_BEHIND):
            raw, source = picked
            values["write_behind"] = _parse_bool(raw, source=source)
        if picked := pick(ENV_MAX_PENDING_WRITES, EXTRA_MAX_PENDING_WRITES):
            raw, source = picked
            values["max_pending_writes"] = _parse_positive_int(
                raw, source=source, allow_zero=False
            )
        if picked := pick(ENV_TELEMETRY, EXTRA_TELEMETRY):
            raw, source = picked
            values["telemetry"] = _parse_bool(raw, source=source)
        if picked := pick(ENV_TELEMETRY_LOG, EXTRA_TELEMETRY_LOG):
            raw, source = picked
            values["telemetry_log"] = _parse_slab_path(raw, source=source)
        if picked := pick(ENV_FAST_PATH, EXTRA_FAST_PATH):
            raw, source = picked
            values["fast_path"] = _parse_bool(raw, source=source)

        # Default slab path: derive from TELYS_PATH so a local Telys engine gets
        # a colocated slab automatically.  Remote engines (TELYS_ADDRESS) must
        # set TELYS_VLLM_SLAB_PATH explicitly.
        if "slab_path" not in values:
            telys_path = env.get("TELYS_PATH", "").strip()
            if telys_path:
                values["slab_path"] = f"{telys_path}_kv.slab"

        return cls(**values)
