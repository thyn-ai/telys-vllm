# SPDX-License-Identifier: Apache-2.0
"""Config surface: defaults, env/extra precedence, fail-fast validation."""

import pytest

from telys_vllm.config import (
    ENV_LOAD_ASYNC,
    ENV_MAX_CHUNK_BLOCKS,
    ENV_MAX_LOOKUP_BLOCKS,
    ENV_NAMESPACE,
    TelysConfigError,
    TelysConnectorConfig,
)


def test_defaults_when_nothing_set():
    config = TelysConnectorConfig.from_sources(env={}, extra={})
    assert config.namespace == "telys-vllm"
    assert config.load_async is False
    assert config.max_lookup_blocks == 0
    assert config.max_chunk_blocks == 0


def test_env_overrides():
    config = TelysConnectorConfig.from_sources(
        env={
            ENV_NAMESPACE: "prod-ns",
            ENV_LOAD_ASYNC: "1",
            ENV_MAX_LOOKUP_BLOCKS: "42",
            ENV_MAX_CHUNK_BLOCKS: "8",
        },
        extra={},
    )
    assert config.namespace == "prod-ns"
    assert config.load_async is True
    assert config.max_lookup_blocks == 42
    assert config.max_chunk_blocks == 8


def test_extra_config_beats_env():
    config = TelysConnectorConfig.from_sources(
        env={ENV_NAMESPACE: "env-ns", ENV_MAX_LOOKUP_BLOCKS: "5"},
        extra={
            "telys_namespace": "extra-ns",
            "telys_max_lookup_blocks": 9,
            "telys_max_chunk_blocks": 16,
        },
    )
    assert config.namespace == "extra-ns"
    assert config.max_lookup_blocks == 9
    assert config.max_chunk_blocks == 16


@pytest.mark.parametrize("raw", ["maybe", "2", "", "yest"])
def test_load_async_rejects_non_boolean(raw):
    with pytest.raises(TelysConfigError, match="telys_load_async"):
        TelysConnectorConfig.from_sources(env={}, extra={"telys_load_async": raw})


@pytest.mark.parametrize("raw", ["", "   ", " padded", "padded "])
def test_namespace_rejects_empty_or_padded(raw):
    with pytest.raises(TelysConfigError, match="telys_namespace"):
        TelysConnectorConfig.from_sources(env={}, extra={"telys_namespace": raw})


def test_max_lookup_blocks_allows_zero_rejects_negative():
    config = TelysConnectorConfig.from_sources(
        env={}, extra={"telys_max_lookup_blocks": 0}
    )
    assert config.max_lookup_blocks == 0
    with pytest.raises(TelysConfigError, match="telys_max_lookup_blocks"):
        TelysConnectorConfig.from_sources(env={}, extra={"telys_max_lookup_blocks": -1})


def test_max_chunk_blocks_allows_zero_rejects_negative():
    config = TelysConnectorConfig.from_sources(
        env={}, extra={"telys_max_chunk_blocks": 0}
    )
    assert config.max_chunk_blocks == 0
    with pytest.raises(TelysConfigError, match="telys_max_chunk_blocks"):
        TelysConnectorConfig.from_sources(env={}, extra={"telys_max_chunk_blocks": -1})
