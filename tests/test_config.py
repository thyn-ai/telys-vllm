# SPDX-License-Identifier: Apache-2.0
"""Config surface: defaults, env/extra precedence, fail-fast validation."""

import os
from unittest.mock import patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.config import (
    ENV_BLOBSTORE_NO_SLAB,
    ENV_LOAD_ASYNC,
    ENV_MAX_CHUNK_BLOCKS,
    ENV_MAX_LOOKUP_BLOCKS,
    ENV_NAMESPACE,
    TelysConfigError,
    TelysConnectorConfig,
)
from telys_vllm.connector import TelysKVConnector
from tests.fakes import CREATED_CLIENTS, make_kv_cache_config, make_vllm_config


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
            ENV_LOAD_ASYNC: "0",
            ENV_MAX_LOOKUP_BLOCKS: "42",
            ENV_MAX_CHUNK_BLOCKS: "8",
        },
        extra={},
    )
    assert config.namespace == "prod-ns"
    assert config.load_async is False
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


def test_blobstore_no_slab_is_opt_in_and_extra_beats_env():
    assert (
        TelysConnectorConfig.from_sources(env={}, extra={}).blobstore_no_slab is False
    )

    from_env = TelysConnectorConfig.from_sources(
        env={ENV_BLOBSTORE_NO_SLAB: "true"}, extra={}
    )
    assert from_env.blobstore_no_slab is True

    overridden = TelysConnectorConfig.from_sources(
        env={ENV_BLOBSTORE_NO_SLAB: "true"},
        extra={"telys_blobstore_no_slab": False},
    )
    assert overridden.blobstore_no_slab is False


@pytest.mark.parametrize("raw", ["maybe", "2", ""])
def test_blobstore_no_slab_rejects_non_boolean(raw):
    with pytest.raises(TelysConfigError, match="telys_blobstore_no_slab"):
        TelysConnectorConfig.from_sources(
            env={}, extra={"telys_blobstore_no_slab": raw}
        )


_ASYNC_UNSUPPORTED = (
    "async loading is unsupported in this revision; load_async=false is required"
)


@pytest.mark.parametrize("source", ["env", "extra"])
def test_load_async_true_fails_fast_before_connector_state_is_created(source):
    env = {ENV_LOAD_ASYNC: "true"} if source == "env" else {}
    extra = {"telys_load_async": True} if source == "extra" else {}

    with patch.object(
        TelysConnectorConfig, "__init__", autospec=True, return_value=None
    ) as config_init:
        with pytest.raises(TelysConfigError, match=_ASYNC_UNSUPPORTED):
            TelysConnectorConfig.from_sources(env=env, extra=extra)
        config_init.assert_not_called()

    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(TelysConfigError, match=_ASYNC_UNSUPPORTED):
            TelysKVConnector(
                make_vllm_config(extra=extra),
                KVConnectorRole.SCHEDULER,
                make_kv_cache_config(),
            )
    assert not CREATED_CLIENTS


def test_load_async_false_can_override_environment_true():
    config = TelysConnectorConfig.from_sources(
        env={ENV_LOAD_ASYNC: "true"}, extra={"telys_load_async": False}
    )
    assert config.load_async is False
