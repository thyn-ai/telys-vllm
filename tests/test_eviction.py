# SPDX-License-Identifier: Apache-2.0
"""Eviction hooks: put-cadence compaction and reset_cache namespace clear."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import TelysKVConnector
from tests.fakes import (
    CREATED_CLIENTS,
    make_hashes,
    make_kv_cache_config,
    make_vllm_config,
)


def make_connector(extra=None) -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(extra), KVConnectorRole.WORKER, make_kv_cache_config()
    )


def test_no_auto_compaction_during_puts():
    connector = make_connector()
    client = CREATED_CLIENTS[-1]
    hashes = make_hashes(10)

    connector.store.save_chunk(hashes[0], hashes, {"layer.0": [b"x"] * len(hashes)})
    # Compaction must not fire automatically during normal writes.
    assert client.compact_calls == 0


def test_reset_cache_deletes_written_keys_and_compacts():
    connector = make_connector()
    client = CREATED_CLIENTS[-1]
    hashes = make_hashes(4)
    connector.store.save_chunk(hashes[0], hashes, {"layer.0": [b"x"] * len(hashes)})
    compacts_before = client.compact_calls

    assert connector.reset_cache() is True
    assert sorted(client.deleted) == sorted(
        connector.store.chunk_key(hashes[0], len(hashes)) for h in [hashes]
    )
    assert client.compact_calls == compacts_before + 1
    # The namespace reads as empty afterwards.
    assert connector.store.lookup_prefix(hashes) == 0


def test_close_compacts_once():
    connector = make_connector()
    client = CREATED_CLIENTS[-1]
    hashes = make_hashes(2)
    connector.store.save_chunk(hashes[0], hashes, {"layer.0": [b"x"] * len(hashes)})

    connector.shutdown()
    assert client.compact_calls == 1


def test_reset_cache_clears_scheduler_bookkeeping():
    connector = TelysKVConnector(
        make_vllm_config(), KVConnectorRole.SCHEDULER, make_kv_cache_config()
    )
    from tests.fakes import FakeRequest

    request = FakeRequest("req-1", 40)
    connector.on_new_request(request)
    connector.reset_cache()
    # No stale request state survives a reset; finishing the request is clean.
    assert connector.request_finished(request, []) == (False, None)
