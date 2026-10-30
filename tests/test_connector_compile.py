# SPDX-License-Identifier: Apache-2.0
"""Compile-safe save path: extraction from registered KV caches in wait_for_save."""

from unittest.mock import patch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import TelysKVConnector
from tests.fakes import (
    FakeNewReq,
    FakeRequest,
    FakeSchedulerOutput,
    make_hashes,
    make_kv_cache_config,
    make_vllm_config,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_TOKENS = NUM_BLOCKS * BLOCK_SIZE + 3


def make_worker() -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(), KVConnectorRole.WORKER, make_kv_cache_config()
    )


def make_save_meta(worker: TelysKVConnector, request: FakeRequest):
    scheduler = TelysKVConnector(
        make_vllm_config(), KVConnectorRole.SCHEDULER, make_kv_cache_config()
    )
    scheduler.on_new_request(request)
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, None, 0)
    return scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewReq(request.request_id, (list(range(NUM_BLOCKS + 1)),))
            ],
            num_scheduled_tokens={request.request_id: NUM_TOKENS},
        )
    )


def test_wait_for_save_extracts_from_registered_kv_caches():
    """When KV caches are registered, wait_for_save can save without save_kv_layer.

    This mirrors the ``torch.compile`` path where the per-layer save hook is
    skipped: the connector still extracts and persists blocks at step end.
    """
    worker = make_worker()
    worker_client = worker.store._client
    request = FakeRequest("req-1", NUM_TOKENS, block_hashes=make_hashes(NUM_BLOCKS))
    meta = make_save_meta(worker, request)

    worker.bind_connector_metadata(meta)
    # Register fake KV caches exactly as vLLM would at model-load time.
    worker.register_kv_caches({"layer.0": object(), "layer.1": object()})
    # Provide fake attention metadata so the extraction path is not skipped.
    worker._step_attn_metadata = {"layer.0": object(), "layer.1": object()}

    def fake_extract(kv_layer, block_ids, block_size, attn_metadata):
        return [b"serialized-block"] * len(block_ids)

    def fake_serialize(tensor):
        return tensor

    with (
        patch("telys_vllm.connector.extract_kv_blocks", fake_extract),
        patch("telys_vllm.connector.serialize_kv_tensor", fake_serialize),
    ):
        # No save_kv_layer calls are needed when caches are registered.
        worker.wait_for_save()

    # All blocks are one contiguous run -> one chunk.
    assert worker_client.put_count == 1
