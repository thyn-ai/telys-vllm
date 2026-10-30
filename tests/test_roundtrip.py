# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU round-trip through the connector:

miss -> save -> hit -> load with provenance, plus worker-side save errors.
Both connector roles share one fake Telys store (the single on-device
deployment), created through the configured client factory.
"""

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.codec import LengthPrefixedCodec
from telys_vllm.connector import (
    TelysConnectorError,
    TelysConnectorMetadata,
    TelysKVConnector,
)
from tests.fakes import (
    BLOCK_SIZE,
    CREATED_CLIENTS,
    FakeCachedReqs,
    FakeNewReq,
    FakeRequest,
    FakeSchedulerOutput,
    make_hashes,
    make_kv_cache_config,
    make_vllm_config,
)

NUM_BLOCKS = 8
TAIL_TOKENS = 5
NUM_TOKENS = NUM_BLOCKS * BLOCK_SIZE + TAIL_TOKENS


def make_connector(role: KVConnectorRole) -> TelysKVConnector:
    return TelysKVConnector(make_vllm_config(), role, make_kv_cache_config())


def make_layer_blobs(hashes, marker: bytes) -> dict[bytes, bytes]:
    return {block_hash: marker + block_hash for block_hash in hashes}


def save_prompt_through_worker(
    scheduler: TelysKVConnector,
    worker: TelysKVConnector,
    request: FakeRequest,
) -> None:
    """Drive one full miss-path step that ends with the prompt stored."""
    scheduler.on_new_request(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, None, 0)
    meta = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewReq(request.request_id, (list(range(NUM_BLOCKS + 1)),))
            ],
            num_scheduled_tokens={request.request_id: NUM_TOKENS},
        )
    )
    assert [len(spec.block_hashes) for spec in meta.saves] == [NUM_BLOCKS]

    worker.bind_connector_metadata(meta)
    hashes = request.block_hashes
    worker.save_kv_layer(
        "layer.0", None, None, block_payloads=make_layer_blobs(hashes, b"L0|")
    )
    worker.save_kv_layer(
        "layer.1", None, None, block_payloads=make_layer_blobs(hashes, b"L1|")
    )
    # Serialization is deferred to wait_for_save; no Telys writes yet.
    worker.wait_for_save()
    worker.clear_connector_metadata()


class TestRoundTrip:
    def test_save_then_hit_then_load_with_provenance(self):
        scheduler = make_connector(KVConnectorRole.SCHEDULER)
        worker = make_connector(KVConnectorRole.WORKER)
        hashes = make_hashes(NUM_BLOCKS)

        # Step 1: first request misses and gets stored by the worker.
        first = FakeRequest("req-1", NUM_TOKENS, block_hashes=hashes)
        save_prompt_through_worker(scheduler, worker, first)
        scheduler.request_finished(first, [])

        client = CREATED_CLIENTS[-1]
        # The 8 blocks were saved as one chunk.
        assert client.put_count == 1

        # Step 2: identical prompt hits the external store.
        second = FakeRequest("req-2", NUM_TOKENS, block_hashes=hashes)
        scheduler.on_new_request(second)
        assert scheduler.get_num_new_matched_tokens(second, 0) == (
            NUM_BLOCKS * BLOCK_SIZE,
            False,
        )
        scheduler.update_state_after_alloc(second, None, NUM_BLOCKS * BLOCK_SIZE)
        meta = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[
                    FakeNewReq(
                        "req-2",
                        ([100 + i for i in range(NUM_BLOCKS + 1)],),
                        num_computed_tokens=NUM_BLOCKS * BLOCK_SIZE,
                    )
                ],
                num_scheduled_tokens={"req-2": TAIL_TOKENS},
            )
        )

        # The load spec carries the hashes and the allocated destination blocks.
        assert len(meta.loads) == 1
        load_spec = meta.loads[0]
        assert load_spec.request_id == "req-2"
        assert load_spec.block_hashes == hashes
        assert load_spec.num_tokens == NUM_BLOCKS * BLOCK_SIZE
        # block_ids are sliced to the loaded region (first NUM_BLOCKS entries).
        assert load_spec.block_ids == [100 + i for i in range(NUM_BLOCKS)]

        # The loaded prefix is not re-saved (save cursor starts past it).
        assert meta.saves == []

        # Worker fetches exactly the stored blocks, with provenance.
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(None)
        records = worker.fetched_records["req-2"]
        assert len(records) == NUM_BLOCKS

        codec = LengthPrefixedCodec()
        for record, block_hash in zip(records, hashes):
            assert codec.decode_block(record.value) == {
                "layer.0": b"L0|" + block_hash,
                "layer.1": b"L1|" + block_hash,
            }
            assert record.provenance.plan_hash
            assert record.provenance.result_id
            assert record.metadata["vllm.block_hash"] == block_hash.hex()
            assert record.metadata["vllm.model"] == "fake-model-v0"

    def test_save_cursor_advances_across_steps_without_resaving(self):
        scheduler = make_connector(KVConnectorRole.SCHEDULER)
        request = FakeRequest("req-1", NUM_TOKENS, block_hashes=make_hashes(NUM_BLOCKS))
        scheduler.on_new_request(request)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.update_state_after_alloc(request, None, 0)

        # Step 1: only half the prompt scheduled.
        meta1 = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[FakeNewReq("req-1", (list(range(9)),))],
                num_scheduled_tokens={"req-1": 4 * BLOCK_SIZE},
            )
        )
        assert [h for spec in meta1.saves for h in spec.block_hashes] == (
            request.block_hashes[:4]
        )

        # Step 2: the rest; only newly completed full blocks are saved.
        meta2 = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_cached_reqs=FakeCachedReqs(
                    req_ids=["req-1"],
                    num_computed_tokens=[4 * BLOCK_SIZE],
                ),
                num_scheduled_tokens={"req-1": NUM_TOKENS - 4 * BLOCK_SIZE},
            )
        )
        assert [h for spec in meta2.saves for h in spec.block_hashes] == (
            request.block_hashes[4:]
        )

    def test_deferred_save_emits_one_chunk_per_contiguous_run(self):
        scheduler = make_connector(KVConnectorRole.SCHEDULER)
        worker = make_connector(KVConnectorRole.WORKER)
        request = FakeRequest("req-1", NUM_TOKENS, block_hashes=make_hashes(NUM_BLOCKS))
        scheduler.on_new_request(request)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.update_state_after_alloc(request, None, 0)
        meta = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[
                    FakeNewReq("req-1", (list(range(NUM_BLOCKS + 1)),))
                ],
                num_scheduled_tokens={"req-1": NUM_TOKENS},
            )
        )

        worker.bind_connector_metadata(meta)
        hashes = request.block_hashes
        worker.save_kv_layer(
            "layer.0", None, None, block_payloads=make_layer_blobs(hashes, b"L0|")
        )
        worker.save_kv_layer(
            "layer.1", None, None, block_payloads=make_layer_blobs(hashes, b"L1|")
        )
        # No Telys activity until wait_for_save.
        client = CREATED_CLIENTS[-1]
        assert client.put_count == 0
        worker.wait_for_save()
        assert client.put_count == 1


class TestWorkerSaveErrors:
    def make_worker_pair(self):
        scheduler = make_connector(KVConnectorRole.SCHEDULER)
        worker = make_connector(KVConnectorRole.WORKER)
        request = FakeRequest("req-1", NUM_TOKENS, block_hashes=make_hashes(NUM_BLOCKS))
        scheduler.on_new_request(request)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.update_state_after_alloc(request, None, 0)
        meta = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[FakeNewReq("req-1", (list(range(9)),))],
                num_scheduled_tokens={"req-1": NUM_TOKENS},
            )
        )
        return worker, request, meta

    def test_save_kv_layer_without_block_payloads_raises_when_no_kv_caches(self):
        worker, _, meta = self.make_worker_pair()
        worker.bind_connector_metadata(meta)
        with pytest.raises(TelysConnectorError, match="block_payloads"):
            worker.save_kv_layer("layer.0", None, None)

    def test_wait_for_save_without_layer_buffers_raises(self):
        worker, _, meta = self.make_worker_pair()
        worker.bind_connector_metadata(meta)
        with pytest.raises(TelysConnectorError, match="layer data"):
            worker.wait_for_save()

    def test_wait_for_save_rejects_layer_missing_a_planned_block(self):
        worker, request, meta = self.make_worker_pair()
        worker.bind_connector_metadata(meta)
        incomplete = make_layer_blobs(request.block_hashes[1:], b"L0|")
        worker.save_kv_layer("layer.0", None, None, block_payloads=incomplete)
        with pytest.raises(TelysConnectorError, match="missing"):
            worker.wait_for_save()

    def test_wait_for_save_noops_without_save_work(self):
        worker = make_connector(KVConnectorRole.WORKER)
        worker.bind_connector_metadata(TelysConnectorMetadata())
        worker.wait_for_save()  # no saves planned: must not raise
