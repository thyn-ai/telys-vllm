# SPDX-License-Identifier: Apache-2.0
"""Compile-safe save path: extraction from registered KV caches in wait_for_save."""

import json
from unittest.mock import patch

import torch
from telys_vllm.connector import (
    TelysConnectorMetadata,
    TelysKVConnector,
    TelysSaveSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

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


def make_worker(*, max_chunk_blocks: int = 0) -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(extra={"telys_max_chunk_blocks": max_chunk_blocks}),
        KVConnectorRole.WORKER,
        make_kv_cache_config(),
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


class _ComparisonCountingInt(int):
    comparisons = 0

    def __eq__(self, other):
        type(self).comparisons += 1
        return int(self) == int(other)

    __hash__ = int.__hash__


def test_collect_layer_blobs_uses_linear_block_lookup_and_preserves_mapping():
    worker = make_worker()
    num_blocks = 32
    hashes = make_hashes(num_blocks)
    block_hash_to_id = {
        block_hash: _ComparisonCountingInt(block_id)
        for block_hash, block_id in zip(hashes, reversed(range(num_blocks)))
    }
    worker.register_kv_caches({"layer.0": "layer.0", "layer.1": "layer.1"})

    def fake_extract(kv_layer, block_ids, block_size, attn_metadata):
        return [f"{kv_layer}:{int(block_id)}".encode() for block_id in block_ids]

    expected = {
        layer_name: {
            block_hash: f"{layer_name}:{int(block_id)}".encode()
            for block_hash, block_id in block_hash_to_id.items()
        }
        for layer_name in worker._kv_caches
    }
    _ComparisonCountingInt.comparisons = 0

    with (
        patch("telys_vllm.connector.extract_kv_blocks", fake_extract),
        patch(
            "telys_vllm.connector.serialize_kv_tensor",
            side_effect=lambda blob: blob,
        ),
    ):
        actual = worker._collect_layer_blobs_from_kv_caches(block_hash_to_id)

    assert actual == expected
    assert _ComparisonCountingInt.comparisons == 0


def _save_existing_run(worker: TelysKVConnector, run: list[bytes]) -> None:
    worker.store.save_chunk(run[0], run, {"layer.0": [b"existing"] * len(run)})


def test_wait_for_save_extracts_only_blocks_from_pending_runs_in_order():
    worker = make_worker(max_chunk_blocks=2)
    hashes = make_hashes(6)
    block_ids = [10, 11, 12, 13, 14, 15]
    runs = [hashes[index : index + 2] for index in range(0, 6, 2)]
    _save_existing_run(worker, runs[1])
    worker.bind_connector_metadata(
        TelysConnectorMetadata(
            saves=[TelysSaveSpec("req-1", hashes, block_ids)]
        )
    )
    collected = []

    def fake_collect(block_hash_to_id):
        collected.append(block_hash_to_id)
        return {
            "layer.0": {
                block_hash: b"new" + block_hash for block_hash in block_hash_to_id
            }
        }

    with (
        patch.object(worker, "_collect_layer_blobs", side_effect=fake_collect),
        patch.object(
            worker.store,
            "save_chunk_async",
            wraps=worker.store.save_chunk_async,
        ) as save_async,
    ):
        worker.wait_for_save()

    pending_runs = [runs[0], runs[2]]
    assert collected == [
        {
            hashes[index]: block_ids[index]
            for index in (0, 1, 4, 5)
        }
    ]
    assert [call.args[0] for call in save_async.call_args_list] == [
        run[0] for run in pending_runs
    ]
    assert [call.args[1] for call in save_async.call_args_list] == pending_runs
    assert worker.store._client.put_count == 3
    for run in pending_runs:
        key = worker.store.chunk_key(run[0], len(run), run[-1])
        record = worker.store._client.get(key)
        assert record is not None
        assert record.metadata["vllm.block_start_hash"] == run[0].hex()
        assert record.metadata["vllm.block_count"] == "2"
        assert record.metadata["vllm.block_size"] == str(BLOCK_SIZE)


def test_wait_for_save_all_present_runs_skips_extraction_and_rewriting():
    worker = make_worker(max_chunk_blocks=2)
    hashes = make_hashes(4)
    runs = [hashes[:2], hashes[2:]]
    for run in runs:
        _save_existing_run(worker, run)
    worker.bind_connector_metadata(
        TelysConnectorMetadata(
            saves=[TelysSaveSpec("req-1", hashes, [10, 11, 12, 13])]
        )
    )

    with (
        patch.object(
            worker,
            "_collect_layer_blobs",
            side_effect=AssertionError("present runs must not be extracted"),
        ),
        patch.object(worker.store, "save_chunk_async") as save_async,
    ):
        worker.wait_for_save()

    save_async.assert_not_called()
    assert worker.store._client.put_count == 2


def test_fast_save_emits_run_tensor_timing_and_preserves_bytes(tmp_path):
    telemetry_log = tmp_path / "fast-save.jsonl"
    worker = TelysKVConnector(
        make_vllm_config(
            extra={
                "telys_telemetry": True,
                "telys_telemetry_log": str(telemetry_log),
            },
            tmp_path=tmp_path,
        ),
        KVConnectorRole.WORKER,
        make_kv_cache_config(),
    )
    hashes = make_hashes(2)
    kv_layer = torch.arange(2 * BLOCK_SIZE * 2, dtype=torch.bfloat16).reshape(
        2, 1, BLOCK_SIZE, 2
    )
    worker.register_kv_caches({"layer.0": kv_layer})
    worker.bind_connector_metadata(
        TelysConnectorMetadata(
            saves=[TelysSaveSpec("req-1", hashes, [1, 0])]
        )
    )

    worker.wait_for_save()
    chunks = worker.store.load_decoded_blocks_fast(hashes)
    worker.shutdown()
    worker._telemetry.close()

    assert len(chunks) == 1
    expected = kv_layer[[1, 0]].permute(0, 2, 1, 3).contiguous()
    assert torch.equal(chunks[0].layer_tensors["layer.0"], expected)
    records = [json.loads(line) for line in telemetry_log.read_text().splitlines()]
    phases = {record["phase"] for record in records}
    assert {
        "save_device_gather",
        "save_d2h_submit",
        "save_run_tensors",
        "save_encode_chunk",
    } <= phases
    assert "save_pinned_alloc" not in phases
    assert "save_event_sync" not in phases
    run_span = next(record for record in records if record["phase"] == "save_run_tensors")
    assert run_span["context"] == {"block_count": 2, "layer_count": 1}
