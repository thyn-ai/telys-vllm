# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side contract tests for the Telys KV connector.

These tests prove that the connector reports external prefix hits with the
correct token boundaries, so a vLLM v1 scheduler that trusts the connector
will set ``request.num_computed_tokens`` to the hit prefix length and skip
recomputing those tokens.
"""

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import TelysConnectorError
from tests.fakes import (
    BLOCK_SIZE,
    FakeCachedReqs,
    FakeNewReq,
    FakeRequest,
    FakeSchedulerOutput,
    make_hashes,
)
from tests.test_roundtrip import (
    NUM_BLOCKS,
    NUM_TOKENS,
    TAIL_TOKENS,
    make_connector,
    save_prompt_through_worker,
)


def test_full_prefix_hit_reports_prefix_length_as_computed():
    """A full external hit must advertise exactly the prefix length as computed.

    The vLLM v1 scheduler adds the connector's ``num_external_computed_tokens``
    to ``num_new_local_computed_tokens`` and stores the sum in
    ``request.num_computed_tokens`` (scheduler.py:1045-1047, 1369).  The
    connector therefore must return a hit boundary that is a multiple of
    ``block_size`` and equals the full-block prefix length.
    """
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    worker = make_connector(KVConnectorRole.WORKER)
    hashes = make_hashes(NUM_BLOCKS)

    # Seed the store by saving an identical prompt first.
    first = FakeRequest("req-1", NUM_TOKENS, block_hashes=hashes)
    save_prompt_through_worker(scheduler, worker, first)

    # A second identical request should hit the entire full-block prefix.
    second = FakeRequest("req-2", NUM_TOKENS, block_hashes=hashes)
    scheduler.on_new_request(second)

    hit_tokens, load_async = scheduler.get_num_new_matched_tokens(second, 0)
    assert hit_tokens == NUM_BLOCKS * BLOCK_SIZE
    assert load_async is False

    # After allocation the scheduler tells the connector how many external
    # tokens were adopted; the connector records the load plan.
    scheduler.update_state_after_alloc(second, None, hit_tokens)

    # Simulate the SchedulerOutput the vLLM scheduler would emit: the new
    # request carries ``num_computed_tokens == hit_tokens`` and only the
    # trailing tokens remain to be computed.
    meta = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewReq(
                    "req-2",
                    ([100 + i for i in range(NUM_BLOCKS + 1)],),
                    num_computed_tokens=hit_tokens,
                )
            ],
            num_scheduled_tokens={"req-2": TAIL_TOKENS},
        )
    )

    assert len(meta.loads) == 1
    load_spec = meta.loads[0]
    assert load_spec.request_id == "req-2"
    assert load_spec.num_tokens == hit_tokens
    assert load_spec.block_hashes == hashes
    # Destination blocks are sliced to the loaded region.
    assert load_spec.block_ids == [100 + i for i in range(NUM_BLOCKS)]
    # The adopted prefix must not be re-saved.
    assert meta.saves == []


def _build_new_request_meta(scheduler, request, num_computed_tokens: int):
    return scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewReq(
                    request.request_id,
                    (list(range(NUM_BLOCKS + 1)),),
                    num_computed_tokens=num_computed_tokens,
                )
            ],
            num_scheduled_tokens={request.request_id: NUM_TOKENS - num_computed_tokens},
        )
    )


def _build_cached_request_meta(
    scheduler,
    request,
    num_computed_tokens: int,
    new_block_ids=None,
    resolved_block_ids=None,
    resumed: bool = False,
):
    scheduler_output = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedReqs(
            req_ids=[request.request_id],
            resumed_req_ids={request.request_id} if resumed else set(),
            num_computed_tokens=[num_computed_tokens],
            new_block_ids=[new_block_ids] if new_block_ids is not None else [],
        ),
        num_scheduled_tokens={request.request_id: NUM_TOKENS - num_computed_tokens},
    )
    if resolved_block_ids is not None:
        scheduler_output.kv_connector_block_state = SimpleNamespace(
            get_block_ids=lambda _req_id: resolved_block_ids
        )
    return scheduler.build_connector_meta(scheduler_output)


def _prepare_resumed_external_hit():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    for block_hash in hashes:
        scheduler.store.save_chunk(block_hash, [block_hash], {"layer.0": [b"cached"]})

    request = FakeRequest("req-resumed-hit", NUM_TOKENS, block_hashes=hashes)
    scheduler.on_new_request(request)
    local_blocks = 2
    hit_tokens, _ = scheduler.get_num_new_matched_tokens(
        request, local_blocks * BLOCK_SIZE
    )
    assert hit_tokens == (NUM_BLOCKS - local_blocks) * BLOCK_SIZE
    scheduler.update_state_after_alloc(request, None, hit_tokens)
    return scheduler, request, hashes, local_blocks, hit_tokens


@pytest.mark.parametrize("use_block_state", [True, False])
def test_resumed_external_hit_emits_load_once_without_resaving(use_block_state):
    scheduler, request, hashes, local_blocks, hit_tokens = (
        _prepare_resumed_external_hit()
    )
    allocated = [100 + i for i in range(NUM_BLOCKS + 1)]
    block_sources = (
        {"new_block_ids": ([999],), "resolved_block_ids": (allocated, [888])}
        if use_block_state
        else {"new_block_ids": (allocated,)}
    )

    meta = _build_cached_request_meta(
        scheduler,
        request,
        NUM_BLOCKS * BLOCK_SIZE,
        resumed=True,
        **block_sources,
    )

    assert len(meta.loads) == 1
    load_spec = meta.loads[0]
    assert load_spec.request_id == request.request_id
    assert load_spec.start_block == local_blocks
    assert load_spec.block_hashes == hashes[local_blocks:]
    assert load_spec.block_ids == allocated[local_blocks:NUM_BLOCKS]
    assert load_spec.num_tokens == hit_tokens
    assert meta.saves == []
    assert scheduler._save_cursors[request.request_id] == NUM_BLOCKS
    assert scheduler._load_plans == {}

    repeated = _build_cached_request_meta(
        scheduler,
        request,
        NUM_BLOCKS * BLOCK_SIZE,
        resumed=True,
        **block_sources,
    )
    assert repeated.loads == []
    assert repeated.saves == []


def test_resumed_external_hit_without_full_block_ids_fails_explicitly():
    scheduler, request, _, _, _ = _prepare_resumed_external_hit()

    with pytest.raises(TelysConnectorError, match=request.request_id):
        _build_cached_request_meta(
            scheduler, request, NUM_BLOCKS * BLOCK_SIZE, resumed=True
        )
    with pytest.raises(TelysConnectorError, match=request.request_id):
        _build_cached_request_meta(
            scheduler,
            request,
            NUM_BLOCKS * BLOCK_SIZE,
            new_block_ids=([999],),
            resumed=True,
        )

    assert request.request_id in scheduler._load_plans


def test_new_native_hit_with_stale_allocation_saves_only_the_new_suffix():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-new-native", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 3
    local_tokens = local_blocks * BLOCK_SIZE
    scheduler.on_new_request(request)

    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    scheduler.update_state_after_alloc(request, None, 0)
    assert request.num_computed_tokens == 0
    meta = _build_new_request_meta(scheduler, request, local_tokens)

    assert meta.loads == []
    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes[local_blocks:]
    assert meta.saves[0].block_ids == list(range(local_blocks, NUM_BLOCKS))


def test_cached_continuation_uses_scheduler_group_zero_block_ids():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-cached-ids", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 3
    local_tokens = local_blocks * BLOCK_SIZE
    allocated = list(range(100, 100 + NUM_BLOCKS + 1))
    scheduler.on_new_request(request)
    scheduler.get_num_new_matched_tokens(request, local_tokens)
    scheduler.update_state_after_alloc(request, None, 0)

    meta = _build_cached_request_meta(
        scheduler,
        request,
        local_tokens,
        new_block_ids=([999],),
        resolved_block_ids=(allocated, [888]),
    )

    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes[local_blocks:]
    assert meta.saves[0].block_ids == allocated[local_blocks:NUM_BLOCKS]


def test_cached_continuation_rejects_delta_ids_shorter_than_save_boundary():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-short-delta", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 3
    local_tokens = local_blocks * BLOCK_SIZE
    delta_ids = list(range(100, 100 + local_blocks + 2))
    scheduler.on_new_request(request)
    scheduler.get_num_new_matched_tokens(request, local_tokens)
    scheduler.update_state_after_alloc(request, None, 0)

    meta = _build_cached_request_meta(
        scheduler, request, local_tokens, new_block_ids=(delta_ids,)
    )

    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes[local_blocks:]
    assert meta.saves[0].block_ids == []


def test_cached_continuation_falls_back_to_new_group_zero_block_ids():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-cached-fallback", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 3
    local_tokens = local_blocks * BLOCK_SIZE
    allocated = list(range(100, 100 + NUM_BLOCKS + 1))
    scheduler.on_new_request(request)
    scheduler.get_num_new_matched_tokens(request, local_tokens)
    scheduler.update_state_after_alloc(request, None, 0)

    meta = _build_cached_request_meta(
        scheduler, request, local_tokens, new_block_ids=(allocated,)
    )

    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes[local_blocks:]
    assert meta.saves[0].block_ids == allocated[local_blocks:NUM_BLOCKS]


def test_cached_native_hit_with_stale_allocation_saves_only_the_new_suffix():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-native", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 3
    local_tokens = local_blocks * BLOCK_SIZE
    scheduler.on_new_request(request)

    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    scheduler.update_state_after_alloc(request, None, 0)
    assert request.num_computed_tokens == 0
    meta = _build_cached_request_meta(scheduler, request, local_tokens)

    assert meta.loads == []
    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes[local_blocks:]
    assert meta.saves[0].block_ids == []


def test_zero_external_miss_still_saves_the_whole_prompt():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-miss", NUM_TOKENS, block_hashes=hashes)
    scheduler.on_new_request(request)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, None, 0)
    meta = _build_new_request_meta(scheduler, request, 0)

    assert meta.loads == []
    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == hashes
    assert meta.saves[0].block_ids == list(range(NUM_BLOCKS))


def test_cached_zero_external_allocation_updates_are_idempotent():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    hashes = make_hashes(NUM_BLOCKS)
    request = FakeRequest("req-repeat", NUM_TOKENS, block_hashes=hashes)
    local_blocks = 2
    local_tokens = local_blocks * BLOCK_SIZE
    scheduler.on_new_request(request)
    scheduler.get_num_new_matched_tokens(request, local_tokens)

    scheduler.update_state_after_alloc(request, None, 0)
    scheduler.update_state_after_alloc(request, None, 0)
    assert request.num_computed_tokens == 0
    first_meta = _build_cached_request_meta(scheduler, request, local_tokens)
    assert len(first_meta.saves) == 1
    assert first_meta.saves[0].block_hashes == hashes[local_blocks:]

    scheduler.update_state_after_alloc(request, None, 0)
    second_meta = _build_cached_request_meta(scheduler, request, local_tokens)
    assert second_meta.saves == []


def _cached_decode_meta(scheduler, request, computed, scheduled, block_ids):
    scheduler_output = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedReqs(
            req_ids=[request.request_id],
            num_computed_tokens=[computed],
        ),
        num_scheduled_tokens={request.request_id: scheduled},
    )
    scheduler_output.kv_connector_block_state = SimpleNamespace(
        get_block_ids=lambda _req_id: (block_ids,)
    )
    return scheduler.build_connector_meta(scheduler_output)


def test_exact_aligned_prompt_block_is_saved_on_first_decode():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    request = FakeRequest("req-aligned-decode", 4 * BLOCK_SIZE)
    block_ids = list(range(20))
    scheduler.on_new_request(request)

    prefill = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewReq(request.request_id, (block_ids,))],
            num_scheduled_tokens={request.request_id: request.num_prompt_tokens},
        )
    )
    assert len(prefill.saves) == 1
    assert len(prefill.saves[0].block_hashes) == 3
    assert prefill.saves[0].block_hashes == request.block_hashes[:3]

    request.append_output_token_ids([100])
    decode = _cached_decode_meta(scheduler, request, 4 * BLOCK_SIZE, 1, block_ids)

    assert len(decode.saves) == 1
    assert decode.saves[0].block_hashes == [request.block_hashes[3]]
    assert decode.saves[0].block_ids == [block_ids[3]]


def test_decode_completion_saves_prompt_boundary_and_generated_blocks():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    prompt_tokens = 3 * BLOCK_SIZE + 12
    request = FakeRequest("req-generated-decode", prompt_tokens)
    block_ids = list(range(20))
    scheduler.on_new_request(request)

    prefill = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewReq(request.request_id, (block_ids,))],
            num_scheduled_tokens={request.request_id: prompt_tokens},
        )
    )
    assert len(prefill.saves) == 1
    assert len(prefill.saves[0].block_hashes) == 3

    request.append_output_token_ids(list(range(5)))
    boundary = _cached_decode_meta(scheduler, request, prompt_tokens + 4, 1, block_ids)
    assert len(boundary.saves) == 1
    assert boundary.saves[0].block_hashes == [request.block_hashes[3]]
    assert boundary.saves[0].block_ids == [block_ids[3]]

    request.append_output_token_ids(list(range(5, 69)))
    generated = _cached_decode_meta(
        scheduler, request, prompt_tokens + 68, 1, block_ids
    )
    assert len(generated.saves) == 1
    assert generated.saves[0].block_hashes == request.block_hashes[4:8]
    assert generated.saves[0].block_ids == block_ids[4:8]
    assert len(request.block_hashes) == 8
    assert len(generated.saves[0].block_hashes) == 4


def test_async_final_schedule_uses_available_hash_boundary():
    scheduler = make_connector(KVConnectorRole.SCHEDULER)
    request = FakeRequest("req-async-final", 50)
    request.append_output_token_ids(list(range(62)))
    block_ids = list(range(len(request.block_hashes)))
    scheduler.on_new_request(request)
    scheduler._save_cursors[request.request_id] = 6

    assert request.num_tokens == 112
    assert len(request.block_hashes) == 7
    final = _cached_decode_meta(scheduler, request, 113, 1, block_ids)

    assert len(final.saves) == 1
    assert final.saves[0].block_hashes == [request.block_hashes[6]]
    assert final.saves[0].block_ids == [block_ids[6]]
    assert scheduler._save_cursors[request.request_id] == 7
    assert _cached_decode_meta(scheduler, request, 113, 1, block_ids).saves == []
