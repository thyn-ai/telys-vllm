# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side contract tests for the Telys KV connector.

These tests prove that the connector reports external prefix hits with the
correct token boundaries, so a vLLM v1 scheduler that trusts the connector
will set ``request.num_computed_tokens`` to the hit prefix length and skip
recomputing those tokens.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tests.fakes import (
    BLOCK_SIZE,
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
