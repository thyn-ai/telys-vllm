# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side hit/miss semantics of TelysKVConnector.

Miss must be exactly ``(0, False)`` and hits ``(n_tokens, load_async)``,
matching the KVConnectorBase_V1 contract (v1/base.py:515-548).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import TelysKVConnector
from telys_vllm.store import TelysBlockStore
from tests.fakes import (
    BLOCK_SIZE,
    FakeRequest,
    make_hashes,
    make_kv_cache_config,
    make_vllm_config,
)


def make_scheduler_connector(extra=None) -> TelysKVConnector:
    return TelysKVConnector(
        make_vllm_config(extra), KVConnectorRole.SCHEDULER, make_kv_cache_config()
    )


def seed_store(store: TelysBlockStore, hashes, payload=b"x") -> None:
    for block_hash in hashes:
        store.save_chunk(
            block_hash,
            [block_hash],
            {"layer.0": [payload]},
        )


class TestMissPath:
    def test_empty_store_is_exact_miss_signal(self):
        connector = make_scheduler_connector()
        request = FakeRequest("r1", num_tokens=4 * BLOCK_SIZE + 3)
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    def test_mid_chain_hole_caps_hit_at_first_miss(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(6)
        seed_store(connector.store, [hashes[0], hashes[1], hashes[3]])
        request = FakeRequest("r1", num_tokens=6 * BLOCK_SIZE + 3, block_hashes=hashes)
        # Blocks 0-1 hit, block 2 misses: contiguous prefix stops there.
        assert connector.get_num_new_matched_tokens(request, 0) == (
            2 * BLOCK_SIZE,
            False,
        )

    def test_offset_beyond_stored_prefix_is_miss(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(4)
        seed_store(connector.store, hashes[:2])
        request = FakeRequest("r1", num_tokens=4 * BLOCK_SIZE + 3, block_hashes=hashes)
        assert connector.get_num_new_matched_tokens(request, 2 * BLOCK_SIZE) == (
            0,
            False,
        )

    def test_short_prompt_without_full_blocks_is_miss(self):
        connector = make_scheduler_connector()
        request = FakeRequest("r1", num_tokens=BLOCK_SIZE - 1)
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

    def test_skip_reading_prefix_cache_forces_miss(self):
        connector = make_scheduler_connector()
        request = FakeRequest(
            "r1", num_tokens=4 * BLOCK_SIZE + 3, skip_reading_prefix_cache=True
        )
        seed_store(connector.store, request.block_hashes)
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)


class TestHitPath:
    def test_full_prefix_hit(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(4)
        seed_store(connector.store, hashes)
        request = FakeRequest("r1", num_tokens=4 * BLOCK_SIZE + 3, block_hashes=hashes)
        assert connector.get_num_new_matched_tokens(request, 0) == (
            4 * BLOCK_SIZE,
            False,
        )

    def test_hit_respects_num_computed_tokens_offset(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(4)
        seed_store(connector.store, hashes)
        request = FakeRequest("r1", num_tokens=4 * BLOCK_SIZE + 3, block_hashes=hashes)
        # One block already computed locally: only the tail counts.
        assert connector.get_num_new_matched_tokens(request, BLOCK_SIZE) == (
            3 * BLOCK_SIZE,
            False,
        )

    def test_last_full_block_never_offered_when_prompt_is_block_aligned(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(4)
        seed_store(connector.store, hashes)
        request = FakeRequest("r1", num_tokens=4 * BLOCK_SIZE, block_hashes=hashes)
        # (num_tokens - 1) // BLOCK_SIZE == 3: the final block's last token
        # must be computed to produce output.
        assert connector.get_num_new_matched_tokens(request, 0) == (
            3 * BLOCK_SIZE,
            False,
        )

    def test_hit_capped_by_max_lookup_blocks(self):
        connector = make_scheduler_connector({"telys_max_lookup_blocks": 2})
        hashes = make_hashes(5)
        seed_store(connector.store, hashes)
        request = FakeRequest("r1", num_tokens=5 * BLOCK_SIZE + 3, block_hashes=hashes)
        assert connector.get_num_new_matched_tokens(request, 0) == (
            2 * BLOCK_SIZE,
            False,
        )

    def test_hit_reports_async_flag_only_when_configured(self):
        hashes = make_hashes(3)
        sync_connector = make_scheduler_connector()
        async_connector = make_scheduler_connector({"telys_load_async": True})
        seed_store(sync_connector.store, hashes)
        request = FakeRequest("r1", num_tokens=3 * BLOCK_SIZE + 3, block_hashes=hashes)
        assert sync_connector.get_num_new_matched_tokens(request, 0) == (
            3 * BLOCK_SIZE,
            False,
        )
        assert async_connector.get_num_new_matched_tokens(request, 0) == (
            3 * BLOCK_SIZE,
            True,
        )

    def test_repeated_lookup_is_idempotent(self):
        connector = make_scheduler_connector()
        hashes = make_hashes(3)
        seed_store(connector.store, hashes)
        request = FakeRequest("r1", num_tokens=3 * BLOCK_SIZE + 3, block_hashes=hashes)
        first = connector.get_num_new_matched_tokens(request, 0)
        second = connector.get_num_new_matched_tokens(request, 0)
        assert first == second == (3 * BLOCK_SIZE, False)
