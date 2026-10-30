# SPDX-License-Identifier: Apache-2.0
"""TelysBlockStore + codec: chunk key mapping, round-trip, errors, async writes."""

import pytest

from telys_vllm.codec import LengthPrefixedCodec, PayloadCodecError, TelysChunkCodec
from telys_vllm.config import TelysConnectorConfig
from telys_vllm.store import TelysBlockStore, TelysStoreError
from tests.fakes import FakeTelysClient, make_fake_store, make_hashes

LAYER = "layer.0"


def make_store(tmp_path, client=None, **config_kwargs):
    config_kwargs.setdefault("slab_path", str(tmp_path / "kv.slab"))
    config = TelysConnectorConfig(**config_kwargs)
    return TelysBlockStore(
        client or FakeTelysClient(),
        config,
        model_id="fake-model-v0",
        engine_id="test-engine",
        block_size=16,
    )


def seed_blocks(
    store: TelysBlockStore, block_hashes: list[bytes], payload: bytes = b"x"
):
    """Store each block as its own single-block chunk (test convenience)."""
    for block_hash in block_hashes:
        store.save_chunk(
            block_hash,
            [block_hash],
            {LAYER: [payload]},
        )


def seed_chunk(
    store: TelysBlockStore, block_hashes: list[bytes], payload: bytes = b"x"
):
    """Store ``block_hashes`` as one contiguous multi-block chunk."""
    store.save_chunk(
        block_hashes[0],
        block_hashes,
        {LAYER: [payload] * len(block_hashes)},
    )


def decode_record_value(record) -> dict[str, bytes]:
    return LengthPrefixedCodec().decode_block(record.value)


def test_block_key_mapping_is_deterministic(tmp_path):
    store = make_store(tmp_path)
    (hash_a,) = make_hashes(1)
    assert store.block_key(hash_a) == f"telys-vllm/kv/{hash_a.hex()}"
    assert store.block_key(hash_a) == store.block_key(hash_a)


def test_chunk_key_mapping_is_deterministic(tmp_path):
    store = make_store(tmp_path)
    hashes = make_hashes(3)
    assert store.chunk_key(hashes[0], 3) == f"telys-vllm/chunk/{hashes[0].hex()}/3"


def test_namespace_scopes_keys(tmp_path):
    store = make_store(tmp_path, namespace="other-ns")
    (hash_a,) = make_hashes(1)
    assert store.block_key(hash_a).startswith("other-ns/kv/")
    assert store.chunk_start_key(hash_a).startswith("other-ns/chunk/")


def test_save_load_round_trip_with_provenance_and_metadata(tmp_path):
    client = FakeTelysClient()
    store = make_store(tmp_path, client)
    hash_a, hash_b = make_hashes(2)
    seed_blocks(store, [hash_a, hash_b], b"payload")
    store.drain()

    records = store.load_blocks([hash_a, hash_b])
    assert decode_record_value(records[0]) == {LAYER: b"payload"}
    assert decode_record_value(records[1]) == {LAYER: b"payload"}
    for record, block_hash in zip(records, (hash_a, hash_b)):
        # Telys read provenance is present and deterministic per key.
        assert record.provenance.plan_hash
        assert record.provenance.result_id
        # vLLM-side provenance was written as metadata.
        assert record.metadata["vllm.block_hash"] == block_hash.hex()
        assert record.metadata["vllm.block_size"] == "16"
        assert record.metadata["vllm.model"] == "fake-model-v0"
        assert record.metadata["vllm.engine_id"] == "test-engine"
        # Slab pointer is present and points to real data.
        assert int(record.metadata["vllm.slab_offset"]) >= 16
        assert int(record.metadata["vllm.slab_length"]) > 0


def test_load_blocks_raises_on_vanished_key(tmp_path):
    store = make_store(tmp_path)
    hash_a, hash_b = make_hashes(2)
    seed_blocks(store, [hash_a], b"a")
    store.drain()
    with pytest.raises(TelysStoreError, match="no longer present"):
        store.load_blocks([hash_a, hash_b])


def test_lookup_prefix_stops_at_first_miss(tmp_path):
    store = make_store(tmp_path)
    hashes = make_hashes(4)
    seed_blocks(store, [hashes[0], hashes[1]], b"x")
    store.drain()
    # hashes[2] missing, hashes[3] present: a hole breaks the prefix.
    seed_blocks(store, [hashes[3]], b"x")
    store.drain()
    assert store.lookup_prefix(hashes) == 2
    assert store.lookup_prefix(hashes[1:]) == 1
    assert store.lookup_prefix(hashes[2:]) == 0
    assert store.lookup_prefix([]) == 0


def test_lookup_prefix_respects_max_lookup_blocks(tmp_path):
    store = make_store(tmp_path, max_lookup_blocks=2)
    hashes = make_hashes(4)
    seed_blocks(store, hashes, b"x")
    store.drain()
    assert store.lookup_prefix(hashes) == 2


def test_lookup_prefix_uses_chunk_length_not_value_fetch(tmp_path):
    """The prefix scan must be a membership probe, not a per-block read."""
    client = FakeTelysClient()
    store = make_store(tmp_path, client)
    hashes = make_hashes(4)
    seed_blocks(store, hashes, b"x")
    store.drain()

    client.get_count = 0
    assert store.lookup_prefix(hashes) == 4
    # lookup_prefix should only call chunk_length(), never get().
    assert client.get_count == 0


def test_multi_block_chunk_round_trip_and_slicing(tmp_path):
    store = make_store(tmp_path)
    hashes = make_hashes(5)
    seed_chunk(store, hashes, b"block-payload")
    store.drain()

    # Full load returns all blocks.
    records = store.load_blocks(hashes)
    assert len(records) == 5
    for record in records:
        assert decode_record_value(record) == {LAYER: b"block-payload"}

    # Partial load slices the chunk.
    partial = store.load_blocks(hashes[:2])
    assert len(partial) == 2
    for record in partial:
        assert decode_record_value(record) == {LAYER: b"block-payload"}


def test_save_chunk_rejects_mismatched_layer_counts(tmp_path):
    store = make_store(tmp_path)
    hashes = make_hashes(2)
    with pytest.raises(TelysStoreError, match="layer_blobs"):
        store.save_chunk(
            hashes[0],
            hashes,
            {LAYER: [b"a"]},  # only one block of payload for two blocks
        )


def test_write_behind_drain_makes_writes_visible(tmp_path):
    store = make_fake_store(tmp_path, write_behind=True)
    hashes = make_hashes(4)
    seed_blocks(store, hashes, b"async")

    # Before drain, lookup_prefix sees no confirmed chunks.
    assert store.lookup_prefix(hashes) == 0

    store.drain()
    assert store.lookup_prefix(hashes) == 4

    records = store.load_blocks(hashes)
    assert len(records) == 4


def test_write_behind_dedup_within_store(tmp_path):
    store = make_fake_store(tmp_path, write_behind=True)
    hashes = make_hashes(4)
    seed_chunk(store, hashes, b"dedup")
    seed_chunk(store, hashes, b"dedup")  # same chunk again
    store.drain()

    # The fake client should have received exactly one put.
    assert store._client.put_count == 1


def test_contains_sees_pending_writes(tmp_path):
    store = make_fake_store(tmp_path, write_behind=True, max_pending_writes=10)
    hashes = make_hashes(2)
    key = store.chunk_key(hashes[0], 2)
    assert not store.contains(key)
    seed_chunk(store, hashes, b"pending")
    assert store.contains(key)
    store.drain()
    assert store.contains(key)


def test_load_blocks_drains_pending_chunk(tmp_path):
    store = make_fake_store(tmp_path, write_behind=True)
    hashes = make_hashes(2)
    seed_chunk(store, hashes, b"will-drain")
    # No explicit drain: load_blocks must wait for the pending write.
    records = store.load_blocks(hashes)
    assert len(records) == 2
    assert decode_record_value(records[0]) == {LAYER: b"will-drain"}


def test_codec_round_trip_is_byte_deterministic():
    codec = LengthPrefixedCodec()
    blobs = {"layer.1": b"\x01\x02", "layer.0": b"\x03"}
    payload = codec.encode_block(blobs)
    assert payload == codec.encode_block(blobs)
    # Layer order in the input mapping does not affect the framing.
    assert codec.encode_block(dict(reversed(list(blobs.items())))) == payload
    assert codec.decode_block(payload) == blobs


def test_chunk_codec_round_trip():
    codec = TelysChunkCodec()
    layer_blobs = {
        "layer.0": [b"a0", b"a1", b"a2"],
        "layer.1": [b"b0", b"b1", b"b2"],
    }
    payload = codec.encode_chunk(layer_blobs)
    decoded = codec.decode_chunk(payload)
    assert decoded == [
        {"layer.0": b"a0", "layer.1": b"b0"},
        {"layer.0": b"a1", "layer.1": b"b1"},
        {"layer.0": b"a2", "layer.1": b"b2"},
    ]


def test_chunk_codec_rejects_empty_and_non_bytes():
    codec = TelysChunkCodec()
    with pytest.raises(PayloadCodecError):
        codec.encode_chunk({})
    with pytest.raises(PayloadCodecError, match="same block count"):
        codec.encode_chunk({"layer.0": [b"a"], "layer.1": [b"a", b"b"]})


def test_codec_rejects_malformed_payload():
    codec = LengthPrefixedCodec()
    with pytest.raises(PayloadCodecError):
        codec.decode_block(b"\x05\x00")  # truncated
    with pytest.raises(PayloadCodecError):
        codec.decode_block(b"")  # no layers
