# SPDX-License-Identifier: Apache-2.0
"""TelysBlockStore + codec: chunk key mapping, round-trip, errors, async writes."""

import threading
from dataclasses import replace

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


class FakeBlobStoreClient(FakeTelysClient):
    """Metadata-compatible fake that also retains values like BlobStore."""

    stores_payload_values = True

    def __init__(self) -> None:
        super().__init__()
        self.payload_writes: list[bytes] = []
        self._values: dict[str, bytes] = {}

    def put(self, key, value, metadata) -> None:
        self.payload_writes.append(value)
        self._values[key] = value
        super().put(key, value, metadata)

    def get(self, key):
        record = super().get(key)
        if record is None:
            return None
        return replace(record, value=self._values[key])


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


def make_shared_start_chains(block_count: int = 259) -> list[list[bytes]]:
    import hashlib

    chains = []
    for chain_index in range(4):
        parent = b"\xa0" * 32
        chain = [parent]
        for block_index in range(1, block_count):
            parent = hashlib.sha256(
                parent
                + chain_index.to_bytes(2, "big")
                + block_index.to_bytes(4, "big")
            ).digest()
            chain.append(parent)
        chains.append(chain)
    return chains


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
    expected = f"telys-vllm/chunk/{hashes[0].hex()}/{hashes[-1].hex()}/3"
    assert store.chunk_key(hashes[0], 3, hashes[-1]) == expected


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
    # lookup_prefix uses metadata/membership probes, never get() payload reads.
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

    # A shorter prefix requires its own exact terminal-hash key.
    seed_chunk(store, hashes[:2], b"block-payload")
    store.drain()
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
    key = store.chunk_key(hashes[0], 2, hashes[-1])
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


@pytest.mark.parametrize("write_behind", [False, True])
def test_blobstore_no_slab_persists_one_payload_copy(tmp_path, write_behind):
    client = FakeBlobStoreClient()
    slab_path = tmp_path / "duplicate.slab"
    store = make_store(
        tmp_path,
        client,
        slab_path=str(slab_path),
        blobstore_no_slab=True,
        write_behind=write_behind,
    )
    hashes = make_hashes(2)
    expected_payload = TelysChunkCodec().encode_chunk({LAYER: [b"blob"] * 2})

    try:
        seed_chunk(store, hashes, b"blob")
        store.drain()

        assert client.put_count == 1
        assert client.payload_writes == [expected_payload]
        assert not slab_path.exists()

        key = store.chunk_key(hashes[0], len(hashes), hashes[-1])
        chunk_record = client.get(key)
        assert chunk_record is not None
        assert chunk_record.value == expected_payload
        assert chunk_record.metadata == {
            "vllm.block_start_hash": hashes[0].hex(),
            "vllm.block_count": "2",
            "vllm.block_size": "16",
            "vllm.model": "fake-model-v0",
            "vllm.engine_id": "test-engine",
        }

        records = store.load_blocks(hashes)
        assert [decode_record_value(record) for record in records] == [
            {LAYER: b"blob"},
            {LAYER: b"blob"},
        ]
    finally:
        store.close()


def test_blobstore_slab_persistence_remains_default(tmp_path):
    client = FakeBlobStoreClient()
    slab_path = tmp_path / "default.slab"
    store = make_store(
        tmp_path,
        client,
        slab_path=str(slab_path),
        write_behind=False,
    )
    (block_hash,) = make_hashes(1)

    try:
        seed_chunk(store, [block_hash], b"default")

        assert len(client.payload_writes) == 1
        assert slab_path.is_file()
        chunk_record = client.get(store.chunk_key(block_hash, 1, block_hash))
        assert chunk_record is not None
        assert "vllm.slab_offset" in chunk_record.metadata
        assert "vllm.slab_length" in chunk_record.metadata
    finally:
        store.close()


def test_blobstore_no_slab_preserves_legacy_slab_fallback(tmp_path):
    client = FakeTelysClient()
    slab_path = tmp_path / "fallback.slab"
    store = make_store(
        tmp_path,
        client,
        slab_path=str(slab_path),
        blobstore_no_slab=True,
        write_behind=False,
    )
    hashes = make_hashes(2)

    try:
        seed_chunk(store, hashes, b"legacy")

        assert slab_path.is_file()
        key = store.chunk_key(hashes[0], len(hashes), hashes[-1])
        chunk_record = client.get(key)
        assert chunk_record is not None
        assert chunk_record.value == b""
        assert int(chunk_record.metadata["vllm.slab_offset"]) >= 16
        assert int(chunk_record.metadata["vllm.slab_length"]) > 0

        records = store.load_blocks(hashes)
        assert [decode_record_value(record) for record in records] == [
            {LAYER: b"legacy"},
            {LAYER: b"legacy"},
        ]
    finally:
        store.close()


def test_divergent_second_block_does_not_hit_or_load_original_chunk(tmp_path):
    import hashlib

    client = FakeBlobStoreClient()
    store = make_store(
        tmp_path,
        client,
        slab_path=None,
        blobstore_no_slab=True,
        write_behind=False,
    )
    hashes = make_hashes(2)
    divergent = [hashes[0], hashlib.sha256(hashes[0] + b"divergent").digest()]
    seed_chunk(store, hashes, b"original")

    client.get_count = 0
    assert store.lookup_prefix(divergent) == 0
    assert client.get_count == 0
    with pytest.raises(TelysStoreError, match="no longer present"):
        store.load_decoded_blocks(divergent)
    assert store.lookup_prefix(hashes) == 2
    loaded = store.load_decoded_blocks(hashes)
    assert [block.blobs[LAYER] for block in loaded] == [b"original"] * 2


@pytest.mark.parametrize("blob_only", [False, True], ids=["slab", "blob-only"])
def test_shared_start_and_length_chains_load_exact_a40_payloads(tmp_path, blob_only):
    client = FakeBlobStoreClient() if blob_only else FakeTelysClient()
    store = make_store(
        tmp_path,
        client,
        slab_path=None if blob_only else str(tmp_path / "collision.slab"),
        blobstore_no_slab=blob_only,
        write_behind=False,
    )
    chains = make_shared_start_chains()
    payloads = [f"chain-{index}".encode() for index in range(4)]
    keys = [store.chunk_key(chain[0], len(chain), chain[-1]) for chain in chains]
    assert len({chain[-1] for chain in chains}) == 4
    assert len(set(keys)) == 4

    try:
        for chain, payload in zip(chains, payloads):
            seed_chunk(store, chain, payload)
        store.drain()

        start_key = store.chunk_start_key(chains[0][0])
        selected_key = client.chunk_candidates(start_key)[259]
        assert client.chunk_candidates(start_key) == {259: selected_key}

        contains_calls = []
        original_contains = client.contains

        def recording_contains(key):
            contains_calls.append(key)
            return original_contains(key)

        client.contains = recording_contains
        for chain, payload, key in zip(chains, payloads, keys):
            expected_probes = int(key != selected_key)
            probe_count = len(contains_calls)
            get_count = client.get_count
            assert store.lookup_prefix(chain) == 259
            assert len(contains_calls) - probe_count == expected_probes
            assert client.get_count == get_count

            probe_count = len(contains_calls)
            loaded = store.load_decoded_blocks(chain)
            assert len(contains_calls) - probe_count == expected_probes
            assert len(loaded) == 259
            assert [block.blobs[LAYER] for block in loaded] == [payload] * 259
            assert [block.metadata["vllm.block_hash"] for block in loaded] == [
                block_hash.hex() for block_hash in chain
            ]

        import hashlib

        missing = [
            *chains[1][:-1],
            hashlib.sha256(b"nonexistent A40 chain").digest(),
        ]
        missing_key = store.chunk_key(missing[0], len(missing), missing[-1])
        get_count = client.get_count
        assert store.lookup_prefix(missing) == 0
        with pytest.raises(TelysStoreError, match="no longer present"):
            store.load_decoded_blocks(missing)
        assert client.get_count == get_count
        assert contains_calls.count(missing_key) == 2
    finally:
        store.close()


def test_stale_candidate_and_legacy_client_cannot_redirect_exact_chain(tmp_path):
    client = FakeBlobStoreClient()
    store = make_store(
        tmp_path,
        client,
        slab_path=None,
        blobstore_no_slab=True,
        write_behind=False,
    )
    chains = make_shared_start_chains(block_count=3)
    payloads = [f"chain-{index}".encode() for index in range(4)]

    try:
        for chain, payload in zip(chains, payloads):
            seed_chunk(store, chain, payload)

        stale_key = store.chunk_key(chains[0][0], 3, b"\x00" * 32)
        client.chunk_candidates = lambda _start_key: {3: stale_key}
        for chain, payload in zip(chains, payloads):
            assert store.lookup_prefix(chain) == 3
            loaded = store.load_decoded_blocks(chain)
            assert [block.blobs[LAYER] for block in loaded] == [payload] * 3

        client.chunk_candidates = None
        for chain, payload in zip(chains, payloads):
            assert store.lookup_prefix(chain) == 3
            loaded = store.load_decoded_blocks(chain)
            assert [block.blobs[LAYER] for block in loaded] == [payload] * 3
    finally:
        store.close()


def test_legacy_start_and_count_key_misses_safely(tmp_path):
    client = FakeBlobStoreClient()
    store = make_store(
        tmp_path,
        client,
        slab_path=None,
        blobstore_no_slab=True,
        write_behind=False,
    )
    hashes = make_hashes(2)
    legacy_key = f"{store.chunk_start_key(hashes[0])}/{len(hashes)}"
    payload = TelysChunkCodec().encode_chunk({LAYER: [b"legacy"] * 2})
    client.put(legacy_key, payload, {})

    assert store.lookup_prefix(hashes) == 0
    with pytest.raises(TelysStoreError, match="no longer present"):
        store.load_decoded_blocks(hashes)


@pytest.mark.parametrize("blob_only", [False, True], ids=["slab", "blob-only"])
def test_reset_deletes_reopened_data_before_clearing_slab(tmp_path, blob_only):
    client = FakeBlobStoreClient() if blob_only else FakeTelysClient()
    slab_path = None if blob_only else str(tmp_path / "reopened.slab")
    config = {
        "slab_path": slab_path,
        "blobstore_no_slab": blob_only,
        "write_behind": False,
    }
    first = make_store(tmp_path, client, **config)
    hashes = make_hashes(2)
    key = first.save_chunk(hashes[0], hashes, {LAYER: [b"persisted"] * 2})
    reopened = make_store(tmp_path, client, **config)
    slab_sizes_at_delete: list[int] = []
    original_delete = client.delete

    def delete_before_slab_clear(key_to_delete):
        if slab_path is not None:
            slab_sizes_at_delete.append((tmp_path / "reopened.slab").stat().st_size)
        return original_delete(key_to_delete)

    client.delete = delete_before_slab_clear
    try:
        assert key not in reopened._known_keys
        reopened.reset()

        assert client.get(key) is None
        assert client.deleted == [key]
        assert reopened.lookup_prefix(hashes) == 0
        if blob_only:
            assert slab_sizes_at_delete == []
        else:
            assert slab_sizes_at_delete
            assert all(size > 16 for size in slab_sizes_at_delete)
            reopened.save_chunk(hashes[0], hashes, {LAYER: [b"new"] * 2})
            record = client.get(key)
            assert record is not None
            assert record.metadata["vllm.slab_offset"] == "16"
    finally:
        reopened.close()
        first.close()


def test_divergent_lookup_avoids_payload_reads(tmp_path):
    class CountingClient(FakeTelysClient):
        def __init__(self):
            super().__init__()
            self.contains_calls = 0

        def contains(self, key):
            self.contains_calls += 1
            return super().contains(key)

    client = CountingClient()
    store = make_store(tmp_path, client)
    hashes = make_hashes(2)
    seed_chunk(store, hashes)
    store.drain()
    divergent = [hashes[0], make_hashes(1)[0]]

    assert store.lookup_prefix(divergent) == 0
    assert client.contains_calls == 1
    assert client.get_count == 0


def test_stale_max_chunk_length_uses_exact_shorter_candidate(tmp_path):
    class CountingClient(FakeTelysClient):
        def __init__(self):
            super().__init__()
            self.contains_calls = 0

        def contains(self, key):
            self.contains_calls += 1
            return super().contains(key)

    client = CountingClient()
    store = make_store(tmp_path, client)
    hashes = make_hashes(2)
    seed_chunk(store, hashes)
    store.drain()
    longer_hashes = make_hashes(4)
    seed_chunk(store, longer_hashes)
    store.drain()
    client.chunk_length = lambda _start_key: len(longer_hashes)

    assert store.lookup_prefix(make_hashes(3)) == 2
    assert client.contains_calls == 2
    assert client.get_count == 0


def test_fast_load_rejects_empty_decoded_layers(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    hashes = make_hashes(1)
    seed_chunk(store, hashes)
    monkeypatch.setattr(
        store,
        "_decode_chunk_payload",
        lambda *_args, **_kwargs: (True, {}),
    )

    with pytest.raises(TelysStoreError, match="has no layers"):
        store.load_decoded_blocks_fast(hashes)


def _race_hashes() -> tuple[list[bytes], list[bytes]]:
    return [b"\x30" * 32, b"\x31" * 32], [b"\x40" * 32, b"\x41" * 32]


def test_reset_then_concurrent_save_survives_complete_after_cold_reopen(
    tmp_path, monkeypatch
):
    client = FakeTelysClient()
    store = make_store(tmp_path, client, write_behind=False)
    old_hashes, new_hashes = _race_hashes()
    seed_chunk(store, old_hashes, b"old")
    old_key = store.chunk_key(old_hashes[0], len(old_hashes), old_hashes[-1])
    new_key = store.chunk_key(new_hashes[0], len(new_hashes), new_hashes[-1])
    snapshot_ready = threading.Event()
    release_reset = threading.Event()
    save_done = threading.Event()
    errors: list[Exception] = []

    def paused_keys():
        snapshot = client.__class__.keys(client)
        snapshot_ready.set()
        if not release_reset.wait(timeout=5):
            raise TimeoutError("timed out waiting to release reset")
        return snapshot

    def run_reset():
        try:
            store.reset()
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)

    def run_save():
        try:
            store.save_chunk(
                new_hashes[0], new_hashes, {LAYER: [b"new"] * len(new_hashes)}
            )
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)
        finally:
            save_done.set()

    monkeypatch.setattr(client, "keys", paused_keys)
    reset_thread = threading.Thread(target=run_reset)
    save_thread = threading.Thread(target=run_save)
    try:
        reset_thread.start()
        assert snapshot_ready.wait(timeout=5)
        save_thread.start()
        assert not save_done.wait(timeout=0.05)
        release_reset.set()
        reset_thread.join(timeout=5)
        save_thread.join(timeout=5)
        assert not reset_thread.is_alive()
        assert not save_thread.is_alive()
        assert not errors
        assert client.get(old_key) is None
        assert client.get(new_key) is not None
        store.close()

        cold = make_store(tmp_path, client, write_behind=False)
        try:
            assert cold.lookup_prefix(old_hashes) == 0
            loaded = cold.load_blocks(new_hashes)
            assert [decode_record_value(record) for record in loaded] == [
                {LAYER: b"new"},
                {LAYER: b"new"},
            ]
        finally:
            cold.close()
    finally:
        release_reset.set()
        reset_thread.join(timeout=5)
        save_thread.join(timeout=5)
        if not store._closed:
            store.close()


def test_save_during_reset_compaction_survives_cold_reopen(tmp_path, monkeypatch):
    client = FakeTelysClient()
    store = make_store(tmp_path, client, write_behind=False)
    old_hashes, new_hashes = _race_hashes()
    seed_chunk(store, old_hashes, b"old")
    old_key = store.chunk_key(old_hashes[0], len(old_hashes), old_hashes[-1])
    new_key = store.chunk_key(new_hashes[0], len(new_hashes), new_hashes[-1])
    compaction_started = threading.Event()
    release_compaction = threading.Event()
    save_done = threading.Event()
    errors: list[Exception] = []
    original_compact = client.compact

    def paused_compact():
        compaction_started.set()
        if not release_compaction.wait(timeout=5):
            raise TimeoutError("timed out waiting to release compaction")
        return original_compact()

    def run_reset():
        try:
            store.reset()
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)

    def run_save():
        try:
            store.save_chunk(
                new_hashes[0], new_hashes, {LAYER: [b"new"] * len(new_hashes)}
            )
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)
        finally:
            save_done.set()

    monkeypatch.setattr(client, "compact", paused_compact)
    reset_thread = threading.Thread(target=run_reset)
    save_thread = threading.Thread(target=run_save)
    try:
        reset_thread.start()
        assert compaction_started.wait(timeout=5)
        save_thread.start()
        assert save_done.wait(timeout=2), "save blocked behind reset compaction"

        record = client.get(new_key)
        assert record is not None
        assert record.metadata["vllm.slab_offset"] == "16"
        offset = int(record.metadata["vllm.slab_offset"])
        length = int(record.metadata["vllm.slab_length"])
        assert store._slab.read(offset, length) == TelysChunkCodec().encode_chunk(
            {LAYER: [b"new"] * len(new_hashes)}
        )

        release_compaction.set()
        reset_thread.join(timeout=5)
        save_thread.join(timeout=5)
        assert not reset_thread.is_alive()
        assert not save_thread.is_alive()
        assert not errors
        assert client.get(old_key) is None
        store.close()

        cold = make_store(tmp_path, client, write_behind=False)
        try:
            assert cold.lookup_prefix(old_hashes) == 0
            loaded = cold.load_blocks(new_hashes)
            assert [decode_record_value(record) for record in loaded] == [
                {LAYER: b"new"},
                {LAYER: b"new"},
            ]
        finally:
            cold.close()
    finally:
        release_compaction.set()
        reset_thread.join(timeout=5)
        save_thread.join(timeout=5)
        if not store._closed:
            store.close()


def test_save_then_concurrent_reset_deletes_complete_record_after_cold_reopen(
    tmp_path, monkeypatch
):
    client = FakeTelysClient()
    store = make_store(tmp_path, client, write_behind=False)
    old_hashes, new_hashes = _race_hashes()
    seed_chunk(store, old_hashes, b"old")
    old_key = store.chunk_key(old_hashes[0], len(old_hashes), old_hashes[-1])
    new_key = store.chunk_key(new_hashes[0], len(new_hashes), new_hashes[-1])
    put_started = threading.Event()
    release_put = threading.Event()
    reset_done = threading.Event()
    errors: list[Exception] = []
    original_put = client.put

    def paused_put(key, value, metadata):
        if key == new_key:
            put_started.set()
            if not release_put.wait(timeout=5):
                raise TimeoutError("timed out waiting to release put")
        original_put(key, value, metadata)

    def run_save():
        try:
            store.save_chunk(
                new_hashes[0], new_hashes, {LAYER: [b"new"] * len(new_hashes)}
            )
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)

    def run_reset():
        try:
            store.reset()
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)
        finally:
            reset_done.set()

    monkeypatch.setattr(client, "put", paused_put)
    save_thread = threading.Thread(target=run_save)
    reset_thread = threading.Thread(target=run_reset)
    try:
        save_thread.start()
        assert put_started.wait(timeout=5)
        reset_thread.start()
        assert not reset_done.wait(timeout=0.05)
        release_put.set()
        save_thread.join(timeout=5)
        reset_thread.join(timeout=5)
        assert not save_thread.is_alive()
        assert not reset_thread.is_alive()
        assert not errors
        assert client.get(old_key) is None
        assert client.get(new_key) is None
        store.close()

        cold = make_store(tmp_path, client, write_behind=False)
        try:
            assert cold.lookup_prefix(old_hashes) == 0
            assert cold.lookup_prefix(new_hashes) == 0
            assert (tmp_path / "kv.slab").stat().st_size == 16
        finally:
            cold.close()
    finally:
        release_put.set()
        save_thread.join(timeout=5)
        reset_thread.join(timeout=5)
        if not store._closed:
            store.close()


def test_reset_preserves_new_write_behind_pending_accounting(tmp_path, monkeypatch):
    client = FakeTelysClient()
    store = make_store(tmp_path, client, write_behind=True)
    old_hashes, new_hashes = _race_hashes()
    seed_chunk(store, old_hashes, b"old")
    store.drain()
    new_key = store.chunk_key(new_hashes[0], len(new_hashes), new_hashes[-1])
    snapshot_ready = threading.Event()
    release_reset = threading.Event()
    put_started = threading.Event()
    release_put = threading.Event()
    errors: list[Exception] = []
    original_keys = client.keys
    original_put = client.put

    def paused_keys():
        snapshot = original_keys()
        snapshot_ready.set()
        if not release_reset.wait(timeout=5):
            raise TimeoutError("timed out waiting to release reset")
        return snapshot

    def paused_put(key, value, metadata):
        if key == new_key:
            put_started.set()
            if not release_put.wait(timeout=5):
                raise TimeoutError("timed out waiting to release put")
        original_put(key, value, metadata)

    def run_reset():
        try:
            store.reset()
        except Exception as exc:  # noqa: BLE001 - collect thread failures for diagnostics
            errors.append(exc)

    monkeypatch.setattr(client, "keys", paused_keys)
    monkeypatch.setattr(client, "put", paused_put)
    reset_thread = threading.Thread(target=run_reset)
    try:
        reset_thread.start()
        assert snapshot_ready.wait(timeout=5)
        store.save_chunk(
            new_hashes[0], new_hashes, {LAYER: [b"new"] * len(new_hashes)}
        )
        assert store._pending_count == 1
        release_reset.set()
        reset_thread.join(timeout=5)
        assert not reset_thread.is_alive()
        assert put_started.wait(timeout=5)
        assert store._pending_count == 1
        assert new_key in store._pending_keys
        release_put.set()
        store.drain()
        assert store._pending_count == 0
        assert not errors
        store.close()

        cold = make_store(tmp_path, client, write_behind=False)
        try:
            loaded = cold.load_blocks(new_hashes)
            assert [decode_record_value(record) for record in loaded] == [
                {LAYER: b"new"},
                {LAYER: b"new"},
            ]
            assert cold._pending_count == 0
        finally:
            cold.close()
    finally:
        release_reset.set()
        release_put.set()
        reset_thread.join(timeout=5)
        if not store._closed:
            store.close()
