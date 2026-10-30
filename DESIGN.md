# TelysKVConnector — Telys as an external KV/prefix-cache store for vLLM

Design doc for the `telys-vllm` connector. Scope of this revision: design +
CPU-testable skeleton. No GPU code paths are implemented yet (see
[Bench-week open questions](#bench-week-open-questions)).

API facts below were verified against a shallow clone of `vllm-project/vllm`,
commit `b689feb9f22129cb9ba65ea72e3e67e4a9c95355` (2026-10-22). All line
references are to that commit.

---

## 1. The sanctioned API surface (verified)

### 1.1 Where the API actually lives

The brief pointed at `vllm/distributed/kv_transfer/kv_connector/base.py`. That
file is now a 10-line compatibility alias:

- `kv_connector/base.py:5-8` — `KVConnectorBase = KVConnectorBase_V1`,
  re-exported from `vllm/distributed/kv_transfer/kv_connector/v1/base.py`.

The real contract is `KVConnectorBase_V1` in
`vllm/distributed/kv_transfer/kv_connector/v1/base.py` (805 lines). The old v0
`lookup/send/recv/close` connector interface no longer exists anywhere in the
tree; every in-tree connector (lmcache, nixl, mooncake, flexkv, offloading,
example) implements the V1 surface.

### 1.2 Abstract methods a connector MUST implement

`KVConnectorBase_V1` (`v1/base.py:181`), constructor at
`v1/base.py:214-232` — `(vllm_config, role, kv_cache_config)`, raises if
`vllm_config.kv_transfer_config is None` (`v1/base.py:226-229`).

Scheduler-side (runs in the scheduler process):

| Method | Line | Contract |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | `v1/base.py:515-548` | Returns `(int \| None, bool)`. `None` = "ask again later" (scheduler defers the request). Second element = async-load flag, **must be `False` when the first is 0**. Should be side-effect free; only count the largest prefix actually loadable at call time. |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | `v1/base.py:568-592` | Called after block allocation; may be called twice for async loads. Decide to load based on `num_external_tokens`, not on `blocks` being empty. |
| `build_connector_meta(scheduler_output)` | `v1/base.py:594-611` | Builds `KVConnectorMetadata` for the step. Must not mutate `scheduler_output`. Calling it resets per-step connector state. |

Worker-side (runs in each worker process):

| Method | Line | Contract |
|---|---|---|
| `start_load_kv(forward_context, **kwargs)` | `v1/base.py:332-347` | Start loading KV into the paged buffer before the forward pass. |
| `wait_for_layer_load(layer_name)` | `v1/base.py:349-361` | Block until one layer's load is done (layer pipelining). |
| `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)` | `v1/base.py:363-383` | Start saving one layer's KV from the paged buffer. `**kwargs` is a sanctioned extension point. |
| `wait_for_save()` | `v1/base.py:385-396` | Submit/sync saves, once per step. Sync connectors must finish saving before returning. |

Concrete methods we override or rely on:

| Method | Line | Use |
|---|---|---|
| `KVConnectorRole` (`SCHEDULER=0`, `WORKER=1`) | `v1/base.py:139-144` | Two separate instances are constructed per engine. |
| `KVConnectorMetadata` | `v1/base.py:155-160` | Base for scheduler→worker metadata. Must be serializable (crosses process boundary). |
| `bind_connector_metadata` / `clear_connector_metadata` / `_get_connector_metadata` | `v1/base.py:249-290` | Model runner binds metadata each step. |
| `register_kv_caches(kv_caches)` | `v1/base.py:292-300` | Worker hook to pre-register GPU KV tensors. |
| `on_new_request(request)` | `v1/base.py:613-619` | Scheduler bookkeeping hook (called from `scheduler.py:2740`). |
| `request_finished(request, block_ids)` | `v1/base.py:631-650` | Return `(delay_free, kv_transfer_params)`. |
| `get_finished(finished_req_ids)` | `v1/base.py:398-414` | Async send/recv completion tracking. Default `(None, None)` suffices for sync. |
| `get_loaded_kv_cache_group_ids` | `v1/base.py:505-513` | Default returns `prefix_cacheable_group_ids` — correct for a hash-addressed store. |
| `get_block_ids_with_load_errors` | `v1/base.py:426-444` | Load-failure reporting path. |
| `reset_cache()` | `v1/base.py:793-805` | Invoked by `llm.reset_prefix_cache()`. |
| `shutdown()` | `v1/base.py:446-451` | Worker-process cleanup. |
| `requires_kv_delivery` | `v1/base.py:198-207` | Defaults to `is_kv_producer`; best-effort caches override to `False` (pattern: `offloading_connector.py:73-77`). |

### 1.3 The HMA gate (important)

`SupportsHMA` (`v1/base.py:101-129`) with `supports_hma()` check
(`v1/base.py:132-136`). `KVConnectorFactory.create_connector` **raises
ValueError** when the hybrid KV cache manager is enabled (the default) and the
connector class does not implement `SupportsHMA`
(`factory.py:54-60`). `TelysKVConnector` therefore subclasses `SupportsHMA`
and implements `request_finished_all_groups` (pattern:
`offloading_connector.py:223-229`), so users are not forced to pass
`--disable-hybrid-kv-cache-manager`.

### 1.4 Factory discovery mechanism (exact)

`KVConnectorFactory` (`factory.py:27-146`):

1. **In-tree registry**: `register_connector(name, module_path, class_name)`
   (`factory.py:30-40`) stores a lazy loader in `_registry`
   (`factory.py:28`); built-ins registered at `factory.py:149-249`.
2. **External module path (what we use)**: if
   `KVTransferConfig.kv_connector_module_path` is set, it **takes priority
   over the registry** (`factory.py:103-124`): the factory does
   `importlib.import_module(module_path)` + `getattr(module, kv_connector)`.
   It then enforces that the constructor accepts a `kv_cache_config` kwarg
   (`supports_kw` check, `factory.py:116-124`) — 2-argument constructors are
   rejected with a ValueError.
3. `create_connector(config, role, kv_cache_config)` (`factory.py:42-75`)
   instantiates one connector per role (scheduler process / worker process,
   comment at `factory.py:67-74`).

`KVTransferConfig` fields (`vllm/config/kv_transfer.py`): `kv_connector:60`,
`engine_id:64` (auto-uuid at `:111-112`), `kv_role:71` (**required** when
`kv_connector` is set, `:120-122`), `kv_connector_extra_config:78`,
`kv_connector_module_path:81`, `get_from_extra_config:138-139`.

There is **no entry-point discovery** in the factory and **no
`VLLM_KV_CONNECTOR`-style env var** in `vllm/envs.py`. Enabling a connector is
a config-line operation: `--kv-transfer-config '{"kv_connector": ...,
"kv_role": ...}'` on the CLI, or `LLM(..., kv_transfer_config={...})`
(pattern: `examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py:119-122`).

### 1.5 Block-hash path (prefix-cache hit identity)

- `BlockHash = NewType("BlockHash", bytes)` (`vllm/v1/core/kv_cache_utils.py:44-47`).
- Hashes are chained per block: `hash_block_tokens(parent_hash, token_ids, extra_keys)`
  (`kv_cache_utils.py:707+`); sha256 by default with a deterministic default
  seed (`kv_cache_utils.py:90-108`), so identical prefixes hash identically
  across processes/machines — exactly what an external store needs.
- `BlockHashWithGroupId` appends a 4-byte big-endian group id
  (`kv_cache_utils.py:60-79`).
- Requests carry `request.block_hashes: list[BlockHash]`
  (`vllm/v1/request.py:225`), maintained by `update_block_hashes()`
  (`request.py:285-288`). One hash per **full** block; the trailing partial
  block has no hash.
- `request.skip_reading_prefix_cache` (`request.py:314-324`) must be honored
  (scheduler also force-zeroes external hits, `scheduler.py:1040-1041`).

### 1.6 What vLLM calls, in sequence

**Hit path (synchronous load, `kv_role="kv_both"`):**

1. Request admitted → `connector.on_new_request(request)` (`scheduler.py:2740`).
2. Scheduling step, `request.num_computed_tokens == 0`: local prefix lookup
   first (`scheduler.py:1016-1024`), then the connector:
   `ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(request, block_aligned_local)`
   (`scheduler.py:1035-1039`). The local hit is block-aligned before the call
   (`scheduler.py:1031-1034`). `None` defers the request (`scheduler.py:1043-1048`).
3. `kv_cache_manager.allocate_slots(..., num_external_computed_tokens=ext_tokens)`
   (`scheduler.py:1320-1337`).
4. `connector.update_state_after_alloc(request, blocks, ext_tokens)`
   (`scheduler.py:1352-1357`).
5. End of step: `connector.build_connector_meta(scheduler_output)`
   (`scheduler.py:1648`) → metadata rides `SchedulerOutput` to the worker.
6. Worker model runner: `bind_connector_metadata(meta)`, then
   `start_load_kv(forward_context)` before the forward pass
   (`v1/base.py:249`, `332`), per attention layer
   `wait_for_layer_load(layer_name)` (`v1/base.py:349`).
7. Forward pass; per layer `save_kv_layer(layer_name, kv_layer, attn_metadata)`
   (`v1/base.py:363`).
8. After the forward: `wait_for_save()` (`v1/base.py:385`), then
   `clear_connector_metadata()` (`v1/base.py:262`).
9. `get_finished(finished_req_ids)` polled via the executor
   (`v1/base.py:398`); scheduler side `update_connector_output`
   (`v1/base.py:621`).
10. Request completes: `request_finished(request, block_ids)`
    (`scheduler.py:3151` → `v1/base.py:631`). `(False, None)` = free blocks
    immediately.

**Miss path:** identical, except step 2 returns `(0, False)`, step 4 is called
with `num_external_tokens=0`, and step 6 has no load specs in the metadata.
The save path (steps 7-8) still runs — a miss today is a hit tomorrow.

**Async-load variant:** step 2 returns `(N, True)` → the request enters
`WAITING_FOR_REMOTE_KVS` (`scheduler.py:1393`), the worker loads between
steps, `get_finished` reports `finished_recving`, and the scheduler resumes
the request. `update_state_after_alloc` is then called twice
(`v1/base.py:574-578`).

---

## 2. Surprises vs the brief

1. **`kv_connector/base.py` is a shim.** The full abstract surface lives in
   `v1/base.py` (`KVConnectorBase_V1`); the v0-style `lookup/load/save/register/clear`
   method names in the brief do not exist. The V1 surface is what lmcache,
   nixl, mooncake, flexkv, and the native offloading connector all implement.
2. **No entry-point/plugin discovery for connectors.** Registration is either
   in-tree (`register_connector`) or per-deployment via
   `kv_connector_module_path`, which takes priority over the registry
   (`factory.py:103-124`). We use the latter — zero vLLM patches needed.
3. **A constructor-shape check is enforced**: external connectors must accept
   `kv_cache_config` as the third constructor argument
   (`factory.py:116-124`).
4. **The HMA gate is on by default** (`factory.py:54-60`): a connector that
   doesn't implement `SupportsHMA` fails fast at startup. We implement it.
5. **`get_num_new_matched_tokens` has a tri-state return**: `None` means
   "can't determine now, ask again later" and the scheduler defers the request
   (`scheduler.py:1043-1048`). Also, the async flag must be `False` when the
   count is 0 (`v1/base.py:536-538`).
6. **Block hashes are deterministic across processes by default** (sha256
   with a fixed seed, `kv_cache_utils.py:90-108`) — an external store can key
   on them directly without a vLLM-side registry.
7. **No `VLLM_KV_CONNECTOR` env var exists.** "One env var / config line" is
   achievable only as one *config line* (`--kv-transfer-config '{...}'`);
   env vars can only carry our own tunables (FlexKV does the same with
   `FLEXKV_CONFIG_PATH`, example line 83).

---

## 3. Architecture

```
vLLM scheduler process                     vLLM worker process(es)
----------------------                     -----------------------
TelysKVConnector(role=SCHEDULER)           TelysKVConnector(role=WORKER)
  └─ TelysSchedulerSide (in connector)       └─ TelysWorkerSide (in connector)
       └─ TelysBlockStore ─┐                      └─ TelysBlockStore ─┐
                           ▼                                           ▼
                     TelysClient (Protocol)                      TelysClient (Protocol)
                           └──────────►  Telys (on-device store)  ◄──────────┘
```

- **`telys_vllm/client.py`** — the `TelysClient` Protocol (the only Telys
  dependency), plus `TelysRecord` / `TelysProvenance` /
  `TelysMaintenanceReport` value types.
- **`telys_vllm/store.py`** — `TelysBlockStore`: pure-Python core with
  chunk-aware key mapping, contiguous-prefix lookup, save/load with provenance,
  and the compaction hook. No vLLM imports; fully CPU-testable.
- **`telys_vllm/codec.py`** — `LengthPrefixedCodec` for per-block framing and
  `TelysChunkCodec` for contiguous-run framing. GPU tensor→bytes extraction is
  in `gpu_ops.py`.
- **`telys_vllm/config.py`** — `TelysConnectorConfig` from env vars and/or
  `kv_connector_extra_config`, validated, fail-fast.
- **`telys_vllm/connector.py`** — `TelysKVConnector(KVConnectorBase_V1,
  SupportsHMA)`: thin adapter. Worker-side serialization is deferred from
  `save_kv_layer` to `wait_for_save`, and saves can use KV caches registered
  via `register_kv_caches` so they survive `torch.compile`.

### 3.1 Base method → TelysClient mapping

| `KVConnectorBase_V1` method | TelysBlockStore / TelysClient call |
|---|---|
| `get_num_new_matched_tokens` (sched) | `store.lookup_prefix(hashes)` → `client.chunk_length(start_key)` per chunk start, advancing by chunk size |
| `update_state_after_alloc` (sched) | no client call; records the load plan + save cursor |
| `build_connector_meta` (sched) | no client call; emits `TelysConnectorMetadata(loads, saves)`, resets per-step state |
| `on_new_request` (sched) | no client call; keeps the `Request` ref for hash/cursor tracking |
| `start_load_kv` (worker) | `store.load_blocks(hashes)` → fetch covering chunk(s), decode, return per-block records |
| `wait_for_layer_load` (worker) | no-op (sync fetch in `start_load_kv`; layer pipelining is bench-week) |
| `register_kv_caches` (worker) | capture per-layer KV cache tensors for compile-safe saving |
| `save_kv_layer` (worker) | buffers CPU-test bytes (`block_payloads`) or no-ops when KV caches are registered |
| `wait_for_save` (worker) | deduplicate chunk keys within the step (`store.contains` / seen-key set), group unique blocks into contiguous runs, `TelysChunkCodec.encode_chunk(...)` → `store.save_chunk(...)` → `client.put(key, payload, metadata)` once per run |
| `request_finished` / `request_finished_all_groups` (sched) | drops per-request bookkeeping; returns `(False, None)` (best-effort cache, blocks freed immediately) |
| `reset_cache()` (sched) | `store.reset()` → `client.delete(key)` for known chunk keys + `client.compact()`; returns `True` |
| `shutdown()` (worker) | `client.close()` |
| `get_finished` / `take_events` / `get_kv_connector_stats` | inherited defaults (no async transfers, no custom events yet) |

### 3.2 Key mapping and provenance

- Telys chunk key: `f"{namespace}/chunk/{start_block_hash.hex()}/{block_count}"`.
  A chunk holds a contiguous run of blocks; single-block chunks are the
  degenerate case. The key is deterministic, opaque, and namespace-isolated.
- On `put`, metadata carries the vLLM-side provenance:
  `vllm.block_start_hash`, `vllm.block_count`, `vllm.block_size`,
  `vllm.model`, `vllm.engine_id`.
- On `get`, Telys read provenance (`plan_hash`, `result_id`) rides the
  `TelysRecord` and is (a) verified non-empty by the load path, (b) logged on
  the hit path, (c) the anchor for later correctness auditing: any loaded
  block is traceable to the exact Telys read that produced it.
- The chunk payload is all layers' KV bytes for the whole run, framed by
  `TelysChunkCodec` (`[block_count][block_0_framed]...`). Each block inside the
  chunk is framed by `LengthPrefixedCodec`. Inside the connector the payload is
  raw bytes.
- Telys metadata is JSON-serialized over the wire, so `TelysLocalClient`
  base64-encodes the chunk payload once per chunk in the `vllm.payload`
  metadata column (still ~27× cheaper than the v2 per-block base64-JSON
  wrapper). On `get` the base64 string is decoded back to raw bytes before the
  chunk codec sees it. Legacy v2 collections that used a `vllm.meta_blob`
  base64-JSON column remain readable.

### 3.3 Hit/miss semantics (exact)

`get_num_new_matched_tokens(request, num_computed_tokens)`:

```
start_block = num_computed_tokens // block_size        # scheduler pre-aligns (scheduler.py:1031-1034)
max_block   = (request.num_tokens - 1) // block_size   # never offer the trailing partial block
hits        = store.lookup_prefix(request.block_hashes[start_block:max_block])
return hits * block_size, (hits > 0 and config.load_async)
```

- `store.lookup_prefix` walks the prefix in chunk-sized strides using
  `client.chunk_length(start_key)`. The scan stops at the first block that is
  not the start of a stored chunk, at the end of the queried range, or at
  `config.max_lookup_blocks` when non-zero.
- A stored chunk may be longer than the remaining queried blocks; only the
  queried portion is counted, so the returned hit never exceeds the requested
  range.
- Contiguous-prefix rule: a hole anywhere in the prefix breaks the hit, same
  as the reference connector (`offloading/scheduler.py:749-779`).
- Miss → `(0, False)` exactly (contract: `v1/base.py:536-538`).
- The method touches no Telys mutable state (contract: `v1/base.py:9-11`);
  the only bookkeeping is an idempotent note of the lookup boundary.
- We never return `None`: Telys chunk-start lookups are synchronous. The
  deferred-lookup path stays available if a future async Telys binding needs it.

### 3.4 Save planning and chunking

- Per request, a save cursor (`_save_cursors[request_id]`) tracks the next
  unsaved full block. Each step, `build_connector_meta` advances it to
  `min((num_computed + num_scheduled) // block_size, (num_tokens - 1) // block_size)`
  and emits the delta as a `TelysSaveSpec`.
- `register_kv_caches` captures the per-layer paged KV cache tensors once at
  model load, outside the compiled forward region. This lets saves survive
  `torch.compile`, which can skip the per-layer `save_kv_layer` decorator.
- `save_kv_layer` does **no serialization** in the GPU path. It either buffers
  CPU-test bytes (`block_payloads` kwarg) or no-ops when KV caches were
  registered. Real tensor extraction and framing happen once per step in
  `wait_for_save`.
- `wait_for_save` deduplicates chunk keys within the step (a shared prefix may
  appear in multiple save specs), then groups the unique blocks into contiguous
  runs and frames each run as one raw-bytes chunk (`TelysChunkCodec`). One
  Telys `put` is issued per run; a run can be capped with
  `TELYS_VLLM_MAX_CHUNK_BLOCKS`. Extraction uses the registered KV caches when
  available and tolerates missing per-layer attention metadata (used only for
  MLA layout detection; non-MLA models work without it).
- A block served by a Telys hit is not re-saved: the cursor starts past the
  loaded prefix (`update_state_after_alloc`). Telys `put` is idempotent by
  chunk key, so duplicate saves are harmless.

### 3.5 Eviction policy

Telys owns eviction (compaction/IVF maintenance are Telys ops); the connector
only decides **when to ask**:

- **Normal save path**: no compaction. Persisting a block is a single
  `client.put()`; `wait_for_save` never pauses for maintenance mid-run.
- **Reset**: `reset_cache()` (wired to `llm.reset_prefix_cache()`,
  `v1/base.py:793-805`) deletes every key this connector wrote (tracked in
  `store._known_keys`) and compacts once.
- **Process shutdown**: `TelysLocalClient.close()` compacts once when the last
  connector in the process closes, so on-disk state is cleaned up without
  adding latency to any engine step.
- The connector never silently drops blocks Telys still serves. Memory bound of
  `_known_keys` is one chunk-key string per stored chunk — acceptable for the
  skeleton; a namespace-scan delete is the production upgrade
  (bench-week question 5).

### 3.6 Config surface

Enablement (one config line — "users don't switch to us, they add us"):

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "TelysKVConnector",
  "kv_connector_module_path": "telys_vllm.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "telys_client_factory": "telys_vllm.local_client:TelysLocalClient"
  }
}'
```

The connector itself still has zero mandatory runtime dependencies; the real
Telys SDK is only imported when the factory points at
`TelysLocalClient` (see [Real client binding](#38-real-client-binding)).

Connector tunables (env var → `kv_connector_extra_config` key → default;
extra_config wins over env, env wins over default):

| Env var | extra_config key | Default | Meaning |
|---|---|---|---|
| `TELYS_VLLM_NAMESPACE` | `telys_namespace` | `telys-vllm` | Telys key namespace |
| `TELYS_VLLM_LOAD_ASYNC` | `telys_load_async` | `false` | Return async-load flag on hits |
| `TELYS_VLLM_MAX_LOOKUP_BLOCKS` | `telys_max_lookup_blocks` | `0` (unlimited) | Cap on blocks scanned per lookup |
| `TELYS_VLLM_MAX_CHUNK_BLOCKS` | `telys_max_chunk_blocks` | `0` (unlimited) | Max blocks per stored chunk |
| — | `telys_client_factory` | — (required) | Dotted `module:attr` path to a `Callable[[TelysConnectorConfig], TelysClient]` |

The client factory is resolved with the same lazy-import idiom vLLM's own
factory uses (`factory.py:36-38`, `106-110`). No factory → fail fast at
connector construction (`TelysConfigError`). This keeps the package free of
global mutable state and keeps tests hermetic (`tests.fakes:make_fake_client`).

### 3.7 Error handling

- Missing key on the load path (evicted between lookup and load):
  `TelysStoreError` — never a silent partial load (contract:
  `v1/base.py:540-546`; the GPU revision will route this through
  `get_block_ids_with_load_errors`, `v1/base.py:426-444`).
- Invalid config: `TelysConfigError` at construction, before any vLLM state
  is touched.
- Non-bytes payload on the CPU save path: `TelysConnectorError` (the GPU
  extraction point is explicit, not a silent skip).
- All core logic is deterministic: no wall-clock reads, no randomness; the
  only nondeterminism is Telys store content itself, isolated behind
  `TelysClient`.

---

## 3.8 Real client binding

`telys_vllm/local_client.py` implements `TelysLocalClient`, a concrete
`TelysClient` that talks to the real Telys SDK.  Telys is a vector-memory
engine, not a native KV store, so the adapter stores each contiguous run of
blocks as one collection row keyed by the chunk key.

### Process-global shared engine + hot cache

vLLM constructs separate scheduler-side and worker-side connector instances.
If each instance opened its own `Telys(path)` engine, the scheduler's
in-memory collection view would not see rows written through the worker's
collection, so every external prefix-cache lookup would report a miss.  To fix
this, `TelysLocalClient` keeps a process-global map from
`(path, address, collection_name)` to a single shared `(engine, collection)`
pair with reference counting.

On top of the shared collection the client maintains a per-instance,
write-through hot cache:

- `_key_set` — seeded from `collection.ids()` at construction and updated on
  every `put`/`delete`.
- `_record_cache` — caches decoded chunk records after the first fetch, so
  repeated loads of the same chunk avoid Telys searches.
- `_chunk_lengths` — map from chunk start key (`namespace/chunk/<hash>`) to
  the longest stored block count.  This makes `chunk_length(start_key)` an
  in-process dict lookup, so the scheduler's prefix scan is O(number of
  chunks) instead of O(number of blocks).

A chunk written by a **different** process (or before this client was created)
is not in the hot cache; `chunk_length()` returns 0 and the connector treats
that prefix position as a miss until the client is re-seeded. In practice this
is fine for warm-restart hits: the scheduler seeds from disk at startup.

### Protocol method → real Telys call mapping

| `TelysClient` method | Real Telys SDK call(s) |
|---|---|
| `contains(key)` | `_key_set` membership when cached; otherwise one `collection.search(..., where={"__key": key}, with_metadata=True)`. |
| `chunk_length(start_key)` | In-process `_chunk_lengths` lookup; returns 0 on cache miss. |
| `get(key)` | `collection.search(zero_vector, top_k=1, where={"__key": key}, with_metadata=True)`; read the `vllm.payload` column. If the column holds a base64 string it is decoded to raw bytes; if it holds `bytes` it is used directly. Returns `None` when the search yields no ids. |
| `get_batch(keys)` | `get(key)` for each key, in order, using the hot cache when possible. |
| `put(key, value, metadata)` | Base64-encodes `value`, then `collection.upsert(vectors=[[0.0]*dim], ids=[key], metadata=[{"__key": key, "vllm.payload": payload_b64}])`. Idempotent by external id; the record and chunk length are inserted into the hot cache. |
| `delete(key)` | `collection.ids(where={"__key": key})`; if present, `collection.delete([key])` and return `True`, else `False`. |
| `compact()` | `collection.compact()`; parsed into `TelysMaintenanceReport` when the runtime returns a dict, otherwise inferred from `stats()` before/after. |
| `close()` | Reference-counted release of the shared engine. When the last client in the process closes, the collection is compacted once and a remote engine (`TELYS_ADDRESS`) is closed. |

### Divergences from the Protocol and how they are resolved

1. **Telys has no raw get/put/delete/compact by key.** The adapter treats one
   collection as a KV table: the chunk-key string is the external row id, a
   fixed zero vector satisfies the required vector dimension, and the chunk
   bytes are stored in the `vllm.payload` metadata column. Because Telys
   metadata is JSON over the wire, `put` base64-encodes the bytes and `get`
   decodes them; the connector still sees raw bytes.
2. **Telys reads do not expose `plan_hash` / `result_id`.** The SDK's search
   result carries scores and stored metadata but no retrieval provenance pair.
   The adapter derives a deterministic provenance from the key
   (`sha256(key).hexdigest()`) so the connector's audit/observability contract
   (`telys hit: ... plan_hash=... result_id=...`) remains intact.
3. **`with_metadata=True` only returns declared filter columns.** The SDK
   stores arbitrary metadata but `collection.search(..., with_metadata=True)`
   only returns columns named in `filter_columns`.  The payload is stored in
   the single declared `vllm.payload` column; connector metadata lives in the
   chunk payload header so no fixed schema is needed.
4. **`delete` return value.** Telys' own `delete` silently ignores missing ids,
   so the adapter first probes with `ids(where=...)` to return an accurate
   boolean.
5. **`compact` report shape.** The runtime's `compact()` return is not
   documented as a fixed schema, so the adapter parses a dict if it looks like
   a maintenance report and falls back to `stats()`-derived counts otherwise.
6. **Legacy v2 collections** used a base64-JSON `vllm.meta_blob` column. The
   adapter still decodes that format if `vllm.payload` is absent, so existing
   test/integration data is readable; all new writes use raw bytes.

### Configuration and auth model

Connection is configured through environment variables:

| Env var | Meaning | Required |
|---|---|---|
| `TELYS_PATH` | Local on-disk engine directory (`telys.Telys(path)`). | Either `TELYS_PATH` or `TELYS_ADDRESS` |
| `TELYS_ADDRESS` | Self-host server address (`telys.client.connect`). | Either |
| `TELYS_TOKEN` | Access token for a TCP self-host server. | Only with `TELYS_ADDRESS` when the server requires it |
| `TELYS_KV_DIM` | Zero-vector dimension for the backing collection (default `1`). | No |
| `TELYS_COLLECTION` | Override the collection name derived from `telys_namespace`. | No |

Auth is the SDK's own one-time device authorization:

- `import telys`, class inspection, and the headless unit tests work without
  any login.
- Creating or opening a collection loads the signed on-device runtime, which
  requires `telys login` (free community plan, device authorization, fully
  offline thereafter).
- If the runtime is missing, construction fails fast with:
  `Run \`telys login\` once, it's free, device authorization, to fetch the signed on-device runtime.`

### Test coverage split

- **Headless** (verified now): all mapping logic with a stubbed SDK boundary
  (`tests/test_local_client.py`), plus connector/store tests with
  `FakeTelysClient`.
- **Requires login + runtime** (verified now): `test_integration_round_trip_with_real_telys`
  in `tests/test_local_client.py` passes against the logged-in Telys runtime.
- **Requires GPU pod** (bench week): end-to-end vLLM connector correctness and
  performance against the real Telys runtime, using the benchmark plan in
  section 4.

---

## 4. Benchmark plan (GPU pod, later)

Harness: vLLM's own `benchmarks/benchmark_prefix_caching.py` (the script the
FlexKV example points to, example lines 40-41) — "bench_cached_prefix" work
item. Plan:

1. **Pod**: single-GPU pod (e.g. 1×A100/H100 class), vLLM from pip, small
   model first (Llama-3.2-1B/3B class) to isolate connector overhead, then
   the target production model.
2. **A/B/C arms** (mirroring the flexkv example's three-way compare,
   `prefix_caching_flexkv.py:124-216`):
   - A: `--enable-prefix-caching=False` (baseline TTFT).
   - B: vLLM native prefix caching only.
   - C: B + `TelysKVConnector` (`kv_both`, sync loads).
   Correctness gate first: greedy decoding (`temperature=0.0`), identical
   outputs across arms (example lines 184-188).
3. **Workload**: `benchmark_prefix_caching.py --num-prompts 20 --repeat-count 5
   --input-length-range 512:2048` (ShareGPT) plus a fixed long shared-prefix
   prompt set (the script's PROMPT path) to force deep prefix reuse across
   requests *and across engine restarts* (restart = local cache gone, Telys
   persists — our differentiator).
4. **Metrics**: TTFT (primary), prefix hit rate (`connector_prefix_cache_stats`,
   `scheduler.py:1096-1099`), per-get/put latency inside TelysBlockStore,
   compaction pause time, end-to-end throughput.
5. **Tuning knobs to sweep**: `TELYS_VLLM_LOAD_ASYNC` (sync vs async loads),
   `VLLM_ENABLE_V1_MULTIPROCESSING` (shared hot-cache benefit vs isolation),
   `TELYS_VLLM_MAX_CHUNK_BLOCKS`, block payload size vs model/block size.
6. **Success bar**: C ≈ B on warm-cache TTFT (Telys adds ≤ small ε), and C ≫ B
   after engine restart (Telys survives; local cache does not). Arm C now runs
   with `torch.compile` enabled (no `--enforce-eager`).

---

## 5. Bench-week open questions

1. **GPU payload extraction/injection**: implemented in `telys_vllm/gpu_ops.py`
   (tensors extracted in `wait_for_save`, injected in `start_load_kv`) and
   wired through `register_kv_caches`. **Validated** on a Runpod A40 with
   Qwen2.5-1.5B-Instruct under `torch.compile` (Arm C of the v3 benchmark).
2. **Real `TelysClient` binding** (`TelysLocalClient`): implemented in
   `telys_vllm/local_client.py` (see section 3.8). **Validated** on a Runpod
   A40 against a self-host Telys unix-socket server.
3. **Async loads** (`TELYS_VLLM_LOAD_ASYNC=true`): the full
   WAITING_FOR_REMOTE_KVS + `get_finished` flow (`scheduler.py:1384-1409`),
   including `get_block_ids_with_load_errors` wiring.
4. **Namespace-scan delete** for `reset_cache` (drop `_known_keys` tracking).
5. **Multi-group KV caches** (hybrid models): load/save specs currently carry
   group-0 block ids (`example_connector.py:334` does the same); HMA support
   is implemented at the API level but untested against a real hybrid model.
6. **Metrics**: `get_kv_connector_stats` / `build_kv_connector_stats`
   (`v1/base.py:453-455`, `742-750`) for hit-rate/latency export; KV events
   via `take_events` (`v1/base.py:665-672`).
7. **CPU offload tier ordering** vs the native `OffloadingConnector`: can
   they coexist via `MultiConnector` (`factory.py:195-199`), and does the
   longest-hit-wins arbitration behave?
8. **Multi-process vLLM**: with `VLLM_ENABLE_V1_MULTIPROCESSING=1` the
   scheduler and workers live in different processes, so each process has its
   own chunk-length hot cache. The connector stays correct for warm restarts
   (each process seeds from disk). Measured in the v3 benchmark (Arm C MP=1);
   the self-host unix-socket server provides cross-process persistence, with
   MP=1 warm TTFT ~10.9 s vs ~2.8 s for MP=0 on Qwen2.5-1.5B-Instruct. Whether
   a shared remote `TELYS_ADDRESS` server plus periodic refresh improves this
   is left as future tuning.

---

## 6. What this skeleton deliberately does NOT do

- GPU workload has been exercised end-to-end on a Runpod A40 with
  Qwen2.5-1.5B-Instruct (see section 4 / `bench/RESULTS.md` v3). The
  CPU-testable skeleton remains fully functional for local development.
- No async load flow, no KV events, no custom Prometheus metrics.
- No publishing/PRs; the package is importable locally only.

---

## 7. Blob API as the native data path (future)

Telys is adding a first-class binary blob / KV fast path to the Python SDK
(`telys.BlobStore` / `Telys.blob_store(namespace=...)`) specifically to remove
the need for per-chunk base64-over-JSON workarounds. Once that SDK version is
available, the connector should migrate its KV payload storage to the native
blob API and drop the local slab logic.

Until then, the connector's local append-only slab remains the supported
fallback for SDK versions that do not yet expose the blob API. The fallback
must keep the same durability and namespace-isolation invariants so the switch
to the native API is a drop-in replacement.
