# telys-vllm

An open-source external KV / prefix-cache connector for vLLM
(`KVConnectorBase_V1`) backed by [Telys](https://thyn.ai): restart-persistent,
cross-engine KV cache for shared-prefix workloads.

The closed Telys engine stays closed — it is fetched by `telys login` and runs
as a signed local runtime, like the `telys` SDK. This repository is the Apache-2.0
integration layer: the connector, tests, benchmarks, and telemetry that let vLLM
save and load KV blocks through Telys.

## What it does

- Plugs into vLLM V1 as a custom `KVConnectorBase_V1`.
- Hashes per-block KV tokens and stores them in Telys so they survive process
  restarts and can be shared across replicas.
- On a cache hit, loads the stored blocks and injects them into vLLM's paged KV
  cache, skipping prefix recomputation.
- Ships with a fast path that stores stacked per-layer CPU tensors and uses
  batched, per-layer GPU injection.

## Measured results

The current apples-to-apples concurrent matrix is run
`ca40-20261108T181727Z` on one NVIDIA A40. It used
`Qwen/Qwen2.5-7B-Instruct`, vLLM `0.32.0`, BF16, the same direct token-ID
prompts and greedy 64-token decoding in every arm, `max_num_seqs=32`,
`gpu_memory_utilization=0.95`, and `max_model_len=32768`. Each cell has three
repetitions. TTFT p50 pools all requests; wall and output throughput are means
of the three batch-level values. Full settings, reproduction, and exact source
provenance are in [`bench/CONCURRENT.md`](bench/CONCURRENT.md).

| Prefix | C | Arm | Pooled TTFT p50 (s) | Mean wall (s) | Mean output tok/s | Wall speedup vs recompute |
|---:|---:|---|---:|---:|---:|---:|
| 8K | 1 | Forced recompute | 1.165793 | 2.985708 | 21.435457 | 1.0000x |
| 8K | 1 | Fresh-process Telys external hit | 1.104727 | 2.913860 | 21.974601 | **1.0247x** |
| 8K | 1 | Native-hot, same process | 0.060614 | 1.878551 | 34.068840 | **1.5894x** |
| 8K | 4 | Forced recompute | 3.969180 | 6.572369 | 38.950953 | 1.0000x |
| 8K | 4 | Fresh-process Telys external hit | 1.302441 | 3.343736 | 76.573539 | **1.9656x** |
| 8K | 4 | Native-hot, same process | 0.117142 | 2.029236 | 126.156782 | **3.2388x** |
| 8K | 8 | Forced recompute | 6.250244 | 11.383825 | 44.976216 | 1.0000x |
| 8K | 8 | Fresh-process Telys external hit | 1.548653 | 3.716415 | 138.050123 | **3.0631x** |
| 8K | 8 | Native-hot, same process | 0.158475 | 2.146576 | 238.521740 | **5.3032x** |
| 16K | 8 | Forced recompute | 12.908762 | 23.655112 | 21.644457 | 1.0000x |
| 16K | 8 | Fresh-process Telys external hit | 2.706828 | 5.396244 | 94.979480 | **4.3836x** |
| 16K | 8 | Native-hot, same process | 0.252128 | 2.418234 | 211.732989 | **9.7820x** |

The 8K/c1 external result is approximately neutral (**1.0247x** mean-wall
speedup; one repetition was 0.9992x). The measured external wins are at 8K/c4,
8K/c8, and 16K/c8; three repetitions do not establish statistical significance.

Recompute and fresh-process external arms disable native prefix caching.
External uses a separate drained writer and a fresh reader with a warm
persistent store. **Native-hot is a separate same-process arm**, reusing
GPU-resident native prefix KV after an excluded writer. Native-hot is faster
than both other arms in every cell; its ratios are never included in external
persistent-cache speedups. Older sequential/no-cache v5-v7 results are retained
in [`bench/RESULTS.md`](bench/RESULTS.md) as history, not as native-hot wins.

Earlier independent A40 profiles used `telys==0.1.14` and the following
same-run controls:

| Prefix | Arm / phase | Total time | Mean TTFT |
|---|---|---:|---:|
| 16K | No-cache warm | 72.390 s | 35.659 s |
| 16K | Stock filesystem warm | 26.378 s | 17.936 s |
| 16K | Telys cold | 7.417 s | 4.712 s |
| 16K | Telys warm | 42.247 s | 28.955 s |
| 4K | No-cache warm | 20.541 s | 10.333 s |
| 4K | Stock filesystem warm | 13.028 s | 7.612 s |
| 4K | Telys cold | 3.132 s | 1.174 s |
| 4K | Telys warm | 21.779 s | 14.401 s |

At 16K, Telys warm beat same-run no-cache on total time and mean TTFT; stock
filesystem KV remained faster. At 4K, Telys warm did **not** beat no-cache, and
stock filesystem KV was faster. These results do not demonstrate an end-to-end
async extraction improvement; async loading remains disabled.

All A/B/C ordinary correctness and output-equality checks passed on both profiles.
The 16K independent verifier passed **1,081/1,081 cold** and **24,333/24,333 warm**
blocks (**25,414/25,414 combined**), with zero missing blocks, errors, or
inconclusive results, packed shape `[16, 4, 256]`, and clean pending-write
telemetry. The 4K dynamic audit passed **273/273 cold blocks**, **8,056 warm
coverage blocks**, **5,380 reused blocks**, **2,676/2,676 newly persisted blocks**,
and **2,949/2,949 total-store blocks**; every warm prompt hit 269 blocks, with
packed shape `[16, 4, 256]`.

The prior full profiles at `6cbfe82` exercised the real cursor and metadata fixes
for decode and aligned-prompt persistence and for final-decode/restart handling.
Final native checks at `1a100ac26dff6a98b8ceafec5cdaebf0f5d49135` passed v2
`meta_blob` projection and nested payload+metadata decoding, dual-codec fallback,
overlapping-client deletion coherence, and divergent-prefix rejection; evidence is
under `integrations/runpod-worker-vllm/evidence/final-v2-20261108`. **No full
profiles ran at `1a100ac`.** Historical fix6 and corrected pre-fix results remain
separately labeled in
[`integrations/runpod-worker-vllm/RESULTS.md`](integrations/runpod-worker-vllm/RESULTS.md).

### Serverless rc9 lifecycle proof

**Release candidate, not a signed stable release.** Acceptance image
`ghcr.io/thyn-ai/telys-worker-vllm:0.1.2-rc.9` used immutable digest
`sha256:8d84aa45d5578421696f67d2e7ae2e6fb475c4d8ffb5d0429b7ea748dfd0fea2`.
Source commit `1a100ac26dff6a98b8ceafec5cdaebf0f5d49135` matched all 10 compared
Python files in `rc9-source-verification.json`. The `v0.1.2` Git tag exists, but
this evidence does not establish a stable GitHub Release, PyPI wheel, or signed
stable GHCR image; rc9 must not be represented as one. The `v0.1.3` source tag
also exists; its automated publication stopped when the GHCR image push was
denied by the package's repository Actions grant. Downstream GitHub Release and
PyPI jobs were skipped. No stable `0.1.3` GitHub Release, PyPI wheel, or signed
GHCR image is claimed here.

Cold worker `yutiyuykg0g2gc` completed in **7.042 s** and persisted **538 prompt
blocks + 4 generated blocks = 542 blocks**. Fresh worker `tdi0xf9426iu21` returned
exactly matching output; its first lookup hit **538 blocks / 8,608 tokens**. After
actual scale-to-zero with FlashBoot OFF and `running=0`, fresh engine
`4ca28826-b4a8-4edc-9bc0-da0e169fe9e1` on worker `1gsk5j29rzcz0s` returned
matching output in **6.735 s** and again hit **538/8,608**. Fresh engine IDs
differed across starts; the concurrent replica responses shared worker
`tdi0xf9426iu21`.

Initial provisioning reused the existing signed runtime/license on the volume, not
a new device registration. Async loading remained disabled. The endpoint was
deleted; templates have a normal 90-minute platform hold before deletion.

## When it helps / when it doesn't

| Scenario | Verdict | Why |
|---|---|---|
| **Process restart or redeploy** | ✅ Helps | KV blocks live in Telys, so a new process starts warm. |
| **Multiple replicas / horizontal scale** | ✅ Helps | Replicas share one Telys store; a cold replica warm-hits immediately. |
| **Prefixes larger than native GPU KV budget** | ✅ Helps | Telys persists blocks that vLLM's native cache would evict. |
| **Single long-lived engine, prefix fits in GPU cache** | ⚠️ Native caching wins | vLLM's in-memory prefix cache is faster when the engine never restarts and the KV budget is large enough. The connector adds serialization and persistence overhead that only pays off across restarts, replicas, or very long prefixes. |
| **Short prefixes on small models** | ❌ Not helpful | Recomputing a 512-token prefix on a 1.5B model is cheaper than loading it from an external store. |

## Current status

**Working connector with source-scoped validation.** It is not a universal
drop-in win: performance depends on batching, prefix length, model, GPU memory,
and whether the deployment restarts or scales replicas.

- Merged connector validation at `ecbb88e25579bdd1fc93d32c34abc34ee808690f`:
  **370 local tests passed, 2 skipped**, and Ruff passed. Real Telys `0.1.14` /
  vLLM `0.32.0` A40 checks covered persistence, reopen, reset races and two
  opt-in output-correctness passes. Final review regressions through `dbde0b`
  also passed CI.
- Separate 8K/c1 greedy checks at `c42b049` with `VLLM_BATCH_INVARIANT=1`
  proved exact restored KV and 64-token output equality. Non-invariant mode is
  **not bitwise guaranteed**; see the scoped evidence in
  [`bench/CONCURRENT.md`](bench/CONCURRENT.md#correctness-and-validation).
- Telys `0.1.15` adds durable BlobStore deletion tombstones. Use that SDK
  version or newer; the connector also rejects orphaned blob-only records on
  affected older stores instead of injecting deleted KV data.
- Supported vLLM: `>=0.11` (tested most recently on `0.32.0`).
- Supported Python: `>=3.10`.
- Async loading is unsupported in this revision and fails fast; use synchronous
  loads (`TELYS_VLLM_LOAD_ASYNC=0`).

## Install

```bash
pip install "telys>=0.1.15" "telys-vllm[vllm]"
telys login
```

The `[vllm]` extra pulls in `vllm>=0.11`. The connector itself has no runtime
dependency on the closed Telys engine; `telys login` installs and verifies the
signed on-device runtime.

## Usage

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    kv_connector="telys_vllm.connector.TelysKVConnector",
    kv_connector_extra_config={"telys_path": "./telys_kv"},
    enable_prefix_caching=False,  # let Telys be the cache of record
)
```

See `bench/` for reproducible benchmark scripts and `DESIGN.md` for the
architecture and evolution of the connector.

## Repository layout

```
telys_vllm/      # Connector source
  connector.py   # vLLM KVConnectorBase_V1 implementation
  store.py       # Telys-backed block store
  codec.py       # Raw-byte / stacked-tensor codecs
  gpu_ops.py     # GPU injection helpers
  local_client.py # Slab / BlobStore client adapters
  telemetry.py   # JSONL instrumentation used by benches
tests/           # Unit and integration tests
bench/           # Benchmarks, run scripts, and RESULTS.md
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). The Telys engine runtime itself is
closed and distributed separately; contributions here are limited to the open
connector, tests, benchmarks, and docs.

## License

Apache-2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
