# telys-vllm

An open-source external KV / prefix-cache connector for vLLM
(`KVConnectorBase_V1`) backed by [Telys](https://thyn.ai): restart-persistent,
cross-engine KV cache that provably beats recomputation on long shared prefixes.

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

All numbers below come from `bench/RESULTS.md`. Workload: `Qwen/Qwen2.5-7B-Instruct`,
4096-token shared prefix, inputs 4096–8192 tokens, 64 output tokens, vLLM native
prefix caching **OFF** (`--no-enable-prefix-caching`), RunPod A40 (EU-SE-1),
`telys==0.1.10`, `vllm==0.32.0`.

| Version | Arm | TTFT p50 | Total time | Throughput | Speedup vs recompute |
|---|---|---:|---:|---:|---:|
| Benchmark v5 | Recompute (no cache) | 49.3 s | 106.8 s | 60.0 tok/s | 1.0× |
| Benchmark v5 | Telys warm hit | **27.8 s** | **78.8 s** | **81.2 tok/s** | **1.8× TTFT, 1.35× total** |
| Benchmark v6 | Recompute, 8K prefix | 89.3 s | 189.1 s | 33.9 tok/s | 1.0× |
| Benchmark v6 | Telys warm, 8K prefix | 25.1 s | 41.9 s | 152.7 tok/s | 3.6× TTFT, 4.5× total |
| Benchmark v6 | Recompute, 16K prefix | 186.6 s | 383.3 s | 16.7 tok/s | 1.0× |
| Benchmark v6 | Telys warm, 16K prefix | 48.6 s | 74.1 s | 86.4 tok/s | 3.8× TTFT, 5.2× total |
| Benchmark v7 | Recompute (fresh process) | 49.7 s | 103.9 s | 61.6 tok/s | 1.0× |
| Benchmark v7 | Telys warm, second replica | **16.8 s** | **29.0 s** | **220.9 tok/s** | **2.96× TTFT, 3.58× total** |

Key takeaways:

- The connector **beats recomputation on total wall time** at 4K and longer
  prefixes.
- The win **grows with prefix length**: 3.6× TTFT at 8K, 3.8× at 16K.
- The **cross-replica / restart story works**: a second, fresh vLLM process with
  an empty native cache warm-hits the Telys store populated by the first process.

Latest Runpod validation (2026-11-08 final independent A40 profiles and native
checks; `PR #2, package 0.1.2`, all current PR checks green; `telys==0.1.14`, A40):

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
profiles ran at `1a100ac`.** Current PR head `3f39593` is workflow-only after
`1a100ac`. Historical fix6 and corrected pre-fix results remain separately labeled
in [`integrations/runpod-worker-vllm/RESULTS.md`](integrations/runpod-worker-vllm/RESULTS.md).

### Serverless rc9 lifecycle proof

**Release candidate, not a signed final release.** Acceptance image
`ghcr.io/thyn-ai/telys-worker-vllm:0.1.2-rc.9` used immutable digest
`sha256:8d84aa45d5578421696f67d2e7ae2e6fb475c4d8ffb5d0429b7ea748dfd0fea2`.
Source commit `1a100ac26dff6a98b8ceafec5cdaebf0f5d49135` matched all 10 compared
Python files in `rc9-source-verification.json`; later `3f39593` changes are release
workflow only. The final release tag has not been created, and final signed image
publication remains pending.

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

**Proof-level and working.** The connector passes the local suite, correctly
round-trips KV blocks, and the warm-hit path is measurably faster than
recomputation on the workloads above. It is not yet a universal drop-in win:
performance depends on prefix length, model size, GPU memory budget, and whether
the deployment restarts or scales replicas.

- Local tests: **278 passed, 8 skipped** (tests that require a live GPU or the
  full Telys runtime are skipped on headless runners).
- Supported vLLM: `>=0.11` (tested most recently on `0.32.0`).
- Supported Python: `>=3.10`.
- Async loading is unsupported in this revision and fails fast; use synchronous
  loads (`TELYS_VLLM_LOAD_ASYNC=0`).

## Install

```bash
pip install telys-vllm[vllm]
```

The `[vllm]` extra pulls in `vllm>=0.11`. The connector itself has no runtime
dependency on the closed Telys engine; the engine is installed separately by
`pip install telys` and `telys login`.

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
