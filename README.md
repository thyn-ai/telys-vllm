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
| Autopsy v5 | Recompute (no cache) | 49.3 s | 106.8 s | 60.0 tok/s | 1.0× |
| Autopsy v5 | Telys warm hit | **27.8 s** | **78.8 s** | **81.2 tok/s** | **1.8× TTFT, 1.35× total** |
| Autopsy v6 | Recompute, 8K prefix | 89.3 s | 189.1 s | 33.9 tok/s | 1.0× |
| Autopsy v6 | Telys warm, 8K prefix | 25.1 s | 41.9 s | 152.7 tok/s | 3.6× TTFT, 4.5× total |
| Autopsy v6 | Recompute, 16K prefix | 186.6 s | 383.3 s | 16.7 tok/s | 1.0× |
| Autopsy v6 | Telys warm, 16K prefix | 48.6 s | 74.1 s | 86.4 tok/s | 3.8× TTFT, 5.2× total |
| Autopsy v7 | Recompute (fresh process) | 49.7 s | 103.9 s | 61.6 tok/s | 1.0× |
| Autopsy v7 | Telys warm, second replica | **16.8 s** | **29.0 s** | **220.9 tok/s** | **2.96× TTFT, 3.58× total** |

Key takeaways:

- The connector **beats recomputation on total wall time** at 4K and longer
  prefixes.
- The win **grows with prefix length**: 3.6× TTFT at 8K, 3.8× at 16K.
- The **cross-replica / restart story works**: a second, fresh vLLM process with
  an empty native cache warm-hits the Telys store populated by the first process.

Latest Runpod validation (native + Telys, `telys==0.1.14`, A40): **16.623 s total
and 10.754 s mean TTFT**, versus **20.335 s and 9.453 s** for the no-cache arm.
All **2,596/2,596 suffix blocks** persisted and every generated token matched the
no-cache reference with batch-invariant kernels. Stock filesystem KV was faster
at **11.406 s total and 6.221 s mean TTFT**, so Telys is not a universal latency
win. See [`integrations/runpod-worker-vllm/RESULTS.md`](integrations/runpod-worker-vllm/RESULTS.md)
for the full protocol and an actual serverless warm-hit check.

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

- Local tests: **198 passing, 8 skipped** (tests that require a live GPU or the
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
