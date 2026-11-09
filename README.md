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
speedup; one repetition was 0.9992x). Three repetitions do not establish
statistical significance.

Recompute and fresh-process external arms disable native prefix caching.
External uses a separate drained writer and a fresh reader with a warm
persistent store. **Native-hot is a separate same-process arm**, reusing
GPU-resident native prefix KV after an excluded writer. Native-hot had lower
mean wall and pooled TTFT p50 and higher output throughput than both other arms
in every measured cell; its ratios are excluded from external persistent-cache
speedups. Older sequential/no-cache v5-v7 results are retained in
[`bench/RESULTS.md`](bench/RESULTS.md) with their original baselines.

## Validation

- Independent A40 verification passed **25,414/25,414 16K blocks** and
  **2,949/2,949 4K blocks**. All A/B/C ordinary correctness and
  output-equality checks passed on both profiles. See the [full profile
  measurements and limitations](integrations/runpod-worker-vllm/RESULTS.md#2026-11-08-final-independent-a40-profiles-and-native-checks).
- Current local tests and Ruff pass. Real Telys `0.1.14` / vLLM `0.32.0` A40
  validation covered persistence, reopen, reset races, and two opt-in
  output-correctness passes. Bitwise restored-KV
  and 64-token output equality are validated with `VLLM_BATCH_INVARIANT=1`
  and greedy decoding. See [correctness scope and
  provenance](bench/CONCURRENT.md#correctness-and-validation).
- Serverless lifecycle validation returned matching output from fresh workers;
  first lookups hit **538 blocks / 8,608 tokens**, including after
  scale-to-zero. See [lifecycle measurements and
  evidence](integrations/runpod-worker-vllm/RESULTS.md#2026-11-08-serverless-rc9-endpoint-lifecycle-validation).

## Current status

**Working connector with source-scoped validation.** Performance is reported by
workload, including batching, prefix length, model, GPU memory, and process or
replica lifecycle.

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
