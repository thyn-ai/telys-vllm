# Runpod worker validation results

## Environment and protocol

- Runpod A40, `CA-MTL-1`; model `Qwen/Qwen2.5-7B-Instruct`.
- vLLM `0.32.0`, Telys `0.1.14`, derived worker base `runpod/worker-v1-vllm:v2.30.3`.
- Target 4,096-token shared prefix, 64 output tokens, temperature 0, seed 42.
- Native prefix caching disabled; maximum model length 12,288; GPU utilization 0.95.
- One cold prefix-only process, then a fresh warm process with 20 unique suffixes.
- Timing excludes model initialization. Cold and warm phases have different request counts and must not be compared with each other.

## No-cache versus stock filesystem KV

| Arm / phase | Total time (s) | Throughput (tok/s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | First request TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|
| No-cache cold | 2.381 | 26.882 | 0.591 | 0.591 | 0.591 | 0.591 |
| No-cache warm | 20.335 | 62.945 | 9.453 | 9.664 | 16.943 | 0.681 |
| Filesystem cold | 2.411 | 26.541 | 0.619 | 0.619 | 0.619 | 0.619 |
| Filesystem warm | 11.406 | 112.225 | 6.221 | 6.509 | 8.495 | 3.780 |

The filesystem arm used vLLM's unmodified `OffloadingConnector`, a 1 GiB CPU
staging tier, and a filesystem secondary tier on the Runpod network volume. It
reported a **67.4% external prefix-cache hit rate** and left 2,597 files totaling
2,381,841,894 bytes. Warm total time was 1.78x lower than recomputation and
warm mean TTFT was 1.52x lower. These are one-run A40 measurements, not a
performance guarantee.

## Correctness

Without batch-invariant kernels, one fresh-process filesystem run differed from
the no-cache reference in 15 of 64 generated tokens. The alternate tokens were
deterministic across fresh processes. With `VLLM_BATCH_INVARIANT=1`, cold,
same-process warm, and fresh-process warm outputs all matched the no-cache
reference token-for-token. The benchmark now enables batch invariance only for
correctness runs, not for performance timing.

The filesystem arm uses CPU staging and buffered or direct I/O depending on
filesystem support. A fresh process does not clear the OS page cache, so these
measurements are not pure physical-disk cold-read latency. The first warm
request is recorded separately from later requests.

## Final Telys implementation and validation

The final connector uses stacked per-layer GPU-to-CPU extraction, bounded pinned
staging, suffix-only persistence, complete cached-continuation block IDs, and
worker-side fast-chunk encoding. Async loading is disabled by validation because
its lifecycle is incomplete in this revision; enabling it fails fast rather than
deferring a request indefinitely.

Local suite: **198 passed, 8 skipped**. The skipped tests require opt-in live
GPU/runtime resources.

### Final A40 measurements

| Arm / phase | Total time (s) | Throughput (tok/s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | First request TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|
| Telys cold | 3.073 | 20.824 | 1.212 | 1.212 | 1.212 | 1.212 |
| Telys warm, native + external | 16.623 | 77.002 | 10.754 | 11.315 | 13.843 | 4.218 |

The pre-fix native+Telys run took 30.382 s with 21.138 s mean TTFT. The final
implementation is 45.3% lower in total time and 49.1% lower in mean TTFT. It is
also faster than the 20.335 s no-cache total, but stock filesystem KV remains
faster: 11.406 s total and 6.221 s mean TTFT. This is not a universal latency win.

The final run restored 269 blocks / 4,304 tokens on its first request. Later
requests used the native same-process cache. All cold, warm, and fresh-process
outputs matched the no-cache reference token-for-token with batch-invariant
kernels enabled.

### Persistence evidence

- **2,596/2,596 suffix blocks saved**, with no whole-prefix re-save.
- Continuations are complete and split across 24 chunks rather than 20; this is
  fragmentation, not missing data.
- Payload is 2,628,676,910 bytes, down from 6,327,128,826 bytes before the
  cursor fixes (58.45% smaller).
- Worker-side encoding reduced save submission time from 15.442 s to 11.857 s
  (23.22%). Final drain completed without errors, and all 24 payloads were read
  and decoded successfully after shutdown.
- BlobStore-only persistence leaves only a 16-byte empty legacy slab header, not
  a duplicate payload.

### Actual serverless deployment

The derived `ghcr.io/thyn-ai/telys-worker-vllm:0.1.0-fix6-vllm0.32.0` image was
built and imported successfully, including `vllm==0.32.0`, `telys==0.1.14`, and
`TelysKVConnector`. A real Runpod serverless worker started through the intended
endpoint lifecycle and completed inference.

An identical long-prefix request then ran twice through that endpoint. Both
responses matched exactly. Execution times were 6.812 s and 4.718 s, a 30.7%
reduction on the repeated prefix. The test endpoint was deleted after the checks
to avoid idle worker charges.

### Limits

- Stock filesystem KV remains faster on this workload. Telys is valuable for
  restart and replica persistence, not as a universal replacement for native or
  filesystem caching.
- Generation-time metrics exclude the final persistence drain. The drain is
  validated separately and must not be folded into those timings.
- Async loading is unsupported in this revision and fails fast. Multi-writer,
  multi-group, and disaggregated topologies require separate validation.
