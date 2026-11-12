# Runpod worker validation results

## Historical 4K environment and protocol

- Runpod A40, `CA-MTL-1`; model `Qwen/Qwen2.5-7B-Instruct`.
- vLLM `0.32.0`, Telys `0.1.14`, derived worker base `runpod/worker-v1-vllm:v2.30.3`.
- Target 4,096-token shared prefix, 64 output tokens, temperature 0, seed 42.
- Native prefix caching disabled; maximum model length 12,288; GPU utilization 0.95.
- One cold prefix-only process, then a fresh warm process with 20 unique suffixes.
- Timing excludes model initialization. Cold and warm phases have different request counts and must not be compared with each other.

## Historical fix6 no-cache versus stock filesystem KV

| Arm / phase | Total time (s) | Throughput (tok/s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | First request TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|
| No-cache cold | 2.381 | 26.882 | 0.591 | 0.591 | 0.591 | 0.591 |
| No-cache warm | 20.335 | 62.945 | 9.453 | 9.664 | 16.943 | 0.681 |
| Filesystem cold | 2.411 | 26.541 | 0.619 | 0.619 | 0.619 | 0.619 |
| Filesystem warm | 11.406 | 112.225 | 6.221 | 6.509 | 8.495 | 3.780 |

The filesystem arm used vLLM's unmodified `OffloadingConnector`, a 1 GiB CPU
staging tier, and a filesystem secondary tier on the Runpod network volume. It
reported a **67.4% external prefix-cache hit rate** and left 2,597 files totaling
2,381,841,894 bytes. In this historical run, warm total time was 1.78x lower than
recomputation and warm mean TTFT was 1.52x lower. These are one-run A40
measurements, not a performance guarantee.

## Historical fix6 correctness

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

## Historical fix6 Telys implementation and validation

The fix6 connector uses stacked per-layer GPU-to-CPU extraction, bounded pinned
staging, suffix-only persistence, complete cached-continuation block IDs, and
worker-side fast-chunk encoding. Async loading is disabled by validation because
its lifecycle is incomplete in this revision; enabling it fails fast rather than
deferring a request indefinitely.

### Historical fix6 A40 measurements

| Arm / phase | Total time (s) | Throughput (tok/s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | First request TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|
| Telys cold | 3.073 | 20.824 | 1.212 | 1.212 | 1.212 | 1.212 |
| Telys warm, native + external | 16.623 | 77.002 | 10.754 | 11.315 | 13.843 | 4.218 |

The pre-fix native+Telys run took 30.382 s with 21.138 s mean TTFT. The fix6
implementation was 45.3% lower in total time and 49.1% lower in mean TTFT. It was
also faster than the 20.335 s no-cache total, but stock filesystem KV was faster:
11.406 s total and 6.221 s mean TTFT. This historical run was not a universal
latency win.

The fix6 run restored 269 blocks / 4,304 tokens on its first request. Later
requests used the native same-process cache. All cold, warm, and fresh-process
outputs matched the no-cache reference token-for-token with batch-invariant
kernels enabled.

### Historical fix6 persistence evidence

- **2,596/2,596 suffix blocks saved**, with no whole-prefix re-save.
- Continuations are complete and split across 24 chunks rather than 20; this is
  fragmentation, not missing data.
- Payload is 2,628,676,910 bytes, down from 6,327,128,826 bytes before the
  cursor fixes (58.45% smaller).
- Worker-side encoding reduced save submission time from 15.442 s to 11.857 s
  (23.22%). Fix6 drain completed without errors, and all 24 payloads were read
  and decoded successfully after shutdown.
- BlobStore-only persistence leaves only a 16-byte empty legacy slab header, not
  a duplicate payload.

### Historical fix6 serverless deployment

The derived `ghcr.io/thyn-ai/telys-worker-vllm:0.1.0-fix6-vllm0.32.0` image was
built and imported successfully, including `vllm==0.32.0`, `telys==0.1.14`, and
`TelysKVConnector`. A real Runpod serverless worker started through the intended
endpoint lifecycle and completed inference.

An identical long-prefix request then ran twice through that endpoint. Both
responses matched exactly. Execution times were 6.812 s and 4.718 s, a 30.7%
reduction on the repeated prefix. The test endpoint was deleted after the checks
to avoid idle worker charges.

### Historical fix6 limits

- Stock filesystem KV was faster on this historical workload. Telys is valuable
  for restart and replica persistence, not as a universal replacement for native
  or filesystem caching.
- Generation-time metrics exclude the final persistence drain. The drain is
  validated separately and must not be folded into those timings.
- Async loading is unsupported and fails fast. Multi-writer, multi-group, and
  disaggregated topologies require separate validation.

## Historical corrected pre-fix full-harness follow-up (2026-11-08)

Historical source: `PR #2, package 0.1.2`. At this checkpoint, the local suite was
**240 passed, 8 skipped** and its PR checks were green. The skipped tests required
opt-in live GPU/runtime resources. Evidence is under
`integrations/runpod-worker-vllm/evidence/final-corrected-4096-20261108-02`.

### Historical corrected 4K A40 measurements

| Arm / phase | Total time (s) | Throughput (tok/s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) |
|---|---:|---:|---:|---:|---:|
| No-cache cold | 2.368 | 27.025 | 0.580 | 0.580 | 0.580 |
| No-cache warm | 20.378 | 62.814 | 10.262 | 10.217 | 17.647 |
| Filesystem cold | 2.389 | 26.789 | 0.599 | 0.599 | 0.599 |
| Filesystem warm | 13.081 | 97.853 | 7.785 | 7.682 | 10.351 |
| Telys cold | 3.056 | 20.942 | 1.183 | 1.183 | 1.183 |
| Telys warm, native + external | 20.953 | 61.089 | 14.484 | 15.487 | 18.189 |

Telys warm did **not** beat same-run no-cache in total time or mean TTFT, and
stock filesystem KV was faster than both. This run therefore does not demonstrate
an end-to-end async improvement. The historical fix6 figures above remain a
separate run; their controls are not interchangeable with this follow-up.

Both no-cache and filesystem controls moved between runs, indicating run-to-run
variance. These single-run comparisons cannot isolate the cause of the Telys
delta. Fix6 phase telemetry was not retained, preventing phase-level attribution
and storage-time comparison.

### Historical correctness and persistence

All ordinary correctness statuses passed. All recorded output comparisons matched
in both text and tokens; fresh-process warm filesystem and Telys outputs matched
the no-cache reference.

Dynamically derived lookup records confirmed **2,596/2,596 suffix blocks**
persisted in **25 continuation chunks**, plus **269 cold-prefix blocks in one cold
chunk**. The store contained **2,628,678,028 payload bytes** across 26 chunks, all
read and decoded successfully.

### Historical isolated extraction microbenchmarks

Median extraction time on an otherwise idle A40, using 11 interleaved repetitions
after warmup:

| Blocks | Old connector (ms) | Async extraction (ms) | Old / async |
|---:|---:|---:|---:|
| 77 | 58.536 | 9.683 | 6.05x |
| 269 | 124.586 | 15.612 | 7.98x |
| 512 | 203.277 | 25.469 | 7.98x |

Outputs matched for every layer by tensor equality and SHA-256. These isolated
speedups of approximately **6–8×** do not prove an application-level or end-to-end
win: the microbenchmark did not reproduce concurrent model execution. Async loading
remains disabled and fails fast.

## 2026-11-08 final independent A40 profiles and native checks

Current status: `PR #2, package 0.1.2`; local suite **278 passed, 8 skipped**;
all current PR checks are green. Current PR head `3f39593` is workflow-only after
`1a100ac26dff6a98b8ceafec5cdaebf0f5d49135`.

The prior full 16K/4K profiles ran at `6cbfe82`, not at `1a100ac`. Both fresh
profiles used a Runpod A40, `Qwen/Qwen2.5-7B-Instruct`, vLLM `0.32.0`, Telys
`0.1.14`, 64 output tokens, temperature 0, seed 42, and separate cold/warm engines.
Evidence is under
`integrations/runpod-worker-vllm/evidence/final-correctness-16384-20261108` and
`integrations/runpod-worker-vllm/evidence/final-correctness-4096-20261108`.

The prior full profiles exercised the real cursor fixes for decode and
aligned-prompt persistence and the metadata fixes for final-decode and
restart/reopen handling. Reopen checks covered durable and legacy chunk metadata.

### Final-correctness 16K A40 measurements

| Arm / phase | Total time (s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) |
|---|---:|---:|---:|---:|
| No-cache cold | 4.692 | 2.812 | 2.812 | 2.812 |
| No-cache warm | 72.390 | 35.659 | 35.098 | 64.394 |
| Filesystem cold | 4.743 | 2.866 | 2.866 | 2.866 |
| Filesystem warm | 26.378 | 17.936 | 17.506 | 20.797 |
| Telys cold | 7.417 | 4.712 | 4.712 | 4.712 |
| Telys warm, native + external | 42.247 | 28.955 | 30.584 | 36.811 |

Telys warm beat same-run no-cache on total time and mean TTFT. Stock filesystem KV
remained faster on both metrics.

The independent verifier passed **1,081/1,081 cold** and **24,333/24,333 warm**
blocks, **25,414/25,414 combined**, with zero missing blocks, errors, or
inconclusive results. It resolved packed shape `[16, 4, 256]`. Pending-write
telemetry was clean: writes were observed, all chunks persisted, and both harnesses
exited cleanly.

### Final-correctness 4K A40 measurements

| Arm / phase | Total time (s) | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) |
|---|---:|---:|---:|---:|
| No-cache cold | 2.373 | 0.585 | 0.585 | 0.585 |
| No-cache warm | 20.541 | 10.333 | 10.267 | 17.662 |
| Filesystem cold | 2.384 | 0.595 | 0.595 | 0.595 |
| Filesystem warm | 13.028 | 7.612 | 7.279 | 9.954 |
| Telys cold | 3.132 | 1.174 | 1.174 | 1.174 |
| Telys warm, native + external | 21.779 | 14.401 | 15.383 | 18.272 |

Telys warm did **not** beat same-run no-cache on total time or mean TTFT; stock
filesystem KV was faster on both metrics.

The dynamic persistence audit passed **273/273 cold blocks**, **8,056 warm coverage
blocks**, **5,380 reused blocks**, **2,676/2,676 newly persisted blocks**, and
**2,949/2,949 total-store blocks**. Every warm prompt hit 269 blocks. The resolved
packed shape was `[16, 4, 256]`.

### Final correctness and native checks

All A/B/C ordinary correctness statuses and output-equality checks passed on both
profiles. Final native checks at `1a100ac26dff6a98b8ceafec5cdaebf0f5d49135`
passed v2 `meta_blob` projection and nested payload+metadata decoding, dual-codec
fallback, overlapping-client deletion coherence, and divergent-prefix rejection.
Evidence is under `integrations/runpod-worker-vllm/evidence/final-v2-20261108`.
These were native checks only; no full profiles ran at `1a100ac`.

These full-harness measurements do not establish an end-to-end async extraction
improvement; async loading remains disabled and fails fast.

## 2026-11-08 serverless rc9 endpoint lifecycle validation

Evidence is under
`integrations/runpod-worker-vllm/evidence/serverless-rc9-20261108`.

### Release-candidate identity

- Acceptance image: `ghcr.io/thyn-ai/telys-worker-vllm:0.1.2-rc.9`.
- Immutable digest:
  `sha256:8d84aa45d5578421696f67d2e7ae2e6fb475c4d8ffb5d0429b7ea748dfd0fea2`.
- Acceptance source: `1a100ac26dff6a98b8ceafec5cdaebf0f5d49135`.
  `rc9-source-verification.json` reports matching source files for all 10 compared
  Python files, connector `0.1.2`, vLLM `0.32.0`, and Telys `0.1.14`.
- Later `3f39593` changes are release workflow only.

The stable `v0.1.6` release is available on PyPI and GitHub Releases. Its public
image is `ghcr.io/thyn-ai/telys-worker-vllm:0.1.6` at immutable digest
`sha256:1b02b3cdac471064e8b9968fe2ab87f99da2d455e04d9cffea6b74b745a4218f`.
The release workflow verified its Sigstore signature and SLSA provenance; an
independent anonymous registry read returned the same digest.

### Lifecycle and correctness results

| Check | Verified result |
|---|---|
| Cold inference | Worker `yutiyuykg0g2gc` completed in **7.042 s**. |
| Cold persistence | **538 prompt blocks + 4 generated blocks = 542 blocks**. |
| Fresh-engine warm hit | Worker `tdi0xf9426iu21` returned exactly matching output; first lookup hit **538 blocks / 8,608 tokens**. |
| Scale-to-zero | FlashBoot OFF; observed **`running=0`**. |
| Post-zero fresh engine | Engine `4ca28826-b4a8-4edc-9bc0-da0e169fe9e1` on worker `1gsk5j29rzcz0s` returned matching output in **6.735 s** and again hit **538 blocks / 8,608 tokens**. |

All compared outputs matched exactly. Fresh engine IDs differed across starts. Both
concurrent replica responses used worker `tdi0xf9426iu21`. Async loading remained
disabled.

Initial provisioning reused the existing signed runtime/license on the volume; it
did not perform a new device registration. The endpoint was deleted. Templates have
a normal 90-minute platform hold before deletion.

## 2026-11-12 v0.1.6 exact upstream-worker A40 acceptance

The exact upstream worker Dockerfile at commit
`49bfa65d76b6611ed845b48a46317c79633f93bc` built and ran. Acceptance used
connector `0.1.6` at source `2f0c20ccec56afcb613973e296281f51e23cbe26`, SDK
`0.1.15`, vLLM `0.32.0`, one NVIDIA A40, and `Qwen/Qwen2.5-7B-Instruct`. The
runtime image digest was
`sha256:6ce40a639eabb7334d41e73f42858e37add089e574dd6ef3e4d0b21af78c95d8`;
this acceptance image is distinct from the public release image above.

Four prompts shared a 4,096-token prefix and had distinct 64-token suffixes. Each
request generated 32 greedy output tokens (128 total), with batch-invariant mode
enabled and native prefix caching disabled.

| Arm / phase | Mean TTFT (s) | Batch wall (s) | Output throughput (tok/s) |
|---|---:|---:|---:|
| Stock forced recompute | 2.611967 | 4.568364 | 28.019 |
| Telys external cold, empty external cache | 4.310308 | 6.461609 | 19.809 |
| Telys fresh-process warm, persisted external cache | 2.216870 | 4.167074 | 30.717 |

TTFT is the mean across the four requests; wall time and throughput cover the whole
batch. External cold began with an empty external cache, while fresh-process warm
reused persisted external KV. Both are distinct from stock forced recompute, and
the cold persistence drain is reported separately below.

All **1,044 blocks** verified. All four external-cold and fresh-process warm output
token-ID sequences and text matched stock. Fresh warm recorded **16,576 external
hits** (16,384 shared-prefix + 192 suffix), cached tokens per request
`[4144, 4144, 4144, 4144]`, and zero native hits. Cold drain took **0.295375 s**
and finished with zero pending keys, pending writes, and write errors.

Default-off behavior, all 12 override rejections, digest/tamper rejection,
delete/reset, and credential sanitization passed. Pod `p3qu21d37fh3hq`, volume
`7x9gdvpvzh`, and temporary registry resources were deleted. The sanitized
machine-readable record is
[`upstream-vllm-v0.1.6-acceptance.json`](evidence/upstream-vllm-v0.1.6-acceptance.json).

This acceptance covers one A40, one model, and one four-request batch. It does not
establish universal speedups, native-hot superiority, or performance on other GPUs,
models, batch shapes, or workloads.
