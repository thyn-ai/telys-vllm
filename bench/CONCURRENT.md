# Concurrent apples-to-apples A40 validation

Run `ca40-20261108T181727Z` measured batched inference on one Runpod NVIDIA A40
(pod `yzyuv82g45dm5b`). It supersedes the v5-v7 headline as the current
batching/parallelism comparison. The older sequential tables remain below in
[`RESULTS.md`](RESULTS.md) as history, not as native-hot evidence.

## Source and workload identity

- Connector source: exactly
  `d5c8c81cb7ae3d359be48895f9a33784a6b85c53` (`d5c8c81`), with a clean
  public source tree.
- Every Telys process used the runtime-only setting
  `TELYS_HOTPATH_SMALL_SAVE_SYNC_THRESHOLD_BYTES=0`. This selected the batched
  pinned path without changing the public source. Synchronous loading remained
  selected with `TELYS_VLLM_LOAD_ASYNC=0`.
- Model/runtime: `Qwen/Qwen2.5-7B-Instruct`, vLLM `0.32.0`, explicit `bfloat16`.
- Direct token-ID prompts contained an exact 8,192- or 16,384-token shared
  prefix and a distinct 16-token suffix. The same workload files, prompts,
  model, and settings were used for every arm within a cell.
- Greedy sampling produced exactly 64 output tokens with EOS ignored. Other
  settings were `max_num_seqs=32`, `gpu_memory_utilization=0.95`, and
  `max_model_len=32768`.
- Each measured batch was one `llm.generate([...])` call. The completed matrix
  is 8K at concurrency 1, 4, and 8 plus 16K at concurrency 8, with three
  repetitions and three arms per cell (36 measured engines). No 4K cell or 16K
  concurrency-1/4 cell was run.

The two later PR #4 commits are **not** the timing source:

| Commit | Role | Delta from the measured source |
|---|---|---|
| `d5c8c81cb7ae3d359be48895f9a33784a6b85c53` | Concurrent performance matrix | Exact clean source plus the runtime-only threshold above. |
| `c42b0495413a09c603157657f598b1be1e35264b` | Cached-KV correctness investigation | First later commit; the 8K/c1 batch-invariant and non-invariant correctness evidence below ran here. |
| `ecbb88e25579bdd1fc93d32c34abc34ee808690f` | Final local/real-runtime validation described here | Two commits ahead of `d5c8c81`; the concurrent matrix was not rerun at this revision. |

The `d5c8c81..ecbb88e` diff changes `pyproject.toml`, `telys_vllm/connector.py`,
`gpu_ops.py`, `local_client.py`, and `store.py`, plus the async-extraction,
compile, local-client, output-correctness, pinned-load-cache, and store tests.
It does not identify those later persistence fixes as part of the measured
`d5c8c81` timing source.

## Three-repetition results

TTFT p50 is pooled across every request in all three repetitions using
linear-interpolated percentiles. Wall and output throughput are the arithmetic
mean of the three batch-level values. Throughput is
`request_count * 64 / end-to-end generate wall`; engine initialization, writer
time, and final drain are excluded. Wall speedup is the ratio of the mean
recompute wall to the mean wall for the named arm.

| Prefix | Concurrency | Arm | Requests | Pooled TTFT p50 (s) | Mean wall (s) | Mean output tok/s | Wall speedup vs recompute |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8,192 | 1 | Forced recompute | 3 | 1.165793 | 2.985708 | 21.435457 | 1.0000x |
| 8,192 | 1 | Fresh-process Telys external hit | 3 | 1.104727 | 2.913860 | 21.974601 | **1.0247x** |
| 8,192 | 1 | Native-hot, same process | 3 | 0.060614 | 1.878551 | 34.068840 | **1.5894x** |
| 8,192 | 4 | Forced recompute | 12 | 3.969180 | 6.572369 | 38.950953 | 1.0000x |
| 8,192 | 4 | Fresh-process Telys external hit | 12 | 1.302441 | 3.343736 | 76.573539 | **1.9656x** |
| 8,192 | 4 | Native-hot, same process | 12 | 0.117142 | 2.029236 | 126.156782 | **3.2388x** |
| 8,192 | 8 | Forced recompute | 24 | 6.250244 | 11.383825 | 44.976216 | 1.0000x |
| 8,192 | 8 | Fresh-process Telys external hit | 24 | 1.548653 | 3.716415 | 138.050123 | **3.0631x** |
| 8,192 | 8 | Native-hot, same process | 24 | 0.158475 | 2.146576 | 238.521740 | **5.3032x** |
| 16,384 | 8 | Forced recompute | 24 | 12.908762 | 23.655112 | 21.644457 | 1.0000x |
| 16,384 | 8 | Fresh-process Telys external hit | 24 | 2.706828 | 5.396244 | 94.979480 | **4.3836x** |
| 16,384 | 8 | Native-hot, same process | 24 | 0.252128 | 2.418234 | 211.732989 | **9.7820x** |

The 8K/c1 external result is approximately neutral: three per-repetition wall
ratios were 1.0552x, 0.9992x, and 1.0210x, and the ratio of mean walls is only
1.0247x. Three repetitions do not support a statistical-significance claim.
The external benefit is clear in the measured 8K/c4, 8K/c8, and 16K/c8 cells.

**Native-hot is a separate comparison.** Recompute and external arms disable
native prefix caching. External uses a separate, drained writer and a fresh
reader process with initially empty process-local caches and a warm persistent
store. Native-hot instead reuses GPU-resident native prefix KV in the same
engine after an excluded prefix-only writer. Native-hot has lower mean wall and
pooled TTFT p50 and higher output throughput than both other arms in every
measured cell. Its ratios never enter the external-hit speedups, and the older
sequential/no-cache numbers must not be presented as native-hot wins.

## Correctness and validation

The performance matrix ran with `VLLM_BATCH_INVARIANT` unset. Its differing
non-invariant token outputs are not a bitwise-equivalence proof.

A separate 8K/c1 investigation at `c42b049` compared 28 layers x 512 logical
blocks. With `VLLM_BATCH_INVARIANT=1` and greedy decoding, writer GPU KV,
persistent KV, and restored GPU KV had zero mismatches in every compared K/V
layer/block; fresh external-hit and native-hot outputs both matched recompute
for all 64 token IDs. Restored GPU KV also remained bit-identical to persistent
KV in the non-invariant investigation. These scoped runs prove exact restored
KV and output equality for the validated configuration. **Without batch
invariance, bitwise output equality is not guaranteed** because floating-point
execution and reduction order can differ.

At `ecbb88e`, the CI-equivalent local run passed **370 tests with 2 skips**,
and Ruff passed. Real Telys `0.1.14` / vLLM `0.32.0` A40 validation completed
42 passing stages plus four expected nonzero diagnostic stages, including
persistence, reopen, old-schema, overlap, and cross-process race coverage. The
opt-in real output-correctness wrapper passed two tests with batch invariance
set only in its verifier child. No connector real-runtime blocker remained;
the documented raw SDK delete/reopen defect remains an upstream SDK issue and
the connector fails closed. PR #4 remains under review, and these checks do
not claim that it merged or that a stable artifact was published.

## Reproduction and evidence provenance

The preserved run contains `run-config.json`, `bin/run_matrix.sh`,
`bin/concurrent_apples.py`, `bin/summarize_concurrent_apples.py`, the exact
8K/16K workload JSON files, all 36 result JSON files and logs, writer audits,
package/model provenance, and a SHA-256 evidence manifest. The raw result
identity is `ca40-20261108T181727Z` on pod `yzyuv82g45dm5b`.

To reproduce with a preserved copy of that harness, check out the exact source,
make the harness's `SOURCE_ROOT`, `RUN_ROOT`, and `STORE_BASE` paths point to
the checkout and writable run/store directories, and run:

```bash
git checkout d5c8c81cb7ae3d359be48895f9a33784a6b85c53
export PYTHONPATH=/path/to/d5c8c81-source:/path/to/concurrent-apples/bin
export TELYS_HOTPATH_SMALL_SAVE_SYNC_THRESHOLD_BYTES=0
export TELYS_VLLM_LOAD_ASYNC=0
unset VLLM_BATCH_INVARIANT
RUN_ID=<new-run-id> bash /path/to/concurrent-apples/bin/run_matrix.sh
python /path/to/concurrent-apples/bin/summarize_concurrent_apples.py \
  --run-root "/tmp/telys-concurrent-apples-<new-run-id>" --pod-id <pod-id>
```

`run_matrix.sh` creates each workload and drained writer, executes the four
cells in the same arm/repetition order, requires all 36 results, and audits the
store. The equivalent individual measurement invocation is:

```bash
python bin/concurrent_apples.py measure \
  --arm <recompute|external|native-hot> --run-id <new-run-id> \
  --prefix-len <8192|16384> --concurrency <1|4|8> --repetition <1|2|3> \
  --workload workloads/workload-p<prefix>.json --output results/<result>.json \
  [--store-root <persistent-store> --writer-result results/writer-p<prefix>.json]
```

The store-root/writer arguments apply only to the external arm. Keep fresh
processes for every A/B repetition and do not substitute a same-process native
hit for an external persistent hit.

Scope: one A40, one model/runtime/source tuple, fixed short suffixes, and
64-token greedy decoding. Fresh process does not mean cold storage: OS,
network-volume, and storage caches were not flushed. Initialization and writer
costs are excluded, so these are warm-engine steady-state measurements, not
cold-start latency. TTFT includes queueing/scheduling within each offline
batch, and the concurrency-1 pooled percentile contains only three requests.
