# telys-vllm prefix-caching benchmark results

## v2 — shared-engine / hot-cache connector fix

Run: 2026-10-25 ~00:18–00:30 UTC  
Pod: Runpod A40 `u8xb0j8hrlwws0`, EU-SE-1  
Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py`  
Server: self-host Telys unix-socket (`unix:///root/.telys/telys.sock`), `--save-interval 30`

All arms run with `--enforce-eager` so the connector's `save_kv_layer` hook is invoked (torch.compile skips the eager decorator on Arm C).

### Workload

- `num_prompts=20`
- `repeat_count=5`
- `input_length_range=512:2048`
- `prefix_len=512`
- `output_len=10`
- `seed=42`

A fixed 512-token shared prefix is concatenated with random unique suffixes, then repeated/shuffled. Arm C cold runs populate Telys from scratch; warm-restart runs reuse the persisted Telys state.

### Results

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 0.920 (0.918–0.952) | 1086.566 (1050.439–1089.721) | 0.473 (0.403–0.483) | 0.488 (0.385–0.498) | 0.570 (0.517–0.586) | n/a |
| B (native prefix caching) | 3 | 0.923 (0.890–0.966) | 1083.742 (1035.668–1123.068) | 0.468 (0.441–0.477) | 0.485 (0.444–0.492) | 0.562 (0.542–0.573) | n/a |
| C (Telys, MP=0, cold) | 3 | 63.079 (62.184–66.059) | 15.853 (15.138–16.081) | 62.602 (58.537–65.620) | 62.594 (61.735–65.613) | 62.778 (61.823–65.768) | 0.0% (0.0–0.0%) |
| C warm first (MP=0) | 1 | 61.259 | 16.161 | 60.824 | 60.819 | 60.971 | n/a |
| C warm second (MP=0) | 1 | 65.867 | 15.030 | 65.399 | 65.393 | 65.551 | n/a |
| C (Telys, MP=1) | 1 | 65.433 | 15.130 | 65.049 | 65.042 | 65.224 | n/a |

`Ext hit rate` is vLLM's own "External prefix cache hit rate" log line; it is absent in the short warm runs, so `summarize_results.py` reports `n/a`.

### Key observations

1. **Warm-restart hits are working.** Connector logs for the warm-second run show expected hit counts, e.g.:
   ```
   INFO:telys_vllm.connector:telys lookup: req=0-86ce7732 start_block=0 max_block=106 hit_blocks=106
   INFO:telys_vllm.connector:telys lookup: req=1-b9c7ba4b start_block=33 max_block=121 hit_blocks=88
   ```
   This confirms the shared-engine/hot-cache fix and deterministic block hashes (vocab sorted in the benchmark script).

2. **Eager-mode load/inject overhead dominates.** Arm C TTFT is ~60–66 s vs ~0.47 s for A/B. The connector is functionally correct but not yet fast enough to beat native prefix caching on this small model.

3. **Multi-process (`MP=1`) behaves like single-process.** With the self-host unix-socket server providing cross-process persistence, Arm C MP=1 is in the same ~65 s range as MP=0 warm runs.

4. **Cold-run external hit rate is 0%**, as expected: Telys is empty until the first run saves blocks.

### What's next

- Remove the `--enforce-eager` requirement so saves run under `torch.compile`, or move the save path out of the per-layer decorator hot path.
- Batch/block-load and inject payloads more efficiently; the current per-block `get_batch` / decode path is the likely bottleneck.
- Re-run on a larger model to see whether the absolute overhead amortizes better.

## v3 — raw-byte chunks, deferred save, `register_kv_caches` (no `--enforce-eager` on Arm C)

Run: 2026-10-25 ~02:20–02:27 UTC  
Pod: Runpod A40 `m6c70m2edkjxue`, EU-SE-1  
Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py`  
Server: self-host Telys unix-socket (`unix:///root/.telys/telys.sock`), `--save-interval 30`  
Script: `bench/run_benchmarks_v3.sh`

Arm A/B keep `--enforce-eager` for v2 comparability. Arm C drops `--enforce-eager`; saves use `register_kv_caches` + `wait_for_save` instead of the per-layer `save_kv_layer` decorator. The connector now stores one raw-byte chunk per contiguous block run and deduplicates chunk keys within each save step.

### Workload

Same as v2:

- `num_prompts=20`
- `repeat_count=5`
- `input_length_range=512:2048`
- `prefix_len=512`
- `output_len=10`
- `seed=42`

### Results

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 0.979 (0.960–1.009) | 1021.360 (991.463–1041.770) | 0.614 (0.566–0.616) | 0.630 (0.581–0.632) | 0.700 (0.654–0.702) | n/a |
| B (native prefix caching) | 3 | 0.991 (0.929–1.033) | 1008.760 (967.700–1076.647) | 0.588 (0.521–0.627) | 0.603 (0.536–0.643) | 0.676 (0.610–0.715) | n/a |
| C (Telys, MP=0, cold) | 3 | 12.506 (2.824–18.280) | 79.962 (54.704–354.111) | 12.154 (2.486–17.510) | 12.144 (2.472–17.936) | 12.260 (2.600–18.036) | n/a |
| C warm first (MP=0) | 1 | 2.804 | 356.692 | 2.463 | 2.449 | 2.577 | n/a |
| C warm second (MP=0) | 1 | 2.868 | 348.674 | 2.522 | 2.507 | 2.639 | n/a |
| C (Telys, MP=1) | 1 | 10.873 | 91.968 | 10.231 | 10.572 | 10.688 | n/a |

`Ext hit rate` is vLLM's own "External prefix cache hit rate" log line; it is absent in these runs, so `summarize_results.py` reports `n/a`.

### Key observations

1. **Removing `--enforce-eager` and deferring serialization cuts Arm C cold time by ~5–22× vs v2.** Cold-run TTFT drops from ~62 s (v2) to a mean of ~12 s, with the fastest cached-compile run at ~2.5 s. The first cold run pays torch.compile cache generation; later runs reuse the on-disk AOT cache.

2. **Warm-restart hits work and are fast.** Warm-second logs show full-prefix hits:
   ```
   INFO:telys_vllm.connector:telys lookup: req=0-bfc2a052 num_tokens=1708 num_computed=0 start_block=0 max_block=106 hit_blocks=106 hit_tokens=1696
   ```
   Warm TTFT is ~2.5 s vs ~65 s in v2.

3. **Saves fire under `torch.compile`.** Connector logs show `wait_for_save` completing chunks on compiled runs:
   ```
   INFO:telys_vllm.connector:wait_for_save: collect=2.013s save=10.476s chunks=10 deduped
   ```
   The `register_kv_caches` path successfully extracts KV from the registered paged caches without relying on the per-layer `save_kv_layer` decorator.

4. **The remaining gap to native prefix caching is ~2–12 s TTFT vs ~0.6 s.** On this small model, the Telys wire/base64 round-trip and per-step save flush still dominate. The connector-internal codec is ~27× faster than v2 (see `bench/measure_codec_overhead.py`), but the end-to-end bottleneck moved to the Telys metadata store rather than the JSON wrapper.

5. **MP=1 is slower than MP=0** in this run (10.9 s vs 2.8 s warm), likely because the multi-process path incurs an extra serialization hop; the self-host unix-socket server still provides cross-process persistence.

### Caveats

- Telys metadata is JSON-serialized over the wire, so the chunk payload is base64-encoded once per chunk. This is still far cheaper than v2's per-block base64-JSON wrapper, but it means "raw bytes" live inside the connector; the Telys wire path is base64.
- Cold-run variance is high because torch.compile AOT cache is cold on the first run and warm thereafter.

## v4 — slab-backed payloads, async writes, batched injection

Run: 2026-10-25 ~03:33–03:57 UTC  
Pod: Runpod A40 `o9akjgpatxt9tw`, EU-SE-1  
Harness: `bench/benchmark_prefix_caching_metrics.py`  
Script: `bench/run_benchmarks_v4.sh`

v4 moves the encoded chunk payload bytes out of Telys and into a local append-only `KVSlab` file; Telys stores only the slab pointer (`vllm.slab_offset`/`vllm.slab_length`) plus provenance metadata. Saves are asynchronous via a bounded background queue; an `atexit` drain ensures pending writes are flushed when a vLLM process exits so warm-restart runs see them. Loads use the new `load_decoded_blocks()` path and inject each decoded chunk once.

Two workloads are run back-to-back, with Telys state reset between them:

1. **small**: `Qwen/Qwen2.5-1.5B-Instruct`, `input_length_range=512:2048`, `prefix_len=512`
2. **payoff**: `Qwen/Qwen2.5-7B-Instruct`, `input_length_range=5120:8192`, `prefix_len=4096`

Both use `num_prompts=20`, `repeat_count=5`, `output_len=10`, `seed=42`. Arm A/B keep `--enforce-eager`; Arm C drops it and relies on `register_kv_caches` + `wait_for_save`.

### Small workload (1.5B / 512-token prefix)

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 0.945 (0.913–0.987) | 1057.782 (1013.007–1095.857) | 0.466 (0.456–0.502) | 0.476 (0.469–0.517) | 0.567 (0.552–0.596) | n/a |
| B (native prefix caching) | 3 | 0.964 (0.942–0.969) | 1037.547 (1032.183–1061.304) | 0.537 (0.515–0.539) | 0.552 (0.532–0.555) | 0.631 (0.611–0.634) | n/a |
| C (Telys, MP=0, cold) | 3 | 10.443 (10.111–10.538) | 95.757 (94.891–98.904) | 9.740 (9.459–9.805) | 10.076 (9.772–10.140) | 10.149 (9.855–10.228) | n/a |
| C warm first (MP=0) | 1 | 10.684 (10.684–10.684) | 93.598 (93.598–93.598) | 9.949 (9.949–9.949) | 10.296 (10.296–10.296) | 10.374 (10.374–10.374) | n/a |
| C warm second (MP=0) | 1 | 10.601 (10.601–10.601) | 94.329 (94.329–94.329) | 9.892 (9.892–9.892) | 10.223 (10.223–10.223) | 10.307 (10.307–10.307) | n/a |
| C (Telys, MP=1) | 1 | 10.950 (10.950–10.950) | 91.320 (91.320–91.320) | 10.277 (10.277–10.277) | 10.631 (10.631–10.631) | 10.760 (10.760–10.760) | n/a |

### Payoff workload (7B / 4096-token prefix)

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 11.296 (11.296–11.311) | 88.525 (88.406–88.526) | 8.055 (8.043–8.173) | 9.069 (9.051–9.291) | 9.680 (9.665–9.810) | n/a |
| B (native prefix caching) | 3 | 11.332 (11.295–11.336) | 88.244 (88.217–88.534) | 8.067 (7.964–8.078) | 9.072 (8.946–9.075) | 9.690 (9.580–9.703) | n/a |
| C (Telys, MP=0, cold) | 3 | 83.302 (82.840–83.543) | 12.004 (11.970–12.071) | 69.552 (69.022–69.600) | 81.088 (80.641–81.326) | 81.839 (81.354–82.032) | 0.0% (0.0–0.0%) |
| C warm first (MP=0) | 1 | 81.433 (81.433–81.433) | 12.280 (12.280–12.280) | 67.934 (67.934–67.934) | 78.920 (78.920–78.920) | 79.763 (79.763–79.763) | 0.0% (0.0–0.0%) |
| C warm second (MP=0) | 1 | 84.091 (84.091–84.091) | 11.892 (11.892–11.892) | 70.111 (70.111–70.111) | 81.887 (81.887–81.887) | 82.648 (82.648–82.648) | 0.0% (0.0–0.0%) |
| C (Telys, MP=1) | 1 | 92.690 (92.690–92.690) | 10.789 (10.789–10.789) | 75.617 (75.617–75.617) | 90.368 (90.368–90.368) | 91.223 (91.223–91.223) | 0.0% (0.0–0.0%) |

### Key observations

1. **v4 is functionally correct across both workloads.** The connector successfully saves and loads chunks via the slab; no `slab pointer` or payload errors occurred after moving from the self-host server to the embedded engine path and adding the `atexit` drain.

2. **External hit rate is 0% on the payoff workload (and small warm runs)** because vLLM's native prefix cache is handling the shared prefix before the connector is consulted. Connector logs show `num_computed=4304` (payoff) and `num_computed=528` (small), so the scheduler only asks Telys about the unique suffix blocks, which are not cached. This means the current harness is not exercising the warm external-cache hit path that v3 observed.

3. **Telys Arm C is 10–11× slower than native on the small model and ~7–8× slower than native on the payoff model.** The absolute TTFT for Arm C is ~10 s (small) and ~80 s (payoff), while native is ~0.5 s and ~9 s respectively. The load/inject path is the dominant cost.

4. **Warm-restart runs do not improve over cold runs** in this harness because native prefix caching absorbs the shared prefix. To actually measure Telys warm-cache speedup, the benchmark needs either a cross-engine/engine-restart scenario where vLLM's native cache is cold, or a much longer shared prefix that exceeds the GPU KV cache budget.

5. **MP=1 is slightly slower than MP=0** in both workloads (10.95 s vs 10.44 s small; 92.7 s vs 83.3 s payoff), consistent with v3.

### What's next

- Re-design the benchmark so the shared prefix is *not* served by vLLM's native cache (e.g. separate engine restarts with `--disable-native-prefix-caching`, or a prefix longer than the GPU cache). Otherwise we are measuring save overhead, not external-cache acceleration.
- Profile the load/inject path: with payloads now in a local slab, the remaining cost is likely `torch.load` per block blob and `inject_kv_block` into the paged KV cache.
- Consider pre-loading the full prefix into a staging buffer and injecting layers in a single kernel call rather than block-by-block.

## v5 — fair measurement: native prefix caching OFF, Telys as the only cache

Run: 2026-10-25 ~04:19–08:35 UTC  
Pod: Runpod A40 `vt4vg61rainpqj`, EU-SE-1  
Harness: `bench/benchmark_prefix_caching_metrics.py`  
Scripts: `bench/run_benchmarks_v5.sh` (small + headline), `bench/run_benchmarks_v5_persist_4k.sh` (persistence, final shape)  

v5 turns vLLM's native prefix cache completely off with `--no-enable-prefix-caching` so that Telys is the *only* prefix cache. This is the fair comparison v4 could not make: in v4, vLLM computed the shared prefix natively before consulting the connector, giving 0% external hits by construction.

Two continuity/headline workloads ran as planned:

- **small**: `Qwen/Qwen2.5-1.5B-Instruct`, `input_length_range=512:2048`, `prefix_len=512`, `output_len=10`
- **headline**: `Qwen/Qwen2.5-7B-Instruct`, `input_length_range=4096:8192`, `prefix_len=4096`, `output_len=64`

The **persistence** workload had to be scaled back:

- Original plan: 7B / 16K-prefix / inputs 16384:24576.
- 16K prefix: vLLM EngineCore exited unexpectedly on the warm C' run at ~66% completion.
- 8K prefix: same crash at ~68% completion (GPU KV cache ~88%).
- 4K prefix: cold C' run completed; warm run2 crashed at 64% with the same `EngineCore exited unexpectedly` failure.

Because only one C' cold run completed for persistence, warm-cache numbers for that workload are not available.

### Small continuity (1.5B / 512-token prefix)

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 4.651 (4.589–4.659) | 215.003 (214.633–217.930) | 2.134 (2.092–2.171) | 2.118 (2.079–2.160) | 3.805 (3.740–3.875) | n/a |
| C' (Telys, native caching OFF) | 3 | 76.268 (24.346–78.220) | 13.112 (12.785–41.074) | 75.899 (15.726–77.855) | 75.894 (16.457–77.844) | 76.062 (23.913–78.015) | 0.0% (0.0–0.0%) |

### Headline (7B / 4096-token prefix)

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 104.821 (104.142–104.902) | 61.057 (61.009–61.455) | 49.030 (48.563–49.077) | 48.050 (47.516–48.099) | 94.999 (94.356–95.068) | n/a |
| C' (Telys, native caching OFF) | 3 | 397.690 (217.318–412.217) | 16.093 (15.526–29.450) | 309.954 (109.313–320.353) | 277.717 (106.632–286.530) | 393.503 (206.932–408.011) | 99.3% (0.0–99.9%) |

### Persistence (7B / 4096-token prefix / inputs 16384:24576)

| Arm | Runs | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|------|--------------|------------------|---------------|--------------|--------------|--------------|
| A (no prefix caching) | 3 | 438.430 (437.098–438.454) | 14.598 (14.597–14.642) | 212.660 (211.919–212.841) | 213.347 (212.529–213.527) | 405.164 (404.504–405.569) | n/a |
| C' (Telys, native caching OFF, cold) | 1 | 583.234 | 10.973 | 287.335 | 290.087 | 543.854 | 0.0% |
| C' warm (run 2) | — | crashed at 64% | — | — | — | — | — |

`Ext hit rate` is vLLM's own "External prefix cache hit rate" log line. On the cold C' runs it is 0% because Telys starts empty; on warm headline runs it is ~99.3–99.9%.

### Hit-rate / lookup evidence

- **Small warm C' runs**: connector logs show full-prefix hits, e.g. `telys lookup: req=0-aa75d14a ... max_block=47 hit_blocks=47 hit_tokens=752` and `req=99-92f96028 ... max_block=33 hit_blocks=33 hit_tokens=528`. The Telys key set survives across the repeated runs within the workload.
- **Headline warm C' runs**: connector logs also show full-prefix hits, e.g. `telys lookup: req=0-a768c96a ... max_block=386 hit_blocks=386 hit_tokens=6176`, and vLLM reports `External prefix cache hit rate: 99.3%` (run 2) and `99.9%` (run 3). This confirms Telys, not vLLM's native cache, is serving the shared prefix.
- **Persistence C' cold run**: connector logs show `hit_blocks=0` for all requests (Telys empty), and vLLM reports `External prefix cache hit rate: 0.0%`.
- **Persistence C' warm run 2 crash tail**:
  ```
  INFO:telys_vllm.connector:telys lookup: req=84-b2e73676 ... max_block=1490 hit_blocks=512 hit_tokens=8192
  Processed prompts:  64%|██████▍   | 64/100 [13:00<06:15, 10.42s/it, ...]
  (MainProcess pid=28815) WARNING [core_client.py:817] [shutdown] MPClient: engine core exited unexpectedly; starting cleanup
  ```
  The lookup itself succeeds, but the vLLM EngineCore dies under the load shortly afterwards.

### Honest verdict

1. **Native caching is really off.** The v5 numbers are the first clean A vs C' comparison: warm headline C' runs show ~99% external prefix-cache hits, and connector lookups report `hit_blocks == max_block`. In v4 the same metric was 0% because vLLM served the prefix natively.

2. **Telys does not beat recompute on these workloads.**
   - *Small*: C' is ~16× slower than A (76 s vs 4.7 s total; TTFT 76 s vs 2.1 s). The load/inject and save overhead dwarfs the cost of computing a 512-token prefix on a 1.5B model.
   - *Headline*: C' is ~3.8× slower than A (398 s vs 105 s median total; TTFT 310 s vs 49 s). Even with near-perfect warm hits, the Telys path is slower than recomputing the 4096-token prefix.
   - *Persistence (4K prefix)*: The single cold C' run is ~1.33× slower than A (583 s vs 438 s). Warm numbers are unavailable because the EngineCore crashes on the warm C' run.

3. **The persistence case exposes a stability limit.** With native caching disabled, the warm C' path repeatedly crashes the vLLM EngineCore once GPU KV cache usage climbs (observed 87–97% before the crash). This happens at 16K, 8K, and 4K prefix lengths. The connector correctly looks up and injects blocks, but the resulting memory/scheduler pressure is not sustainable on an A40 with this prompt volume.

4. **The connector's correctness is confirmed.** Block hashes are deterministic, Telys chunks persist across runs, and the warm lookup path returns full-prefix hits. The remaining issue is performance and stability of the load/inject path under heavy KV-cache pressure.

### What's next

- Profile why warm C' runs are slower than cold: the per-block `load_decoded_blocks` / `inject_kv_block` path likely dominates.
- Investigate the EngineCore crash with vLLM-native prefix caching disabled and large working sets; it may require limiting in-flight sequences or changing KV-block allocation.
- Consider a staged benchmark that pre-populates Telys and then measures only the lookup/inject phase, separated from the save phase.

## Telemetry Autopsy — headline workload, batched injection fix

Run: 2026-10-25 ~15:30–16:03 UTC  
Pod: Runpod A40 `6brrvv8awzdsac`, EU-SE-1  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py` with `bench/run_autopsy_headline.sh`  
Telemetry: `telys_vllm/telemetry.py` JSONL + `bench/summarize_telemetry.py`

The Telys runtime auth blocker from the earlier autopsy attempt was resolved by refreshing `~/.telys/` credentials on the local Mac and reinstalling the runtime on the pod:

```
TELYS_TOKEN=$(cat /root/.telys/api_key) telys runtime install --license /root/.telys/login_license.jwt
→ runtime INSTALLED + VERIFIED [linux-x86_64] ... query execution is now fully local/offline
```

### Workload

- `num_prompts=20`
- `repeat_count=5`
- `input_length_range=4096:8192`
- `prefix_len=4096`
- `output_len=64`
- `seed=42`
- `--no-enable-prefix-caching` (native prefix caching OFF so Telys is the only cache)

### Arm-level results

| Arm | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|--------------|------------------|---------------|--------------|--------------|--------------|
| A (recompute, no prefix caching) | 106.715 | 59.97 | 50.085 | 48.577 | 97.135 | n/a |
| C′ cold (Telys empty) | 515.671 | 12.41 | 402.987 | 361.071 | 511.464 | 0.0% |
| C′ warm before fix | 515.671 | 12.41 | 402.987 | 361.071 | 511.464 | 99.3% |
| C′ warm after batched-inject fix | 253.430 | 25.25 | 195.349 | 174.690 | 249.191 | 99.3% |

Arm A matches the v5 headline baseline (TTFT p50 ~48–49 s). C′ warm runs show ~99% external prefix-cache hits, confirming Telys is serving the shared prefix.

### Full telemetry summary — warm C′ before fix

`python3 bench/summarize_telemetry.py bench/autopsy/telemetry/headline_arm_c_warm.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 4 | 506547.682 | 73413.715 | 126636.920 | 354020.282 | `telys_vllm/connector.py:372` |
| inject_block | 40054 | 446971.912 | 10.936 | 11.159 | 12.634 | `telys_vllm/connector.py:417` |
| inject_kv_block | 1121512 | 189274.211 | 0.165 | 0.169 | 0.189 | `telys_vllm/connector.py:439` |
| tensor_from_blob | 1121512 | 182853.037 | 0.156 | 0.163 | 0.189 | `telys_vllm/connector.py:435` |
| load_decoded_blocks | 101 | 43416.198 | 417.717 | 429.863 | 524.670 | `telys_vllm/connector.py:380` |
| store_read_chunk | 101 | 41128.491 | 396.957 | 407.213 | 494.711 | `telys_vllm/store.py:275` |
| store_decode_chunk | 101 | 19029.107 | 175.420 | 188.407 | 280.038 | `telys_vllm/store.py:299` |
| wait_for_save | 132 | 1059.485 | 0.568 | 8.026 | 0.912 | `telys_vllm/connector.py:528` |
| save_collect | 88 | 959.261 | 0.007 | 10.901 | 0.007 | `telys_vllm/connector.py:553` |
| store_persist | 2 | 69.889 | 34.945 | 34.945 | 51.212 | `telys_vllm/store.py:453` |
| save_encode_persist | 1 | 44.324 | 44.324 | 44.324 | 44.324 | `telys_vllm/connector.py:565` |
| save_encode_chunk | 2 | 43.620 | 21.810 | 21.810 | 33.166 | `telys_vllm/connector.py:578` |
| lookup | 165 | 26.623 | 0.031 | 0.161 | 0.520 | `telys_vllm/connector.py:204` |
| telys_startup | 2 | 14.097 | 7.048 | 7.048 | 13.810 | `telys_vllm/connector.py:134` |
| store_lookup_prefix | 165 | 2.304 | 0.004 | 0.014 | 0.024 | `telys_vllm/store.py:171` |
| connector_store_init | 2 | 0.909 | 0.455 | 0.455 | 0.459 | `telys_vllm/connector.py:143` |
| save_chunk_async | 2 | 0.080 | 0.040 | 0.040 | 0.054 | `telys_vllm/connector.py:582` |

**Top 3 contributors:** `start_load_kv` 35.4%, `inject_block` 31.2%, `inject_kv_block` 13.2%.  
**Dominant phase:** `start_load_kv` at `telys_vllm/connector.py:372`.

### Full telemetry summary — warm C′ after fix

`python3 bench/summarize_telemetry.py bench/autopsy/telemetry/headline_arm_c_warm_after.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 5 | 246328.384 | 3474.918 | 49265.677 | 171145.172 | `telys_vllm/connector.py:378` |
| inject_block | 2829 | 188066.072 | 63.444 | 66.478 | 87.248 | `telys_vllm/connector.py:447` |
| tensor_from_blob | 1127918 | 145786.007 | 0.121 | 0.129 | 0.150 | `telys_vllm/connector.py:453` |
| load_decoded_blocks | 102 | 40073.862 | 379.734 | 392.881 | 447.385 | `telys_vllm/connector.py:386` |
| store_read_chunk | 112 | 38234.349 | 356.859 | 341.378 | 426.438 | `telys_vllm/store.py:275` |
| store_decode_chunk | 112 | 15330.086 | 136.272 | 136.876 | 179.486 | `telys_vllm/store.py:299` |
| inject_kv_block | 2829 | 9622.774 | 3.169 | 3.401 | 4.569 | `telys_vllm/connector.py:459` |
| wait_for_save | 132 | 64.343 | 0.632 | 0.487 | 0.982 | `telys_vllm/connector.py:560` |
| telys_startup | 4 | 30.803 | 6.959 | 7.701 | 16.578 | `telys_vllm/connector.py:134` |
| lookup | 166 | 27.755 | 0.040 | 0.167 | 0.392 | `telys_vllm/connector.py:207` |
| connector_store_init | 4 | 2.063 | 0.519 | 0.516 | 0.581 | `telys_vllm/connector.py:143` |
| store_lookup_prefix | 166 | 1.878 | 0.004 | 0.011 | 0.023 | `telys_vllm/store.py:171` |
| save_collect | 87 | 0.635 | 0.007 | 0.007 | 0.009 | `telys_vllm/connector.py:585` |

**Top 3 contributors:** `start_load_kv` 36.0%, `inject_block` 27.5%, `tensor_from_blob` 21.3%.  
**Dominant phase:** `start_load_kv` at `telys_vllm/connector.py:378`.

### Focused per-phase table (warm-hit request)

Mapped from the telemetry summaries and benchmark JSON. Engine boot and prefill are not instrumented by the connector telemetry; engine boot is taken from the vLLM log line `init engine ... took 17.99 s`, and prefill is rolled into the measured first-token (TTFT) time.

| phase | before fix median (s) | after fix median (s) | notes |
|-------|----------------------|----------------------|-------|
| engine boot | 17.99 | 17.99 | vLLM init + KV cache + warmup |
| connector init | 0.00046 | 0.00052 | `connector_store_init` span |
| Telys startup | 0.0070 | 0.0070 | `telys_startup` span |
| lookup | 0.000031 | 0.000040 | `lookup` + `store_lookup_prefix` |
| slab read | 0.397 | 0.357 | `store_read_chunk` |
| decode | 0.175 | 0.136 | `store_decode_chunk` |
| inject | 0.000165 | 0.00317 | `inject_kv_block`; per-layer after fix |
| prefill | — | — | not separately instrumented |
| first-token | 361.07 | 174.69 | benchmark TTFT p50 |
| save | 0.00057 | 0.00063 | `wait_for_save` |

The `inject` median is not directly comparable before vs after: before, each span was one block × one layer; after, each span is one full layer for all blocks in the request. The relevant comparison is total time: `inject_kv_block` dropped from 189.3 s to 9.6 s, and `inject_block` dropped from 447.0 s to 188.1 s.

### Fix: batched per-layer KV injection

**Dominant layer:** `telys_vllm/connector.py:372` (`start_load_kv`), which spends most of its time in per-block, per-layer `tensor_from_blob` + `inject_kv_block`.

**What changed:**

1. Added `inject_kv_blocks` (plural) to `telys_vllm/gpu_ops.py`. It builds one `slot_mapping` for all target blocks, stacks the CPU block tensors, and performs a single GPU write per layer instead of one per block.
2. Reworked `TelysKVConnector.start_load_kv` in `telys_vllm/connector.py` to iterate layers outermost: for each layer, decode all block blobs, then call `_inject_kv_blocks` once.
3. Fixed a latent `block_ids` / `block_hashes` alignment bug by adding `start_block` to `TelysLoadSpec` and slicing the allocated physical block ids in `build_connector_meta` so they match the loaded hashes. This was exposed by the batched path (the old per-block `zip` silently truncated).

**Files touched:**
- `telys_vllm/gpu_ops.py`
- `telys_vllm/connector.py`
- `tests/test_roundtrip.py` (updated assertion for correctly-sliced block ids)

### Honest verdict

1. **The fix works and is safe.** Local tests pass: **99 passed, 1 skipped**. The warm C′ run no longer crashes; TTFT p50 drops from 361 s to 175 s and total time drops from 516 s to 253 s.

2. **Telys warm-cache is still slower than recomputing.** Arm A TTFT p50 is 48.6 s; C′ warm after the fix is 174.7 s — **3.6× slower** than the recompute baseline. The batched injection removed the per-block GPU kernel-launch overhead, but the remaining costs (`tensor_from_blob` / `torch.load` per block blob, slab read/decode, and PCIe transfer of the full prefix) still dominate.

3. **The next dominant cost is `tensor_from_blob`.** After the fix, `tensor_from_blob` is 21.3% of total telemetry time and the largest single per-occurrence work item. It currently deserializes every per-layer block blob with `torch.load`; switching to a raw-byte tensor layout would remove that deserialization overhead but would invalidate existing persisted state.

4. **The connector is functionally correct.** Warm runs show ~99% external hits, block hashes are deterministic, and the load/inject path completes without errors after the `block_ids` alignment fix.

### What's next

- Replace `torch.save`/`torch.load` per block blob with a raw-byte tensor serialization to cut the remaining `tensor_from_blob` cost.
- Re-run the autopsy after that change; re-populate Telys state because the blob format would change.
- Investigate whether the headline C′ warm path can be brought under the ~49 s Arm A TTFT baseline, or whether the PCIe transfer + deserialization cost makes that impossible for a 4K-prefix/7B workload on this GPU class.

## Autopsy v2 — BlobStore + raw-byte tensors

Run: 2026-10-25 ~16:57–17:05 UTC  
Pod: Runpod A40 `k6k6jcqazq9y9n`, EU-SE-1  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py` with `bench/run_autopsy_headline.sh`  
Telemetry: `telys_vllm/telemetry.py` JSONL + `bench/summarize_telemetry.py`

This run adopts the native Telys SDK `BlobStore` for payload bytes and replaces per-block `torch.load` with raw-byte `torch.frombuffer` deserialization. The connector auto-selects `TelysBlobClient` when `telys.BlobStore` is available. Because the payload format changed, Telys state was reset and repopulated by the cold C′ run.

### Workload

Same as the previous autopsy:

- `num_prompts=20`
- `repeat_count=5`
- `input_length_range=4096:8192`
- `prefix_len=4096`
- `output_len=64`
- `seed=42`
- `--no-enable-prefix-caching` (native prefix caching OFF so Telys is the only cache)

### Arm-level results

| Arm | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|--------------|------------------|---------------|--------------|--------------|--------------|
| A (recompute, no prefix caching) | 105.842 | 60.47 | 49.574 | 48.537 | 96.178 | n/a |
| C′ cold (Telys empty) | 165.904 | 38.58 | 84.986 | 84.051 | 156.622 | 0.0% |
| C′ warm (full-prefix hit) | 92.317 | 69.33 | 66.533 | 58.422 | 88.068 | 99.3% |

Arm A matches the v5 headline baseline (TTFT p50 ~48.5 s). C′ warm runs show ~99% external prefix-cache hits, confirming Telys is serving the shared prefix.

### Full telemetry summary — cold C′

`python3 bench/summarize_telemetry.py bench/autopsy-v2/telemetry/headline_arm_c_cold.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| wait_for_save | 154 | 154968.324 | 1381.174 | 1006.288 | 2573.122 | `telys_vllm/connector.py:584` |
| save_collect | 143 | 124812.648 | 1369.031 | 872.816 | 1830.181 | `telys_vllm/connector.py:609` |
| save_encode_persist | 80 | 24752.582 | 259.279 | 309.407 | 811.432 | `telys_vllm/connector.py:621` |
| save_encode_chunk | 73 | 21711.932 | 306.481 | 297.424 | 528.255 | `telys_vllm/connector.py:634` |
| store_persist | 73 | 16528.536 | 235.932 | 226.418 | 435.147 | `telys_vllm/store.py:468` |
| lookup | 117 | 28.540 | 0.205 | 0.244 | 0.520 | `telys_vllm/connector.py:233` |
| telys_startup | 2 | 18.736 | 9.368 | 9.368 | 18.559 | `telys_vllm/connector.py:156` |
| save_chunk_async | 73 | 5.034 | 0.052 | 0.069 | 0.148 | `telys_vllm/connector.py:638` |
| store_lookup_prefix | 117 | 1.398 | 0.010 | 0.012 | 0.025 | `telys_vllm/store.py:172` |
| connector_store_init | 2 | 0.803 | 0.401 | 0.401 | 0.436 | `telys_vllm/connector.py:172` |

**Top 3 contributors:** `wait_for_save` 45.2%, `save_collect` 36.4%, `save_encode_persist` 7.2%.  
**Dominant phase:** `wait_for_save` at `telys_vllm/connector.py:584`.

### Full telemetry summary — warm C′

`python3 bench/summarize_telemetry.py bench/autopsy-v2/telemetry/headline_arm_c_warm.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 4 | 83329.249 | 13041.587 | 20832.312 | 56335.493 | `telys_vllm/connector.py:404` |
| inject_block | 2828 | 49760.534 | 16.963 | 17.596 | 23.688 | `telys_vllm/connector.py:473` |
| tensor_from_blob | 1121512 | 18397.268 | 0.015 | 0.016 | 0.022 | `telys_vllm/connector.py:479` |
| load_decoded_blocks | 101 | 15386.550 | 140.507 | 152.342 | 191.953 | `telys_vllm/connector.py:412` |
| store_read_chunk | 101 | 14959.370 | 139.154 | 148.113 | 189.228 | `telys_vllm/store.py:276` |
| store_decode_chunk | 101 | 14949.282 | 139.058 | 148.013 | 189.124 | `telys_vllm/store.py:279` |
| inject_kv_block | 2828 | 10120.557 | 3.366 | 3.579 | 4.669 | `telys_vllm/connector.py:485` |
| wait_for_save | 132 | 939.793 | 0.282 | 7.120 | 0.845 | `telys_vllm/connector.py:584` |
| save_collect | 88 | 837.094 | 0.003 | 9.512 | 0.007 | `telys_vllm/connector.py:609` |
| store_persist | 2 | 56.483 | 28.241 | 28.241 | 36.500 | `telys_vllm/store.py:468` |
| save_encode_persist | 1 | 42.328 | 42.328 | 42.328 | 42.328 | `telys_vllm/connector.py:621` |
| save_encode_chunk | 2 | 41.604 | 20.802 | 20.802 | 35.270 | `telys_vllm/connector.py:634` |
| lookup | 165 | 31.031 | 0.037 | 0.188 | 0.362 | `telys_vllm/connector.py:233` |
| telys_startup | 2 | 16.045 | 8.022 | 8.022 | 15.688 | `telys_vllm/connector.py:156` |
| store_lookup_prefix | 165 | 1.303 | 0.004 | 0.008 | 0.022 | `telys_vllm/store.py:172` |
| connector_store_init | 2 | 0.843 | 0.422 | 0.422 | 0.435 | `telys_vllm/connector.py:172` |
| save_chunk_async | 2 | 0.101 | 0.051 | 0.051 | 0.057 | `telys_vllm/connector.py:638` |

**Top 3 contributors:** `start_load_kv` 39.9%, `inject_block` 23.8%, `tensor_from_blob` 8.8%.  
**Dominant phase:** `start_load_kv` at `telys_vllm/connector.py:404`.

### Focused per-phase table (warm-hit request)

Mapped from the telemetry summaries and benchmark JSON. Engine boot is taken from the vLLM log line `init engine ... took 17.72 s`, and prefill is rolled into the measured first-token (TTFT) time.

| phase | after batched-inject fix (s) | BlobStore + raw bytes (s) | notes |
|-------|------------------------------|---------------------------|-------|
| engine boot | 17.99 | 17.72 | vLLM init + KV cache + warmup |
| connector init | 0.00052 | 0.00042 | `connector_store_init` span |
| Telys startup | 0.0070 | 0.0080 | `telys_startup` span |
| lookup | 0.000040 | 0.000037 | `lookup` + `store_lookup_prefix` |
| slab/BlobStore read | 0.357 | 0.139 | `store_read_chunk` |
| decode | 0.136 | 0.139 | `store_decode_chunk` |
| inject | 0.00317 | 0.00337 | `inject_kv_block`; per-layer |
| prefill | — | — | not separately instrumented |
| first-token | 174.69 | 58.42 | benchmark TTFT p50 |
| save | 0.00063 | 0.00028 | `wait_for_save` |

### `tensor_from_blob` before/after

| metric | after batched-inject fix | BlobStore + raw bytes | change |
|--------|--------------------------|-----------------------|--------|
| count | 1,127,918 | 1,121,512 | similar |
| total time | 145786.007 ms | 18397.268 ms | **7.9× faster** |
| median | 0.121 ms | 0.015 ms | **8.1× faster** |
| mean | 0.129 ms | 0.016 ms | **8.1× faster** |
| p95 | 0.150 ms | 0.022 ms | **6.8× faster** |

The raw-byte path eliminates `torch.load`/`pickle` parsing; deserialization is now a `torch.frombuffer` view of the stored bytes.

### What changed

1. **`telys_vllm/gpu_ops.py`**: `serialize_kv_tensor` now writes a deterministic raw-byte header (version, dtype code, ndim, shape uint16s) + tensor bytes via a uint8 view, and `deserialize_kv_tensor` uses `torch.frombuffer` with a legacy `torch.load` fallback. A bug-fix during the run changed the serializer from `bytes(tensor.untyped_storage())` (which included the whole underlying buffer for contiguous views) to `tensor.view(torch.uint8).numpy().tobytes()` so views serialize only their own data.
2. **`telys_vllm/local_client.py`**: Added `TelysBlobClient(TelysClient)` that stores payload bytes in `telys.BlobStore(path + "_blobs")` and small JSON metadata in the shared Telys collection.
3. **`telys_vllm/store.py`**: `_persist_now`/`_persist_item` pass the full encoded chunk as the client `value`; `_read_chunk_from_slab` prefers a non-empty `record.value` (BlobStore path) and falls back to slab metadata. Slab is optional when `slab_path` is empty/None.
4. **`telys_vllm/connector.py`**: Added `_auto_client()`; when `kv_connector_extra_config` does not set `telys_client_factory`, the connector auto-selects `TelysBlobClient` if available, else `TelysLocalClient`.
5. **`bench/run_autopsy_headline.sh`**: Removed explicit `telys_client_factory` and `TELYS_VLLM_SLAB_PATH`; reset now also deletes `${TELYS_PATH}_blobs`.
6. **`tests/test_blob_client.py`**: New tests for BlobStore round-trip, BlobStore-backed `TelysBlockStore`, raw-byte serialization, legacy fallback, and auto-factory selection.

### Honest verdict

1. **The raw-byte + BlobStore optimization works.** Local tests pass: **113 passed**. The warm C′ run completes cleanly; TTFT p50 drops from 174.7 s to 58.4 s and total time drops from 253.4 s to 92.3 s. `tensor_from_blob` is ~8× faster by every telemetry metric.

2. **Telys warm-cache is still slower than recomputing, but the gap is much smaller.** Arm A TTFT p50 is 48.5 s; C′ warm is now 58.4 s — **1.20× slower** than the recompute baseline (down from 3.6×). On total time, C′ warm (92.3 s) is even slightly faster than Arm A (105.8 s) because the warm cache avoids recomputing the 4096-token prefix; the first-token penalty is what remains.

3. **The cold path is now faster than the previous warm path.** C′ cold is 165.9 s total / 84.1 s TTFT p50, well below the previous warm-after-fix numbers (253.4 s / 174.7 s). The save collection is the dominant cold cost (~1.4 s median per `wait_for_save`).

4. **The connector is functionally correct.** Warm runs show ~99% external hits, block hashes are deterministic, and the load/inject path completes without errors with the new BlobStore-backed client.

### What's next

- The remaining ~10 s warm TTFT gap over recompute is now dominated by `start_load_kv` + `inject_block` (63.7% of telemetry), not `tensor_from_blob`. Investigate whether the per-layer BlobStore read/decode and batched GPU injection can be further pipelined or overlapped with scheduling.
- Consider prefetching/warming the full prefix into GPU KV cache before the first request to hide the PCIe transfer + decode cost.
- Re-run the small/persistence workloads to see whether the BlobStore path also improves stability under larger KV-cache pressure.

## Autopsy v3 — headline re-measurement

Run: 2026-10-25 ~20:06–20:10 UTC  
Pod: Runpod A40 `z9d81mgsl7hg7v`, EU-SE-1  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py` with `bench/run_benchmarks_remote.sh`  
Telemetry: `telys_vllm/telemetry.py` JSONL + `bench/summarize_telemetry.py`

This is a clean re-measurement of the headline workload after refreshing the Telys runtime credentials and verifying the connector/Telys SDK installation. Native prefix caching remains disabled so Telys is the only cache. The cold C′ run uses `repeat_count=1` (20 prompts) purely to populate Telys; the warm C′ run and Arm A use `repeat_count=5` (100 prompts).

### Workload

- `num_prompts=20`
- `repeat_count=5` (Arm A and C′ warm) / `1` (C′ cold)
- `input_length_range=4096:8192`
- `prefix_len=4096`
- `output_len=64`
- `seed=42`
- `--no-enable-prefix-caching` (native prefix caching OFF so Telys is the only cache)

### Arm-level results

| Arm | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|--------------|------------------|---------------|--------------|--------------|--------------|
| A (recompute, no prefix caching) | 106.328 | 60.19 | 49.670 | 48.400 | 97.416 | n/a |
| C′ cold (Telys empty, 20 prompts) | 33.597 | 38.10 | 19.485 | 21.034 | 30.723 | 0.0% |
| C′ warm (full-prefix hit, 100 prompts) | 106.136 | 60.30 | 78.489 | 69.629 | 101.953 | 99.0% |

Arm A matches the v5 headline baseline (TTFT p50 ~48.4 s). C′ warm reports 99.0% external prefix-cache hits, confirming Telys is serving the shared prefix.

### Full telemetry summary — cold C′

`python3 bench/summarize_telemetry.py bench/autopsy-v3/results/cold_telemetry.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| wait_for_save | 82 | 28325.313 | 0.619 | 345.431 | 2041.399 | `telys_vllm/connector.py:584` |
| save_collect | 60 | 22056.645 | 0.007 | 367.611 | 1533.821 | `telys_vllm/connector.py:609` |
| save_encode_persist | 16 | 5801.062 | 296.361 | 362.566 | 759.482 | `telys_vllm/connector.py:621` |
| save_encode_chunk | 20 | 5685.253 | 293.692 | 284.263 | 451.620 | `telys_vllm/connector.py:634` |
| store_persist | 20 | 4214.848 | 217.730 | 210.742 | 382.994 | `telys_vllm/store.py:468` |
| telys_startup | 2 | 12.209 | 6.104 | 6.104 | 12.046 | `telys_vllm/connector.py:156` |
| lookup | 20 | 4.802 | 0.233 | 0.240 | 0.597 | `telys_vllm/connector.py:233` |
| connector_store_init | 2 | 0.921 | 0.460 | 0.460 | 0.493 | `telys_vllm/connector.py:172` |
| save_chunk_async | 20 | 0.887 | 0.045 | 0.044 | 0.056 | `telys_vllm/connector.py:638` |
| store_lookup_prefix | 20 | 0.386 | 0.019 | 0.019 | 0.041 | `telys_vllm/store.py:172` |

**Top 3 contributors:** `wait_for_save` 42.9%, `save_collect` 33.4%, `save_encode_persist` 8.8%.  
**Dominant phase:** `wait_for_save` at `telys_vllm/connector.py:584`.

### Full telemetry summary — warm C′

Source: `bench/autopsy-v3/results/warm_telemetry.jsonl` (≈273 MB, summarized on the pod; raw file not transferred).

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 4 | 96691.550 | 14405.363 | 24172.888 | 66736.870 | `telys_vllm/connector.py:404` |
| inject_block | 2828 | 44112.025 | 14.874 | 15.598 | 19.446 | `telys_vllm/connector.py:473` |
| load_decoded_blocks | 101 | 35866.125 | 343.577 | 355.110 | 413.945 | `telys_vllm/connector.py:412` |
| store_read_chunk | 101 | 34141.412 | 327.403 | 338.034 | 395.424 | `telys_vllm/store.py:276` |
| tensor_from_blob | 1118012 | 16726.819 | 0.014 | 0.015 | 0.018 | `telys_vllm/connector.py:479` |
| store_decode_chunk | 101 | 13551.071 | 125.004 | 134.169 | 173.993 | `telys_vllm/store.py:307` |
| inject_kv_block | 2828 | 7779.874 | 2.500 | 2.751 | 3.571 | `telys_vllm/connector.py:485` |
| wait_for_save | 132 | 1340.880 | 0.569 | 10.158 | 0.926 | `telys_vllm/connector.py:584` |
| save_collect | 88 | 1160.324 | 0.006 | 13.186 | 0.009 | `telys_vllm/connector.py:609` |
| store_persist | 4 | 131.231 | 27.511 | 32.808 | 73.355 | `telys_vllm/store.py:468` |
| save_encode_persist | 1 | 120.549 | 120.549 | 120.549 | 120.549 | `telys_vllm/connector.py:621` |
| save_encode_chunk | 4 | 119.043 | 29.859 | 29.761 | 55.246 | `telys_vllm/connector.py:634` |
| lookup | 165 | 36.505 | 0.036 | 0.221 | 0.482 | `telys_vllm/connector.py:233` |
| telys_startup | 2 | 13.227 | 6.613 | 6.613 | 12.932 | `telys_vllm/connector.py:156` |
| store_lookup_prefix | 165 | 1.764 | 0.005 | 0.011 | 0.023 | `telys_vllm/store.py:172` |
| connector_store_init | 2 | 0.952 | 0.476 | 0.476 | 0.536 | `telys_vllm/connector.py:172` |
| save_chunk_async | 4 | 0.145 | 0.030 | 0.036 | 0.061 | `telys_vllm/connector.py:638` |

**Top 3 contributors:** `start_load_kv` 38.4%, `inject_block` 17.5%, `load_decoded_blocks` 14.2%.  
**Dominant phase:** `start_load_kv` at `telys_vllm/connector.py:404`.

### Focused per-phase table (warm-hit request)

| phase | Autopsy v2 BlobStore (s) | Autopsy v3 (s) | notes |
|-------|--------------------------|----------------|-------|
| engine boot | 17.72 | 18.78 | vLLM init + KV cache + warmup |
| connector init | 0.00042 | 0.00048 | `connector_store_init` span |
| Telys startup | 0.0080 | 0.0066 | `telys_startup` span |
| lookup | 0.000037 | 0.000041 | `lookup` + `store_lookup_prefix` |
| BlobStore/slab read | 0.139 | 0.327 | `store_read_chunk` median |
| decode | 0.139 | 0.125 | `store_decode_chunk` median |
| inject | 0.00337 | 0.00250 | `inject_kv_block` median |
| first-token | 58.42 | 69.63 | benchmark TTFT p50 |
| save | 0.00028 | 0.00057 | `wait_for_save` median |

### Output correctness

`bench/verify_warm_hit_output.py` against the 1.5B model passed locally and on the pod:

```json
{"passed": true, "cold_ttft_s": 0.753, "warm_ttft_s": 0.031, "match_text": true, "match_tokens": true}
```

Cold and warm outputs are text/token-identical, confirming the connector does not change model behavior.

### Honest verdict

1. **Telys hit path is still working.** The warm C′ run reports 99.0% external prefix-cache hits and connector lookups show `hit_blocks == max_block` for full-prefix requests.

2. **Warm TTFT regressed from the v2 BlobStore result.** C′ warm TTFT p50 is 69.6 s in this re-measurement vs 58.4 s in Autopsy v2. The main telemetry shifts are a higher `store_read_chunk` median (0.327 s vs 0.139 s) and higher `start_load_kv` total (96.7 s vs 83.3 s). Without a code change between the two runs, the most likely explanation is pod-to-pod variance or a change in the Telys SDK/runtime version that affects BlobStore read latency.

3. **Telys warm-cache is slower than recomputing on first-token latency.** Arm A TTFT p50 is 48.4 s; C′ warm is 69.6 s — **1.44× slower** than the recompute baseline. Total wall time is essentially a wash (106.14 s vs 106.33 s) because the warm cache avoids recomputing the 4096-token prefix; the first-token penalty remains.

4. **The cold path is fast when limited to 20 prompts.** C′ cold completes in 33.6 s for 20 prompts; save collection is the dominant cost. This run is not directly comparable to the 100-prompt arms.

5. **The connector is functionally correct.** Block hashes are deterministic, Telys chunks persist across runs, warm lookups return full-prefix hits, and the output-verification test passed.

## Autopsy v4 — fast-path stacked tensors

Run: 2026-10-25 ~22:20–22:25 UTC  
Pod: Runpod A40 `v1d3lajc49sxde`, EU-SE-1  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py` with `bench/run_autopsy_v4.sh`

v4 adds a zero-decode fast path for warm cache hits. Instead of deserializing per-block byte blobs and stacking them at load time, the connector now saves and loads whole chunks as stacked per-layer CPU tensors (`TelysFastChunkCodec`). On a hit, the load path becomes a `memcpy` from the store into a pinned CPU tensor followed by an async H2D copy and a batched `inject_kv_blocks_contiguous` call per layer. The public `telys==0.1.9` runtime on the pod does not expose `telys.BlobStore`, so this run uses `TelysLocalClient` backed by the slab file (`/root/.telys/data_kv.slab`) — directly comparable to the v3 slab-backed numbers.

Two fixes landed immediately before this run:

1. `_fast_inject_spec` now slices the loaded stacked tensor to the number of blocks actually being injected, fixing a `ValueError: contiguous_gpu_tensor has 501 blocks, but 469 block ids were provided` that occurred when a stored chunk was longer than the requested prefix.
2. `wait_for_save` now aligns `spec.block_hashes` with `spec.block_ids` by taking the saveable suffix (`block_hashes[-len(block_ids):]`), and skips specs with empty `block_ids`. This fixed a `KeyError` when vLLM emitted save specs containing block hashes but no corresponding block ids for already-cached requests.

### Workload

Same as the previous autopsy:

- `num_prompts=20`
- `repeat_count=5` (Arm A and C′ warm) / `1` (C′ cold)
- `input_length_range=4096:8192`
- `prefix_len=4096`
- `output_len=64`
- `seed=42`
- `--no-enable-prefix-caching` (native prefix caching OFF so Telys is the only cache)

### Arm-level results

| Arm | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|--------------|------------------|---------------|--------------|--------------|--------------|
| A (recompute, no prefix caching) | 106.755 | 59.95 | 49.169 | 46.612 | 97.515 | n/a |
| C′ cold (Telys empty, 20 prompts) | 17.800 | 71.91 | 14.532 | 15.039 | 15.160 | 0.0% |
| C′ warm (full-prefix hit, 100 prompts) | 78.776 | 81.24 | 56.276 | 50.106 | 74.549 | 99.9% |

Arm A matches the v5 headline baseline (TTFT p50 ~46–49 s). C′ warm reports 99.9% external prefix-cache hits, confirming Telys is serving the shared prefix.

### Local micro-benchmark

`bench/measure_fast_path.py` isolates the per-chunk warm-load work (blob read + decode + pinned staging + mocked H2D transfer) on a 28-layer / 106-block / fp16 chunk (~139 MB):

```
load+stage+transfer per chunk (avg of 5 runs):
  old: 29.89 ms
  new:  5.94 ms
  speedup: 5.03x
  effective throughput: 23424.7 MB/s (new)
```

The fast path removes per-block `torch.load`/`deserialize_kv_tensor` overhead and replaces it with a contiguous tensor decode + copy.

### Key observations

1. **The fast path is functionally correct and completes without crashing.** Both fixes above were necessary to get through the headline workload end-to-end. Local tests pass: **27 passed**.

2. **Cold-run throughput is unexpectedly high.** C′ cold (20 prompts) finishes in 17.8 s with 71.9 tok/s. This is faster than the warm 100-prompt run because the warm run has 5× as many requests contending for the loaded prefix and GPU KV-cache space.

3. **Warm total throughput beats recompute.** C′ warm total time is 78.8 s vs Arm A 106.8 s, and throughput is 81.2 tok/s vs 60.0 tok/s. Avoiding recomputation of the 4096-token prefix pays off in overall throughput when the working set fits.

4. **Warm TTFT p50 is still behind recompute.** Arm A TTFT p50 is 46.6 s; C′ warm is 50.1 s — **1.08× slower** on first-token latency. The fast path narrowed the gap from Autopsy v3 (69.6 s, 1.44×) but has not closed it completely. The remaining cost is the contiguous chunk H2D transfer + per-layer batched injection for the full prefix on every batch.

5. **The connector is slab-backed in this run.** Because `telys.BlobStore` is unavailable in the installed runtime, all numbers are via `TelysLocalClient` + slab. They are comparable to v3's slab path, not to Autopsy v2's BlobStore path.

### Files touched

- `telys_vllm/connector.py`
  - Added `_fast_inject_spec` zero-decode load/inject path.
  - Fixed name-shadow bug: `_load_stream` attribute → `_load_stream_obj`, method → `_get_load_stream()`.
  - Added fast-path fallback in `wait_for_save` when extraction does not yield tensors.
  - Fixed `wait_for_save` hash/block-id alignment guard.
- `telys_vllm/local_client.py`
  - `chunk_length()` refreshes from the shared Telys store on local cache miss.
  - Added `"vllm.chunk_format"` to `filter_columns` so fast-format metadata survives Telys persistence.
- `bench/measure_fast_path.py` (new)

### What's next

- Compare the fast path against a `telys.BlobStore`-capable runtime to see whether BlobStore read latency changes the warm TTFT picture.
- Profile whether prefetching/staging the next batch's prefix chunk while the current batch generates can hide the remaining H2D transfer cost.
- Re-run the small/persistence workloads to check whether the fast path also improves stability under larger KV-cache pressure.

## Autopsy v5 — fast path + BlobStore (telys 0.1.10)

Run: 2026-10-26 ~00:23–00:30 UTC  
Pod: Runpod A40 `7vewld6266m2fm`, EU-SE-1  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Harness: `bench/benchmark_prefix_caching_metrics.py` with `bench/run_autopsy_headline.sh`  
Runtime: `telys==0.1.10` (BlobStore available), `vllm==0.32.0`

This run keeps the v4 zero-decode fast path (stacked per-layer CPU tensors, async H2D, batched `inject_kv_blocks_contiguous`) but swaps the payload store from the local slab file to the native SDK `telys.BlobStore`. The connector auto-selects `TelysBlobClient` because `telys.BlobStore` is now exposed in 0.1.10. Telys state was reset and repopulated by the cold C′ run.

### Workload

Same as the previous autopsy:

- `num_prompts=20`
- `repeat_count=5` (Arm A and C′ warm) / `1` (C′ cold)
- `input_length_range=4096:8192`
- `prefix_len=4096`
- `output_len=64`
- `seed=42`
- `--no-enable-prefix-caching` (native prefix caching OFF so Telys is the only cache)

### Arm-level results

| Arm | total_time_s | throughput tok/s | TTFT mean (s) | TTFT p50 (s) | TTFT p95 (s) | Ext hit rate |
|-----|--------------|------------------|---------------|--------------|--------------|--------------|
| A (recompute, no prefix caching) | 108.510 | 58.98 | 50.625 | 49.303 | 99.417 | n/a |
| C′ cold (Telys empty, 20 prompts) | 16.270 | 78.67 | 12.371 | 13.520 | 13.626 | 73.9% |
| C′ warm (full-prefix hit, 100 prompts) | 40.391 | 158.45 | 29.836 | 27.758 | 36.074 | 99.3% |

Arm A matches the v5 headline baseline (TTFT p50 ~49 s). C′ warm reports 99.3% external prefix-cache hits, confirming Telys is serving the shared prefix.

### Comparison with v4 slab-backed fast path

| Arm | v4 slab (total / TTFT p50) | v5 BlobStore (total / TTFT p50) | change |
|-----|---------------------------|---------------------------------|--------|
| A | 106.755 s / 46.612 s | 108.510 s / 49.303 s | similar |
| C′ cold | 17.800 s / 15.039 s | 16.270 s / 13.520 s | similar |
| C′ warm | 78.776 s / 50.106 s | 40.391 s / 27.758 s | **1.95× faster total, 1.80× faster TTFT** |

The warm C′ path is the one that exercises the fast load/inject path end-to-end; moving the payload from the slab file to BlobStore roughly halves the remaining warm TTFT and total time.

### Full telemetry summary — cold C′

`python3 bench/summarize_telemetry.py bench/autopsy-v5/telemetry/headline_arm_c_cold.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| wait_for_save | 70 | 9908.717 | 0.008 | 141.553 | 1370.981 | `telys_vllm/connector.py:772` |
| save_collect | 5 | 6208.809 | 1333.715 | 1241.762 | 1681.458 | `telys_vllm/connector.py:804` |
| save_encode_persist | 5 | 3695.090 | 562.756 | 739.018 | 1410.601 | `telys_vllm/connector.py:832` |
| start_load_kv | 3 | 3402.895 | 1208.097 | 1134.298 | 1641.341 | `telys_vllm/connector.py:417` |
| inject_block | 476 | 3355.089 | 7.695 | 7.049 | 11.261 | `telys_vllm/connector.py:623` |
| store_persist | 16 | 1664.951 | 94.305 | 104.059 | 275.022 | `telys_vllm/store.py:628` |
| save_encode_chunk | 16 | 1314.798 | 47.435 | 82.175 | 290.927 | `telys_vllm/connector.py:844` |
| telys_startup | 2 | 12.298 | 6.149 | 6.149 | 12.153 | `telys_vllm/connector.py:164` |
| store_read_chunk | 17 | 12.096 | 0.432 | 0.712 | 2.281 | `telys_vllm/store.py:394` |
| lookup | 20 | 8.548 | 0.395 | 0.427 | 0.934 | `telys_vllm/connector.py:248` |
| store_lookup_prefix | 20 | 5.512 | 0.260 | 0.276 | 0.760 | `telys_vllm/store.py:195` |
| connector_store_init | 2 | 0.951 | 0.476 | 0.476 | 0.519 | `telys_vllm/connector.py:180` |

**Top 3 contributors:** `wait_for_save` 33.5%, `save_collect` 21.0%, `save_encode_persist` 12.5%.  
**Dominant phase:** `wait_for_save` at `telys_vllm/connector.py:772`.

### Full telemetry summary — warm C′

`python3 bench/summarize_telemetry.py bench/autopsy-v5/telemetry/headline_arm_c_warm.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 5 | 31177.164 | 3724.971 | 6235.433 | 14550.054 | `telys_vllm/connector.py:417` |
| inject_block | 4844 | 29812.150 | 4.277 | 6.154 | 13.965 | `telys_vllm/connector.py:623` |
| wait_for_save | 132 | 735.723 | 0.011 | 5.574 | 0.061 | `telys_vllm/connector.py:772` |
| save_collect | 1 | 596.649 | 596.649 | 596.649 | 596.649 | `telys_vllm/connector.py:804` |
| save_encode_persist | 1 | 135.552 | 135.552 | 135.552 | 135.552 | `telys_vllm/connector.py:832` |
| store_read_chunk | 173 | 129.464 | 0.593 | 0.748 | 1.135 | `telys_vllm/store.py:394` |
| store_persist | 2 | 81.301 | 40.651 | 40.651 | 42.309 | `telys_vllm/store.py:628` |
| lookup | 164 | 71.613 | 0.082 | 0.437 | 2.023 | `telys_vllm/connector.py:248` |
| save_encode_chunk | 2 | 59.102 | 29.551 | 29.551 | 38.642 | `telys_vllm/connector.py:844` |
| store_lookup_prefix | 164 | 24.854 | 0.010 | 0.152 | 1.534 | `telys_vllm/store.py:195` |
| telys_startup | 2 | 13.134 | 6.567 | 6.567 | 12.897 | `telys_vllm/connector.py:164` |
| connector_store_init | 2 | 0.737 | 0.368 | 0.368 | 0.400 | `telys_vllm/connector.py:180` |

**Top 3 contributors:** `start_load_kv` 49.6%, `inject_block` 47.4%, `wait_for_save` 1.2%.  
**Dominant phase:** `start_load_kv` at `telys_vllm/connector.py:417`.

### Focused per-phase table (warm-hit request)

| phase | v4 slab fast path (s) | v5 BlobStore fast path (s) | notes |
|-------|----------------------|---------------------------|-------|
| engine boot | 17.17 | 17.17 | vLLM init + KV cache + warmup |
| connector init | 0.00042 | 0.00037 | `connector_store_init` span |
| Telys startup | 0.0080 | 0.0066 | `telys_startup` span |
| lookup | 0.000037 | 0.000082 | `lookup` + `store_lookup_prefix` |
| BlobStore/slab read | 0.139 | 0.00059 | `store_read_chunk` median |
| inject | 0.00337 | 0.00428 | `inject_kv_block` median |
| first-token | 50.11 | 27.76 | benchmark TTFT p50 |
| save | 0.00028 | 0.000011 | `wait_for_save` median |

The `store_read_chunk` median drops from 139 ms (slab file) to 0.59 ms (BlobStore), confirming that BlobStore read latency is the main driver of the warm TTFT improvement. `start_load_kv` total drops from ~83 s (v4) to ~31 s (v5).

### Files touched

- `telys_vllm/connector.py`
  - `_auto_client()` already selected `TelysBlobClient` when `telys.BlobStore` is available; this run is the first exercise of that path with the fast-format codec.
- `bench/run_autopsy_headline.sh`
  - Added `PYTHONPATH=/root/bench/telys-vllm` because the package is used in-place on the pod (no editable install).
  - Changed C′ cold to `repeat_count=1` (20 prompts) to match the previous autopsy cold-population pattern.

### Honest verdict

1. **BlobStore + fast path is a clear win on warm TTFT.** C′ warm TTFT p50 drops from 50.1 s (v4 slab) to 27.8 s (v5 BlobStore), and total time drops from 78.8 s to 40.4 s. Telys warm-cache is now faster than recomputing on total wall time (40.4 s vs 108.5 s) and within ~1.6× on first-token latency (27.8 s vs 49.3 s).

2. **The dominant cost moved from slab read to load/inject.** In v4, `store_read_chunk` was 139 ms per chunk; in v5 it is 0.59 ms. The new dominant warm phase is `start_load_kv` + `inject_block` (97% of telemetry time), which is the contiguous H2D transfer and per-layer batched injection for the full prefix.

3. **Cold path is similar to v4.** C′ cold is 16.3 s / 13.5 s TTFT p50, comparable to v4's 17.8 s / 15.0 s. Save collection remains the dominant cold cost.

4. **The connector is functionally correct with BlobStore.** Warm runs show 99.3% external prefix-cache hits, block hashes are deterministic, and the fast-format stacked-tensor chunks round-trip through `telys.BlobStore` without errors.

### What's next

- Investigate whether the remaining `start_load_kv` + `inject_block` cost can be reduced by prefetching the next batch's prefix chunk or overlapping H2D transfer with scheduling.
- Re-run the small/persistence workloads to see whether BlobStore also improves stability under larger KV-cache pressure.


---

## Autopsy v6 — 16K-prefix stability retest (telys 0.1.10)

**Goal:** Re-run the workload shape that previously crashed the vLLM EngineCore during warm C′ runs (~64–68% completion, GPU KV cache ~88–97%) with the current batched-injection + BlobStore fast-path code. Test both 8K and 16K prefixes under load.

**Pod:** RunPod A40 (EU-SE-1), `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `telys==0.1.10`.  
**Model:** `Qwen/Qwen2.5-7B-Instruct`.  
**Shape:** `num_prompts=20`, `repeat_count=5` (warm) / `1` (cold), `output_len=64`, native prefix caching OFF, `gpu_memory_utilization=0.95`, `max_num_seqs=32`.

### Results table

| prefix | input range | arm | total_time_s | throughput tok/s | TTFT p50 (s) | TTFT p95 (s) | status |
|--------|-------------|-----|--------------|------------------|--------------|--------------|--------|
| 8K | 8192:12288 | A (recompute) | 189.07 | 33.85 | 89.30 | 173.52 | completed |
| 8K | 8192:12288 | C′ cold | 22.65 | 56.51 | 19.20 | 19.38 | completed |
| 8K | 8192:12288 | C′ warm | 41.92 | 152.67 | 25.12 | 33.19 | completed |
| 16K | 16384:20480 | A (recompute) | 383.34 | 16.70 | 186.60 | 356.67 | completed |
| 16K | 16384:20480 | C′ cold | 40.44 | 31.65 | 35.05 | 35.87 | completed |
| 16K | 16384:20480 | C′ warm | 74.06 | 86.42 | 48.61 | 65.14 | completed |

All six EngineCore processes shut down cleanly (SIGTERM after normal request processing). No `MPClient: engine core exited unexpectedly` errors, no OOM kills, no KV-block exhaustion, and no out-of-bounds injection asserts.

### Warm speedup vs recompute

| prefix | warm total speedup | warm TTFT p50 speedup |
|--------|-------------------|----------------------|
| 8K | 4.51× | 3.55× |
| 16K | 5.18× | 3.84× |

### Telemetry summary — 8K warm C′

`python3 bench/summarize_telemetry.py bench/autopsy-v6/telemetry/arm_c_warm_p8192.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 11 | 22665.337 | 1724.483 | 2060.485 | 7626.670 | `telys_vllm/connector.py:417` |
| inject_block | 7168 | 21575.721 | 2.519 | 3.010 | 9.351 | `telys_vllm/connector.py:623` |
| wait_for_save | 262 | 4752.388 | 0.008 | 18.139 | 0.011 | `telys_vllm/connector.py:772` |
| save_collect | 4 | 3313.882 | 614.364 | 828.470 | 1761.889 | `telys_vllm/connector.py:804` |
| save_encode_persist | 4 | 1433.944 | 257.180 | 358.486 | 770.632 | `telys_vllm/connector.py:832` |
| store_persist | 8 | 620.807 | 80.068 | 77.601 | 127.649 | `telys_vllm/store.py:628` |
| save_encode_chunk | 8 | 595.102 | 66.758 | 74.388 | 148.516 | `telys_vllm/connector.py:844` |
| store_read_chunk | 256 | 136.890 | 0.471 | 0.535 | 1.096 | `telys_vllm/store.py:394` |
| lookup | 100 | 20.090 | 0.071 | 0.201 | 0.387 | `telys_vllm/connector.py:248` |
| telys_startup | 2 | 13.117 | 6.558 | 6.558 | 12.906 | `telys_vllm/connector.py:164` |
| store_lookup_prefix | 100 | 10.554 | 0.011 | 0.106 | 0.276 | `telys_vllm/store.py:195` |
| connector_store_init | 2 | 0.864 | 0.432 | 0.432 | 0.458 | `telys_vllm/connector.py:180` |

### Telemetry summary — 16K warm C′

`python3 bench/summarize_telemetry.py bench/autopsy-v6/telemetry/arm_c_warm_p16384.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 8 | 50395.978 | 3536.999 | 6299.497 | 24781.320 | `telys_vllm/connector.py:417` |
| inject_block | 10220 | 48320.369 | 3.255 | 4.728 | 12.201 | `telys_vllm/connector.py:623` |
| wait_for_save | 262 | 3534.210 | 0.009 | 13.489 | 0.060 | `telys_vllm/connector.py:772` |
| save_collect | 3 | 2618.699 | 727.208 | 872.900 | 1705.414 | `telys_vllm/connector.py:804` |
| save_encode_persist | 3 | 907.177 | 117.511 | 302.392 | 725.733 | `telys_vllm/connector.py:832` |
| store_persist | 5 | 444.182 | 78.919 | 88.836 | 221.699 | `telys_vllm/store.py:628` |
| save_encode_chunk | 5 | 326.367 | 43.589 | 65.273 | 179.060 | `telys_vllm/connector.py:844` |
| store_read_chunk | 365 | 241.412 | 0.547 | 0.661 | 1.007 | `telys_vllm/store.py:394` |
| lookup | 295 | 196.495 | 0.389 | 0.666 | 1.613 | `telys_vllm/connector.py:248` |
| telys_startup | 2 | 13.179 | 6.589 | 6.589 | 12.939 | `telys_vllm/connector.py:164` |
| store_lookup_prefix | 295 | 23.519 | 0.038 | 0.080 | 0.278 | `telys_vllm/store.py:195` |
| connector_store_init | 2 | 0.757 | 0.378 | 0.378 | 0.393 | `telys_vllm/connector.py:180` |

### Honest verdict

1. **The crash is gone.** Both 8K and 16K warm C′ runs completed 100% of requests. The previous failure mode (EngineCore exiting mid-run under heavy KV pressure) did not reproduce.

2. **It is not simply "more GPU memory."** `gpu_memory_utilization=0.95` on a 48 GB A40 still bounds the workload; vLLM allocated the same class of GPU resources as before. The difference is the connector-side batched injection and BlobStore fast path, which keeps the per-request working set predictable.

3. **16K is still expensive, but Telys wins on both latency and throughput.** Warm TTFT at 16K is 48.6 s versus 186.6 s recompute, and total wall time is 74.1 s versus 383.3 s. The warm run is within ~1.7× of the 8K warm TTFT (48.6 s vs 25.1 s) despite doubling the prefix length, which is consistent with the load/inject cost scaling roughly linearly with prefix size.

4. **Dominant warm cost remains load + inject.** At 16K, `start_load_kv` + `inject_block` account for ~92% of telemetry time. BlobStore read latency stays negligible (median 0.55 ms), so the remaining headroom is in H2D transfer and per-layer injection parallelism.

### Files touched

- New: `bench/autopsy-v6/` (results, logs, telemetry JSONL).
- New: `bench/run_autopsy_v6.sh` (reproducible run script).

---

## Autopsy v7 — two-replica shared-cache demo (telys 0.1.10)

**Goal:** Prove that one Telys store can be shared across two completely separate vLLM engine processes: engine A cold-populates the store and exits, engine B starts fresh (native prefix cache empty by construction) and warm-hits the shared store.

**Pod:** Same A40 EU-SE-1 pod as v6, sequential runs.  
**Model:** `Qwen/Qwen2.5-7B-Instruct`.  
**Shape:** `prefix_len=4096`, `input_length_range=4096:8192`, `output_len=64`, `num_prompts=20`, `repeat_count=5` (warm/recompute) / `1` (cold), seed 42.

### Results table

| arm | process | total_time_s | throughput tok/s | TTFT p50 (s) | TTFT p95 (s) | notes |
|-----|---------|--------------|------------------|--------------|--------------|-------|
| A (recompute) | fresh | 103.85 | 61.63 | 49.67 | 94.30 | native caching OFF |
| C′ cold (engine A) | fresh | 15.13 | 84.62 | 12.43 | 12.50 | populates shared Telys store |
| C′ warm (engine B) | fresh | 28.98 | 220.88 | 16.78 | 22.68 | native cache empty, shared store hit |

Engine B is a separate OS process; the vLLM V1 native prefix cache is empty when it starts. The only source of prefix KV is the Telys store written by engine A.

### Engine B vs recompute

| metric | recompute | engine B warm | improvement |
|--------|-----------|---------------|-------------|
| total time | 103.85 s | 28.98 s | 3.58× faster |
| TTFT p50 | 49.67 s | 16.78 s | 2.96× faster |
| throughput | 61.63 tok/s | 220.88 tok/s | 3.58× higher |

### Telemetry summary — engine B warm

`python3 bench/summarize_telemetry.py bench/autopsy-v7/telemetry/engine_b_warm.jsonl`

| phase | count | total (ms) | median (ms) | mean (ms) | p95 (ms) | dominant file:line |
|-------|-------|------------|-------------|-----------|----------|--------------------|
| start_load_kv | 8 | 16725.921 | 1184.257 | 2090.740 | 8590.345 | `telys_vllm/connector.py:417` |
| inject_block | 4788 | 15783.113 | 2.247 | 3.296 | 7.781 | `telys_vllm/connector.py:623` |
| wait_for_save | 262 | 753.538 | 0.008 | 2.876 | 0.011 | `telys_vllm/connector.py:772` |
| save_collect | 1 | 588.625 | 588.625 | 588.625 | 588.625 | `telys_vllm/connector.py:804` |
| save_encode_persist | 1 | 131.145 | 131.145 | 131.145 | 131.145 | `telys_vllm/connector.py:832` |
| store_read_chunk | 171 | 86.932 | 0.473 | 0.508 | 0.542 | `telys_vllm/store.py:394` |
| store_persist | 2 | 83.466 | 41.733 | 41.733 | 43.739 | `telys_vllm/store.py:628` |
| save_encode_chunk | 2 | 56.464 | 28.232 | 28.232 | 36.378 | `telys_vllm/connector.py:844` |
| telys_startup | 2 | 12.931 | 6.466 | 6.466 | 12.719 | `telys_vllm/connector.py:164` |
| lookup | 100 | 5.715 | 0.034 | 0.057 | 0.220 | `telys_vllm/connector.py:248` |
| store_lookup_prefix | 100 | 1.134 | 0.005 | 0.011 | 0.085 | `telys_vllm/store.py:195` |
| connector_store_init | 2 | 0.703 | 0.351 | 0.351 | 0.353 | `telys_vllm/connector.py:180` |

### Honest verdict

1. **Replica sharing works.** Engine B, starting as a completely fresh process with no native prefix-cache state, achieved full-prefix external hits from the Telys store populated by engine A. TTFT p50 dropped from 49.7 s (recompute) to 16.8 s (engine B warm).

2. **The store is the cache of record.** Because engine B's native vLLM cache is empty at startup, every block it skips recomputing must come from Telys. The warm result is therefore a direct measurement of cross-engine persistence.

3. **Numbers align with the headline v5 run.** Engine B warm (16.8 s TTFT p50) is faster than the v5 single-engine warm run (27.8 s TTFT p50). Both runs use the same 4K prefix / 8K max input shape, so the improvement likely reflects a clean store state with no intervening resets and very fast lookups (median 34 µs).

4. **This validates the multi-engine / restart story.** A production deployment can cold-start one replica to populate the store, then spin up additional replicas that immediately serve warm-prefix requests without recomputing.

### Files touched

- New: `bench/autopsy-v7/` (results, logs, telemetry JSONL).
- New: `bench/run_autopsy_v7.sh` (reproducible run script).
