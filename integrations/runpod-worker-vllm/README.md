# Runpod worker image for Telys and stock filesystem KV

This directory packages the existing `telys_vllm` connector into a derived
[`runpod-workers/worker-vllm`](https://github.com/runpod-workers/worker-vllm/tree/v2.30.3)
image. It preserves the worker's stock vLLM 0.32.0 installation and original
`/src/main.py` startup path. The same image can run either the Telys connector
or vLLM's built-in filesystem offload tier; configure only one connector per
endpoint.

## Use the release image

The signed `v0.1.6` image is public on GHCR. Pull it by immutable digest:

```bash
docker pull ghcr.io/thyn-ai/telys-worker-vllm@sha256:1b02b3cdac471064e8b9968fe2ab87f99da2d455e04d9cffea6b74b745a4218f
```

Its Sigstore signature and SLSA provenance are attached to the [GitHub Release](https://github.com/thyn-ai/telys-vllm/releases/tag/v0.1.6).

## Build a custom image

Run the build from the **repository root**, not from this directory. Use an
immutable, versioned tag of your own and an x86-64 target for Runpod:

```bash
cd /path/to/telys-vllm
export IMAGE="your-dockerhub-user/telys-worker-vllm:0.1.0"
docker build --platform=linux/amd64 \
  --file integrations/runpod-worker-vllm/Dockerfile \
  --tag "$IMAGE" .
```

The Dockerfile installs `telys==0.1.14` and the local connector package with
`--no-deps`, then fails the build unless the installed distribution is exactly
`vllm==0.32.0` and `telys_vllm.connector.TelysKVConnector` imports. It does not
run `telys login`, accept a Telys token as a build argument, or bake a model or
signed Telys runtime into the image. Publish the resulting image to a registry
that the Runpod endpoint can pull.

## Common Runpod configuration

Attach a Runpod network volume at `/runpod-volume` to both arms. Configure the
same model, model revision, GPU type, dtype, KV dtype, block size, context
length, parallelism, and request workload in each arm. Set the worker's normal
variables, including:

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
# HF_TOKEN=<only for a gated/private model; configure it as a secret>
```

The examples below disable native GPU prefix caching and V1 multiprocessing in
both arms. This matches the connector-only comparison used by this repository's
benchmarks: native GPU hits cannot mask external-cache hits, and multi-process
hot-cache coherence is excluded as an additional variable. Set
`ENABLE_PREFIX_CACHING=true` in **both** arms instead if native caching is part
of the intended comparison.

Use separate cache directories for the arms. Do not enable both
`KV_TRANSFER_CONFIG` values at once, and do not put JSON in `VLLM_EXTRA_ARGS`
unless the endpoint cannot pass `KV_TRANSFER_CONFIG` directly.

## Telys arm

Set these environment variables on the Telys endpoint or worker template:

```bash
ENABLE_PREFIX_CACHING=false
VLLM_ENABLE_V1_MULTIPROCESSING=0
TELYS_HOME=/runpod-volume/telys-home
TELYS_PATH=/runpod-volume/telys-data
KV_TRANSFER_CONFIG={"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both"}
```

For a runtime that is not installed yet, configure exactly one of these
headless provisioning mechanisms as a Runpod secret:

- **`TELYS_API_KEY` (direct `de_` API-key provisioning).** An inline Python step
  reads the key from the environment and calls
  `telys.login.register_device(api=telys.paths.api_url(), api_key=key)`. It
  atomically writes the returned license with mode `0600` to
  `login_license.jwt` under `telys.paths.telys_home()`—the current
  `TELYS_HOME`, or `~/.telys` by default—then calls
  `telys.installer.install_from_host(token=key, license_path=license_path)`.
  This path does not invoke `telys login --token`.
- **`TELYS_OAUTH_TOKEN` (OAuth provisioning).** This token is accepted by the
  OAuth-only CLI flow, which the entrypoint runs exactly as:

  ```bash
  telys login --token "$TELYS_OAUTH_TOKEN" --no-browser --plan telys_developer
  ```

A `de_` API key is supported only through the direct `TELYS_API_KEY` path; it
is not an OAuth token and must not be supplied through `TELYS_OAUTH_TOKEN` or
to `telys login --token`. If both variables are set, API-key provisioning takes
precedence and the OAuth CLI is not invoked. Failure in the selected path stops
startup immediately: there is no fallback to the other mechanism and no silent
fall back to an unauthenticated runtime.

After successful provisioning, the entrypoint unsets both credential variables
and execs `python3 /src/main.py`. Shell tracing is disabled, and neither secret
is written to the Dockerfile or image. Do not place either secret in
`KV_TRANSFER_CONFIG`, `VLLM_EXTRA_ARGS`, a build arg, or an `ENV` instruction.

`TELYS_HOME` persists the SDK's signed runtime and its own login/license state.
After that state has been installed on the volume, both credential variables
can be removed: the entrypoint skips login when both are unset or empty and
starts the worker with the already-installed runtime. Keep the same
`TELYS_HOME` setting in either mode. Perform the initial provisioning with one
worker before allowing concurrent workers to initialize the same directory.
The SDK state contains credentials, so restrict access to the volume.

The connector selects `TelysBlobClient` when this SDK exposes `BlobStore`, and
otherwise uses `TelysLocalClient`. With the path above, embedded engine data is
under `/runpod-volume/telys-data`; BlobStore payloads use
`/runpod-volume/telys-data_blobs`. The default fallback slab path is
`/runpod-volume/telys-data_kv.slab`. Set `TELYS_VLLM_SLAB_PATH` explicitly if a
different slab location is required. Keep all of these paths on persistent
storage.

`TELYS_ADDRESS` and `TELYS_TOKEN` can select a self-host Telys server instead
of `TELYS_PATH`; see [the design document](../../DESIGN.md#38-real-client-binding).
When using a remote engine with the slab fallback, set an explicit local
`TELYS_VLLM_SLAB_PATH` on the volume.

## Stock filesystem arm

The same derived image can run the stock arm without Telys login. Leave
`TELYS_API_KEY`, `TELYS_OAUTH_TOKEN`, `TELYS_PATH`, and `TELYS_ADDRESS` unset,
and use:

```bash
ENABLE_PREFIX_CACHING=false
VLLM_ENABLE_V1_MULTIPROCESSING=0
KV_TRANSFER_CONFIG={"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":10737418240,"secondary_tiers":[{"type":"fs","root_dir":"/runpod-volume/vllm-fs-kv"}]}}
```

This is vLLM 0.32.0's unmodified `OffloadingConnector` with its
`TieringOffloadingSpec` and built-in `fs` secondary tier; no Telys code handles
these requests. The filesystem root is `/runpod-volume/vllm-fs-kv`.

`cpu_bytes_to_use` is **required** even when the desired persistence tier is
the filesystem: vLLM stages filesystem blocks through a CPU primary tier. The
example reserves 10 GiB across all workers, not 10 GiB per worker. Tune it for
the model and available host RAM, leaving headroom for vLLM and the worker.
Optional `n_read_threads` and `n_write_threads` keys on the `fs` entry default
to 16 each. See the
[vLLM filesystem-offload reference](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/features/kv_offloading_usage.md)
for chunking and layout details.

## Persistent storage and cache lifecycle

- The base worker already places Hugging Face downloads under
  `/runpod-volume/huggingface-cache` and its compile cache under
  `/runpod-volume/vllm-cache` when the volume is attached.
- Telys runtime/login state, engine metadata, blob or slab payloads, and stock
  FS block files are independent data sets. Size and monitor the volume for
  all of them; a full or read-only volume can prevent startup or cache writes.
- The stock FS tier namespaces files by model and run-configuration digest.
  Old directories are not automatically reclaimed when configurations change.
  Telys data likewise needs an explicit capacity and deletion policy. Stop all
  writers before deleting either cache.
- Do not share a Telys namespace or cache directory across incompatible model
  revisions, dtypes, quantization, KV layouts, block sizes, or parallelism
  configurations. Use separate paths and `TELYS_VLLM_NAMESPACE` values. Treat
  KV data and SDK login state as tenant-sensitive, not as a public shared
  cache.

## Limitations and required validation

This integration is packaging and configuration only. The build-time import
check does not exercise a GPU, model, signed runtime, network volume, or warm
KV hit. Before production use, validate cold misses, warm hits after a fresh
worker start, output equivalence against the same uncached model, disk usage,
and restart behavior on the target Runpod GPU and model.

The stock FS arm uses CPU staging and storage I/O, so its latency is not a
pure disk measurement. It may use `O_DIRECT` where supported and otherwise
falls back to buffered I/O. Telys has different serialization and persistence
costs; neither arm is universally faster than recomputation or native GPU
prefix caching, especially for short prefixes.

Do not assume that an embedded Telys engine on a network volume supports
concurrent writers or immediate cross-process visibility. The connector's
chunk-length hot cache is process-local; fresh processes seed from persisted
state, while live multi-process updates have different coherence behavior.
Use a supported self-host Telys deployment for shared live state and validate
that topology separately. Hybrid/multi-group models and multi-writer deployments
require specific validation rather than being guaranteed by this image.

Asynchronous Telys loading is disabled by design in this revision, not offered
as an optional tuning knob. `load_async=false` is required; setting
`TELYS_VLLM_LOAD_ASYNC=true` or `telys_load_async=True` fails configuration
validation before connector or client state is created.

## Reproducible benchmark

`benchmark.sh` runs the no-cache, stock-filesystem, and Telys arms with fresh
processes and refuses to reuse cache roots. `summarize_results.py` checks
output equivalence and reports timing and cache-hit evidence. Use
`./integrations/runpod-worker-vllm/benchmark.sh --dry-run 4096 run1` to inspect
the commands, then choose a new run ID for each live run. The connector's
architecture and existing measured workloads are documented in the repository
[README](../../README.md) and [DESIGN](../../DESIGN.md).
