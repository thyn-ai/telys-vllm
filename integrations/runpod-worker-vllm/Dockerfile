FROM runpod/worker-v1-vllm:v2.30.3

COPY pyproject.toml README.md /tmp/telys-vllm/
COPY telys_vllm /tmp/telys-vllm/telys_vllm

RUN python3 -m pip install --no-cache-dir "telys==0.1.14" \
    && python3 -m pip install --no-cache-dir --no-deps /tmp/telys-vllm \
    && rm -rf /tmp/telys-vllm \
    && python3 -c 'from importlib.metadata import version; vllm_version = version("vllm"); assert vllm_version == "0.32.0", f"expected vllm==0.32.0, found {vllm_version}"; telys_version = version("telys"); assert telys_version == "0.1.14", f"expected telys==0.1.14, found {telys_version}"; from telys_vllm.connector import TelysKVConnector'

COPY --chmod=0755 integrations/runpod-worker-vllm/entrypoint.sh /usr/local/bin/telys-vllm-entrypoint

ENTRYPOINT ["/usr/local/bin/telys-vllm-entrypoint"]
