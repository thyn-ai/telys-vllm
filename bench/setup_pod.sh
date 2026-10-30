#!/usr/bin/env bash
# One-time environment setup for the Runpod A40 benchmark pod.
set -euo pipefail

python3 -m venv /root/bench-venv
source /root/bench-venv/bin/activate

python3 -m pip install --upgrade pip wheel

# Pin to the versions used in the v1 bench run.
python3 -m pip install "vllm==0.32.0" "telys==0.1.9"

# Editable install of the connector code (must already be rsynced to /root/bench/telys-vllm).
python3 -m pip install -e /root/bench/telys-vllm

mkdir -p /root/.telys
chmod 700 /root/.telys

# Headless login with the OAuth/API token copied from the local workstation.
# This downloads the signed on-device runtime needed by TelysLocalClient.
telys login --token "$(cat /root/.telys/api_key)" --no-browser --plan telys_developer

# Runtime path used by the benchmark harness.
mkdir -p /root/.telys/runtime

echo "=== setup complete ==="
