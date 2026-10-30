# SPDX-License-Identifier: Apache-2.0
"""Telys external KV/prefix-cache connector for vLLM.

Enables Telys (on-device memory/retrieval) as vLLM's external KV store via
the KVConnectorBase_V1 API. Users add it through one config line:

    --kv-transfer-config '{"kv_connector": "TelysKVConnector",
        "kv_connector_module_path": "telys_vllm.connector",
        "kv_role": "kv_both"}'

See DESIGN.md for the full design.
"""

from telys_vllm.client import (
    TelysClient,
    TelysMaintenanceReport,
    TelysProvenance,
    TelysRecord,
)
from telys_vllm.config import TelysConfigError, TelysConnectorConfig
from telys_vllm.store import TelysBlockStore, TelysStoreError

__all__ = [
    "TelysBlockStore",
    "TelysClient",
    "TelysConfigError",
    "TelysConnectorConfig",
    "TelysMaintenanceReport",
    "TelysProvenance",
    "TelysRecord",
    "TelysStoreError",
]

__version__ = "0.1.0"
