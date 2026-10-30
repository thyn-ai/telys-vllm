# SPDX-License-Identifier: Apache-2.0
"""pytest bootstrap: stub the vLLM modules telys_vllm imports at runtime.

The connector is an external package against KVConnectorBase_V1
(vllm/distributed/kv_transfer/kv_connector/v1/base.py). Building vLLM is out
of scope on this CPU-only machine, so this conftest installs faithful stubs
of exactly the names the connector imports, mirroring the real semantics
(line refs are to vllm commit b689feb9):

- vllm.logger.init_logger
- vllm.config.kv_transfer.KVTransferConfig      (config/kv_transfer.py)
- ...kv_connector.v1.base: KVConnectorRole (base.py:139), KVConnectorMetadata
  (base.py:155), SupportsHMA (base.py:101), KVConnectorBase_V1 (base.py:181)
- ...kv_connector.factory.KVConnectorFactory    (factory.py:27-146)

The stubs are installed before any test module imports telys_vllm.connector.
"""

import contextlib
import enum
import importlib
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, ClassVar


def _install(name: str, module: ModuleType) -> ModuleType:
    sys.modules[name] = module
    parent, _, attr = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], attr, module)
    return module


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # mark as package
    return _install(name, module)


# ---------------------------------------------------------------------------
# vllm.logger
# ---------------------------------------------------------------------------


def _install_logger() -> None:
    module = ModuleType("vllm.logger")
    module.init_logger = logging.getLogger
    _install("vllm.logger", module)


# ---------------------------------------------------------------------------
# vllm.config.kv_transfer (config/kv_transfer.py)
# ---------------------------------------------------------------------------


@dataclass
class KVTransferConfig:
    """Mirror of the fields the connector/factory read."""

    kv_connector: str | None = None
    engine_id: str | None = None
    kv_role: str | None = None
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    kv_connector_module_path: str | None = None

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_role in ("kv_producer", "kv_both")

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)


def _install_kv_transfer_config() -> None:
    module = ModuleType("vllm.config.kv_transfer")
    module.KVTransferConfig = KVTransferConfig
    _install("vllm.config.kv_transfer", module)


# ---------------------------------------------------------------------------
# vllm.distributed.kv_transfer.kv_connector.v1.base (v1/base.py)
# ---------------------------------------------------------------------------


class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1


class KVConnectorMetadata(ABC):
    pass


class SupportsHMA(ABC):
    @abstractmethod
    def request_finished_all_groups(self, request, block_ids):
        raise NotImplementedError


class KVConnectorBase_V1(ABC):
    """Mirror of the abstract surface + concrete helpers the connector uses."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self._connector_metadata = None
        self._vllm_config = vllm_config
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        self._kv_cache_config = kv_cache_config
        self._kv_cache_manager = None
        self._role = role

    @property
    def role(self):
        return self._role

    @property
    def requires_kv_delivery(self) -> bool:
        return self._kv_transfer_config.is_kv_producer

    # -- metadata plumbing (base.py:249-290) --------------------------------

    def bind_connector_metadata(self, connector_metadata) -> None:
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def _get_connector_metadata(self):
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        return self._connector_metadata is not None

    def register_kv_caches(self, kv_caches):
        return

    # -- abstract worker-side methods ----------------------------------------

    @abstractmethod
    def start_load_kv(self, forward_context, **kwargs):
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    @abstractmethod
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        pass

    @abstractmethod
    def wait_for_save(self):
        pass

    def get_finished(self, finished_req_ids):
        return None, None

    def shutdown(self):
        return None

    # -- abstract scheduler-side methods --------------------------------------

    @abstractmethod
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        pass

    @abstractmethod
    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        pass

    @abstractmethod
    def build_connector_meta(self, scheduler_output):
        pass

    def on_new_request(self, request) -> None:
        return

    def request_finished(self, request, block_ids):
        return False, None

    def get_loaded_kv_cache_group_ids(self, request):
        return self._kv_cache_config.prefix_cacheable_group_ids

    def reset_cache(self):
        return None


def _install_v1_base() -> None:
    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    module.KVConnectorRole = KVConnectorRole
    module.KVConnectorMetadata = KVConnectorMetadata
    module.SupportsHMA = SupportsHMA
    module.KVConnectorBase_V1 = KVConnectorBase_V1
    _install("vllm.distributed.kv_transfer.kv_connector.v1.base", module)


# ---------------------------------------------------------------------------
# vllm.distributed.kv_transfer.kv_connector.factory (factory.py:27-146)
# ---------------------------------------------------------------------------


def _supports_kw(callable_obj: Any, kw_name: str) -> bool:
    """Minimal mirror of vllm.utils.func_utils.supports_kw."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kw_name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


class KVConnectorFactory:
    """Mirror of factory.py: registry + external-module-path priority."""

    _registry: ClassVar[dict[str, Any]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader():
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, config, role, kv_cache_config):
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class(kv_transfer_config)
        return connector_cls(config, role, kv_cache_config)

    @classmethod
    def get_connector_class(cls, kv_transfer_config):
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        connector_module_path = kv_transfer_config.kv_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError("kv_connector_module_path cannot be an empty string.")
        if connector_module_path:
            # External module path takes priority over internal registry.
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            if not _supports_kw(connector_cls, "kv_cache_config"):
                msg = (
                    f"Connector {connector_cls.__name__} uses deprecated "
                    "2-argument constructor signature. External v1 KV "
                    "connectors must accept kv_cache_config as the third "
                    "constructor argument and pass it to super().__init__()."
                )
                raise ValueError(msg)
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls


def _install_factory() -> None:
    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    module.KVConnectorFactory = KVConnectorFactory
    _install("vllm.distributed.kv_transfer.kv_connector.factory", module)


# ---------------------------------------------------------------------------
# Install everything (package shells first)
# ---------------------------------------------------------------------------

for pkg in (
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
):
    if pkg not in sys.modules:
        _package(pkg)

_install_logger()
_install_kv_transfer_config()
_install_v1_base()
_install_factory()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


import pytest


@pytest.fixture(autouse=True)
def _clean_fake_state():
    """Isolate tests: fresh shared Telys store and factory registry."""
    from tests import fakes

    fakes.reset_fakes()
    KVConnectorFactory._registry.clear()
    yield
    fakes.reset_fakes()
    KVConnectorFactory._registry.clear()
    for td in list(fakes._FALLBACK_TMP_DIRS):
        with contextlib.suppress(OSError):
            td.cleanup()
        fakes._FALLBACK_TMP_DIRS.remove(td)
