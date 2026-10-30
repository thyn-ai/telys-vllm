# SPDX-License-Identifier: Apache-2.0
"""Factory registration shapes, mirroring KVConnectorFactory semantics
(factory.py:30-40 register_connector, factory.py:96-129 module-path
priority + constructor-shape check, factory.py:42-75 create_connector).
"""

import pytest
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from telys_vllm.connector import TelysKVConnector
from tests.fakes import (
    make_kv_cache_config,
    make_kv_transfer_config,
    make_vllm_config,
)

CONNECTOR_NAME = "TelysKVConnector"
CONNECTOR_MODULE = "telys_vllm.connector"


def test_register_connector_makes_class_resolvable_by_name():
    KVConnectorFactory.register_connector(
        CONNECTOR_NAME, CONNECTOR_MODULE, CONNECTOR_NAME
    )
    resolved = KVConnectorFactory.get_connector_class(
        KVTransferConfig(kv_connector=CONNECTOR_NAME, kv_role="kv_both")
    )
    assert resolved is TelysKVConnector


def test_duplicate_registration_raises():
    KVConnectorFactory.register_connector(
        CONNECTOR_NAME, CONNECTOR_MODULE, CONNECTOR_NAME
    )
    with pytest.raises(ValueError, match="already registered"):
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME, CONNECTOR_MODULE, CONNECTOR_NAME
        )


def test_external_module_path_resolves_without_registry_entry():
    resolved = KVConnectorFactory.get_connector_class(
        make_kv_transfer_config(kv_connector_module_path=CONNECTOR_MODULE)
    )
    assert resolved is TelysKVConnector


def test_external_module_path_takes_priority_over_registry():
    # Registry maps the name to a different class entirely...
    KVConnectorFactory.register_connector(
        CONNECTOR_NAME, "tests.fakes", "LegacyTwoArgConnector"
    )
    # ...but the module path wins (factory.py:106-107).
    resolved = KVConnectorFactory.get_connector_class(
        make_kv_transfer_config(kv_connector_module_path=CONNECTOR_MODULE)
    )
    assert resolved is TelysKVConnector


def test_unknown_connector_name_raises():
    with pytest.raises(ValueError, match="Unsupported connector type"):
        KVConnectorFactory.get_connector_class(
            KVTransferConfig(kv_connector="NoSuchConnector", kv_role="kv_both")
        )


def test_empty_module_path_raises():
    with pytest.raises(ValueError, match="cannot be an empty string"):
        KVConnectorFactory.get_connector_class(
            make_kv_transfer_config(kv_connector_module_path="")
        )


def test_module_path_rejects_two_arg_constructor():
    # factory.py:116-124: external connectors must accept kv_cache_config.
    with pytest.raises(ValueError, match="kv_cache_config"):
        KVConnectorFactory.get_connector_class(
            KVTransferConfig(
                kv_connector="LegacyTwoArgConnector",
                kv_role="kv_both",
                kv_connector_module_path="tests.fakes",
            )
        )


def test_create_connector_instantiates_both_roles():
    config = make_vllm_config(
        kv_transfer_config=make_kv_transfer_config(
            kv_connector_module_path=CONNECTOR_MODULE
        )
    )
    scheduler = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, make_kv_cache_config()
    )
    worker = KVConnectorFactory.create_connector(
        config, KVConnectorRole.WORKER, make_kv_cache_config()
    )
    assert isinstance(scheduler, TelysKVConnector)
    assert isinstance(worker, TelysKVConnector)
    assert scheduler.role is KVConnectorRole.SCHEDULER
    assert worker.role is KVConnectorRole.WORKER


def test_create_connector_requires_kv_transfer_config():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="kv_transfer_config must be set"):
        KVConnectorFactory.create_connector(
            SimpleNamespace(kv_transfer_config=None),
            KVConnectorRole.SCHEDULER,
            make_kv_cache_config(),
        )
