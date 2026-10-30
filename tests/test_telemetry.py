# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in telemetry: zero overhead when off, JSONL when on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telys_vllm.telemetry import (
    ENV_TELEMETRY,
    ENV_TELEMETRY_LOG,
    Telemetry,
    get_telemetry,
    is_enabled,
    telemetry_from_env,
)
from tests.fakes import (
    BLOCK_SIZE,
    FakeNewReq,
    FakeRequest,
    FakeSchedulerOutput,
    make_fake_store,
    make_hashes,
    make_kv_cache_config,
    make_vllm_config,
)


@pytest.fixture
def telemetry_log(tmp_path: Path) -> str:
    return str(tmp_path / "telemetry.jsonl")


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv(ENV_TELEMETRY, raising=False)
    monkeypatch.delenv(ENV_TELEMETRY_LOG, raising=False)


class TestTelemetryParsing:
    def test_is_enabled_accepts_truthy_strings(self):
        for raw in ("1", "true", "True", "YES", "on"):
            assert is_enabled(raw) is True

    def test_is_enabled_rejects_everything_else(self):
        for raw in ("0", "false", "no", "off", "", "maybe", None, 42):
            assert is_enabled(raw) is False

    def test_telemetry_from_env_disabled_by_default(self, tmp_path: Path):
        env = {"TELYS_VLLM_TELEMETRY": "0"}
        tel = telemetry_from_env(env)
        assert tel.enabled is False

    def test_telemetry_from_env_enabled(self, tmp_path: Path):
        log = tmp_path / "out.jsonl"
        env = {"TELYS_VLLM_TELEMETRY": "1", "TELYS_VLLM_TELEMETRY_LOG": str(log)}
        tel = telemetry_from_env(env)
        assert tel.enabled is True
        assert tel._log_path == str(log)


class TestTelemetryDisabled:
    def test_disabled_instance_is_noop(self, telemetry_log: str):
        tel = Telemetry(enabled=False, log_path=telemetry_log)
        with tel.span("should_not_appear"):
            pass
        tel.record("also_not_appearing", 123.0)
        assert not Path(telemetry_log).exists()

    def test_get_telemetry_disabled_returns_noop(self, telemetry_log: str):
        tel = get_telemetry(enabled=False, log_path=telemetry_log)
        with tel.span("should_not_appear"):
            pass
        assert not Path(telemetry_log).exists()


class TestTelemetryEnabled:
    def test_span_emits_jsonl(self, telemetry_log: str):
        tel = Telemetry(enabled=True, log_path=telemetry_log)
        with tel.span("test_phase") as ctx:
            ctx["extra"] = "value"
        tel.close()

        lines = Path(telemetry_log).read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["phase"] == "test_phase"
        assert payload["context"]["extra"] == "value"
        assert "elapsed_us" in payload
        assert "ts" in payload
        assert "pid" in payload

    def test_record_emits_jsonl(self, telemetry_log: str):
        tel = Telemetry(enabled=True, log_path=telemetry_log)
        tel.record("manual", 42.0, block_id=7)
        tel.close()

        lines = Path(telemetry_log).read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["phase"] == "manual"
        assert payload["elapsed_us"] == 42.0
        assert payload["context"]["block_id"] == 7


class TestStoreTelemetry:
    def test_store_roundtrip_emits_phases(
        self, tmp_path: Path, telemetry_log: str
    ) -> None:
        store = make_fake_store(
            tmp_path,
            telemetry=True,
            telemetry_log=telemetry_log,
            write_behind=False,
        )
        hashes = make_hashes(4)
        layer_blobs = {
            "layer_0": [b"block0", b"block1", b"block2", b"block3"],
            "layer_1": [b"BLOCK0", b"BLOCK1", b"BLOCK2", b"BLOCK3"],
        }

        store.save_chunk(hashes[0], hashes, layer_blobs)
        hits = store.lookup_prefix(hashes)
        blocks = store.load_decoded_blocks(hashes)
        store.close()

        assert hits == 4
        assert len(blocks) == 4
        lines = Path(telemetry_log).read_text().strip().splitlines()
        phases = [json.loads(line)["phase"] for line in lines]

        assert "store_lookup_prefix" in phases
        assert "store_read_chunk" in phases
        assert "store_decode_chunk" in phases
        assert "store_persist" in phases

    def test_store_disabled_telemetry_does_not_create_log(
        self, tmp_path: Path, telemetry_log: str
    ) -> None:
        store = make_fake_store(
            tmp_path,
            telemetry=False,
            telemetry_log=telemetry_log,
            write_behind=False,
        )
        hashes = make_hashes(2)
        store.save_chunk(hashes[0], hashes, {"layer_0": [b"a", b"b"]})
        store.load_decoded_blocks(hashes)
        store.close()
        assert not Path(telemetry_log).exists()


class TestConnectorTelemetry:
    def test_connector_save_load_emits_phases(
        self, tmp_path: Path, telemetry_log: str
    ) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        from telys_vllm.connector import TelysKVConnector

        extra = {
            "telys_client_factory": "tests.fakes:make_fake_client",
            "telys_slab_path": str(tmp_path / "kv.slab"),
            "telys_telemetry": True,
            "telys_telemetry_log": telemetry_log,
            "telys_write_behind": False,
        }
        vllm_config = make_vllm_config(extra=extra, tmp_path=tmp_path)
        scheduler = TelysKVConnector(
            vllm_config, KVConnectorRole.SCHEDULER, make_kv_cache_config()
        )
        worker = TelysKVConnector(
            vllm_config, KVConnectorRole.WORKER, make_kv_cache_config()
        )

        num_blocks = 4
        num_tokens = num_blocks * BLOCK_SIZE + 3
        hashes = make_hashes(num_blocks)
        request = FakeRequest("req-1", num_tokens, block_hashes=hashes)

        scheduler.on_new_request(request)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.update_state_after_alloc(request, None, 0)
        meta = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[
                    FakeNewReq("req-1", (list(range(num_blocks + 1)),))
                ],
                num_scheduled_tokens={"req-1": num_tokens},
            )
        )

        worker.bind_connector_metadata(meta)
        worker.save_kv_layer(
            "layer.0", None, None, block_payloads={h: b"L0" + h for h in hashes}
        )
        worker.wait_for_save()
        worker.clear_connector_metadata()

        second = FakeRequest("req-2", num_tokens, block_hashes=hashes)
        scheduler.on_new_request(second)
        scheduler.get_num_new_matched_tokens(second, 0)
        scheduler.update_state_after_alloc(second, None, num_blocks * BLOCK_SIZE)
        meta2 = scheduler.build_connector_meta(
            FakeSchedulerOutput(
                scheduled_new_reqs=[
                    FakeNewReq(
                        "req-2",
                        ([100 + i for i in range(num_blocks + 1)],),
                        num_computed_tokens=num_blocks * BLOCK_SIZE,
                    )
                ],
                num_scheduled_tokens={"req-2": 3},
            )
        )
        worker.bind_connector_metadata(meta2)
        worker.start_load_kv(None)
        worker.clear_connector_metadata()
        worker.shutdown()
        scheduler.shutdown()

        lines = Path(telemetry_log).read_text().strip().splitlines()
        phases = [json.loads(line)["phase"] for line in lines]

        assert "connector_store_init" in phases
        assert "telys_startup" in phases
        assert "lookup" in phases
        assert "start_load_kv" in phases
        assert "load_decoded_blocks" in phases
        assert "wait_for_save" in phases
        assert "save_collect" in phases
        assert "store_persist" in phases
