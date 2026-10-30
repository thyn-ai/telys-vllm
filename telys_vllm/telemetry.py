# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-phase telemetry for the Telys vLLM connector.

Zero overhead when disabled: the default module-level instance is a null
object; importing and calling it costs a Python attribute lookup and a no-op
function call.  When ``TELYS_VLLM_TELEMETRY=1`` is set, a real instance is
constructed and emits JSONL lines to ``TELYS_VLLM_TELEMETRY_LOG`` (or a
sensible default).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from typing_extensions import Self

ENV_TELEMETRY = "TELYS_VLLM_TELEMETRY"
ENV_TELEMETRY_LOG = "TELYS_VLLM_TELEMETRY_LOG"

DEFAULT_LOG_PATH = "/tmp/telys_vllm_telemetry.jsonl"

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class Telemetry:
    """Per-phase timer that writes JSONL when enabled.

    Instances are thread-safe; multiple worker threads can record spans
    concurrently.  When ``enabled`` is False, every method is a no-op and
    returns falsy/do-nothing objects.
    """

    def __init__(
        self,
        enabled: bool = False,
        log_path: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._log_path = log_path or DEFAULT_LOG_PATH
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._pid = os.getpid()
        if self._enabled:
            self._open_file()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _open_file(self) -> int | None:
        """Open the telemetry log; disable telemetry on I/O failure.

        Uses an unbuffered O_APPEND file descriptor so each record is an
        atomic POSIX append (safe across threads and, with O_APPEND, across
        processes sharing the same path).
        """
        try:
            Path(self._log_path).parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(
                self._log_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
                0o644,
            )
            return self._fd
        except OSError:
            # Fail safe: disable telemetry if we cannot open the log.
            self._enabled = False
            self._fd = None
            return None

    @contextmanager
    def span(
        self,
        phase: str,
        **context: Any,
    ):
        """Context manager that records elapsed time for ``phase``.

        Yields a mutable dict that the caller can populate with extra context
        before the span closes.
        """
        if not self._enabled:
            yield {}
            return

        span_context: dict[str, Any] = dict(context)
        start_ns = time.perf_counter_ns()
        try:
            yield span_context
        finally:
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            self._record(phase, elapsed_us, span_context)

    def record(
        self,
        phase: str,
        elapsed_us: float,
        **context: Any,
    ) -> None:
        """Record a pre-computed elapsed time for ``phase``."""
        if not self._enabled:
            return
        self._record(phase, elapsed_us, context)

    def _record(
        self,
        phase: str,
        elapsed_us: float,
        context: Mapping[str, Any],
    ) -> None:
        caller_file: str | None = None
        caller_line: int | None = None
        # Only pay for frame inspection when enabled and the caller did not
        # explicitly provide file/line.  Walk up the stack skipping this
        # module and the contextlib machinery so the caller points at the
        # real instrumentation site.
        if "caller_file" not in context or "caller_line" not in context:
            try:
                depth = 2
                this_file = __file__
                while True:
                    frame = sys._getframe(depth)
                    filename = frame.f_code.co_filename
                    if filename != this_file and "contextlib" not in filename:
                        caller_file = filename
                        caller_line = frame.f_lineno
                        break
                    depth += 1
                    if depth > 20:
                        break
            except ValueError:
                pass

        payload: dict[str, Any] = {
            "ts": time.time_ns(),
            "pid": self._pid,
            "phase": phase,
            "elapsed_us": round(elapsed_us, 1),
        }
        if context:
            ctx = dict(context)
            ctx.pop("caller_file", None)
            ctx.pop("caller_line", None)
            payload["context"] = ctx
        if caller_file:
            payload["caller_file"] = caller_file
        if caller_line:
            payload["caller_line"] = caller_line

        line = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            if self._fd is None:
                return
            os.write(self._fd, (line + "\n").encode("utf-8"))

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                finally:
                    self._fd = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


# The canonical disabled instance.  Imports should bind to this lazily.
NOOP = Telemetry(enabled=False)


def is_enabled(value: object) -> bool:
    """Parse an env-var/extra-config value as a telemetry boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES
    return False


def get_telemetry(enabled: bool = False, log_path: str | None = None) -> Telemetry:
    """Return a telemetry instance.

    Callers normally receive a Telemetry through ``TelysConnectorConfig`` so
    the env vars are parsed once.  This helper is exposed for ad-hoc tests.
    """
    if not enabled:
        return NOOP
    return Telemetry(enabled=True, log_path=log_path)


def telemetry_from_env(env: Mapping[str, str] | None = None) -> Telemetry:
    """Build a telemetry instance from environment variables.

    This is intentionally separate from ``TelysConnectorConfig`` so tests can
    create a telemetry instance without constructing a full connector config.
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_TELEMETRY, "")
    enabled = raw.strip().lower() in TRUE_VALUES
    log_path = env.get(ENV_TELEMETRY_LOG) or None
    return get_telemetry(enabled=enabled, log_path=log_path)
