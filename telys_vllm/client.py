# SPDX-License-Identifier: Apache-2.0
"""Abstract Telys client boundary.

The connector is designed against this Protocol only; the concrete on-device
Telys binding is provided at deployment time through the
``telys_client_factory`` config key (see DESIGN.md section 3.6).

Provenance model: every read returns the Telys ``plan_hash`` / ``result_id``
pair identifying the exact retrieval that produced the record, so any KV
block the connector injects is traceable to a Telys read.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TelysProvenance:
    """Provenance attached by Telys to a single-key read."""

    plan_hash: str
    result_id: str


@dataclass(frozen=True)
class TelysRecord:
    """One key/value entry returned by Telys, with read provenance."""

    key: str
    value: bytes
    provenance: TelysProvenance
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TelysMaintenanceReport:
    """Result of a compaction / IVF maintenance pass."""

    entries_scanned: int
    entries_dropped: int


@runtime_checkable
class TelysClient(Protocol):
    """Single-key, filtered-query interface to the on-device Telys store.

    Implementations must be deterministic with respect to store content:
    the same key always reads back the same value and metadata until the
    key is overwritten or deleted.
    """

    def contains(self, key: str) -> bool:
        """Return True if ``key`` is present without fetching its value."""
        ...

    def chunk_length(self, start_key: str) -> int:
        """Return the number of blocks in the stored chunk starting at ``start_key``.

        A return value of 0 means no chunk starts at this key. This lets the
        connector walk the prefix in chunk-sized strides instead of one block at
        a time.
        """
        return 0

    def get(self, key: str) -> TelysRecord | None:
        """Return the record for ``key`` with provenance, or None on miss."""
        ...

    def get_batch(self, keys: list[str]) -> list[TelysRecord | None]:
        """Return records for ``keys`` in order; None for missing keys."""
        ...

    def put(self, key: str, value: bytes, metadata: Mapping[str, str]) -> None:
        """Store ``value`` under ``key`` with string metadata.

        Idempotent per key: re-putting an identical value is a no-op for
        readers.
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete ``key``. Returns True when the key existed."""
        ...

    def compact(self) -> TelysMaintenanceReport:
        """Run a compaction / index maintenance pass (the eviction hook)."""
        ...

    def close(self) -> None:
        """Release client resources. Called from the worker shutdown path."""
        ...
