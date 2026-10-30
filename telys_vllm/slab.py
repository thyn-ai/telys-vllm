# SPDX-License-Identifier: Apache-2.0
"""Append-only slab file for KV payload bytes.

Telys stores the index (chunk key → slab offset/length + provenance metadata).
The slab stores the actual encoded chunk bytes.  Writes are atomic append so
multiple processes can share one slab; reads are pread-style and concurrent.
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path
from typing import BinaryIO

_SLAB_MAGIC = b"TELYKV01"
_SLAB_VERSION = 1
_HEADER = struct.Struct(f"<{len(_SLAB_MAGIC)}sQ")  # magic + version
_RECORD_LEN = struct.Struct("<Q")  # payload length prefix


class SlabError(RuntimeError):
    """Raised when the slab cannot honor a read or write."""


class KVSlab:
    """Thread-safe append-only slab for opaque payload bytes.

    Format on disk::

        [magic:8][version:8]
        [length:8][payload bytes] * N

    Each ``append`` returns the byte offset of the length prefix and the payload
    length.  Callers store that (offset, length) pair in the Telys index.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._read_fd: BinaryIO | None = None
        self._write_fd: int | None = None
        self._closed = False

        # Open/create the slab and write/read header if new.
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Create the slab file with a header if it does not exist."""
        if self._path.exists():
            self._read_fd = open(self._path, "rb")  # noqa: SIM115
            self._validate_header()
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            f.write(_HEADER.pack(_SLAB_MAGIC, _SLAB_VERSION))
        self._read_fd = open(self._path, "rb")  # noqa: SIM115

    def _validate_header(self) -> None:
        """Ensure the existing file has the expected header."""
        if self._read_fd is None:
            raise SlabError("slab not open")
        self._read_fd.seek(0)
        header = self._read_fd.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise SlabError("slab file is truncated (header)")
        magic, version = _HEADER.unpack(header)
        if magic != _SLAB_MAGIC:
            raise SlabError(f"slab has unexpected magic {magic!r}")
        if version != _SLAB_VERSION:
            raise SlabError(f"slab version {version} is not supported")

    def append(self, payload: bytes) -> tuple[int, int]:
        """Atomically append ``payload`` and return (offset, length).

        The returned ``offset`` is the file position of the length prefix for
        this record, so ``read(offset, length)`` reconstructs exactly this
        payload.
        """
        if self._closed:
            raise SlabError("slab is closed")
        payload_bytes = bytes(payload)
        length = len(payload_bytes)
        record = _RECORD_LEN.pack(length) + payload_bytes

        with self._lock:
            if self._write_fd is None:
                # O_APPEND guarantees atomic append writes on POSIX.
                self._write_fd = os.open(
                    self._path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                    0o644,
                )
            # os.write with O_APPEND ignores the file offset and appends atomically.
            offset = os.lseek(self._write_fd, 0, os.SEEK_END)
            written = 0
            while written < len(record):
                n = os.write(self._write_fd, record[written:])
                if n == 0:
                    raise SlabError("short write to slab")
                written += n

        return offset, length

    def read(self, offset: int, length: int) -> bytes:
        """Read the payload stored at ``offset`` with ``length`` bytes."""
        if self._closed:
            raise SlabError("slab is closed")
        if offset < _HEADER.size:
            raise SlabError(f"read offset {offset} is inside the slab header")
        if length < 0:
            raise SlabError(f"read length {length} is negative")
        if self._read_fd is None:
            raise SlabError("slab not open")

        self._read_fd.seek(offset)
        len_bytes = self._read_fd.read(_RECORD_LEN.size)
        if len(len_bytes) != _RECORD_LEN.size:
            raise SlabError("slab record length prefix is truncated")
        (stored_length,) = _RECORD_LEN.unpack(len_bytes)
        if stored_length != length:
            raise SlabError(
                f"slab record length mismatch: index says {length}, "
                f"slab says {stored_length}"
            )
        payload = self._read_fd.read(length)
        if len(payload) != length:
            raise SlabError("slab payload is truncated")
        return payload

    def fsync(self) -> None:
        """Flush slab data to stable storage."""
        if self._write_fd is not None:
            os.fsync(self._write_fd)

    def clear(self) -> None:
        """Truncate the slab, leaving only a fresh header."""
        with self._lock:
            if self._write_fd is not None:
                os.close(self._write_fd)
                self._write_fd = None
            if self._read_fd is not None:
                self._read_fd.close()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                f.write(_HEADER.pack(_SLAB_MAGIC, _SLAB_VERSION))
            self._read_fd = open(self._path, "rb")  # noqa: SIM115

    def close(self) -> None:
        """Close file descriptors."""
        self._closed = True
        with self._lock:
            if self._write_fd is not None:
                try:
                    os.fsync(self._write_fd)
                finally:
                    os.close(self._write_fd)
                    self._write_fd = None
            if self._read_fd is not None:
                self._read_fd.close()
                self._read_fd = None
