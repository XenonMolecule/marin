# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Object-store operations the claimant pipeline is built on.

Every coordination primitive in ``rdedup`` reduces to four object-store guarantees:

* ``create`` succeeds for exactly one writer (GCS ``if_generation_match=0``);
* ``replace`` succeeds only against the generation the caller last saw, so a stale claim has
  exactly one taker and a worker that lost its claim learns so on its next heartbeat;
* a ranged read can be pinned to a generation, so a reader never mixes bytes of two writes;
* a listing either returns every object under a prefix or raises. A failed listing must never
  read as "no objects" -- that has faked data loss on this project before.

``GcsStore`` provides them on GCS. ``LocalStore`` provides them on a local directory for tests
and laptop smoke runs, with a lock file standing in for GCS's server-side atomicity.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import requests
from google.api_core import exceptions as gexc
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY, DEFAULT_RETRY_IF_GENERATION_SPECIFIED

# Uploads above this size stream through a resumable session in chunks of this many bytes, so a
# multi-GB shuffle object never needs a second in-memory copy.
_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
_READ_TIMEOUT = (30, 600)
_GCS_PROJECT = "hai-gcp-models"
_POOL_SIZE = 128


@dataclasses.dataclass(frozen=True)
class ObjectInfo:
    """Metadata for one object, as a listing or stat reports it."""

    path: str
    generation: int
    updated: float
    size: int


class PreconditionFailed(Exception):
    """A generation precondition did not hold: another writer got there first."""


class ObjectStore(Protocol):
    """The storage operations the pipeline needs. Paths are full URLs (``gs://b/k``) or local paths."""

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        """Bytes ``[start, end)`` of ``path``; with ``generation``, fail unless the object is that write."""
        ...

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        """Overwrite ``path`` unconditionally and return the new generation."""
        ...

    def create(self, path: str, data: bytes) -> int:
        """Create ``path`` only if absent; raise :class:`PreconditionFailed` if it exists."""
        ...

    def replace(self, path: str, data: bytes, generation: int) -> int:
        """Overwrite ``path`` only if its generation is ``generation``; else :class:`PreconditionFailed`."""
        ...

    def stat(self, path: str) -> ObjectInfo | None:
        """Metadata for ``path``, or ``None`` if it does not exist."""
        ...

    def list(self, prefix: str) -> list[ObjectInfo]:
        """Every object under ``prefix``. Raises on any listing error, never returns a partial list."""
        ...

    def delete(self, path: str) -> None:
        """Delete ``path`` if present."""
        ...

    def now(self) -> float:
        """Current time on the clock that ``ObjectInfo.updated`` is measured against."""
        ...


def read_json(store: ObjectStore, path: str) -> dict:
    return json.loads(store.read(path))


def dumps(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode()


def delete_prefix(store: ObjectStore, prefix: str, threads: int = 32) -> int:
    """Delete every object under ``prefix``; returns how many were deleted."""
    objects = store.list(prefix)
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda info: store.delete(info.path), objects))
    return len(objects)


def _new_client() -> storage.Client:
    """A client whose connection pool fits the ranged-read thread pools (default is 10)."""
    client = storage.Client(project=_GCS_PROJECT)
    adapter = requests.adapters.HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
    client._http.mount("https://", adapter)
    return client


def _split_gs(path: str) -> tuple[str, str]:
    if not path.startswith("gs://"):
        raise ValueError(f"not a gs:// path: {path!r}")
    bucket, _, key = path.removeprefix("gs://").partition("/")
    return bucket, key


class GcsStore:
    """:class:`ObjectStore` on Google Cloud Storage.

    One ``storage.Client`` per process: the client is not fork-safe, so a forked or spawned child
    builds its own on first use.
    """

    def __init__(self) -> None:
        self._client: storage.Client | None = None
        self._pid = -1
        self._lock = threading.Lock()

    def _blob(self, path: str, generation: int | None = None) -> storage.Blob:
        with self._lock:
            if self._client is None or self._pid != os.getpid():
                self._client = _new_client()
                self._pid = os.getpid()
            client = self._client
        bucket, key = _split_gs(path)
        return client.bucket(bucket).blob(key, generation=generation)

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        if start is not None and end is not None and end <= start:
            return b""
        blob = self._blob(path)
        return blob.download_as_bytes(
            start=start,
            # google-cloud-storage takes an INCLUSIVE end offset.
            end=None if end is None else end - 1,
            if_generation_match=generation,
            timeout=_READ_TIMEOUT,
            retry=DEFAULT_RETRY,
            checksum=None if start is not None or end is not None else "md5",
        )

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        blob = self._blob(path)
        if isinstance(data, bytes | bytearray | memoryview):
            blob.upload_from_string(bytes(data), timeout=_READ_TIMEOUT, retry=DEFAULT_RETRY)
            return int(blob.generation)
        # Streaming writes: the payload is deterministic, so a resumable session that restarts
        # from scratch after an error rewrites identical bytes.
        blob.chunk_size = _UPLOAD_CHUNK_BYTES
        with blob.open("wb", retry=DEFAULT_RETRY, ignore_flush=True) as handle:
            for piece in data:
                handle.write(piece)
        blob.reload()
        return int(blob.generation)

    def create(self, path: str, data: bytes) -> int:
        blob = self._blob(path)
        try:
            blob.upload_from_string(
                data, if_generation_match=0, timeout=_READ_TIMEOUT, retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED
            )
        except gexc.PreconditionFailed as e:
            raise PreconditionFailed(path) from e
        return int(blob.generation)

    def replace(self, path: str, data: bytes, generation: int) -> int:
        blob = self._blob(path)
        try:
            blob.upload_from_string(
                data,
                if_generation_match=generation,
                timeout=_READ_TIMEOUT,
                retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
            )
        except gexc.PreconditionFailed as e:
            raise PreconditionFailed(path) from e
        return int(blob.generation)

    def stat(self, path: str) -> ObjectInfo | None:
        blob = self._blob(path)
        try:
            blob.reload(timeout=_READ_TIMEOUT, retry=DEFAULT_RETRY)
        except gexc.NotFound:
            return None
        return ObjectInfo(path, int(blob.generation), blob.updated.timestamp(), int(blob.size))

    def list(self, prefix: str) -> list[ObjectInfo]:
        bucket, key = _split_gs(prefix)
        with self._lock:
            if self._client is None or self._pid != os.getpid():
                self._client = _new_client()
                self._pid = os.getpid()
            client = self._client
        blobs = client.list_blobs(
            bucket,
            prefix=key,
            fields="items(name,generation,updated,size),nextPageToken",
            page_size=5000,
            timeout=_READ_TIMEOUT,
            retry=DEFAULT_RETRY,
        )
        return [
            ObjectInfo(f"gs://{bucket}/{b.name}", int(b.generation), b.updated.timestamp(), int(b.size)) for b in blobs
        ]

    def delete(self, path: str) -> None:
        try:
            self._blob(path).delete(timeout=_READ_TIMEOUT, retry=DEFAULT_RETRY)
        except gexc.NotFound:
            pass

    def now(self) -> float:
        return time.time()


class LocalStore:
    """:class:`ObjectStore` on a local directory, for tests and laptop smoke runs.

    Generations and update times live in a sidecar file per object; a lock file serializes every
    mutation so concurrent processes see the same create/replace atomicity GCS gives. ``clock``
    lets tests age claims without sleeping.
    """

    _META_SUFFIX = ".__meta__"

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock_path = os.path.join(tempfile.gettempdir(), "rdedup_localstore.lock")

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with open(self._lock_path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _meta(self, path: str) -> dict | None:
        """Sidecar metadata; a plain file written outside the store reads as generation 0."""
        try:
            with open(path + self._META_SUFFIX) as f:
                return json.load(f)
        except FileNotFoundError:
            pass
        try:
            return {"generation": 0, "updated": os.path.getmtime(path)}
        except FileNotFoundError:
            return None

    def _put(self, path: str, data: bytes) -> int:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = self._meta(path)
        generation = (meta["generation"] if meta else 0) + 1
        tmp = f"{path}.__tmp__{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        with open(path + self._META_SUFFIX, "w") as f:
            json.dump({"generation": generation, "updated": self._clock()}, f)
        return generation

    def read(self, path: str, start: int | None = None, end: int | None = None, generation: int | None = None) -> bytes:
        with self._locked():
            meta = self._meta(path)
            if meta is None:
                raise FileNotFoundError(path)
            if generation is not None and meta["generation"] != generation:
                raise PreconditionFailed(path)
            with open(path, "rb") as f:
                data = f.read()
        return data[start:end]

    def write(self, path: str, data: bytes | Iterable[bytes]) -> int:
        payload = bytes(data) if isinstance(data, bytes | bytearray | memoryview) else b"".join(data)
        with self._locked():
            return self._put(path, payload)

    def create(self, path: str, data: bytes) -> int:
        with self._locked():
            if self._meta(path) is not None:
                raise PreconditionFailed(path)
            return self._put(path, data)

    def replace(self, path: str, data: bytes, generation: int) -> int:
        with self._locked():
            meta = self._meta(path)
            if meta is None or meta["generation"] != generation:
                raise PreconditionFailed(path)
            return self._put(path, data)

    def stat(self, path: str) -> ObjectInfo | None:
        meta = self._meta(path)
        if meta is None:
            return None
        return ObjectInfo(path, meta["generation"], meta["updated"], os.path.getsize(path))

    def list(self, prefix: str) -> list[ObjectInfo]:
        directory = prefix if prefix.endswith(os.sep) else os.path.dirname(prefix)
        out = []
        for root, _dirs, files in _walk(directory):
            for name in files:
                full = os.path.join(root, name)
                if name.endswith(self._META_SUFFIX) or "__tmp__" in name or not full.startswith(prefix):
                    continue
                info = self.stat(full)
                if info is not None:
                    out.append(info)
        return sorted(out, key=lambda info: info.path)

    def delete(self, path: str) -> None:
        with self._locked():
            for p in (path, path + self._META_SUFFIX):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    def now(self) -> float:
        return self._clock()


def _walk(directory: str) -> Iterator[tuple[str, list[str], list[str]]]:
    if not os.path.isdir(directory):
        return iter(())
    return os.walk(directory)


def store_for(path: str) -> ObjectStore:
    """The store that serves ``path``: GCS for ``gs://`` URLs, the local filesystem otherwise."""
    return GcsStore() if path.startswith("gs://") else LocalStore()
