# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cross-region replicator.

The bug this guards against shipped once already: ``gcsfs.find()`` returns keys
with the ``gs://`` scheme stripped, but ``_listing`` sliced relative keys using
the length of the scheme-bearing root. Every key came out truncated by
``len("gs://")`` characters — ``.artifact.json`` became ``fact.json`` — and the
resulting copy call 404'd against a path that was never the source object. A
stub filesystem reproduces gcsfs's scheme-stripping behavior exactly, so this
runs with no network and no real bucket.
"""

from __future__ import annotations

import pytest

from experiments.baseline_collection import replicate_prefix

BUCKET = "fake-bucket"


class _StubFS:
    """Minimal fsspec-shaped filesystem that strips ``gs://`` like gcsfs does.

    ``store`` maps bucket-relative paths (no scheme) to CRC32C strings.
    """

    def __init__(self, store: dict[str, dict]):
        self.store = store
        self.copied: dict[str, str] = {}

    def _strip_protocol(self, path: str) -> str:
        return path.removeprefix("gs://").rstrip("/")

    def find(self, root: str, detail: bool = False):
        root = self._strip_protocol(root)
        return {
            path: {"type": "file", "size": info["size"], "crc32c": info["crc32c"]}
            for path, info in self.store.items()
            if path.startswith(root + "/")
        }

    def exists(self, path: str) -> bool:
        return any(p.startswith(self._strip_protocol(path)) for p in self.store)

    def info(self, path: str) -> dict:
        stripped = self._strip_protocol(path)
        info = self.store[stripped]
        return {"type": "file", "size": info["size"], "crc32c": info["crc32c"]}

    def cp_file(self, src: str, dst: str) -> None:
        stripped_src, stripped_dst = self._strip_protocol(src), self._strip_protocol(dst)
        self.store[stripped_dst] = self.store[stripped_src]
        self.copied[dst] = self.store[stripped_src]["crc32c"]


@pytest.fixture(autouse=True)
def _stub_url_to_fs(monkeypatch):
    """Route ``fsspec.core.url_to_fs`` to the stub for every test in this module."""
    fs_holder: dict[str, _StubFS] = {}

    def install(store: dict[str, dict]) -> _StubFS:
        fs = _StubFS(store)
        fs_holder["fs"] = fs
        return fs

    monkeypatch.setattr(
        replicate_prefix.fsspec.core, "url_to_fs", lambda url: (fs_holder["fs"], fs_holder["fs"]._strip_protocol(url))
    )
    return install


def test_keys_are_not_truncated_by_the_scheme_length(_stub_url_to_fs):
    """The regression case: a leading-dot filename must survive listing intact.

    Before the fix, ``.artifact.json`` (14 chars) was sliced 5 chars short of
    where it should be — exactly ``len("gs://")`` — producing ``fact.json``.
    """
    store = {
        f"{BUCKET}/store/dclm/.artifact.json": {"size": 512, "crc32c": "AAAA=="},
        f"{BUCKET}/store/dclm/cluster=0/quality=0/part-00000": {"size": 4096, "crc32c": "BBBB=="},
    }
    _stub_url_to_fs(store)

    listing = replicate_prefix._listing(replicate_prefix.fsspec.core.url_to_fs("gs://x")[0], f"gs://{BUCKET}/store/dclm")

    assert set(listing) == {".artifact.json", "cluster=0/quality=0/part-00000"}


def test_replicate_copies_and_verifies_a_fresh_destination(_stub_url_to_fs):
    store = {
        f"{BUCKET}/store/dclm/.artifact.json": {"size": 512, "crc32c": "AAAA=="},
        f"{BUCKET}/store/dclm/cluster=0/quality=0/part-00000": {"size": 4096, "crc32c": "BBBB=="},
    }
    fs = _stub_url_to_fs(store)

    replicate_prefix.replicate(f"gs://{BUCKET}/store/dclm", f"gs://{BUCKET}-east5/store/dclm", verify_only=False)

    assert fs.copied == {
        f"gs://{BUCKET}-east5/store/dclm/.artifact.json": "AAAA==",
        f"gs://{BUCKET}-east5/store/dclm/cluster=0/quality=0/part-00000": "BBBB==",
    }


def test_verify_only_raises_on_a_missing_destination(_stub_url_to_fs):
    store = {f"{BUCKET}/store/dclm/.artifact.json": {"size": 512, "crc32c": "AAAA=="}}
    _stub_url_to_fs(store)

    with pytest.raises(RuntimeError, match="verify-only"):
        replicate_prefix.replicate(f"gs://{BUCKET}/store/dclm", f"gs://{BUCKET}-east5/store/dclm", verify_only=True)
