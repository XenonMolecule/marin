# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the completed-registry cache (summary object + monotonic union)."""

import io
import json

import pytest

from experiments.baseline_collection import run_extract_standalone as res

PREFIX = "gs://bucket/ns/_b_done"


class FakeGcs:
    """Registry-shaped fake: tracks ls/open calls, serves sentinel names and a summary blob."""

    def __init__(self, entries, summary=None):
        self.entries = set(entries)
        self.summary = summary  # None -> no summary object
        self.ls_calls = 0
        self.open_calls = 0

    def ls(self, prefix, refresh=False):
        self.ls_calls += 1
        return [f"{prefix}/data-{e}" for e in sorted(self.entries)]

    def open(self, path, mode="r"):
        self.open_calls += 1
        assert path.endswith(res.REGISTRY_SUMMARY_SUFFIX) and "_b_done/" not in path
        if "w" in mode:
            buf = io.StringIO()
            close = buf.close
            buf.close = lambda: (self._store(buf.getvalue()), close())
            return buf
        if self.summary is None:
            raise FileNotFoundError(path)
        return io.StringIO(self.summary)

    def _store(self, value):
        self.summary = value


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 1_000.0}
    monkeypatch.setattr(res.time, "time", lambda: state["now"])
    return state


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def use(fs):
        holder["fs"] = fs
        monkeypatch.setattr(res.fsspec, "filesystem", lambda proto: fs)
        monkeypatch.setattr(res.fsspec, "open", lambda path, mode="r": fs.open(path, mode))
        return fs

    monkeypatch.setattr(res, "_registry_cache", {})
    return use


def test_first_load_lists_then_summary_get_replaces_listing(clock, fake):
    fs = fake(FakeGcs({"a", "b"}, summary=json.dumps({"written": 0, "hashes": ["a", "b", "c"]})))
    assert res._load_completed_registry(PREFIX) == {"a", "b"}
    assert (fs.ls_calls, fs.open_calls) == (1, 0)  # first call: full listing, no summary read

    clock["now"] += res.REGISTRY_FRESH_SECONDS + 1
    assert res._load_completed_registry(PREFIX) == {"a", "b", "c"}
    assert (fs.ls_calls, fs.open_calls) == (1, 1)  # refresh served by ONE summary GET


def test_no_summary_falls_back_to_listing(clock, fake):
    fs = fake(FakeGcs({"a"}, summary=None))
    res._load_completed_registry(PREFIX)
    clock["now"] += res.REGISTRY_FRESH_SECONDS + 1
    fs.entries.add("b")
    assert res._load_completed_registry(PREFIX) == {"a", "b"}
    assert fs.ls_calls == 2  # summary 404 -> paged listing, same behavior as before


def test_union_is_monotonic_against_shrunken_summary(clock, fake):
    fs = fake(FakeGcs({"a", "b"}, summary=json.dumps({"written": 0, "hashes": ["b"]})))
    assert res._load_completed_registry(PREFIX) == {"a", "b"}
    clock["now"] += res.REGISTRY_FRESH_SECONDS + 1
    # Summary under-reports (lost-update race); previously seen entries must survive.
    assert res._load_completed_registry(PREFIX) == {"a", "b"}


def test_fresh_window_serves_cache_with_no_calls(clock, fake):
    fs = fake(FakeGcs({"a"}, summary=None))
    res._load_completed_registry(PREFIX)
    calls = (fs.ls_calls, fs.open_calls)
    clock["now"] += res.REGISTRY_FRESH_SECONDS / 2
    res._load_completed_registry(PREFIX)
    assert (fs.ls_calls, fs.open_calls) == calls


def test_full_listing_returns_after_belt_interval(clock, fake):
    fs = fake(FakeGcs({"a"}, summary=json.dumps({"written": 0, "hashes": ["a"]})))
    res._load_completed_registry(PREFIX)
    clock["now"] += res.REGISTRY_FULL_LIST_SECONDS + 1
    fs.entries.add("z")  # visible only to a real listing
    assert "z" in res._load_completed_registry(PREFIX)
    assert fs.ls_calls == 2


def test_write_summary_roundtrips_through_reader(clock, fake):
    fs = fake(FakeGcs({"a", "b"}, summary=None))
    res._write_registry_summary(PREFIX)
    assert json.loads(fs.summary)["hashes"] == ["a", "b"]
    res._load_completed_registry(PREFIX)  # full listing (cold cache)
    clock["now"] += res.REGISTRY_FRESH_SECONDS + 1
    fs.entries.add("c")
    res._write_registry_summary(PREFIX)
    clock["now"] += 1
    assert res._load_completed_registry(PREFIX) == {"a", "b", "c"}
