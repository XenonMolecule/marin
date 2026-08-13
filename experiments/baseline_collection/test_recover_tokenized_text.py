# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the token-cache -> source-text recovery rules.

The framing rule is the load-bearing piece of the whole reconstruction: getting it
wrong by a single character took the content-hash match rate from 100% to 0%
against the real corpus, so it is pinned here rather than left to a smoke run.
"""

from types import SimpleNamespace

import pytest
from marin.datakit.normalize import generate_id

from experiments.baseline_collection.recover_tokenized_text import id_bucket, strip_framing

BOS, EOS, SPACE = 128000, 128001, 220


@pytest.fixture
def framing():
    return SimpleNamespace(bos=BOS, eos=EOS, space=SPACE)


def test_strip_framing_removes_bos_eos_and_the_appended_space(framing):
    core, anomaly = strip_framing([BOS, 1, 2, 3, SPACE, EOS], framing)
    assert core == [1, 2, 3]
    assert anomaly is None


def test_strip_framing_keeps_a_documents_own_trailing_space(framing):
    """BatchTokenizer appends its space unconditionally, so a document that already
    ended in whitespace is stored with two. Exactly one belongs to the framing."""
    core, anomaly = strip_framing([BOS, 1, 2, SPACE, SPACE, EOS], framing)
    assert core == [1, 2, SPACE]
    assert anomaly is None


def test_strip_framing_flags_a_row_with_no_trailing_space(framing):
    core, anomaly = strip_framing([BOS, 1, 2, EOS], framing)
    assert core == [1, 2]
    assert anomaly == "missing-trailing-space"


def test_strip_framing_flags_missing_bos(framing):
    _, anomaly = strip_framing([1, 2, SPACE, EOS], framing)
    assert anomaly == "missing-bos"


def test_strip_framing_flags_missing_eos(framing):
    _, anomaly = strip_framing([BOS, 1, 2, SPACE], framing)
    assert anomaly == "missing-eos"


def _content_ids(n: int) -> list[str]:
    """Ids shaped like real ones: `generate_id` returns 32 hex chars of xxh3_128,
    so the leading digits that `id_bucket` reads are uniformly distributed. Ids
    built by zero-padding small integers are NOT representative — every one of
    them starts `00000000` and lands in bucket 0."""
    return [generate_id(f"document-{i}") for i in range(n)]


def test_id_bucket_is_stable_and_in_range():
    """Both sides of the join bucket independently, in different jobs, so the
    function must depend only on the id text."""
    ids = _content_ids(500)
    buckets = [id_bucket(i, 64) for i in ids]
    assert all(0 <= b < 64 for b in buckets)
    assert buckets == [id_bucket(i, 64) for i in ids]


def test_id_bucket_spreads_across_all_buckets():
    assert len({id_bucket(i, 64) for i in _content_ids(20000)}) == 64
