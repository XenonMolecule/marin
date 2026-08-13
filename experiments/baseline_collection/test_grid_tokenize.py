# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the co-partitioned tokenizer.

The property that matters is the one ``datakit_store`` enforces with a hard
failure: tokenized output must be one file per input shard, same basename, same
row order, with ids matching the grid tables exactly. Everything else about this
module is replaceable; that invariant is not.

These call :func:`grid_tokenize.tokenize_shard` directly rather than driving a
Zephyr context — the fan-out is Zephyr's job and is covered by Zephyr, while the
shape and alignment of the output is ours. The tokenizer is stubbed for the same
reason: ``BatchTokenizer`` belongs to Levanter, and downloading a real one would
make these slow and network-dependent.
"""

from __future__ import annotations

import gzip
import json

import pyarrow.parquet as pq
import pytest

from experiments.baseline_collection import grid_corpora, grid_tokenize
from experiments.baseline_collection.grid_corpora import Format, GridCorpus, output_stem, read_shard
from experiments.fsspec_paths import fsspec_glob

DATASET = "dclm_10k"
N_SHARDS = 3
DOCS_PER_SHARD = 7
EMPTY_SHARD = 1
NO_CAP = 10_000_000


def _stub_batcher(batch):
    """One token per word, wrapped in BOS/EOS sentinels."""
    return [{"input_ids": [1, *range(10, 10 + len(r["text"].split())), 2]} for r in batch]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny corpus that includes an empty shard."""
    source = tmp_path / "src"
    source.mkdir()
    for shard_idx in range(N_SHARDS):
        path = source / f"data-{shard_idx:05d}-of-{N_SHARDS:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            if shard_idx == EMPTY_SHARD:
                continue
            for doc_idx in range(DOCS_PER_SHARD):
                fh.write(json.dumps({"text": f"shard {shard_idx} doc {doc_idx} words words"}) + "\n")

    stub = GridCorpus(str(source), "us-central1", Format.JSONL_GZ)
    monkeypatch.setitem(grid_corpora.GRID_CORPORA, DATASET, stub)
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "out"))
    monkeypatch.setattr(grid_tokenize, "OUTPUT_BASE", str(tmp_path / "out"))
    return stub


def _tokenize_all(corpus) -> list[str]:
    shards = grid_corpora.list_shards(corpus)
    for shard in shards:
        grid_tokenize.tokenize_shard(DATASET, shard, corpus, _stub_batcher)
    return shards


def test_output_is_copartitioned_with_the_source(corpus):
    """One file per input shard, same stem, same ids in the same order.

    This is exactly what ``datakit_store``'s positional join asserts, so a
    violation here would surface later as a store crash or, worse, a silently
    mis-aligned grid.
    """
    shards = _tokenize_all(corpus)
    written = fsspec_glob(f"{grid_tokenize.tokenize_dir(DATASET)}/*.parquet")
    assert {output_stem(p) for p in written} == {output_stem(s) for s in shards}

    for shard in shards:
        table = pq.read_table(grid_tokenize.tokenize_output(DATASET, shard)).to_pydict()
        assert table["id"] == read_shard(shard, corpus, NO_CAP).ids, f"id order diverged for {output_stem(shard)}"


def test_empty_shard_still_gets_a_file(corpus):
    """An empty input shard must produce a zero-row file, not a missing one."""
    shards = _tokenize_all(corpus)
    empty = pq.read_table(grid_tokenize.tokenize_output(DATASET, shards[EMPTY_SHARD]))
    full = pq.read_table(grid_tokenize.tokenize_output(DATASET, shards[0]))
    assert empty.num_rows == 0
    assert full.num_rows == DOCS_PER_SHARD
    # Schema must survive the empty case or the join breaks on column mismatch.
    assert empty.schema == full.schema


def test_rerun_skips_finished_shards(corpus):
    """Resumability: after one pass nothing is pending."""
    shards = _tokenize_all(corpus)
    assert grid_tokenize.pending(DATASET, shards) == []


def test_bos_and_eos_are_present(corpus):
    """Sequences must carry the sentinels, since training expects them.

    Guards the reason this module delegates to Levanter's ``BatchTokenizer``
    instead of calling an HF tokenizer: the registry still carries a
    ``nemotron_full_bos_fixed`` cache from the last time BOS went wrong.
    """
    shards = _tokenize_all(corpus)
    table = pq.read_table(grid_tokenize.tokenize_output(DATASET, shards[0])).to_pydict()
    assert table["input_ids"], "expected non-empty shard"
    for seq in table["input_ids"]:
        assert seq[0] == 1, "missing BOS"
        assert seq[-1] == 2, "missing EOS"
        assert len(seq) > 2
