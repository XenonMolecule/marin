# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the quality x domain grid scorer.

The property under test is the one the whole design rests on: **running the two
stages fused must be indistinguishable from running them separately**, so a
topic-only pass tonight and a quality catch-up months later add up to exactly
what a single fused run would have produced.

Everything runs against a local temp directory (fsspec treats local paths and
gs:// alike) with stubbed models, so these are fast and hermetic. The models
themselves are covered by the parity test, not here.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.baseline_collection import grid_corpora, grid_label
from experiments.baseline_collection.grid_corpora import (
    Format,
    GridCorpus,
    Stage,
    assign_shards,
    output_stem,
    read_shard,
)

DATASET = "dclm_10k"
N_SHARDS = 4
DOCS_PER_SHARD = 25


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny on-disk corpus, with all grid outputs redirected under tmp_path."""
    source = tmp_path / "source"
    source.mkdir()
    for shard_idx in range(N_SHARDS):
        path = source / f"data-{shard_idx:05d}-of-{N_SHARDS:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for doc_idx in range(DOCS_PER_SHARD):
                fh.write(
                    json.dumps(
                        {
                            "text": f"shard {shard_idx} doc {doc_idx} " + "lorem ipsum " * (doc_idx % 5),
                            "url": f"http://example.com/{shard_idx}/{doc_idx}",
                            "warc_record_id": f"warc-{shard_idx}-{doc_idx}",
                        }
                    )
                    + "\n"
                )

    stub = GridCorpus(str(source), "us-central1", Format.JSONL_GZ, native_id_field="warc_record_id")
    monkeypatch.setitem(grid_corpora.GRID_CORPORA, DATASET, stub)
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "out"))
    return stub


@pytest.fixture
def corpus_with_empty_shard(tmp_path, monkeypatch):
    """A corpus where one shard is a valid but empty gzip.

    Not a hypothetical: 7 of fineweb_edu's 1,469 shards are 20-byte empty gzips —
    WARCs from which nothing survived the upstream filter.
    """
    source = tmp_path / "source_empty"
    source.mkdir()
    for shard_idx in range(3):
        path = source / f"data-{shard_idx:05d}-of-00003.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            if shard_idx == 1:  # the empty one
                continue
            for doc_idx in range(10):
                fh.write(json.dumps({"text": f"s{shard_idx} d{doc_idx}", "url": "http://e.com"}) + "\n")

    stub = GridCorpus(str(source), "us-central1", Format.JSONL_GZ)
    monkeypatch.setitem(grid_corpora.GRID_CORPORA, DATASET, stub)
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "out_empty"))
    return stub


@pytest.fixture
def stub_models(monkeypatch):
    """Deterministic stand-ins for both models, keyed on document text.

    Keying on text (not on position) is what makes the fused-vs-separate
    comparison meaningful: if the scorer ever reordered rows, the stubs would
    still return the same value per document and the row-order bug would show up
    as a mismatch against the source shard rather than being masked.
    """

    def fake_load_topic(max_length, mesh):
        return "params", "tokenizer", [f"topic-{i}" for i in range(grid_corpora.NUM_TOPICS)]

    def fake_score_topic(docs, params, tokenizer, max_length, mesh):
        topics = np.array([len(t) % grid_corpora.NUM_TOPICS for t in docs.texts], dtype=np.int64)
        probs = np.array([0.5 + (len(t) % 50) / 100.0 for t in docs.texts], dtype=np.float32)
        lengths = np.array([max(1, len(t) // 4) for t in docs.texts], dtype=np.int64)
        return topics, probs, lengths

    def fake_score_quality(scorer, texts):
        scores = np.array([(len(t) % 100) / 100.0 for t in texts], dtype=np.float32)
        return scores, np.digitize(scores, grid_label.BUCKET_EDGES).astype(np.int8)

    monkeypatch.setattr(grid_label, "load_topic", fake_load_topic)
    monkeypatch.setattr(grid_label, "score_topic", fake_score_topic)
    monkeypatch.setattr(grid_label, "load_quality", lambda model_dir: "scorer")
    monkeypatch.setattr(grid_label, "score_quality", fake_score_quality)
    monkeypatch.setattr(grid_label, "build_mesh", lambda scope: _FakeMesh())
    monkeypatch.setattr(grid_label, "process_partition", lambda shards, scope: shards)


class _FakeMesh:
    """Stands in for a jax Mesh; only ``devices.size`` is read outside the model."""

    devices = np.zeros(1)


def _label(stages: str, quality_model: str | None = "unused") -> None:
    grid_label.run_label(
        DATASET,
        grid_label.parse_stages(stages),
        source="native",
        num_chunks=1,
        chunk_idx=0,
        max_length=512,
        quality_model=quality_model,
        mesh_scope="local",
    )


def _read_stage_tables(stage: Stage) -> dict[str, dict]:
    """Every output table for ``stage``, keyed by shard stem."""
    from marin.utils import fsspec_glob

    directory = grid_corpora.stage_output_dir(DATASET, stage)
    pattern = f"{directory}/outputs/main/*.parquet" if stage is Stage.QUALITY else f"{directory}/*.parquet"
    return {output_stem(path): pq.read_table(path).to_pydict() for path in sorted(fsspec_glob(pattern))}


def test_fused_and_separate_runs_produce_identical_output(corpus, stub_models, tmp_path, monkeypatch):
    """The load-bearing property: order of operations must not change the result."""
    _label("topic,quality")
    fused = {stage: _read_stage_tables(stage) for stage in (Stage.TOPIC, Stage.QUALITY)}

    # Same corpus, a clean output tree, but the stages run as two separate passes.
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "out_split"))
    _label("topic", quality_model=None)
    _label("quality")
    split = {stage: _read_stage_tables(stage) for stage in (Stage.TOPIC, Stage.QUALITY)}

    assert fused == split


def test_stages_are_independently_resumable(corpus, stub_models):
    """A topic-only run must leave every quality shard still pending, and vice versa."""
    shards = assign_shards(grid_corpora.list_shards(corpus), 1, 0)

    _label("topic", quality_model=None)
    assert grid_corpora.pending_shards(DATASET, Stage.TOPIC, shards) == []
    assert grid_corpora.pending_shards(DATASET, Stage.QUALITY, shards) == shards

    _label("quality")
    assert grid_corpora.pending_shards(DATASET, Stage.QUALITY, shards) == []


def test_rerun_is_a_noop(corpus, stub_models):
    """Re-running a finished stage must not rewrite anything."""
    _label("topic,quality")
    before = {stage: _read_stage_tables(stage) for stage in (Stage.TOPIC, Stage.QUALITY)}

    _label("topic,quality")
    assert {stage: _read_stage_tables(stage) for stage in (Stage.TOPIC, Stage.QUALITY)} == before


def test_outputs_are_copartitioned_with_the_source(corpus, stub_models):
    """One shard in, one file out per stage: same stem, same rows, same ids.

    This is what ``datakit_store``'s positional 5-way join will later enforce
    with a hard failure, so it is cheaper to assert it here.
    """
    _label("topic,quality")
    shards = assign_shards(grid_corpora.list_shards(corpus), 1, 0)
    topic = _read_stage_tables(Stage.TOPIC)
    quality = _read_stage_tables(Stage.QUALITY)

    assert set(topic) == set(quality) == {output_stem(s) for s in shards}
    for shard in shards:
        stem = output_stem(shard)
        source_ids = read_shard(shard, corpus, grid_label.LABEL_TEXT_CHARS).ids
        assert topic[stem]["id"] == source_ids
        assert quality[stem]["id"] == source_ids
        assert len(source_ids) == DOCS_PER_SHARD


def test_merge_builds_the_grid(corpus, stub_models):
    """The 24 x 5 cross-tab must account for every document exactly once."""
    _label("topic,quality")
    grid_label.run_merge(DATASET)

    path = f"{grid_corpora.OUTPUT_BASE}/{grid_corpora.METADATA_ROOT}/{DATASET}/distribution.json"
    with open(path) as fh:
        summary = json.load(fh)

    total = N_SHARDS * DOCS_PER_SHARD
    docs_grid = np.array(summary["grid"]["docs"])
    assert docs_grid.shape == (grid_corpora.NUM_TOPICS, len(grid_label.BUCKET_EDGES) + 1)
    assert docs_grid.sum() == total
    assert summary["topic"]["n_docs"] == total
    assert summary["quality"]["n_docs"] == total
    assert sum(summary["quality"]["bucket_counts"]) == total
    # Row and column marginals must agree with the per-stage tallies.
    assert docs_grid.sum(axis=1).tolist() == [
        summary["topic"]["counts"][f"topic-{i}"] for i in range(grid_corpora.NUM_TOPICS)
    ]
    assert docs_grid.sum(axis=0).tolist() == summary["quality"]["bucket_counts"]


def test_merge_works_before_quality_has_run(corpus, stub_models):
    """A topic-only corpus still merges — the grid is simply absent."""
    _label("topic", quality_model=None)
    grid_label.run_merge(DATASET)

    path = f"{grid_corpora.OUTPUT_BASE}/{grid_corpora.METADATA_ROOT}/{DATASET}/distribution.json"
    with open(path) as fh:
        summary = json.load(fh)

    assert summary["topic"]["n_docs"] == N_SHARDS * DOCS_PER_SHARD
    assert "quality" not in summary
    assert "grid" not in summary


def test_quality_stage_requires_a_model(corpus, stub_models):
    """Asking for quality without an artifact must fail loudly, not silently skip."""
    with pytest.raises(ValueError, match="--quality-model is required"):
        _label("quality", quality_model=None)


def test_merge_rejects_broken_copartitioning(corpus, stub_models):
    """A truncated attribute table must fail the merge rather than skew the grid."""
    _label("topic,quality")
    directory = grid_corpora.stage_output_dir(DATASET, Stage.QUALITY)
    from marin.utils import fsspec_glob

    victim = sorted(fsspec_glob(f"{directory}/outputs/main/*.parquet"))[0]
    table = pq.read_table(victim)
    pq.write_table(table.slice(0, table.num_rows - 1), victim)

    with pytest.raises(ValueError, match="co-partitioning broken"):
        grid_label.run_merge(DATASET)


def test_empty_shard_still_produces_an_output_file(corpus_with_empty_shard, stub_models):
    """An empty input shard must yield an empty OUTPUT shard, not a missing one.

    The co-partitioning contract is one output file per *input* shard. A
    downstream positional join needs a zero-row file where the input was empty;
    skipping it silently shifts every subsequent shard's alignment. Failing on it
    is also wrong — an empty shard is normal for a filtered corpus.
    """
    _label("topic,quality")
    shards = assign_shards(grid_corpora.list_shards(corpus_with_empty_shard), 1, 0)
    assert len(shards) == 3

    topic = _read_stage_tables(Stage.TOPIC)
    quality = _read_stage_tables(Stage.QUALITY)
    # Every input shard has an output, including the empty one.
    assert set(topic) == set(quality) == {output_stem(s) for s in shards}

    empty_stem = output_stem(shards[1])
    assert topic[empty_stem]["id"] == []
    assert quality[empty_stem]["id"] == []
    # Schema must survive the empty case, or the join breaks on column mismatch.
    assert set(topic[empty_stem]) == set(topic[output_stem(shards[0])])
    assert set(quality[empty_stem]) == set(quality[output_stem(shards[0])])
    # And the non-empty shards are unaffected.
    assert len(topic[output_stem(shards[0])]["id"]) == 10


def test_merge_tolerates_empty_shards(corpus_with_empty_shard, stub_models):
    """Totals must count only real documents, with the empty shard contributing zero."""
    _label("topic,quality")
    grid_label.run_merge(DATASET)
    path = f"{grid_corpora.OUTPUT_BASE}/{grid_corpora.METADATA_ROOT}/{DATASET}/distribution.json"
    with open(path) as fh:
        summary = json.load(fh)
    assert summary["topic"]["n_shards"] == 3
    assert summary["topic"]["n_docs"] == 20
    assert np.array(summary["grid"]["docs"]).sum() == 20


def test_output_base_override_does_not_move_the_mirror(monkeypatch, tmp_path):
    """Redirecting outputs must not redirect the INPUT mirror.

    ``--output-base`` exists so a smoke run can write to a scratch tree. Folding
    the mirror path into the same constant made that flag silently repoint the
    scorer at an empty scratch directory, which surfaced as "no shards found"
    rather than as a misconfiguration.
    """
    before = grid_corpora.mirrored(DATASET).path
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "scratch"))
    after = grid_corpora.mirrored(DATASET).path
    assert after == before, "mirror input path must be independent of the output root"
    # ...while the output tree does move.
    assert str(tmp_path / "scratch") in grid_corpora.stage_output_dir(DATASET, Stage.TOPIC)


def test_score_quality_applies_the_calibration(monkeypatch):
    """Buckets must come from CALIBRATED scores, never raw ones.

    ``BUCKET_EDGES`` are cutpoints in calibrated space. Digitizing a raw score
    against them mis-buckets — worst at the tails, which is exactly where the
    junk/excellent decisions live. This regressed once: the scorer was loaded
    without its calibration and the raw score was digitized directly.
    """
    xk = [0.0428, 0.2495, 0.4419, 0.6194, 0.7514, 0.8794]
    yk = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    raw = np.array([0.20, 0.55, 0.87], dtype=np.float64)
    monkeypatch.setattr(grid_label, "score_bme", lambda model, texts: raw[: len(texts)], raising=False)
    import experiments.datakit.cluster.quality.fast_transformer.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "score_bme", lambda model, texts: raw[: len(texts)])

    scores, buckets = grid_label.score_quality(("model", np.array(xk), np.array(yk)), ["a", "b", "c"])

    expected = np.interp(raw, xk, yk)
    np.testing.assert_allclose(scores, expected)
    assert buckets.tolist() == np.digitize(expected, grid_label.BUCKET_EDGES).tolist()
    # raw 0.20 digitizes to bucket 1, but calibrates to 0.152 -> bucket 0.
    assert np.digitize(raw, grid_label.BUCKET_EDGES)[0] == 1
    assert buckets[0] == 0, "calibration must move this doc out of bucket 1"
