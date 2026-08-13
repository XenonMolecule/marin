# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the store verifier.

A verifier that cannot fail is worse than no verifier: it converts "unchecked"
into "certified". So the cases here are half positive and half deliberate
sabotage — each one a mistake the real pipeline could actually make.

The store under test is built by the real ``build_clustered_store`` against a
``LocalClient``, not stubbed, because the verifier's whole job is to compare a
real store against real attribute tables.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig

from experiments.baseline_collection import grid_corpora, grid_store, grid_tokenize, verify_grid_store
from experiments.datakit.store.datakit_store import (
    STAT_SIDECAR,
    SUBSHARD_PLAN,
    ClusteredStoreData,
    _load_or_persist_plan,
    build_clustered_store,
)
from experiments.datakit.store.store_compat import (
    AssignmentAttrData,
    QualityScores,
    TokenizedAttrData,
    read_artifact,
)

DATASET = "dclm_10k"
N_SHARDS = 3
DOCS_PER_SHARD = 10
N_TOPICS = 24
TOKENIZE_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("input_ids", pa.list_(pa.int32()))])


def _topic(shard: int, doc: int) -> int:
    return (shard * DOCS_PER_SHARD + doc) % 6


def _quality(shard: int, doc: int) -> int:
    return (shard + doc) % 5


@pytest.fixture(scope="module")
def local_client():
    client = LocalClient()
    yield client
    client.shutdown(wait=True)


def _build_store(*, tok_dir, topic_dir, quality_dir, store_path, client, resume=False):
    """The same build the fixture performs, repeatable so resume can be tested."""
    return build_clustered_store(
        tokenize={
            DATASET: TokenizedAttrData(
                output_dirs={"train": str(tok_dir)},
                source_main_dirs={"train": str(pathlib.Path(store_path).parent / "docs")},
                tokenizer="stub",
            )
        },
        cluster_assign={
            DATASET: AssignmentAttrData(
                output_dir=str(topic_dir),
                source_main_dir=str(pathlib.Path(store_path).parent / "docs"),
                k_train=N_TOPICS,
            )
        },
        quality={
            DATASET: QualityScores(
                main_output_dir=str(quality_dir),
                model_dir="gs://fake/model",
                calib_file="calib_bme.json",
                bucket_edges=[0.2, 0.4, 0.6, 0.8],
            )
        },
        output_path=store_path,
        cluster_view=N_TOPICS,
        client=client,
        chunk_storage_prefix=str(pathlib.Path(store_path).parent / "chunks"),
        worker_resources=ResourceConfig(cpu=1, ram="512m"),
        max_workers=2,
        reduce_shards=4,
        default_subshards=1,
        resume=resume,
    )


def _rebuild(built, client, *, resume):
    return _build_store(
        tok_dir=built["tokenize"],
        topic_dir=built["topic"],
        quality_dir=built["quality"],
        store_path=built["store"],
        client=client,
        resume=resume,
    )


@pytest.fixture
def built(tmp_path, monkeypatch, local_client):
    """A real store plus the tables it was built from, with paths redirected to tmp."""
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path / "out"))
    monkeypatch.setattr(verify_grid_store, "NUM_TOPICS", N_TOPICS)

    tok_dir = tmp_path / "tokenize"
    topic_dir = tmp_path / "out" / grid_corpora.TOPIC_ROOT / f"{DATASET}_{grid_corpora.GRID_SUFFIX}"
    quality_dir = (
        tmp_path / "out" / grid_corpora.QUALITY_ROOT / f"{DATASET}_{grid_corpora.GRID_SUFFIX}" / "outputs" / "main"
    )
    for directory in (tok_dir, topic_dir, quality_dir):
        directory.mkdir(parents=True)

    for shard in range(N_SHARDS):
        name = f"part-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        ids = [f"{shard:02d}-{d:02d}" for d in range(DOCS_PER_SHARD)]
        topics = [_topic(shard, d) for d in range(DOCS_PER_SHARD)]
        buckets = [_quality(shard, d) for d in range(DOCS_PER_SHARD)]
        pq.write_table(
            pa.table(
                {"id": ids, "input_ids": [[1, shard, d, 2] for d in range(DOCS_PER_SHARD)]},
                schema=TOKENIZE_SCHEMA,
            ),
            tok_dir / name,
        )
        pq.write_table(pa.table({"id": ids, f"cluster_{N_TOPICS}": pa.array(topics, pa.int32())}), topic_dir / name)
        pq.write_table(pa.table({"id": ids, "quality_bucket": pa.array(buckets, pa.int8())}), quality_dir / name)

    store_path = str(tmp_path / "store")
    _build_store(
        tok_dir=tok_dir, topic_dir=topic_dir, quality_dir=quality_dir, store_path=store_path, client=local_client
    )
    return {
        "store": store_path,
        "tokenize": str(tok_dir),
        "topic": topic_dir,
        "quality": quality_dir,
        "tmp": tmp_path,
    }


def test_a_correct_store_verifies(built):
    verify_grid_store.verify(DATASET, built["store"], built["tokenize"])


def test_a_missing_tokenized_shard_is_caught(built):
    """Dropping a shard from the join leaves the store short and self-consistent.

    This is the resumability failure mode: a tokenize wave that quietly finished
    2 of 3 shards produces a store whose own artifact adds up perfectly.
    """
    name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
    (built["tmp"] / "tokenize" / name).unlink()
    with pytest.raises(RuntimeError, match=r"mismatch|docs"):
        verify_grid_store.verify(DATASET, built["store"], built["tokenize"])


def test_relabelled_topics_are_caught(built):
    """If the labels change after the store was built, the store is stale.

    Compares against the tables as they are *now*, which is what makes the check
    independent of the artifact rather than a restatement of it.
    """
    name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
    path = built["topic"] / name
    table = pq.read_table(path)
    shifted = pa.array([(t + 1) % 6 for t in table.column(f"cluster_{N_TOPICS}").to_pylist()], pa.int32())
    pq.write_table(table.set_column(1, f"cluster_{N_TOPICS}", shifted), path)

    with pytest.raises(RuntimeError, match="mismatch"):
        verify_grid_store.verify(DATASET, built["store"], built["tokenize"])


def test_output_base_rebinds_every_module_that_resolves_paths(monkeypatch):
    """--output-base must move the tokenize path too, not just the store's own.

    grid_tokenize holds its own import-time copy of OUTPUT_BASE, so rebinding
    only grid_store and grid_corpora left the join looking for the tokenization
    in the default region while every other path resolved correctly.
    """
    for mod in (grid_store, grid_corpora, grid_tokenize):
        monkeypatch.setattr(mod, "OUTPUT_BASE", "gs://default-region")

    grid_store.rebind_output_base("gs://elsewhere")

    assert grid_store.store_output("dclm_10k").startswith("gs://elsewhere")
    assert grid_store.tokenize_dir("dclm_10k").startswith("gs://elsewhere")
    assert grid_corpora.stage_output_dir("dclm_10k", grid_corpora.Stage.TOPIC).startswith("gs://elsewhere")


def test_resume_skips_finished_cells_and_reproduces_the_artifact(built, local_client, monkeypatch):
    """A fully resumed build must move no data and still report the same grid.

    Finished cells are dropped at the map, so their reducers never run. The
    artifact must still list them -- the driver reads their stats back from the
    sidecars -- or a resumed store would silently lose every cell it reused.
    """
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(built["tmp"] / "out"))
    before = read_artifact(built["store"], ClusteredStoreData)

    resumed = _rebuild(built, local_client, resume=True)

    assert resumed.counters.get("datakit_store/tokens_out", 0) == 0, "finished cells must not be shuffled again"
    assert resumed.counters.get("datakit_store/reduce_rows", 0) == 0, "a fully resumed build writes no rows"

    def cells(artifact):
        return sorted((b.cluster_id, b.quality_bucket, b.total_elements, b.total_tokens) for b in artifact.buckets)

    assert cells(resumed) == cells(before), "resumed store must report the same cells, rows and tokens"


def test_resume_is_opt_in(built, local_client, monkeypatch):
    """Without --resume the shuffle and reducers run, so other pipelines are unaffected."""
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(built["tmp"] / "out"))
    rebuilt = _rebuild(built, local_client, resume=False)
    assert rebuilt.counters.get("datakit_store/tokens_out", 0) > 0
    assert rebuilt.counters.get("datakit_store/reduce_rows", 0) > 0


def test_partial_resume_rebuilds_only_the_missing_cells(built, local_client, monkeypatch):
    """The realistic case: a run died partway, so some cells have sidecars and some do not."""
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(built["tmp"] / "out"))
    before = read_artifact(built["store"], ClusteredStoreData)

    # Simulate a cell that never finished: drop one cell's directory entirely.
    victim = sorted(pathlib.Path(built["store"]).glob("cluster=*/quality=*/sub=*/train"))[0]
    dropped_rows = int(json.loads((victim / STAT_SIDECAR).read_text())["rows"])
    shutil.rmtree(victim)

    resumed = _rebuild(built, local_client, resume=True)

    assert resumed.counters.get("datakit_store/reduce_rows", 0) == dropped_rows, "only the missing cell is rebuilt"

    def cells(artifact):
        return sorted((b.cluster_id, b.quality_bucket, b.total_elements, b.total_tokens) for b in artifact.buckets)

    assert cells(resumed) == cells(before), "the rebuilt store must match the original exactly"


def test_subshard_plan_is_pinned_on_first_build_and_reused(tmp_path):
    """The plan must survive re-derivation, or resume writes docs twice.

    sub = hash(doc_id) % k, so a changed k puts a bucket's docs in subshards the
    resume filter never matches: they are rewritten under the new numbering while
    the old subshards still hold them, and finalize sums both.
    """
    store = str(tmp_path / "store")
    (tmp_path / "store").mkdir()
    first = _load_or_persist_plan(store, {(0, 0): 3, (1, 1): 9})
    assert first == {(0, 0): 3, (1, 1): 9}

    # A later run derives a different plan (finer target, changed hint, whatever).
    reused = _load_or_persist_plan(store, {(0, 0): 1, (1, 1): 1})
    assert reused == {(0, 0): 3, (1, 1): 9}, "the pinned plan must win over any recomputed one"


def test_pinned_plan_round_trips_through_the_store(built, local_client, monkeypatch):
    """A real build pins a plan, and resuming reuses it rather than replanning."""
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(built["tmp"] / "out"))
    pinned = pathlib.Path(built["store"]) / SUBSHARD_PLAN
    assert pinned.exists(), "a build must pin its subshard plan"
    before = pinned.read_text()

    _rebuild(built, local_client, resume=True)

    assert pinned.read_text() == before, "resuming must not rewrite the pinned plan"
