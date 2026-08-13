# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the forked ``datakit_store``.

These run the real shuffle against a Zephyr ``LocalClient`` — no mocks — because
the properties worth checking are only observable end to end:

* every ``(topic, quality)`` cell is a cache **levanter can actually load**, and
  its documents are exactly the ones the attribute tables assign to it. Upstream
  writes a bucket-level ledger over child caches; this levanter cannot read that,
  and the failure mode is a cell that looks fine and trains on nothing, so
  loading each cell back is the test that matters.
* token sequences arrive intact and in order. Regrouping documents into cells
  must not disturb anything inside a document.
* the positional join fails loudly on broken co-partitioning rather than
  silently pairing a document with another document's labels.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray import ResourceConfig
from fray.local_backend import LocalClient
from levanter.store.cache import TreeCache

from experiments.datakit.store.datakit_store import build_clustered_store
from experiments.datakit.store.store_compat import (
    AssignmentAttrData,
    QualityScores,
    TokenizedAttrData,
)

SOURCE = "smoke"
N_SHARDS = 3
DOCS_PER_SHARD = 12
N_TOPICS = 4
N_QUALITY = 5
MODEL_DIR = "gs://fake/pooled_junkgate2"
CALIB_FILE = "calib_bme.json"
BUCKET_EDGES = [0.2, 0.4, 0.6, 0.8]
TOKENIZE_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("input_ids", pa.list_(pa.int32()))])
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


def _doc_id(shard: int, doc: int) -> str:
    return f"{shard:02d}-{doc:02d}"


def _topic(shard: int, doc: int) -> int:
    return (shard * DOCS_PER_SHARD + doc) % N_TOPICS


def _quality(shard: int, doc: int) -> int:
    return (shard + doc) % N_QUALITY


def _tokens(shard: int, doc: int) -> list[int]:
    """A sequence whose values encode its identity, so misrouting is detectable."""
    base = 1000 * shard + 10 * doc
    return [1, *(base + i for i in range(3 + doc % 4)), 2]


@pytest.fixture(scope="module")
def local_client():
    client = LocalClient()
    yield client
    client.shutdown(wait=True)


@pytest.fixture
def inputs(tmp_path):
    """Three co-partitioned shards of tokenize/cluster/quality tables."""
    tok_dir = tmp_path / "tokenize"
    cluster_dir = tmp_path / "cluster"
    quality_dir = tmp_path / "quality"
    for directory in (tok_dir, cluster_dir, quality_dir):
        directory.mkdir()

    for shard in range(N_SHARDS):
        name = f"part-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        ids = [_doc_id(shard, d) for d in range(DOCS_PER_SHARD)]
        pq.write_table(
            pa.table(
                {"id": ids, "input_ids": [_tokens(shard, d) for d in range(DOCS_PER_SHARD)]},
                # int32 token lists, matching grid_tokenize's schema exactly -- the
                # dtype propagates into the cache and into consolidation's exemplar.
                schema=TOKENIZE_SCHEMA,
            ),
            tok_dir / name,
        )
        pq.write_table(
            pa.table(
                {
                    "id": ids,
                    "cluster_4": pa.array([_topic(shard, d) for d in range(DOCS_PER_SHARD)], pa.int32()),
                }
            ),
            cluster_dir / name,
        )
        pq.write_table(
            pa.table(
                {
                    "id": ids,
                    "quality_bucket": pa.array([_quality(shard, d) for d in range(DOCS_PER_SHARD)], pa.int8()),
                }
            ),
            quality_dir / name,
        )

    return {
        "tokenize": {
            SOURCE: TokenizedAttrData(
                output_dirs={"train": str(tok_dir)},
                source_main_dirs={"train": str(tmp_path / "docs")},
                tokenizer="stub-tokenizer",
            )
        },
        "cluster_assign": {
            SOURCE: AssignmentAttrData(
                output_dir=str(cluster_dir),
                source_main_dir=str(tmp_path / "docs"),
                k_train=N_TOPICS,
            )
        },
        "quality": {
            SOURCE: QualityScores(
                main_output_dir=str(quality_dir),
                model_dir=MODEL_DIR,
                calib_file=CALIB_FILE,
                bucket_edges=BUCKET_EDGES,
            )
        },
    }


def _build(inputs, tmp_path, local_client, **kwargs):
    return build_clustered_store(
        output_path=str(tmp_path / "store"),
        cluster_view=N_TOPICS,
        client=local_client,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        worker_resources=ResourceConfig(cpu=1, ram="512m"),
        max_workers=2,
        reduce_shards=4,
        default_subshards=1,
        **inputs,
        **kwargs,
    )


def _expected_cells() -> dict[tuple[int, int], dict[str, list[int]]]:
    cells: dict[tuple[int, int], dict[str, list[int]]] = {}
    for shard in range(N_SHARDS):
        for doc in range(DOCS_PER_SHARD):
            cell = cells.setdefault((_topic(shard, doc), _quality(shard, doc)), {})
            cell[_doc_id(shard, doc)] = _tokens(shard, doc)
    return cells


def test_every_cell_is_a_loadable_cache_with_the_right_documents(inputs, tmp_path, local_client):
    """The load-bearing test: read each cell back through levanter and compare contents.

    Cells are compared as *sets* of sequences. Which order documents land in
    within a cell is the shuffle's business; which documents land there, and that
    each one's tokens survive unaltered, is ours.
    """
    artifact = _build(inputs, tmp_path, local_client)
    expected = _expected_cells()
    assert {(b.cluster_id, b.quality_bucket) for b in artifact.buckets} == set(expected)

    for bucket in artifact.buckets:
        want = expected[(bucket.cluster_id, bucket.quality_bucket)]
        cache = TreeCache.load(bucket.path, EXEMPLAR)
        assert len(cache) == len(want) == bucket.total_elements

        got = {tuple(cache[i]["input_ids"].tolist()) for i in range(len(cache))}
        assert got == {tuple(seq) for seq in want.values()}
        assert bucket.total_tokens == sum(len(seq) for seq in want.values())

    assert sum(b.total_elements for b in artifact.buckets) == N_SHARDS * DOCS_PER_SHARD


def test_artifact_records_the_provenance_a_mixture_needs(inputs, tmp_path, local_client):
    """A cell is only comparable across corpora if the scorer and view are pinned."""
    artifact = _build(inputs, tmp_path, local_client)
    assert artifact.cluster_view == N_TOPICS
    assert artifact.bucket_edges == BUCKET_EDGES
    assert artifact.tokenizer == "stub-tokenizer"
    assert artifact.source_names == [SOURCE]
    assert artifact.counters["datakit_store/records_out"] == N_SHARDS * DOCS_PER_SHARD


def test_empty_shards_are_skipped_not_fatal(inputs, tmp_path, local_client):
    """Zero-row shards are normal input, not corruption.

    13,541 of fineweb_cc's 21,531 shards are legitimately empty, so a store that
    treats an empty shard as an error cannot run on the largest corpus.
    """
    for kind, directory in (
        ("tokenize", inputs["tokenize"][SOURCE].output_dirs["train"]),
        ("cluster", inputs["cluster_assign"][SOURCE].output_dir),
        ("quality", inputs["quality"][SOURCE].main_output_dir),
    ):
        name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
        schema = pq.read_schema(f"{directory}/{name}")
        pq.write_table(schema.empty_table(), f"{directory}/{name}")
        assert kind  # loop variable is only for readability of failures

    artifact = _build(inputs, tmp_path, local_client)
    survivors = (N_SHARDS - 1) * DOCS_PER_SHARD
    assert sum(b.total_elements for b in artifact.buckets) == survivors


def test_empty_shard_with_null_typed_id_column_is_not_fatal(inputs, tmp_path, local_client):
    """The shape an empty shard's ``id`` column actually has in production.

    ``grid_label.py`` builds each attribute table as
    ``pa.table({"id": docs.ids, ...})`` from a plain Python list, not
    ``pa.array(docs.ids, type=pa.string())``. For a genuinely empty shard that
    infers the column as pyarrow's ``null`` type rather than ``string`` -- there
    are no values for pyarrow to infer a type from -- and ``pc.equal`` has no
    kernel for ``(null, null)``. ``test_empty_shards_are_skipped_not_fatal``
    above rewrites via ``schema.empty_table()``, which preserves the ORIGINAL
    (string) column type and so does not exercise this path; this test
    reconstructs the table the way the real writer does.
    """
    name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
    # Only "id" is left untyped, matching write_topic_shard/write_quality_shard:
    # every OTHER column is built with an explicit pa.array(..., type=...), so
    # only "id" (a bare Python list, `docs.ids`) is exposed to this inference gap.
    for directory, col, dtype in (
        (inputs["cluster_assign"][SOURCE].output_dir, f"cluster_{N_TOPICS}", pa.int32()),
        (inputs["quality"][SOURCE].main_output_dir, "quality_bucket", pa.int8()),
    ):
        path = f"{directory}/{name}"
        pq.write_table(pa.table({"id": [], col: pa.array([], type=dtype)}), path)
    tok_path = f"{inputs['tokenize'][SOURCE].output_dirs['train']}/{name}"
    pq.write_table(pa.table({"id": [], "input_ids": []}, schema=TOKENIZE_SCHEMA), tok_path)

    artifact = _build(inputs, tmp_path, local_client)
    survivors = (N_SHARDS - 1) * DOCS_PER_SHARD
    assert sum(b.total_elements for b in artifact.buckets) == survivors


def test_subshard_split_produces_one_consolidated_cache_per_cell(inputs, tmp_path, local_client):
    """With k>1 a cell must still be ONE loadable cache holding all its documents.

    Exercises the consolidation branch of :func:`_finalize_buckets`. Our corpora
    are small enough that ``_plan_subshards`` leaves k=1, so without this test that
    branch would only ever run for the first time on a corpus large enough to need
    it — the worst moment to discover it.
    """
    artifact = build_clustered_store(
        output_path=str(tmp_path / "store"),
        cluster_view=N_TOPICS,
        client=local_client,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        worker_resources=ResourceConfig(cpu=1, ram="512m"),
        max_workers=2,
        reduce_shards=4,
        default_subshards=3,
        **inputs,
    )
    expected = _expected_cells()
    multi = [b for b in artifact.buckets if b.n_shards > 1]
    assert multi, "expected at least one cell to have been split"

    for bucket in artifact.buckets:
        want = expected[(bucket.cluster_id, bucket.quality_bucket)]
        cache = TreeCache.load(bucket.path, EXEMPLAR)
        assert len(cache) == len(want)
        got = {tuple(cache[i]["input_ids"].tolist()) for i in range(len(cache))}
        assert got == {tuple(seq) for seq in want.values()}


@pytest.mark.parametrize("default_subshards", [1, 3])
def test_cell_path_carries_the_split_level_a_mixture_component_needs(inputs, tmp_path, local_client, default_subshards):
    """Pin the path shape Levanter requires of a mixture component.

    Levanter resolves a ``DatasetComponent`` by appending the split itself --
    ``load_lm_dataset_cache(os.path.join(cache_dir, split), ...)`` in
    ``levanter/data/text/datasets.py`` -- so a cell whose cache sits at
    ``.../quality=<Q>/sub=0`` with no ``train/`` beneath it cannot be loaded by a
    mixture at all: it raises "No source and no cache found for component". Both
    the single-subshard (cell *is* the sub cache) and consolidated (k>1) branches
    have to agree on that shape, hence the parametrization.
    """
    artifact = build_clustered_store(
        output_path=str(tmp_path / "store"),
        cluster_view=N_TOPICS,
        client=local_client,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        worker_resources=ResourceConfig(cpu=1, ram="512m"),
        max_workers=2,
        reduce_shards=4,
        default_subshards=default_subshards,
        **inputs,
    )
    assert artifact.buckets
    for bucket in artifact.buckets:
        assert bucket.path == os.path.join(bucket.component_cache_dir, artifact.split)
        # ...and the path Levanter would compose is the one actually materialized.
        assert TreeCache.load(os.path.join(bucket.component_cache_dir, artifact.split), EXEMPLAR) is not None


def test_misaligned_quality_table_is_rejected(inputs, tmp_path, local_client):
    """Same row count, different id order — the case a length check would miss.

    Left undetected this pairs each document with a *different* document's quality
    label, producing a grid that is wrong everywhere and looks right.
    """
    quality_dir = inputs["quality"][SOURCE].main_output_dir
    name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
    table = pq.read_table(f"{quality_dir}/{name}")
    reversed_ids = pa.array(table.column("id").to_pylist()[::-1])
    pq.write_table(table.set_column(0, "id", reversed_ids), f"{quality_dir}/{name}")

    with pytest.raises(Exception, match="co-partitioning broken"):
        _build(inputs, tmp_path, local_client)


def test_extra_tokenize_rows_are_rejected(inputs, tmp_path, local_client):
    """A tokenize shard longer than its attribute shards must fail, not truncate."""
    tok_dir = inputs["tokenize"][SOURCE].output_dirs["train"]
    name = f"part-{0:05d}-of-{N_SHARDS:05d}.parquet"
    table = pq.read_table(f"{tok_dir}/{name}")
    extra = pa.table({"id": ["99-99"], "input_ids": [[1, 7, 2]]}, schema=TOKENIZE_SCHEMA)
    pq.write_table(pa.concat_tables([table, extra]), f"{tok_dir}/{name}")

    with pytest.raises(Exception, match="co-partitioning broken"):
        _build(inputs, tmp_path, local_client)


def test_cluster_view_must_have_been_materialized(inputs, tmp_path, local_client):
    """Asking for a view the assignment table never produced is a config error."""
    with pytest.raises(ValueError, match="cluster_view"):
        build_clustered_store(
            output_path=str(tmp_path / "store"),
            cluster_view=40,
            client=local_client,
            chunk_storage_prefix=str(tmp_path / "chunks"),
            default_subshards=1,
            **inputs,
        )
