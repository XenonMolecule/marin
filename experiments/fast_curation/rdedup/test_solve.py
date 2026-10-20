# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.processing.classification.deduplication import cluster_verify

from experiments.baseline_collection.verified_dedup import C075_PARAMS
from experiments.fast_curation.rdedup import stages


def _clusters(rng: random.Random) -> list[dict]:
    words = [f"t{i}" for i in range(300)]
    rows = []
    d = 0
    for c in range(12):
        base = [rng.choice(words) for _ in range(rng.randint(20, 120))]
        size = 300 if c == 0 else rng.randint(1, 9)
        for _ in range(size):
            # Truncations, light edits and unrelated texts: contained, near and unrelated members.
            kind = rng.random()
            if kind < 0.4:
                text = base[: rng.randint(5, len(base))]
            elif kind < 0.8:
                text = [w if rng.random() > 0.1 else rng.choice(words) for w in base]
            else:
                text = [rng.choice(words) for _ in range(rng.randint(5, 60))]
            rows.append(
                {"cluster_key": f"{c:03d}", "id": f"{rng.getrandbits(128):032x}", "text": " ".join(text), "d": d}
            )
            d += 1
    return sorted(rows, key=lambda r: (r["cluster_key"], r["id"]))


@pytest.mark.parametrize("cap", [256 * 1024 * 1024, 2500])
def test_solve_matches_cluster_verify(tmp_path, monkeypatch, cap):
    # A small cap drives the byte-bounded batches and mid-cluster flushes through both solvers.
    monkeypatch.setattr(cluster_verify, "MAXIMUM_CLUSTER_CHARS", cap)
    monkeypatch.setattr(stages, "SOLVE_MAX_CLUSTER_CHARS", cap)
    monkeypatch.setattr(cluster_verify, "MAXIMUM_DOCUMENT_CHARS", 400)
    monkeypatch.setattr(stages, "SOLVE_MAX_DOCUMENT_CHARS", 400)
    rows = _clusters(random.Random(cap))
    path = tmp_path / "text.parquet"
    pq.write_table(
        pa.table(
            {
                "cluster_key": [r["cluster_key"] for r in rows],
                "dup_cluster_id": [r["cluster_key"] for r in rows],
                "id": [r["id"] for r in rows],
                "text": pa.array([r["text"] for r in rows], type=pa.large_string()),
                "file_idx": [0] * len(rows),
                "source_tag": ["source_000"] * len(rows),
            }
        ),
        path,
    )
    reference = {m["id"] for m in cluster_verify.solve_text_shard(str(path), C075_PARAMS)}
    id_of = {r["d"]: r["id"] for r in rows}
    stats = stages.SolveStats()
    ours = {id_of[d] for d in stages.solve_sorted_rows([dict(r) for r in rows], stats)}
    assert ours == reference and reference
    assert stats.duplicates == len(reference)
    if cap < 10_000:
        assert stats.mid_cluster_flushes > 0
