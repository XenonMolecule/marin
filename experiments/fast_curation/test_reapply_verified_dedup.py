# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import write_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams

from experiments.baseline_collection.verified_dedup import ClusterStageResources
from experiments.fast_curation.reapply_verified_dedup import ReapplyOutput, build_steps

RESOURCES = ClusterStageResources(
    max_workers=2,
    worker_cpu=1,
    worker_ram="1g",
    worker_disk="1g",
    task_ram="256m",
    task_disk="256m",
    reduce_task_ram="512m",
    output_shards=2,
    shards_per_task=1,
)


@pytest.fixture
def local_client():
    client = LocalClient()
    try:
        with set_current_client(client):
            yield client
    finally:
        client.shutdown()


def _built_tree(root: Path) -> Path:
    """A dedup tree with finished normalize/ and fuzzy/ steps and nothing upstream of them.

    Component ``c1`` chains two unrelated families, each a document plus a shorter copy of it.
    Component ``c2`` is pure chaining: two documents that share nothing.
    """
    rng = random.Random(0)
    vocabulary = [f"w{i}" for i in range(5000)]

    def prose(words: int) -> list[str]:
        return [rng.choice(vocabulary) for _ in range(words)]

    family_a, family_b = prose(400), prose(400)
    text = {
        "a_full": " ".join(family_a),
        "a_copy": " ".join(family_a[:380] + prose(5)),
        "b_full": " ".join(family_b),
        "b_copy": " ".join(family_b[10:390]),
        "x": " ".join(prose(300)),
        "y": " ".join(prose(300)),
        "solo": " ".join(prose(200)),
    }
    component = {"a_full": "c1", "a_copy": "c1", "b_full": "c1", "b_copy": "c1", "x": "c2", "y": "c2"}
    canonical = {"a_full", "x"}
    shards = {"part-00000.parquet": ["a_full", "b_copy", "x", "solo"], "part-00001.parquet": ["a_copy", "b_full", "y"]}

    tree = root / "documents" / "baseline_demo_deduped" / "7warcs"
    main = tree / "normalize" / "outputs" / "main"
    attrs = tree / "fuzzy" / "outputs" / "source_000"
    main.mkdir(parents=True)
    attrs.mkdir(parents=True)
    for name, ids in shards.items():
        pq.write_table(
            pa.table({"id": ids, "text": [text[i] for i in ids], "url": ids, "modernbert_prob": [0.5] * len(ids)}),
            main / name,
        )
        members = [i for i in ids if i in component]
        pq.write_table(
            pa.table(
                {
                    "id": members,
                    "dup_cluster_id": [component[i] for i in members],
                    "is_cluster_canonical": [i in canonical for i in members],
                }
            ),
            attrs / name,
        )
    write_artifact(
        NormalizedData(
            main_output_dir=str(main), dup_output_dir=str(tree / "normalize" / "outputs" / "dups"), counters={}
        ),
        str(tree / "normalize"),
    )
    write_artifact(
        FuzzyDupsAttrData(
            params=MinHashParams(num_perms=286, num_bands=26, ngram_size=5, seed=42),
            sources={str(main.relative_to(root)): FuzzyDupsPerSource(attr_dir=str(attrs))},
            counters={"dedup/fuzzy/document/cluster_members": 6, "dedup/fuzzy/document/canonicals": 2},
        ),
        str(tree / "fuzzy"),
    )
    for step in ("normalize", "fuzzy"):
        StatusFile(str(tree / step), worker_id="test").write_status(STATUS_SUCCESS)
    return tree


def _urls(directory: Path) -> list[str]:
    urls = []
    for path in sorted(directory.glob("data-*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            urls.extend(json.loads(line)["url"] for line in handle)
    return sorted(urls)


def test_a_chained_component_loses_only_the_documents_a_longer_one_contains(tmp_path, monkeypatch, local_client):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tree = _built_tree(tmp_path)

    StepRunner().run(build_steps(str(tree), ReapplyOutput.TREE, RESOURCES))

    # cc_canonical would keep a_full, x and solo: one survivor per component plus the singleton.
    assert _urls(tree / "deduped_c075") == ["a_full", "b_full", "solo", "x", "y"]


def test_counts_output_measures_both_rules_without_writing_a_corpus(tmp_path, monkeypatch, local_client):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tree = _built_tree(tmp_path)

    StepRunner().run(build_steps(str(tree), ReapplyOutput.COUNTS, RESOURCES))

    survival = json.loads((tree / "stats_c075" / "survival.json").read_text())
    assert (survival["docs"], survival["component_members"]) == (7, 6)
    assert (survival["cc_canonical_removed_docs"], survival["c075_removed_docs"]) == (4, 2)
    # The two c075 removals are the 385- and 380-word copies; cc_canonical also takes b_full, x or y.
    assert survival["c075_removed_chars"] < survival["cc_canonical_removed_chars"] < survival["chars"]
    assert survival["c075_doc_survival"] == pytest.approx(5 / 7)
    assert not (tree / "deduped_c075").exists()


def test_a_tree_without_a_finished_fuzzy_step_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tree = _built_tree(tmp_path)
    (tree / "fuzzy" / ".executor_status").unlink()

    with pytest.raises(FileNotFoundError, match="needs a finished fuzzy step"):
        build_steps(str(tree), ReapplyOutput.COUNTS, RESOURCES)
