# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import write_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)

from experiments.baseline_collection.verified_dedup import (
    C075_PARAMS,
    _marin_prefix_and_key,
    apply_verified_dups,
)

SHARDS = {
    "part-00000.parquet": ["a_full", "b_copy", "solo"],
    "part-00001.parquet": ["a_copy", "b_full"],
}


@pytest.fixture
def local_client():
    client = LocalClient()
    try:
        with set_current_client(client):
            yield client
    finally:
        client.shutdown()


def _step(path: Path) -> StepSpec:
    return StepSpec(name=path.name, override_output_path=str(path), deps=[], fn=lambda op: None)


def _corpus(root: Path, markers: dict[str, list[str]], recorded_markers: int) -> tuple[StepSpec, StepSpec]:
    """A two-shard normalized source and a sparse marker tree naming ``markers`` as removed."""
    main = root / "normalize" / "outputs" / "main"
    main.mkdir(parents=True)
    for name, ids in SHARDS.items():
        pq.write_table(pa.table({"id": ids, "text": [f"text of {i}" for i in ids], "url": ids}), main / name)
    write_artifact(
        NormalizedData(
            main_output_dir=str(main), dup_output_dir=str(root / "normalize" / "outputs" / "dups"), counters={}
        ),
        str(root / "normalize"),
    )

    attr_dir = root / "verified" / "outputs" / "source_000"
    attr_dir.mkdir(parents=True)
    for name, ids in markers.items():
        pq.write_table(pa.table({"id": ids, "dup_doc": [True] * len(ids)}), attr_dir / name)
    write_artifact(
        VerifiedFuzzyDupsAttrData(
            rule=C075_PARAMS,
            sources={
                "normalize/outputs/main": VerifiedFuzzyDupsPerSource(attr_dir=str(attr_dir), source_tag="source_000")
            },
            counters={"fuzzy/cluster_verify/markers": recorded_markers},
        ),
        str(root / "verified"),
    )
    return _step(root / "normalize"), _step(root / "verified")


def _written_urls(output: Path) -> list[str]:
    urls = []
    for path in sorted(output.glob("data-*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            urls.extend(json.loads(line)["url"] for line in handle)
    return sorted(urls)


def _carry(record: dict) -> dict:
    return {"text": record["text"], "url": record["url"]}


def test_only_marked_documents_are_dropped_and_unmarked_shards_pass_through(tmp_path, monkeypatch, local_client):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # Shard 0 has no marker file at all: the verifier writes none for a shard with no removals.
    normalize, verified = _corpus(tmp_path, {"part-00001.parquet": ["a_copy"]}, recorded_markers=1)

    result = apply_verified_dups(normalize, verified, str(tmp_path / "deduped"), _carry)

    assert _written_urls(tmp_path / "deduped") == ["a_full", "b_copy", "b_full", "solo"]
    assert (result["kept"], result["dropped"]) == (4, 1)


def test_a_lost_marker_shard_fails_the_run_instead_of_keeping_its_duplicates(tmp_path, monkeypatch, local_client):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # The verifier recorded two removals, but only one marker shard survives on disk.
    normalize, verified = _corpus(tmp_path, {"part-00001.parquet": ["a_copy"]}, recorded_markers=2)

    with pytest.raises(RuntimeError, match="wrote 2 markers but apply dropped 1"):
        apply_verified_dups(normalize, verified, str(tmp_path / "deduped"), _carry)


def test_a_marker_shard_without_a_normalized_shard_is_rejected(tmp_path, monkeypatch, local_client):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalize, verified = _corpus(tmp_path, {"part-00007.parquet": ["a_copy"]}, recorded_markers=1)

    with pytest.raises(RuntimeError, match="marker shards with no normalized shard"):
        apply_verified_dups(normalize, verified, str(tmp_path / "deduped"), _carry)


def test_step_paths_split_into_the_prefix_and_key_the_hero_scripts_take():
    path = "gs://marin-us-east5/documents/baseline_x_deduped/10000warcs/fuzzy"
    assert _marin_prefix_and_key(path) == ("gs://marin-us-east5", "documents/baseline_x_deduped/10000warcs/fuzzy")
