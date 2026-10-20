# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: the claimant pipeline against the Zephyr dedup path it replaces.

A small kept corpus with exact duplicates (within and across files), near-duplicate
chains, blank and whitespace-run texts and NaN scores goes through both pipelines; exact
duplicate counts, cluster membership, canonical choice and the surviving records must agree.
"""

import dataclasses
import glob
import gzip
import json
import math
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import normalize_to_parquet
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs
from zephyr.readers import load_parquet
from zephyr.writers import write_jsonl_file

from experiments.fast_curation.dedup import _carry_provenance
from experiments.fast_curation.project_text_only import DROP_COLUMNS
from experiments.fast_curation.rdedup import stages, worker
from experiments.fast_curation.rdedup.plan import build_plan, write_plan
from experiments.fast_curation.rdedup.store import LocalStore
from experiments.fast_curation.rdedup.tasks import ClaimConfig
from experiments.fast_curation.rdedup.worker import slot_loop

_WORDS = [f"w{i}" for i in range(400)] + ["the", "and", "of", "to", "in", "is", "that", "for"]
_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),
        ("n_tokens", pa.int32()),
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),
    ]
)


def _edit(rng: random.Random, words: list[str], fraction: float) -> list[str]:
    out = list(words)
    for _ in range(max(1, int(len(out) * fraction))):
        out[rng.randrange(len(out))] = rng.choice(_WORDS)
    return out


def _corpus(root: Path, n_files: int = 9, seed: int = 0) -> str:
    """Write kept-shaped parquet files (with ``n_tokens``); returns the kept directory."""
    rng = random.Random(seed)
    bases = [[rng.choice(_WORDS) for _ in range(rng.randint(60, 250))] for _ in range(40)]
    pool: list[str] = []
    for base in bases:
        pool.append(" ".join(base))
        chain = base
        for _ in range(rng.choice([0, 1, 3])):
            chain = _edit(rng, chain, 0.04)
            pool.append(" ".join(chain))
    pool.append("x" + " " * 400 + " ".join(bases[0]))
    # A long drifting chain of near-duplicates: a path in the LSH graph that takes connected
    # components several rounds, spread across every file.
    drift = [rng.choice(_WORDS) for _ in range(300)]
    chain_docs = []
    for _ in range(60):
        drift = _edit(rng, drift, 0.02)
        chain_docs.append(" ".join(drift))
    kept = root / "kept"
    kept.mkdir(parents=True)
    for f in range(n_files):
        texts: list[str | None] = [rng.choice(pool) for _ in range(rng.randint(15, 30))]
        texts += [rng.choice(pool)] * 2 + [None, "", " \n "] + chain_docs[f::n_files]
        rows = []
        for i, text in enumerate(texts):
            rows.append(
                {
                    "doc_id": f"{f}-{i}",
                    "url": f"https://example.com/{f}/{i}",
                    "warc_hash": f"{f:012x}",
                    "snapshot": "CC-MAIN-2020-10",
                    "fasttext_score": rng.random(),
                    "text": text,
                    "n_tokens": len(text.split()) if text else 0,
                    "pooled_prob": float("nan") if i % 5 == 0 else rng.random(),
                    "modernbert_prob": float("nan") if i % 3 == 0 else rng.random(),
                }
            )
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), kept / f"data-{f:012x}.parquet")
    return str(kept)


def _write_kept(kept: Path, files: list[list[str]]) -> str:
    kept.mkdir(parents=True)
    for f, texts in enumerate(files):
        rows = [
            {
                "doc_id": f"{f}-{i}",
                "url": f"https://example.com/{f}/{i}",
                "warc_hash": f"{f:012x}",
                "snapshot": "CC-MAIN-2020-10",
                "fasttext_score": 0.5,
                "text": text,
                "n_tokens": len(text.split()) if text else 0,
                "pooled_prob": 0.5,
                "modernbert_prob": 0.5,
            }
            for i, text in enumerate(texts)
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), kept / f"data-{f:012x}.parquet")
    return str(kept)


def _random_words(rng: random.Random, n: int) -> list[str]:
    """Words of random letters: unrelated texts share almost no character 5-grams."""
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) for _ in range(n)]


def _reference(kept: str, work: Path, monkeypatch) -> dict:
    """Run reshape -> normalize -> minhash -> fuzzy -> cc_canonical apply as dedup.py does."""
    monkeypatch.setenv("MARIN_PREFIX", str(work))
    files = sorted(glob.glob(f"{kept}/*.parquet"))
    # dedup.py read kept_text/, which is kept/ minus DROP_COLUMNS.
    records = [
        _carry_provenance({k: v for k, v in r.items() if k not in DROP_COLUMNS})
        for f in files
        for r in load_parquet(f)
        if r.get("text")
    ]
    write_jsonl_file(records, str(work / "reshape" / "data-00000.jsonl.gz"))
    with set_current_client(LocalClient()):
        norm = normalize_to_parquet(
            input_path=str(work / "reshape"), output_path=str(work / "normalize"), target_partition_bytes=2000
        )
        minhash = compute_minhash_attrs(source=norm, output_path=str(work / "minhash"), text_cap_chars=None)
        fuzzy = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=str(work / "fuzzy"), max_parallelism=4)
    attr_dir = next(iter(fuzzy.sources.values())).attr_dir
    attrs = [r for p in sorted(glob.glob(f"{attr_dir}/*.parquet")) for r in load_parquet(p)]
    dropped = {r["id"] for r in attrs if r["is_cluster_canonical"] is False}
    survivors = [
        _carry_provenance(r)
        for p in sorted(glob.glob(f"{norm.main_output_dir}/*.parquet"))
        for r in load_parquet(p)
        if r["id"] not in dropped
    ]
    return {
        "exact_dups": norm.counters["normalize/duplicate_records_out"],
        "unique": norm.counters["normalize/unique_records_out"],
        "members": len(attrs),
        "canonicals": sum(1 for r in attrs if r["is_cluster_canonical"]),
        "survivors": survivors,
    }


def _run(kept: str, root: Path, claims: ClaimConfig = ClaimConfig()) -> tuple[dict, list[dict], dict[str, bytes]]:
    store = LocalStore()
    paths = sorted(glob.glob(f"{kept}/*.parquet"))
    sizes = [os.path.getsize(p) for p in paths]
    plan = build_plan(
        run_root=str(root / "run"),
        output_root=str(root / "out"),
        spec_id="test",
        sizes=sizes,
        unit_bytes=3 * max(sizes),
        rules=["cc_canonical", "c075"],
        node_partitions=2,
        files_per_output=2,
    )
    plan = dataclasses.replace(plan, exact_partitions=3, lsh_partitions=5, c075_partitions=3)
    write_plan(store, plan, paths, sizes)
    assert slot_loop(0, plan.run_root, cpus=1, memory_gb=64, claims=claims) == 0
    report = json.loads(Path(plan.output_root, "stats", "rdedup_stats.json").read_text())
    outputs = {p: Path(p).read_bytes() for p in sorted(glob.glob(f"{plan.output_root}/deduped*/*.jsonl.gz"))}
    records = [
        json.loads(line)
        for p in sorted(glob.glob(f"{plan.output_root}/deduped/*.jsonl.gz"))
        for line in gzip.decompress(Path(p).read_bytes()).splitlines()
    ]
    return report, records, outputs


def _normalized(record: dict) -> str:
    return json.dumps(
        {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in record.items()}, sort_keys=True
    )


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(worker, "IDLE_POLL_SECONDS", 0.2)
    monkeypatch.setattr(worker, "FAILURE_BACKOFF_SECONDS", (0.0, 0.0))


@pytest.mark.timeout(300)
def test_claimant_pipeline_matches_zephyr_dedup(tmp_path, monkeypatch):
    kept = _corpus(tmp_path / "corpus")
    reference = _reference(kept, tmp_path / "reference", monkeypatch)
    # A failing task would otherwise wait out the production stale timeout before a retry.
    report, records, _ = _run(kept, tmp_path / "rdedup", ClaimConfig(heartbeat_seconds=0.05, stale_seconds=0.3))

    assert report["exact_duplicates"] == reference["exact_dups"] > 0
    assert report["unique_docs"] == reference["unique"]
    assert report["cluster_members"] == reference["members"] > 0
    assert report["components"] == reference["canonicals"]
    # Exact duplicates keep an arbitrary copy upstream, so compare survivors by text, and whole
    # records wherever the text has a single copy in the input.
    assert sorted(r["text"] for r in records) == sorted(r["text"] for r in reference["survivors"])
    copies: dict[str, int] = {}
    for f in sorted(glob.glob(f"{kept}/*.parquet")):
        for r in load_parquet(f):
            if r["text"]:
                copies[r["text"]] = copies.get(r["text"], 0) + 1
    # The reference writes no copy or cluster-size fields; every other field must match.
    added = {stages.EXACT_COPIES_FIELD, stages.CLUSTER_SIZE_FIELD}
    singles_ours = sorted(
        _normalized({k: v for k, v in r.items() if k not in added}) for r in records if copies.get(r["text"]) == 1
    )
    singles_ref = sorted(_normalized(r) for r in reference["survivors"] if copies.get(r["text"]) == 1)
    assert singles_ours == singles_ref and singles_ours
    assert report["c075"]["docs"] >= report["cc_canonical"]["docs"]
    assert report["stages"]["components"]["final_round"] >= 3


@pytest.mark.timeout(300)
def test_failed_tasks_are_retried_and_outputs_are_deterministic(tmp_path, monkeypatch):
    kept = _corpus(tmp_path / "corpus", n_files=6, seed=3)
    clean_report, _, clean_outputs = _run(kept, tmp_path / "clean")

    # Every task of these stages fails on its first attempt; the stale claim must be taken over
    # and the rerun must land byte-identical outputs.
    failed: set[tuple[str, int]] = set()
    originals = {
        name: getattr(stages, name)
        for name in ("run_pass1", "run_bucket", "run_cc_a", "run_cc_c", "run_apply_cc", "run_c075_solve")
    }

    def flaky(name):
        def wrapper(ctx, task, *args):
            key = (name, task, *args)
            if key not in failed:
                failed.add(key)
                raise RuntimeError(f"injected failure in {name}")
            return originals[name](ctx, task, *args)

        return wrapper

    for name in originals:
        monkeypatch.setattr(stages, name, flaky(name))
    flaky_report, _, flaky_outputs = _run(
        kept, tmp_path / "flaky", ClaimConfig(heartbeat_seconds=0.05, stale_seconds=0.3)
    )

    assert failed
    assert {Path(p).relative_to(tmp_path / "flaky") for p in flaky_outputs} == {
        Path(p).relative_to(tmp_path / "clean") for p in clean_outputs
    }
    assert list(flaky_outputs.values()) == list(clean_outputs.values())
    for key in ("exact_duplicates", "cluster_members", "components", "cc_canonical", "c075"):
        assert flaky_report[key] == clean_report[key]


@pytest.mark.timeout(300)
def test_multipass_c075_solve_gives_identical_outputs(tmp_path, monkeypatch):
    kept = _corpus(tmp_path / "corpus", n_files=6, seed=5)
    one_report, _, one_outputs = _run(kept, tmp_path / "one")
    # A one-byte budget forces a pass per compressed byte's worth of blocks: many passes.
    monkeypatch.setattr(stages, "_SOLVE_PASS_COMPRESSED_BYTES", 1)
    many_report, _, many_outputs = _run(kept, tmp_path / "many")
    assert many_report["stages"]["c075_solve"]["duplicates"] == one_report["stages"]["c075_solve"]["duplicates"] > 0
    assert list(many_outputs.values()) == list(one_outputs.values())


@pytest.mark.timeout(300)
def test_output_docs_carry_exact_copy_counts_and_fuzzy_cluster_sizes(tmp_path):
    rng = random.Random(7)
    fillers = iter(" ".join(_random_words(rng, 80)) for _ in range(60))
    a, b = (" ".join(_random_words(rng, 120)) for _ in range(2))
    base = _random_words(rng, 300)
    family = [" ".join(_edit(rng, base, 0.005)) for _ in range(4)]
    spaced = _random_words(rng, 100)
    # Whitespace runs over 128 characters are cut to 128, so these two normalize to one text.
    long_gap = " ".join(spaced[:50]) + " " * 300 + " ".join(spaced[50:])
    gap_128 = " ".join(spaced[:50]) + " " * 128 + " ".join(spaced[50:])
    files = [
        [a, a, family[0], long_gap] + [next(fillers) for _ in range(8)],
        [a, b] + [next(fillers) for _ in range(8)],
        [family[1], gap_128] + [next(fillers) for _ in range(8)],
        [family[2], b] + [next(fillers) for _ in range(8)],
        [family[2], "", None] + [next(fillers) for _ in range(8)],
        [a, family[3]] + [next(fillers) for _ in range(8)],
    ]
    kept = _write_kept(tmp_path / "corpus", files)

    report, _, outputs = _run(kept, tmp_path / "rdedup")

    # Both collapse paths are exercised: copies inside a unit and copies across units.
    assert report["stages"]["pass1"]["unit_dups"] > 0
    assert report["stages"]["exact"]["cross_unit_dups"] > 0
    expected_copies = {a: 4, b: 2, family[2]: 2, gap_128: 2}
    by_tree: dict[str, dict[str, dict]] = {}
    for path, payload in outputs.items():
        tree = Path(path).parent.name
        for line in gzip.decompress(payload).splitlines():
            record = json.loads(line)
            assert record["text"] not in by_tree.setdefault(tree, {})
            by_tree[tree][record["text"]] = record
    assert set(by_tree) == {"deduped", "deduped_c075"}
    for tree, records in by_tree.items():
        family_kept = [t for t in records if t in family]
        assert family_kept, tree
        for text, record in records.items():
            assert record["exact_copies"] == expected_copies.get(text, 1), (tree, text[:40])
            assert record["fuzzy_cluster_size"] == (4 if text in family else 1), (tree, text[:40])
