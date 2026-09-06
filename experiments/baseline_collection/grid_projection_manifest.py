# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pick a random WARC sample from an in-flight extraction run and shard it by region.

An extraction run that has not been consolidated yet has no single corpus
directory to point a labeller at. Its output is ``data-{hash}/batch_NNNN.jsonl.gz``
spread across five regional buckets, and steal mode means one WARC's batches can
be split across several of them — and, when a stale claim gets overridden, the
same batch can exist in two regions at once.

This builds the input :mod:`experiments.baseline_collection.grid_projection`
needs: per-region JSONL group manifests, where a group is "the batches of WARC
``h`` that live in region ``r``". Partitioning by region is what keeps the
labelling jobs from ever reading across a region boundary.

Three correctness rules, each of which would silently skew a token projection if
skipped:

* **Only WARCs in the ``_completed`` registry.** The marker is written after the
  last batch lands, so a WARC without one may be missing batches, and a missing
  batch is missing tokens.
* **Batch indices must be contiguous from 0.** The registry says the *run*
  finished the WARC; contiguity says *this inventory* found every batch of it.
  A gap means a region was missed or a listing was truncated.
* **Duplicate (warc, batch) keys are resolved to exactly one copy.** Counting
  both copies of a stolen batch would double-count those documents. The winner
  is the largest object, matching the ``sum_text_chars`` cascade in
  ``consolidate/resolve_duplicates.py`` — for the same batch of the same WARC,
  more compressed bytes means more surviving records and/or longer text.

Every GCS listing is taken as a cached file rather than re-listed on each run.
That is not only because the region sweep is slow (~2.2M objects, tens of
minutes): ``gcsfs`` cannot complete the TLS handshake to storage.googleapis.com
from the dev machine even though the ``gcloud`` CLI can, so building the manifest
locally has to go through the CLI regardless::

    for r in us-central1 us-east1 us-east5 us-west4 eu-west4; do
      gcloud storage ls --long --recursive \\
        "gs://marin-$r/documents/baseline_llm_extraction/llm_pipeline_v1_1/**" > listings/$r.txt &
    done; wait
    gcloud storage ls \\
      "gs://marin-us-central1/documents/baseline_llm_extraction/llm_pipeline_v1_1/_completed/" \\
      > listings/registry.txt

    python -m experiments.baseline_collection.grid_projection_manifest \\
        --listings listings --num-warcs 1000 --out-dir out
    gcloud storage cp out/'*' gs://marin-us-central1/metadata/lpv1_1_projection/
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import random
import re
from collections import defaultdict

import fsspec

logger = logging.getLogger(__name__)

REGIONS: tuple[str, ...] = ("us-central1", "us-east1", "us-east5", "us-west4", "eu-west4")
# `--num-warcs 0`: take every eligible WARC instead of a fixed-size sample (a full-run inventory).
ALL_WARCS = 0
# Iris region label -> the bucket short-name the listings and manifests are keyed by. They differ
# for europe-west4 only (bucket `marin-eu-west4`).
REGION_MANIFEST_KEY: dict[str, str] = {
    "us-central1": "us-central1",
    "us-east5": "us-east5",
    "us-east1": "us-east1",
    "us-west4": "us-west4",
    "europe-west4": "eu-west4",
}
# `gcloud storage ls --long` prefixes each line with size and creation time; plain
# `ls` gives the URL alone. Accept either so a listing taken without --long still
# parses, at the cost of losing the duplicate tiebreaker.
BATCH_RE = re.compile(r"(gs://[^\s]*/data-(?P<warc>[0-9a-f]+)/batch_(?P<idx>\d{4})\.jsonl\.gz)$")
SIZE_RE = re.compile(r"^\s*(?P<size>\d+)\s")


def parse_listing(path: str) -> dict[tuple[str, int], tuple[str, int]]:
    """Map ``(warc_hash, batch_idx)`` to ``(url, size)`` for one region's listing."""
    found: dict[tuple[str, int], tuple[str, int]] = {}
    with open(path) as fh:
        for line in fh:
            match = BATCH_RE.search(line.rstrip())
            if not match:
                continue
            size_match = SIZE_RE.match(line)
            size = int(size_match.group("size")) if size_match else 0
            found[(match.group("warc"), int(match.group("idx")))] = (match.group(1), size)
    return found


def load_completed(registry_listing: str) -> set[str]:
    """Hashes of WARCs the run marked complete, from a cached listing of ``_completed/``."""
    with open(registry_listing) as fh:
        hashes = {
            posixpath.basename(line.strip())[len("data-") :]
            for line in fh
            if posixpath.basename(line.strip()).startswith("data-")
        }
    if not hashes:
        raise ValueError(f"{registry_listing} holds no data-* markers")
    logger.info("completed registry: %d WARCs", len(hashes))
    return hashes


def resolve_batches(
    per_region: dict[str, dict[tuple[str, int], tuple[str, int]]],
) -> tuple[dict[str, dict[int, tuple[str, str]]], int]:
    """Collapse the per-region views into one winner per ``(warc, batch)``.

    Returns ``(by_warc, num_duplicates)`` where ``by_warc[hash][idx]`` is
    ``(url, region)``.
    """
    candidates: dict[tuple[str, int], list[tuple[int, str, str]]] = defaultdict(list)
    for region, found in per_region.items():
        for key, (url, size) in found.items():
            candidates[key].append((size, region, url))

    by_warc: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    duplicates = 0
    for (warc, idx), copies in candidates.items():
        if len(copies) > 1:
            duplicates += 1
        # Largest object wins; region name breaks a size tie deterministically.
        size, region, url = max(copies, key=lambda c: (c[0], c[1]))
        by_warc[warc][idx] = (url, region)
    return by_warc, duplicates


def eligible_warcs(by_warc: dict[str, dict[int, tuple[str, str]]], completed: set[str]) -> list[str]:
    """WARCs that finished the run AND whose batch indices are contiguous from 0."""
    eligible = []
    gapped = 0
    for warc, batches in by_warc.items():
        if warc not in completed:
            continue
        if sorted(batches) != list(range(len(batches))):
            gapped += 1
            continue
        eligible.append(warc)
    logger.info("eligible: %d WARCs (%d completed-but-gapped were dropped)", len(eligible), gapped)
    return sorted(eligible)


def build_single_region(listings_dir: str, region: str, out_dir: str) -> None:
    """Group manifest of ONE region's batches of every completed WARC, for a same-region sample.

    Steal mode spreads a WARC's batches over ~3 regions, and which fleet claimed a WARC has nothing
    to do with its content, so one region's share of the run is a random subsample of it. Reading it
    from that region needs no cross-region traffic and no consolidation. Contiguity is not checked:
    a WARC's other batches are, by construction, in other regions. Only WARCs in the `_completed`
    registry are kept, so no in-flight WARC contributes a partial view.
    """
    found = parse_listing(f"{listings_dir}/{region}.txt")
    completed = load_completed(f"{listings_dir}/registry.txt")
    by_warc: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for (warc, _idx), (url, _size) in sorted(found.items()):
        if warc not in completed:
            skipped += 1
            continue
        by_warc[warc].append(url)
    path = f"{out_dir}/manifest_{region}.jsonl"
    with fsspec.open(path, "w") as fh:
        for warc in sorted(by_warc):
            fh.write(json.dumps({"warc": warc, "region": region, "shards": by_warc[warc]}) + "\n")
    total_shards = sum(len(v) for v in by_warc.values())
    summary = {
        "region": region,
        "num_warcs": len(by_warc),
        "total_shards": total_shards,
        "batches_of_incomplete_warcs_skipped": skipped,
    }
    with fsspec.open(f"{out_dir}/sample.json", "w") as fh:
        json.dump(summary, fh)
    logger.info(
        "%s: %d WARCs, %d batches (%d batches of incomplete WARCs skipped) -> %s",
        region,
        len(by_warc),
        total_shards,
        skipped,
        path,
    )


def build(listings_dir: str, num_warcs: int, seed: int, out_dir: str) -> None:
    per_region = {}
    sized = 0
    for region in REGIONS:
        found = parse_listing(f"{listings_dir}/{region}.txt")
        per_region[region] = found
        sized += sum(1 for _, size in found.values() if size)
        logger.info("%s: %d batch files", region, len(found))
    if not sized:
        logger.warning(
            "listings carry no object sizes (taken without --long); duplicate (warc,batch) keys "
            "will be resolved by region name rather than by which copy holds more text"
        )

    by_warc, duplicates = resolve_batches(per_region)
    logger.info("%d WARCs seen, %d (warc,batch) keys had duplicate copies", len(by_warc), duplicates)

    pool = eligible_warcs(by_warc, load_completed(f"{listings_dir}/registry.txt"))
    if num_warcs == ALL_WARCS:
        num_warcs = len(pool)
    if len(pool) < num_warcs:
        raise ValueError(f"asked for {num_warcs} WARCs but only {len(pool)} are eligible")

    rng = random.Random(seed)
    sample = rng.sample(pool, num_warcs)
    # Shuffle again so each region's manifest is in random order: a run that is
    # cut short then holds a uniform random subsample rather than whatever the
    # hash ordering happened to put first.
    rng.shuffle(sample)

    groups: dict[str, list[dict]] = defaultdict(list)
    for warc in sample:
        by_region: dict[str, list[str]] = defaultdict(list)
        for idx in sorted(by_warc[warc]):
            url, region = by_warc[warc][idx]
            by_region[region].append(url)
        for region, shards in by_region.items():
            groups[region].append({"warc": warc, "region": region, "shards": shards})

    total_shards = 0
    for region in REGIONS:
        rows = groups.get(region, [])
        path = f"{out_dir}/manifest_{region}.jsonl"
        with fsspec.open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        shards = sum(len(row["shards"]) for row in rows)
        total_shards += shards
        logger.info("%s: %d groups, %d shards -> %s", region, len(rows), shards, path)

    summary = {
        "num_warcs": num_warcs,
        "seed": seed,
        "pool_size": len(pool),
        "total_shards": total_shards,
        "duplicate_keys": duplicates,
        "warcs": sample,
        # How many region-groups each WARC was split into. The report needs this
        # to tell "this WARC is fully tallied" from "one of its regions has not
        # reported yet", which no amount of looking at the tallies can reveal.
        "groups_per_warc": {warc: len({region for _, region in by_warc[warc].values()}) for warc in sample},
    }
    with fsspec.open(f"{out_dir}/sample.json", "w") as fh:
        json.dump(summary, fh)
    logger.info("sample of %d WARCs, %d shards total", num_warcs, total_shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listings", required=True, help="directory of cached {region}.txt listings")
    parser.add_argument("--num-warcs", type=int, default=1000, help="WARCs to sample; 0 = every eligible WARC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True, help="where the manifests are written")
    parser.add_argument(
        "--only-region",
        choices=REGIONS,
        help="Index one region's batches of every completed WARC instead of sampling whole WARCs across regions.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.only_region:
        build_single_region(args.listings, args.only_region, args.out_dir.rstrip("/"))
        return
    build(args.listings, args.num_warcs, args.seed, args.out_dir.rstrip("/"))


if __name__ == "__main__":
    main()
