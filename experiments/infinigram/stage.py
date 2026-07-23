# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a resolved corpus onto local disk in a form infini-gram-mini can index.

infini-gram-mini reads local files only (globs ``{data_dir}/**/*.json*``) and
takes the ``text`` field of each JSON line as the document, keeping every other
field as searchable metadata. We stream every source shard (jsonl.gz or parquet,
both handled by ``zephyr.load_file``), reading in-region, and write local
``.jsonl.gz`` shards. In the same pass we:

* attach provenance (url / warc ids) to text-only tiers via an exact text-hash
  join (see :mod:`~experiments.infinigram.provenance`), so lookups return an id;
* emit ``url_index.jsonl.gz`` -- one ``{url, warc_record_id, warc_file,
  snapshot}`` row per document -- a compact keyed table for url -> extraction
  lookup, uploaded alongside the index.

Runs in the corpus's own region (the Iris job is pinned there); resolve has
already checked every shard is in-region.
"""

import gzip
import json
import logging
import os
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

from zephyr.readers import load_file

from experiments.infinigram.provenance import PROVENANCE_FIELDS, build_provenance_map, content_hash
from experiments.infinigram.resolve import ResolvedTarget

logger = logging.getLogger(__name__)

URL_INDEX_NAME = "url_index.jsonl.gz"
# Shards are processed concurrently; many-small-shard corpora (nemotron: 24k
# shards) are dominated by per-shard GCS open latency, which threads overlap.
_STAGE_WORKERS = 16


@dataclass(frozen=True)
class StagedCorpus:
    """A corpus written to local disk, ready for the indexer's ``--data_dir``."""

    data_dir: str
    shard_count: int
    byte_count: int
    doc_count: int
    matched_provenance: int
    url_index_path: str


def _local_bytes(data_dir: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(data_dir):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def _doc_text(rec: dict, text_field: str) -> str | None:
    return rec.get(text_field) or rec.get("generated_text")


def _collect_wanted_hashes(shard_urls: tuple[str, ...], text_field: str) -> set[str]:
    """First pass over the (text-only) corpus: the text hashes needing provenance."""
    wanted: set[str] = set()
    for url in shard_urls:
        for rec in load_file(url):
            text = _doc_text(rec, text_field)
            if text:
                wanted.add(content_hash(text))
    return wanted


@contextmanager
def stage_corpus(resolved: ResolvedTarget, local_root: str, *, text_field: str = "text") -> Iterator[StagedCorpus]:
    """Stream ``resolved``'s shards to local disk and yield a :class:`StagedCorpus`.

    Cleans up the staged data dir on exit (the index itself lives elsewhere).
    """
    target = resolved.target
    data_dir = os.path.join(local_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    url_index_path = os.path.join(local_root, URL_INDEX_NAME)

    prov_map: dict[str, dict] = {}
    if target.provenance_globs:
        wanted = _collect_wanted_hashes(resolved.shard_urls, text_field)
        prov_map = build_provenance_map(target.provenance_globs, wanted)

    uidx_dir = os.path.join(local_root, "_uidx")
    os.makedirs(uidx_dir, exist_ok=True)

    def _process(item: tuple[int, str]) -> tuple[int, int]:
        i, url = item
        docs = matched = 0
        out_path = os.path.join(data_dir, f"shard-{i:05d}.jsonl.gz")
        uidx_path = os.path.join(uidx_dir, f"uidx-{i:05d}.jsonl.gz")
        with gzip.open(out_path, "wt", encoding="utf-8") as out, gzip.open(uidx_path, "wt", encoding="utf-8") as uidx:
            for rec in load_file(url):
                text = _doc_text(rec, text_field)
                if not text:
                    continue
                if prov_map:
                    prov = prov_map.get(content_hash(text))
                    if prov:
                        rec = {**rec, **prov}
                        matched += 1
                if text_field != "text":
                    rec = {**rec, "text": text}
                out.write(json.dumps(rec, ensure_ascii=False))
                out.write("\n")
                docs += 1
                if rec.get("url"):
                    uidx.write(json.dumps({f: rec.get(f) or "" for f in PROVENANCE_FIELDS}, ensure_ascii=False))
                    uidx.write("\n")
        return docs, matched

    doc_count = 0
    matched = 0
    try:
        with ThreadPoolExecutor(max_workers=_STAGE_WORKERS) as ex:
            for docs, m in ex.map(_process, enumerate(resolved.shard_urls)):
                doc_count += docs
                matched += m
        # Concatenate per-shard url-index files (gzip streams concatenate).
        with open(url_index_path, "wb") as combined:
            for i in range(len(resolved.shard_urls)):
                part = os.path.join(uidx_dir, f"uidx-{i:05d}.jsonl.gz")
                with open(part, "rb") as pf:
                    shutil.copyfileobj(pf, combined)
        shutil.rmtree(uidx_dir, ignore_errors=True)

        byte_count = _local_bytes(data_dir)
        logger.info(
            "Staged %s: %d docs across %d shards, %.2f GiB local; provenance matched %d",
            data_dir,
            doc_count,
            resolved.shard_count,
            byte_count / 1024**3,
            matched,
        )
        yield StagedCorpus(
            data_dir=data_dir,
            shard_count=resolved.shard_count,
            byte_count=byte_count,
            doc_count=doc_count,
            matched_provenance=matched,
            url_index_path=url_index_path,
        )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
