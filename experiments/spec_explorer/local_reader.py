# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Light-tier local data access for text resolution and BM25 search.

The in-region Iris *actor* worker (``worker.py``) is the intended data path, but
the marin controller's actor proxy is currently unreachable (a deployed-image
version skew: register/list_endpoints work, yet ``/iris.actor.ActorService/*``
404s — even from the ``iris`` CLI). Until that's fixed cluster-side, the light
tier reads the (now us-central1-consolidated) 300-WARC artifacts directly:

- **text resolution** — cache the small ``meta.parquet`` locally, build
  ``url_h -> url_key`` (``url_h = u64(url_key)``), then a pruned ``text.parquet``
  read for just the sampled docs. Cheap: meta is ~440 MB total; each view pulls
  only a few row groups.
- **BM25** — mirror a dataset's SMALL index shards to a local cache once
  (``open_bm25_index``) and query them. Feasible for the core datasets (dclm
  0.5 GB, nemotron 1 GB, high_quality 1.9 GB); the LLM pipelines (8/12 GB) are
  too large to mirror and are left to the in-region worker.

All heavy datasets are consolidated in us-central1, so reads hit one region.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import duckdb

from experiments.infinigram.targets import Collection
from experiments.spec_explorer.catalog import INDEX_DATASET_REGION
from experiments.url_index import layout, lookup
from experiments.url_index.keys import u64

logger = logging.getLogger(__name__)

COLLECTION = Collection.SMALL
_CACHE = Path(__file__).parent / "cache"
_META_DIR = _CACHE / "meta" / COLLECTION.value
_BM25_CACHE = _CACHE / "bm25"

# Datasets small enough to mirror a local BM25 index for (GB-scale). The LLM
# pipelines are excluded (8-12 GB); searching those needs the in-region worker.
BM25_LOCAL_DATASETS = ("dclm", "nemotron_full", "high_quality")

_url_h_maps: dict[str, dict[int, tuple[str, str, str]]] = {}
_bm25_indices: dict[str, object] = {}
_lock = threading.Lock()


def _region(dataset: str) -> str:
    region = INDEX_DATASET_REGION.get(dataset)
    if region is None:
        raise ValueError(f"no index region for dataset {dataset!r}")
    return region


def _local_meta(dataset: str) -> Path:
    return _META_DIR / dataset / layout.META_NAME


def _ensure_meta_local(dataset: str) -> Path:
    dst = _local_meta(dataset)
    if not dst.exists():
        from google.cloud import storage

        dst.parent.mkdir(parents=True, exist_ok=True)
        src = layout.meta_path(_region(dataset), COLLECTION, dataset)
        bucket, blob = src.replace("gs://", "").split("/", 1)
        logger.info("downloading meta for %s", dataset)
        storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dst))
    return dst


def _url_h_map(dataset: str) -> dict[int, tuple[str, str, str]]:
    if dataset not in _url_h_maps:
        with _lock:
            if dataset not in _url_h_maps:
                path = _ensure_meta_local(dataset)
                con = duckdb.connect()
                rows = con.execute(
                    f"SELECT url_key, warc_record_id, snapshot FROM read_parquet('{path}') WHERE url_key IS NOT NULL"
                ).fetchall()
                con.close()
                _url_h_maps[dataset] = {u64(uk): (uk, rid or "", snap or "") for uk, rid, snap in rows}
                logger.info("built url_h map for %s: %d urls", dataset, len(_url_h_maps[dataset]))
    return _url_h_maps[dataset]


def resolve_text(dataset: str, url_hs: list, max_chars: int = 20000) -> list[dict]:
    """Resolve ``url_h`` values to extracted text (same shape as ``worker.resolve``)."""
    wanted = [int(x) for x in url_hs or []]
    if not wanted:
        return []
    hmap = _url_h_map(dataset)
    matched = {h: hmap[h] for h in wanted if h in hmap}
    if not matched:
        return []
    region = _region(dataset)
    rows = [{"dataset": dataset, "region": region, "url_key": uk} for (uk, _r, _s) in matched.values()]
    texts = lookup.fetch_texts(rows, COLLECTION, text_dir=None)  # pruned text.parquet read (gs://)
    out = []
    for h, (uk, rid, snap) in matched.items():
        text = texts.get((dataset, uk), "")
        clipped = text[:max_chars] + (f"\n… (+{len(text) - max_chars} chars)" if len(text) > max_chars else "")
        out.append(
            {
                "url_h": str(h),
                "url_key": uk,
                "warc_record_id": rid,
                "snapshot": snap,
                "text_len": len(text),
                "text": clipped,
            }
        )
    return out


def bm25_available(dataset: str) -> bool:
    return dataset in BM25_LOCAL_DATASETS


def _bm25_index(dataset: str):
    if dataset not in _bm25_indices:
        with _lock:
            if dataset not in _bm25_indices:
                from experiments.infinigram.bm25_query import open_bm25_index

                logger.info("opening local BM25 index for %s (mirrors shards on first use)", dataset)
                _bm25_indices[dataset] = open_bm25_index(dataset, COLLECTION, cache_root=str(_BM25_CACHE))
    return _bm25_indices[dataset]


def bm25_search(dataset: str, query: str, k: int = 10) -> list[dict]:
    """Ranked BM25 hits (same shape as ``worker.bm25``) for a mirrorable dataset."""
    if not bm25_available(dataset):
        raise ValueError(f"BM25 for {dataset!r} not mirrorable locally (too large); needs the in-region worker")
    hits = _bm25_index(dataset).search(query, k=k)
    return [
        {
            "score": h.score,
            "url": h.metadata.get("url"),
            "warc_record_id": h.metadata.get("warc_record_id"),
            "preview": h.metadata.get("preview"),
            "modernbert_prob": h.metadata.get("modernbert_prob"),
            "dataset": dataset,
        }
        for h in hits
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from experiments.spec_explorer import coverage_service as cs

    sample = [int(x) for x in cs.set_expression(["dclm"], ["high_quality"], key="url_h", sample=3)["sample"]]
    print("sample url_h:", sample)
    for r in resolve_text("dclm", sample):
        print(f"  {r['url_key'][:55]!r} len={r['text_len']} :: {r['text'][:90]!r}")


if __name__ == "__main__":
    main()
