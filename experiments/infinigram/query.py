# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open and query infini-gram-mini indices, and smoke-test a freshly built one.

Switching which dataset you search is a one-liner: :func:`open_engine` maps a
``(dataset, collection)`` to its index dir and returns a ready engine. Because
``InfiniGramMiniEngine`` accepts a *list* of index dirs, you can also search
several corpora jointly by passing more than one target.

The engine and its ``cpp_engine`` extension live in the vendored checkout at
``INFINIGRAM_MINI_DIR`` (see ``packaging.md``); we add ``src/`` to ``sys.path``
lazily so importing this module never requires the toolchain unless you query.
"""

import json
import logging
import os
import sys
import tempfile
from typing import Any

import fsspec
from marin.utils import fsspec_exists

from experiments.infinigram.gcs_io import download_dir
from experiments.infinigram.targets import Collection, IndexTarget, get_target

logger = logging.getLogger(__name__)

# Probes for the post-build smoke test: a ubiquitous token and a phrase that must
# resolve to at least one retrievable document with intact provenance metadata.
_SMOKE_COUNT_PROBE = "the"
_SMOKE_FIND_PROBE = "the United States"


def _load_engine_class():
    """Import InfiniGramMiniEngine from the vendored checkout (needs the compiled engine).

    The engine lives in ``<repo>/engine/src/engine.py`` and mixes an absolute
    ``from src.models import ...`` with a relative ``from .cpp_engine import ...``,
    so it must be imported as ``src.engine`` with ``<repo>/engine`` on the path;
    the compiled ``cpp_engine`` extension sits beside it in ``engine/src/``.
    """
    ig_dir = os.environ.get("INFINIGRAM_MINI_DIR")
    if not ig_dir:
        raise RuntimeError("INFINIGRAM_MINI_DIR is not set; cannot load the query engine.")
    engine_root = os.path.join(ig_dir, "engine")
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)
    from src.engine import InfiniGramMiniEngine

    return InfiniGramMiniEngine


def index_dirs_for(target: IndexTarget) -> list[str]:
    """Resolve a target's on-disk shard dirs from its uploaded ``manifest.json``."""
    manifest_url = f"{target.index_dir.rstrip('/')}/manifest.json"
    if not fsspec_exists(manifest_url):
        raise FileNotFoundError(f"no index at {target.index_dir} (missing {manifest_url}); build it first.")
    with fsspec.open(manifest_url, "r") as f:
        return list(json.load(f)["shard_dirs"])


def _localize(index_dirs: list[str], cache_root: str) -> list[str]:
    """Mirror gs:// index dirs to local disk (mmap needs local files)."""
    local: list[str] = []
    for d in index_dirs:
        if not d.startswith("gs://"):
            local.append(d)
            continue
        dest = os.path.join(cache_root, d.replace("gs://", ""))
        download_dir(d, dest)
        local.append(dest)
    return local


def open_engine(
    dataset: str,
    collection: Collection = Collection.FULL,
    *,
    also: list[tuple[str, Collection]] | None = None,
    load_to_ram: bool = False,
    cache_root: str | None = None,
) -> Any:
    """Open the index for ``(dataset, collection)`` (optionally several) for querying.

    Args:
        dataset: registry dataset name, e.g. ``"dclm"``.
        collection: FULL (10k) or SMALL (random-300).
        also: extra ``(dataset, collection)`` pairs to search jointly.
        load_to_ram: load index files into RAM instead of mmap-from-disk.
        cache_root: where to mirror gs:// indices locally (default a temp dir).

    Returns a ready ``InfiniGramMiniEngine``. Switch datasets by calling again
    with a different name.
    """
    targets = [get_target(dataset, collection)]
    for ds, col in also or []:
        targets.append(get_target(ds, col))

    dirs: list[str] = []
    for t in targets:
        dirs.extend(index_dirs_for(t))

    cache_root = cache_root or tempfile.mkdtemp(prefix="infinigram-idx-")
    local_dirs = _localize(dirs, cache_root)
    logger.info("Opening engine over %d index dir(s): %s", len(local_dirs), [t.name for t in targets])
    engine_cls = _load_engine_class()
    return engine_cls(index_dirs=local_dirs, load_to_ram=load_to_ram, get_metadata=True)


def _first_doc(engine, query: str, find_result: dict) -> dict | None:
    """Retrieve the first document matching ``query`` from a ``find()`` result.

    ``find()`` returns ``{'cnt', 'segment_by_shard': [[start, end], ...]}`` (one
    rank range per shard); ``get_doc_by_rank(s, rank, needle_len, max_ctx_len)``
    returns ``{text, metadata: {..., metadata: {<record fields>}}, ...}``.
    """
    needle_len = len(query.encode("utf-8"))
    for s, seg in enumerate(find_result.get("segment_by_shard", [])):
        start, end = seg
        if end > start:
            doc = engine.get_doc_by_rank(s=s, rank=start, needle_len=needle_len, max_ctx_len=40)
            if isinstance(doc, dict) and "error" not in doc:
                return doc
    return None


def record_metadata(doc: dict) -> dict:
    """The original document's stored fields (url, warc ids, ...) from a doc result.

    infini-gram-mini nests them under ``metadata.metadata`` (its outer metadata
    also carries shard ``path``/``linenum``)."""
    meta = doc.get("metadata") or {}
    inner = meta.get("metadata")
    return inner if isinstance(inner, dict) else meta


def smoke_test_index(index_dirs: list[str]) -> dict:
    """Validate a freshly built (local) index before upload.

    Asserts a ubiquitous token counts > 0 and a common phrase retrieves at least
    one document, then surfaces that document's provenance so we can confirm
    ``url``/warc ids round-trip. Returns a report; raises on the hard assertions.
    """
    engine_cls = _load_engine_class()
    engine = engine_cls(index_dirs=index_dirs, load_to_ram=False, get_metadata=True)

    count = engine.count(_SMOKE_COUNT_PROBE)["count"]
    if count <= 0:
        raise AssertionError(f"count({_SMOKE_COUNT_PROBE!r}) == {count}; index looks empty")

    find_result = engine.find(_SMOKE_FIND_PROBE)
    doc = _first_doc(engine, _SMOKE_FIND_PROBE, find_result)
    if doc is None:
        raise AssertionError(f"find({_SMOKE_FIND_PROBE!r}) retrieved no document (cnt={find_result.get('cnt')})")

    meta = record_metadata(doc)
    report = {
        "count_probe": _SMOKE_COUNT_PROBE,
        "count": count,
        "find_probe": _SMOKE_FIND_PROBE,
        "find_hits": find_result.get("cnt"),
        "doc_metadata_keys": sorted(meta.keys()),
        "doc_url": meta.get("url"),
    }
    logger.info("Smoke test OK: %s", report)
    return report
