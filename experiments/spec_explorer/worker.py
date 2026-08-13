# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-region heavy worker for the spec-explorer dashboard (an HTTP endpoint).

Runs as an on-demand Iris CPU job in ONE region and serves that region's
datasets, so the large ``text.parquet`` (up to ~7 GB/dataset) and BM25 indices
(up to ~12 GB) are read where they live — never mirrored or pulled cross-region.

It runs a small ``http.server`` (no extra deps) and registers its address as an
Iris endpoint via ``ctx.registry.register``. The light tier reaches it through
the controller's **generic HTTP endpoint proxy** at
``{dashboard}/proxy/<name-with-slashes-as-dots>/<path>`` (the deployed
controller has this reverse proxy even though it lacks the newer actor proxy).

HTTP methods (JSON in/out):

- ``GET  /health``  -> region, datasets, readiness.
- ``POST /resolve`` -> ``{dataset, url_hs, max_chars}`` -> resolve coverage
  ``url_h`` (from the set-difference explorer) to extracted text: ``url_h`` ->
  ``url_key`` (via this region's small ``meta.parquet``, ``url_h = u64(url_key)``)
  -> ``text`` (pruned ``text.parquet`` read).
- ``POST /bm25``    -> ``{dataset, query, k}`` -> ranked BM25 hits.

Region + served datasets come from env (``SPEC_EXPLORER_REGION``,
``SPEC_EXPLORER_DATASETS``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb

from experiments.infinigram.targets import Collection
from experiments.url_index import layout
from experiments.url_index.duckdb_gcs import cap_memory, maybe_register_gcs
from experiments.url_index.keys import u64

logger = logging.getLogger(__name__)

COLLECTION = Collection.SMALL
ENDPOINT_NAME = "spec_worker"
HEARTBEAT_PREFIX = "spec_explorer/worker_heartbeat"
# Local mirror of text.parquet so resolves read from disk, not per-query gs://
# (the big LLM-pipeline files are 5-7 GB and slow to read over the network).
TEXT_DIR = "/tmp/spec_explorer_text"
# Per-dataset url_key-indexed duckdb built from the mirror. A raw
# `read_parquet(...) WHERE url_key IN (...)` reads the whole text column every
# call (~13s for a 7 GB file); an ART index on url_key makes resolves point
# lookups (ms), which is what keeps search under the latency budget.
INDEX_DIR = "/tmp/spec_explorer_index"


def _heartbeat(region: str, stage: str, info: dict | None = None) -> None:
    """Best-effort GCS marker so the light tier can see startup progress even if finelog is down."""
    try:
        from google.cloud import storage

        blob = f"{HEARTBEAT_PREFIX}/{region}_{stage}.json"
        storage.Client().bucket(f"marin-{region}").blob(blob).upload_from_string(
            json.dumps({"stage": stage, **(info or {})})
        )
    except Exception:
        logger.exception("heartbeat %s failed (ignored)", stage)


class SpecExplorerWorker:
    """Serves one region's datasets: url_h -> text resolution and BM25 search."""

    def __init__(self, region: str, datasets: list[str]):
        self.region = region
        self.datasets = list(datasets)
        self._maps: dict[str, dict[int, tuple[str, str, str]]] = {}
        self._lock = threading.Lock()
        self._bm25: dict[str, object] = {}  # dataset -> Bm25Index (pre-warmed)
        self._bm25_ready: set[str] = set()
        self._bm25_lock = threading.Lock()
        self._text_db: dict[str, str] = {}  # dataset -> path of its url_key-indexed duckdb
        self._text_db_lock = threading.Lock()

    def _build_url_h_map(self, dataset: str) -> dict[int, tuple[str, str, str]]:
        con = duckdb.connect()
        path = layout.meta_path(self.region, COLLECTION, dataset)
        maybe_register_gcs(con, [path])
        cap_memory(con)
        rows = con.execute(
            f"SELECT url_key, warc_record_id, snapshot FROM read_parquet('{path}') WHERE url_key IS NOT NULL"
        ).fetchall()
        con.close()
        out = {u64(uk): (uk, rid or "", snap or "") for uk, rid, snap in rows}
        logger.info("built url_h map for %s: %d urls", dataset, len(out))
        return out

    def _url_h_map(self, dataset: str) -> dict[int, tuple[str, str, str]]:
        if dataset not in self._maps:
            with self._lock:
                if dataset not in self._maps:
                    self._maps[dataset] = self._build_url_h_map(dataset)
        return self._maps[dataset]

    def _check(self, dataset: str) -> None:
        if dataset not in self.datasets:
            raise ValueError(f"dataset {dataset!r} not served in {self.region} (have {self.datasets})")

    def _text_db_path(self, dataset: str) -> str:
        return str(Path(INDEX_DIR) / COLLECTION.value / dataset / "text_lookup.duckdb")

    def _index_blob(self, dataset: str) -> tuple[str, str]:
        """(bucket, blob) of the prebuilt index in this region's bucket (in-region, $0 egress)."""
        return f"marin-{self.region}", f"spec_explorer/text_index/{COLLECTION.value}/{dataset}/text_lookup.duckdb"

    def _ensure_text_index(self, dataset: str) -> None:
        """Make a url_key-indexed duckdb available locally for point-lookup resolves.

        Self-populating cache: prefer a prebuilt index downloaded from in-region
        GCS; if none exists yet, build it from the local text.parquet and upload it
        so every later worker (incl. preemption replacements) just downloads it and
        skips the ~5 min build. The duckdb file is portable (version-pinned).
        Built/fetched atomically via a temp file so a crash never leaves a
        half-written db that looks ready.
        """
        if dataset in self._text_db:
            return
        with self._text_db_lock:
            if dataset in self._text_db:
                return
            dbp = self._text_db_path(dataset)
            if not os.path.exists(dbp):
                Path(dbp).parent.mkdir(parents=True, exist_ok=True)
                if not self._download_prebuilt_index(dataset, dbp):
                    self._build_and_upload_index(dataset, dbp)
            self._text_db[dataset] = dbp

    def _download_prebuilt_index(self, dataset: str, dbp: str) -> bool:
        """Download the prebuilt index from GCS if present. No text.parquet mirror needed."""
        from google.cloud import storage

        bucket, blob = self._index_blob(dataset)
        try:
            b = storage.Client().bucket(bucket).blob(blob)
            if not b.exists():
                return False
            tmp = dbp + ".downloading"
            logger.info("downloading prebuilt text index for %s…", dataset)
            b.download_to_filename(tmp)
            os.rename(tmp, dbp)
            logger.info("prebuilt text index ready for %s", dataset)
            return True
        except Exception:
            logger.exception("downloading prebuilt index for %s failed; will build", dataset)
            return False

    def _build_and_upload_index(self, dataset: str, dbp: str) -> None:
        """Build the index from the local text.parquet, then upload it for future workers."""
        self._ensure_text_local(dataset)
        parquet = str(Path(TEXT_DIR) / COLLECTION.value / dataset / layout.TEXT_NAME)
        tmp = dbp + ".building"
        if os.path.exists(tmp):
            os.remove(tmp)
        logger.info("building text index for %s…", dataset)
        con = duckdb.connect(tmp)
        cap_memory(con)
        con.execute(
            f"CREATE TABLE text_lookup AS SELECT url_key, text FROM read_parquet('{parquet}') WHERE url_key IS NOT NULL"
        )
        con.execute("CREATE INDEX uk_idx ON text_lookup(url_key)")
        con.close()
        os.rename(tmp, dbp)
        logger.info("text index ready for %s", dataset)
        try:
            from google.cloud import storage

            bucket, blob = self._index_blob(dataset)
            storage.Client().bucket(bucket).blob(blob).upload_from_filename(dbp)
            logger.info("uploaded prebuilt text index for %s -> gs://%s/%s", dataset, bucket, blob)
        except Exception:
            logger.exception("uploading text index for %s failed (ignored)", dataset)
        # The parquet mirror was only needed to build the index; resolve reads the
        # duckdb. Delete it so disk holds the text once, not twice — the doubling
        # was overflowing the node with the full us-central1 dataset set.
        try:
            os.remove(parquet)
        except OSError:
            pass

    def _text_by_keys(self, dataset: str, keys: list[str]) -> dict[str, str]:
        """url_key -> text via the dataset's indexed duckdb (point lookups)."""
        if not keys:
            return {}
        self._ensure_text_index(dataset)
        con = duckdb.connect(self._text_db[dataset], read_only=True)
        try:
            placeholders = ", ".join("?" for _ in keys)
            rows = con.execute(
                f"SELECT url_key, text FROM text_lookup WHERE url_key IN ({placeholders})", keys
            ).fetchall()
        finally:
            con.close()
        return {uk: text for uk, text in rows}

    def health(self) -> dict:
        # text is "ready" once its indexed db exists (fast resolves), not merely mirrored.
        text_ready = sorted(ds for ds in self.datasets if os.path.exists(self._text_db_path(ds)))
        return {
            "ok": True,
            "region": self.region,
            "datasets": self.datasets,
            "maps_ready": sorted(self._maps),
            "text_ready": text_ready,
            "bm25_ready": sorted(self._bm25_ready),
            "bm25_pending": sorted(set(self.datasets) - self._bm25_ready),
        }

    def _get_bm25(self, dataset: str):
        """Open (and cache) a dataset's BM25 index; mirrors shards in-region on first use."""
        if dataset not in self._bm25:
            with self._bm25_lock:
                if dataset not in self._bm25:
                    from experiments.infinigram.bm25_query import open_bm25_index

                    logger.info("warming BM25 for %s…", dataset)
                    self._bm25[dataset] = open_bm25_index(dataset, COLLECTION)
                    self._bm25_ready.add(dataset)
                    logger.info("BM25 ready: %s", dataset)
        return self._bm25[dataset]

    def _ensure_text_local(self, dataset: str) -> None:
        """Mirror a dataset's text.parquet to local disk (resolves then read from disk)."""
        dst = Path(TEXT_DIR) / COLLECTION.value / dataset / layout.TEXT_NAME
        if dst.exists():
            return
        from google.cloud import storage

        dst.parent.mkdir(parents=True, exist_ok=True)
        src = layout.text_path(self.region, COLLECTION, dataset)
        bucket, blob = src.replace("gs://", "").split("/", 1)
        logger.info("mirroring text.parquet for %s…", dataset)
        storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dst))
        logger.info("text.parquet local for %s", dataset)

    def warm_all(self) -> None:
        """Pre-warm each dataset's text index + BM25 index, then its url_h map.

        Order matters for memory: the index builds are the heavy step (duckdb loads
        the text column + builds an ART index, ~0.6x the container limit). The
        url_h maps are ~GB-scale resident dicts, so building them FIRST — holding
        all of them while duckdb also runs — OOM-killed the container (exit 137)
        once the us-central1 set grew to 8 datasets. Building maps LAST keeps them
        off the heap during the index builds. Maps are not part of /health warmth;
        they only speed the first resolve. Done in a background thread at startup.
        """
        for ds in self.datasets:
            try:
                self._ensure_text_index(ds)  # heavy: duckdb build — run with no maps resident
            except Exception:
                logger.exception("building text index for %s failed", ds)
        for ds in self.datasets:
            try:
                self._get_bm25(ds)
            except Exception:
                logger.exception("warming BM25 for %s failed", ds)
        for ds in self.datasets:
            try:
                self._url_h_map(ds)  # resident dict, built after the duckdb work is done
            except Exception:
                logger.exception("warming url_h map for %s failed", ds)

    def bm25_multi(self, datasets: list, queries: list, k: int = 20) -> dict:
        """Run every (dataset, query) BM25 search over WARM indices; pool + dedup by url.

        One in-region fan-out call (avoids many proxy round-trips). Skips datasets
        whose index isn't warm yet and reports them in ``pending``.
        """
        wanted = [d for d in datasets or self.datasets if d in self.datasets]
        pooled: dict[str, dict] = {}
        used, pending = [], []
        for ds in wanted:
            if ds not in self._bm25_ready:
                pending.append(ds)
                continue
            used.append(ds)
            index = self._bm25[ds]
            for q in queries:
                for h in index.search(q, k=k):
                    url = h.metadata.get("url") or ""
                    key = url or h.metadata.get("warc_record_id") or f"{ds}:{h.metadata.get('doc_id')}"
                    prev = pooled.get(key)
                    cand = {
                        "url": url,
                        "warc_record_id": h.metadata.get("warc_record_id"),
                        "preview": h.metadata.get("preview"),
                        "modernbert_prob": h.metadata.get("modernbert_prob"),
                        "dataset": ds,
                        "score": float(h.score),
                        "query": q,
                    }
                    # Keep the highest-scoring hit per doc across all (dataset, query) pairs.
                    if prev is None or cand["score"] > prev["score"]:
                        pooled[key] = cand
        hits = sorted(pooled.values(), key=lambda c: -c["score"])
        return {"hits": hits, "used_datasets": used, "pending_datasets": pending}

    def resolve(self, dataset: str, url_hs: list, max_chars: int = 20000) -> dict:
        self._check(dataset)
        wanted = [int(x) for x in url_hs or []]
        if not wanted:
            return {"results": []}
        hmap = self._url_h_map(dataset)
        matched = {h: hmap[h] for h in wanted if h in hmap}
        if not matched:
            return {"results": []}
        text_by_key = self._text_by_keys(dataset, [uk for (uk, _r, _s) in matched.values()])
        results = []
        for h, (uk, rid, snap) in matched.items():
            text = text_by_key.get(uk, "")
            clipped = text[:max_chars] + (f"\n… (+{len(text) - max_chars} chars)" if len(text) > max_chars else "")
            results.append(
                {
                    "url_h": str(h),
                    "url_key": uk,
                    "warc_record_id": rid,
                    "snapshot": snap,
                    "text_len": len(text),
                    "text": clipped,
                }
            )
        return {"results": results}

    def bm25(self, dataset: str, query: str, k: int = 10) -> dict:
        self._check(dataset)
        index = self._get_bm25(dataset)
        hits = [
            {
                "score": h.score,
                "url": h.metadata.get("url"),
                "warc_record_id": h.metadata.get("warc_record_id"),
                "preview": h.metadata.get("preview"),
                "modernbert_prob": h.metadata.get("modernbert_prob"),
                "dataset": dataset,
            }
            for h in index.search(query, k=k)
        ]
        return {"hits": hits}


def _make_handler(worker: SpecExplorerWorker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet the default stderr logging
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, worker.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, {"error": "bad json"})
                return
            route = self.path.rstrip("/").rsplit("/", 1)[-1]  # tolerate proxy prefix
            try:
                if route == "resolve":
                    self._send(
                        200,
                        worker.resolve(body.get("dataset"), body.get("url_hs") or [], int(body.get("max_chars", 20000))),
                    )
                elif route == "bm25":
                    self._send(200, worker.bm25(body.get("dataset"), body.get("query") or "", int(body.get("k", 10))))
                elif route == "bm25_multi":
                    self._send(
                        200,
                        worker.bm25_multi(body.get("datasets") or [], body.get("queries") or [], int(body.get("k", 20))),
                    )
                else:
                    self._send(404, {"error": f"unknown route {route!r}"})
            except Exception as e:  # surface worker errors as JSON
                logger.exception("route %s failed", route)
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    region = os.environ.get("SPEC_EXPLORER_REGION", "")
    datasets = json.loads(os.environ.get("SPEC_EXPLORER_DATASETS", "[]"))
    if not region or not datasets:
        raise SystemExit("set SPEC_EXPLORER_REGION and SPEC_EXPLORER_DATASETS (json list) in the environment")
    _heartbeat(region, "started", {"datasets": datasets})

    from iris.client import iris_ctx
    from iris.cluster.client import get_job_info

    ctx = iris_ctx()
    job_info = get_job_info()
    port = ctx.get_port("actor")
    worker = SpecExplorerWorker(region, datasets)
    server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(worker))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    address = f"http://{job_info.advertise_host}:{port}"
    wire_name = f"{ctx.namespace}/{ENDPOINT_NAME}"

    def _keepalive() -> None:
        """Re-register the endpoint periodically so its lease never expires.

        The controller drops a registered endpoint after a few minutes unless
        it is refreshed. We re-register on a short interval (the worker address
        is stable, so every registration points to the same place) and drop the
        previous registration id to avoid accumulating rows.
        """
        prev_id = None
        first = True
        while True:
            try:
                new_id = ctx.registry.register(ENDPOINT_NAME, address, {"job_id": ctx.job_id.to_wire()})
                if prev_id is not None:
                    try:
                        ctx.registry.unregister(prev_id)
                    except Exception:
                        pass
                prev_id = new_id
                if first:
                    logger.info("registered %s at %s (id=%s)", wire_name, address, new_id)
                    # Per-worker heartbeat (keyed by wire_name, / -> .) so multiple HA
                    # replicas in one region don't overwrite each other's address; the
                    # light tier reads this as a fallback address source.
                    key = wire_name.lstrip("/").replace("/", ".")
                    _heartbeat(region, key, {"address": address, "wire_name": wire_name, "endpoint_id": new_id})
                    first = False
            except Exception:
                logger.exception("endpoint (re)register failed")
            import time as _t

            _t.sleep(45)

    threading.Thread(target=_keepalive, daemon=True).start()

    # Pre-warm BM25 indices in the background so search stays under the proxy's
    # 30s cap; /health reports readiness as datasets finish mirroring in-region.
    threading.Thread(target=worker.warm_all, daemon=True).start()

    threading.Event().wait()  # serve until the container is killed


if __name__ == "__main__":
    main()
