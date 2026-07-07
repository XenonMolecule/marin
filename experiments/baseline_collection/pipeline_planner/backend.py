# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local server for the cascade planner — and an OPTIONAL full-doc API.

The planner is designed to be **fully static** (deployable to Vercel): the browser does all the math
over `web/data/pipeline_matrix_*.json` and lazy-fetches `web/data/docs/shard-*.json.gz` for borderline
inspection. This server exists for two reasons only:

  1. **Local dev** — serve `web/` over http so `fetch()` + `DecompressionStream` work (file:// blocks them).
  2. **Full 100k docs** — an optional `/api/doc/<id>` that reads the full scored parquet for any record
     (beyond the 10k bundled into the static shards). Set ``DOC_PARQUET`` to enable; runs in-region
     (us-east5) where the data lives — NOT something to point at from a laptop (cross-region pull).

CORS is open so a Vercel-hosted frontend could call a separately-hosted instance, but the primary
deployment is static-only with no backend at all.

Run::  python -m experiments.baseline_collection.pipeline_planner.backend --port 8092
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
DOC_PARQUET = os.environ.get("DOC_PARQUET", "")  # gs://.../sample_100k_scored — optional, in-region only
_DOC_INDEX: dict[str, dict] | None = None
_DOC_COLS = ["url", "stripped_html", "raw_html", "text_8b", "text_1p7b", "text_0p6b", "text_justext"]

app = Flask(__name__)


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:rel>")
def static_file(rel):
    return send_from_directory(WEB_DIR, rel)


def _load_doc_index() -> dict[str, dict]:
    """Build warc_record_id -> full-doc dict from the scored parquet (lazy; in-region)."""
    global _DOC_INDEX
    if _DOC_INDEX is None:
        import fsspec
        import pyarrow.parquet as pq
        from marin.utils import fsspec_glob

        idx: dict[str, dict] = {}
        for path in sorted(fsspec_glob(f"{DOC_PARQUET}/*.parquet")):
            with fsspec.open(path, "rb") as fh:
                t = pq.ParquetFile(fh).read(columns=["warc_record_id", *_DOC_COLS])
            ids = t.column("warc_record_id").to_pylist()
            cols = {c: t.column(c).to_pylist() for c in _DOC_COLS}
            for i, rid in enumerate(ids):
                idx[rid] = {c: cols[c][i] for c in _DOC_COLS}
        logger.info("doc index: %d records", len(idx))
        _DOC_INDEX = idx
    return _DOC_INDEX


@app.route("/api/doc/<path:warc_record_id>")
def api_doc(warc_record_id):
    if not DOC_PARQUET:
        return jsonify({"ok": False, "error": "DOC_PARQUET not set; static doc shards serve the default 10k"}), 404
    doc = _load_doc_index().get(warc_record_id)
    if doc is None:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "data": doc})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8092)
    args = p.parse_args()
    logger.info("serving %s on http://127.0.0.1:%d  (DOC_PARQUET=%s)", WEB_DIR, args.port, DOC_PARQUET or "unset")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
