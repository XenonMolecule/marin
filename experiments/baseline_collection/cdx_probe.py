# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""One-off: validate the columnar CDX query works + time it, writing a small json to GCS."""
import json
import time

import duckdb
import fsspec

from marin.datakit.download.commoncrawl.cdx_query_columnar import _build_domain_filter, _get_parquet_urls

OUT = "gs://marin-us-central2/scratch/provenance_10k_devset/cdx_probe.json"
out: dict = {}
t = time.time()
purls = _get_parquet_urls("CC-MAIN-2015-18")
out["paths_time_s"] = round(time.time() - t, 1)
out["n_parquet"] = len(purls) if purls else 0
if purls:
    con = duckdb.connect()
    con.execute("SET threads=4; SET http_retries=20; SET http_retry_wait_ms=2000;")
    rel = con.read_parquet(purls, hive_partitioning=True)
    rel.create_view("_ccindex", replace=True)
    filt = _build_domain_filter(["en.wikipedia.org"], "host")
    t = time.time()
    rows = con.execute(
        f"SELECT url, warc_filename, warc_record_offset, warc_record_length "
        f"FROM _ccindex WHERE {filt} AND fetch_status=200 LIMIT 5"
    ).fetchall()
    out["query_time_s"] = round(time.time() - t, 1)
    out["n_rows"] = len(rows)
    out["sample"] = [[r[0][:60], int(r[2]), int(r[3])] for r in rows[:3]]
with fsspec.open(OUT, "w") as f:
    json.dump(out, f)
print(json.dumps(out))
