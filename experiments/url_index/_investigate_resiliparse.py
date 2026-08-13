# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throwaway: why are some high_quality URLs not in the resiliparse "universe"?

Runs in-region (us-central2) so reading resiliparse's 12M keys is fast. Null-safe
anti-joins high_quality's url_h against resiliparse's, characterizes the missing
set, then probes whether stripping the query string recovers the match (which would
mean the gap is URL-normalization, not resiliparse failing to extract the page).

Writes the CORE answer to GCS immediately, then overwrites with the fuller version
including the probe -- so a slow/preempted probe still leaves the essential result.
"""

import duckdb
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.url_index.keys import u64

OUT = "gs://marin-us-central2/url_index/analysis/resiliparse_investigation.txt"
RESILI = "gs://marin-us-central2/url_index/small/resiliparse/keys.parquet"
HQ = "gs://marin-us-central1/url_index/small/high_quality/meta.parquet"


def _write(lines: list[str]) -> None:
    with fsspec.open(OUT, "wt") as f:
        f.write("\n".join(lines))


def main() -> None:
    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET threads=4")

    # resiliparse url_h universe (only the u64 -- cheap, ~96 MB).
    con.execute(f"CREATE TABLE r AS SELECT DISTINCT url_h FROM read_parquet('{RESILI}')")
    nnull = con.execute("SELECT count(*) FROM r WHERE url_h IS NULL").fetchone()[0]
    lines = [f"resiliparse distinct url_h: {con.execute('SELECT count(*) FROM r').fetchone()[0]:,} (null: {nnull:,})"]

    t = pq.read_table(HQ, columns=["url_key", "domain", "text_len"])
    uk = t.column("url_key").to_pylist()
    dom = t.column("domain").to_pylist()
    tl = t.column("text_len").to_pylist()
    uh = [u64(x) if x else None for x in uk]
    con.register("hq", pa.table({"url_key": uk, "domain": dom, "text_len": tl, "url_h": pa.array(uh, type=pa.uint64())}))

    # Null-safe anti-join (LEFT JOIN + IS NULL; NOT IN would zero out on any NULL).
    con.execute("CREATE TABLE miss AS SELECT hq.* FROM hq LEFT JOIN r USING (url_h) WHERE r.url_h IS NULL")
    con.execute("CREATE TABLE hit  AS SELECT hq.* FROM hq LEFT JOIN r USING (url_h) WHERE r.url_h IS NOT NULL")
    n = len(uk)
    miss = con.execute("SELECT count(*) FROM miss").fetchone()[0]
    dnull = con.execute("SELECT count(*) FROM miss WHERE url_key IS NULL OR url_key=''").fetchone()[0]
    lines.append(f"high_quality: {n:,} docs; NOT in resiliparse (url_h): {miss:,} ({100 * miss / n:.2f}%)")
    lines.append(f"  of the missing, empty/null url_key: {dnull:,}")
    lines.append(
        "  avg text_len -- missing: "
        f"{con.execute('SELECT round(avg(text_len)) FROM miss').fetchone()[0]}"
        " | present: "
        f"{con.execute('SELECT round(avg(text_len)) FROM hit').fetchone()[0]}"
    )
    lines.append("\nTop domains among MISSING:")
    for d, c in con.execute(
        "SELECT domain,count(*) c FROM miss WHERE domain<>'' GROUP BY domain ORDER BY c DESC LIMIT 15"
    ).fetchall():
        lines.append(f"  {c:>6,}  {d}")
    lines.append("\n20 example MISSING url_keys:")
    for (u,) in con.execute("SELECT url_key FROM miss WHERE url_key<>'' LIMIT 20").fetchall():
        lines.append(f"  {u[:110]}")

    # Persist the core answer NOW, before the expensive probe.
    lines.append("\n(probe pending...)")
    _write(lines)
    lines.pop()  # drop the "pending" line for the final version

    # Probe: do the missing url_keys reappear in resiliparse under looser matching?
    # host+path (strip query) and host-only would indicate a normalization mismatch
    # rather than resiliparse genuinely never extracting the page.
    con.execute(f"CREATE TABLE rk AS SELECT DISTINCT url_key FROM read_parquet('{RESILI}') WHERE url_key<>''")
    lines.append("\nProbe -- missing url_keys recovered under looser matching:")
    exact = con.execute("SELECT count(*) FROM miss m JOIN rk USING (url_key)").fetchone()[0]
    lines.append(f"  exact url_key present in resiliparse: {exact:,} / {miss:,}  (sanity: should be ~0)")
    strip = con.execute(
        "SELECT count(*) FROM miss m WHERE split_part(m.url_key,'?',1) IN (SELECT split_part(url_key,'?',1) FROM rk)"
    ).fetchone()[0]
    lines.append(f"  match after stripping '?query': {strip:,} / {miss:,}")
    withq = con.execute("SELECT count(*) FROM miss WHERE url_key LIKE '%?%'").fetchone()[0]
    lines.append(f"  (for reference, missing url_keys that contain a '?': {withq:,})")

    _write(lines)
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
