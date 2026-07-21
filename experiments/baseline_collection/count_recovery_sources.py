# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""One-off: per-source distinct-doc recovery breakdown + cross-source overlap. Writes a small json."""
import json
from collections import defaultdict

import fsspec
import pyarrow.parquet as pq

OUT = "marin-us-central2/scratch/provenance_10k_devset/devset_html_refetch"
REPORT = "gs://marin-us-central2/scratch/provenance_10k_devset/source_breakdown.json"

fs = fsspec.filesystem("gcs")
by_tag: dict[str, set] = defaultdict(set)
for f in fs.glob(f"{OUT}/*.parquet"):
    tag = f.split("/")[-1].split("-")[0]  # r0 / cx0 / targeted / local
    try:
        d = pq.read_table(f, columns=["dev_url", "status", "html"], filesystem=fs).to_pydict()
    except Exception:
        continue
    for u, st, h in zip(d["dev_url"], d["status"], d["html"], strict=True):
        if st == "ok" and h:
            by_tag[tag].add(u)

tags = sorted(by_tag)
union: set = set()
for t in tags:
    union |= by_tag[t]
# how much does each tag add that no OTHER tag has
uniq_contrib = {}
for t in tags:
    others = set()
    for o in tags:
        if o != t:
            others |= by_tag[o]
    uniq_contrib[t] = len(by_tag[t] - others)

out = {
    "per_tag_distinct": {t: len(by_tag[t]) for t in tags},
    "unique_contribution": uniq_contrib,
    "total_union": len(union),
}
with fsspec.open(REPORT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
