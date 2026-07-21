# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent corpus lookup: fetch matching records from a pipeline's corpus by field match.

Non-circular verification/tracing tool. Scans a corpus in-region (Zephyr, workers pinned to the
source region), keeps every record whose `--match-field` (dotted path, e.g. `url` or
`metadata.nemotron_url`) contains `--contains`, and records the full record (text capped) with the
text length + sha256 so the laptop can pull a tiny parquet and compare byte-for-byte.

Source is either a named pipeline (`--method`, resolved via provenance_audit_10k.SOURCES) or an
explicit `--root`/`--glob`/`--format` (e.g. the genuine staged Nemotron-CC dump).

Examples:
    # our 10k nemo output (join corpus)
    python -m experiments.baseline_collection.find_nemo_url \
        --method nemo --match-field url --contains 'export.arxiv.org/list/physics/1907?skip=275'

    # genuine staged Nemotron-CC dump, one partition
    python -m experiments.baseline_collection.find_nemo_url \
        --root gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl \
        --glob 'quality=high/kind=actual/kind2=actual/*.jsonl.gz' --format jsonl --region us-central2 \
        --match-field metadata.nemotron_url --contains 'export.arxiv.org/list/physics/1907?skip=275'
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx

from experiments.baseline_collection.provenance_audit_10k import SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("find_nemo_url")

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k/devset/url_lookup"

_CFG: dict | None = None


def _dig(rec: dict, dotted: str):
    cur = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _match(rec: dict) -> dict | None:
    """Keep a record iff its match-field contains the needle; record full text + digest."""
    global _CFG
    if _CFG is None:
        _CFG = zephyr_worker_ctx().get_shared("cfg")
    field, needles = _CFG["field"], _CFG["needles"]
    val = _dig(rec, field)
    if not isinstance(val, str) or not any(n in val for n in needles):
        return None
    text = rec.get("text") or ""
    return {
        "match_field": field,
        "match_value": val,
        "text": text[:60000],
        "n_chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "id": str(rec.get("id", "")),
        "metadata": str(rec.get("metadata", ""))[:2000],
        # pass through any nemo join-provenance columns present in our 10k output rows
        "nemotron_quality": str(rec.get("nemotron_quality", "")),
        "nemotron_kind": str(rec.get("nemotron_kind", "")),
        "nemotron_kind2": str(rec.get("nemotron_kind2", "")),
        "nemotron_id": str(rec.get("nemotron_id", "")),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(SOURCES), help="named pipeline source (SOURCES)")
    p.add_argument("--root", help="explicit corpus root (overrides --method)")
    p.add_argument("--glob", help="explicit shard glob (with --root)")
    p.add_argument("--format", choices=["jsonl", "parquet"], help="explicit format (with --root)")
    p.add_argument("--region", help="explicit worker region (with --root)")
    p.add_argument(
        "--match-field", required=True, help="dotted record field to match, e.g. url or metadata.nemotron_url"
    )
    p.add_argument(
        "--contains",
        required=True,
        action="append",
        help="substring the match-field must contain (repeatable; any-match)",
    )
    p.add_argument("--label", required=True, help="output subdir label, e.g. nemo or genuine-high-actual")
    p.add_argument("--max-workers", type=int, default=200)
    args = p.parse_args()

    if args.root:
        root, glob, fmt = args.root, args.glob, args.format
        region = args.region or root.split("/")[2].removeprefix("marin-")
    else:
        src = SOURCES[args.method]
        root, glob, fmt = src["root"], src["glob"], src["format"]
        region = src["root"].split("/")[2].removeprefix("marin-")

    out = f"{WORKSPACE}/{args.label}/hit-{{shard:05d}}-of-{{total:05d}}.parquet"
    ds = Dataset.from_files(f"{root}/{glob}")
    ds = ds.load_parquet() if fmt == "parquet" else ds.load_jsonl()
    ds = ds.map(_match).filter(lambda x: x is not None).reshard(1)
    ctx = ZephyrContext(
        name=f"findurl-{args.label}",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=[region], preemptible=True),
    )
    ctx.put("cfg", {"field": args.match_field, "needles": args.contains})
    ctx.execute(ds.write_parquet(out, skip_existing=False))
    logger.info("[find-url] scanned %s/%s (region=%s); hits -> %s", root, glob, region, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
