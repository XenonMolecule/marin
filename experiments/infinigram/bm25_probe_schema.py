# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-cluster probe: verify the real training-data tiers for BM25 indexing.

For each candidate document glob (the tiers the training caches were actually
tokenized from, per the code trace), report shard count, a sample record's keys,
and whether ``url`` is present inline (so we know if BM25 metadata needs a
provenance join). Raises with a compact summary for bug-report capture.
"""

import itertools
import json
import logging

from zephyr.readers import load_file

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# (label, glob). Confighash `-*` segments are resolved by fsspec's plain glob here.
CANDIDATES: dict[str, str] = {
    "dclm-full": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977/data-*.jsonl.gz",
    "nemotron_full-full": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9/*.jsonl.gz",
    "fineweb_edu-full": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-*/*.jsonl.gz",
    "fineweb_cc-full": "gs://marin-us-central2/documents/baseline_fineweb_cc_deduped/10364warcs/deduped/data-*.jsonl.gz",
    "dclm-small": "gs://marin-us-central2/filtered_subsets/dclm_random_300warcs-*/data-*.jsonl.gz",
    "nemotron_full-small": "gs://marin-us-central2/filtered_subsets/nemotron_full_random_300warcs-*/data-*.jsonl.gz",
    "high_quality-small": "gs://marin-us-central1/filtered_subsets/high_quality_random_300warcs-*/data-*.jsonl.gz",
    "resiliparse_dedup_flat": "gs://marin-us-central2/documents/baseline_resiliparse_deduped/data-*.jsonl.gz",
    "resiliparse_dedup_10k": (
        "gs://marin-us-central2/documents/baseline_resiliparse_deduped/10364warcs/**/data-*.jsonl.gz"
    ),
    "resiliparse_dedup_300": "gs://marin-us-central2/documents/baseline_resiliparse_deduped/300warcs/**/data-*.jsonl.gz",
}


def _probe(glob: str) -> dict:
    shards = fsspec_glob(glob)
    info: dict = {"n": len(shards)}
    if shards:
        try:
            rec = next(itertools.islice(load_file(sorted(shards)[0]), 1))
            info["keys"] = sorted(rec.keys())[:18]
            info["has_url"] = "url" in rec
            info["has_text"] = bool(rec.get("text") or rec.get("generated_text"))
        except Exception as e:
            info["read_err"] = str(e)[:80]
    return info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out: dict[str, dict] = {}
    for label, glob in CANDIDATES.items():
        info = _probe(glob)
        out[label] = info
        logger.info("SCHEMA %s -> %s", label, json.dumps(info))
    compact = {k: {"n": v["n"], "url": v.get("has_url"), "text": v.get("has_text")} for k, v in out.items()}
    raise RuntimeError(f"SCHEMA_DONE {json.dumps(compact)}")


if __name__ == "__main__":
    main()
