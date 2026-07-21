# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase C scan (Zephyr): distributed in-region PER-EXAMPLE topical scan of a corpus.

For each hq-worse eval EXAMPLE we have a small keyword set (its topic). A corpus doc is
relevant to that example iff it contains most of that example's keywords (>=3 AND >=60%).
Matching per-example (not against the global vocab) gives topical coherence: a page about
photosynthesis matches the photosynthesis example; a tag-cloud/list page matching many
unrelated keywords matches NO single example and is correctly excluded.

Reads jsonl.gz / parquet shards, emits matched docs {url, domain, subjects, tasks,
n_examples, best_frac, snippet} to parquet, distributed across many CPU workers.

EGRESS-SAFE: workers region-pinned to the corpus's own region (ResourceConfig). The
example index (~few MB) is staged once via ctx.put and read by workers via get_shared.

Per corpus, in-region (hq=us-central1, dclm/nemo/fwedu=us-central2):
    iris --cluster marin job run --region us-central2 --cpu 4 --memory 3GB \\
        -e HQKW_METHOD dclm -e HQKW_WORKERS 300 -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -- python experiments/baseline_collection/keyword_scan_zephyr.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter, defaultdict

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.keyword_corpus_scan import (
    _TOKEN_RE,
    MIN_TOKENS,
    SNIPPET_LEN,
    _bucket_of,
    _build_automaton,
    _hits_dir,
    _match_tokens,
)
from experiments.baseline_collection.provenance_audit_10k import registered_domain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("keyword_scan_zephyr")

EXAMPLES_GCS = "gs://marin-us-central2/scratch/provenance_10k/keyword_examples.json"
SHARED_KEY = "kw_examples"
MIN_HITS_PREFILTER = 3  # cheap gate before per-example check
MIN_FRAC = float(os.environ.get("HQKW_MIN_FRAC", "0.6"))  # a doc must contain >= this frac of an example's keywords

# Per-worker lazy singletons (built once per worker from the shared example index).
_EXAMPLES: list | None = None
_KW2EX: dict[str, list[int]] = {}
_AUTO: dict | None = None


def _ensure() -> None:
    global _EXAMPLES, _KW2EX, _AUTO
    if _EXAMPLES is None:
        from zephyr.execution import zephyr_worker_ctx

        data = zephyr_worker_ctx().get_shared(SHARED_KEY)
        _EXAMPLES = data["examples"]
        _AUTO = _build_automaton(data["match_vocab"])
        kw2ex: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(_EXAMPLES):
            for k in e["kws"]:
                kw2ex[k].append(i)
        _KW2EX = kw2ex


def scan_record(rec: dict) -> dict | None:
    """Emit a doc iff it topically matches >=1 hq-worse eval example (>=3 kws AND >=60%)."""
    url, text = rec.get("url"), rec.get("text")
    if not url or not text:
        return None
    _ensure()
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < MIN_TOKENS:
        return None
    hits = _match_tokens(toks, _AUTO)
    if len(hits) < MIN_HITS_PREFILTER:
        return None
    cand: Counter = Counter()
    for k in hits:
        for i in _KW2EX.get(k, ()):
            cand[i] += 1
    subjects, tasks, best = set(), set(), 0.0
    for i, ov in cand.items():
        e = _EXAMPLES[i]
        need = max(3, -(-len(e["kws"]) * int(MIN_FRAC * 100) // 100))  # >=3 and >=MIN_FRAC of the example's keywords
        if ov >= need:
            subjects.add(e["subject"])
            tasks.add(e["task"])
            best = max(best, ov / len(e["kws"]))
    if not subjects:
        return None
    return {
        "url": url,
        "domain": registered_domain(url),
        "subjects": "|".join(sorted(subjects)[:6]),
        "tasks": "|".join(sorted(tasks)),
        "n_examples": len(subjects),
        "best_frac": float(best),
        "snippet": text[:SNIPPET_LEN],
    }


def main() -> int:
    from experiments.baseline_collection.keyword_corpus_scan import SOURCES, _read_json_gcs

    method = os.environ["HQKW_METHOD"]
    workers = int(os.environ.get("HQKW_WORKERS", "300"))
    region = _bucket_of(method).removeprefix("marin-")
    src = SOURCES[method]
    hits_dir = os.environ.get("HQKW_HITS_DIR", _hits_dir(method))
    out = f"{hits_dir}/hits-{{shard:05d}}-of-{{total:05d}}.parquet"
    logger.info("[zephyr-scan] method=%s region=%s workers=%d → %s", method, region, workers, out)

    examples = _read_json_gcs(EXAMPLES_GCS)
    logger.info("[zephyr-scan] %d eval-example keyword sets", len(examples["examples"]))

    glob = os.environ.get("HQKW_GLOB", f"{src['root']}/{src['glob']}")
    ds = Dataset.from_files(glob)
    ds = ds.load_parquet() if src["format"] == "parquet" else ds.load_jsonl()
    ds = ds.map(scan_record).filter(lambda x: x is not None)
    # For corpora with many tiny input shards (nemo: 24k), consolidate the small matched
    # stream into fewer output files (cheap shuffle over the ~8% that survive the filter).
    reshard_n = int(os.environ.get("HQKW_RESHARD", "0"))
    if reshard_n > 0:
        ds = ds.reshard(reshard_n)
    pipeline = ds.write_parquet(out, skip_existing=True)

    ctx = ZephyrContext(
        name=f"kwscan-{method}",
        max_workers=workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=[region], preemptible=True),
    )
    ctx.put(SHARED_KEY, examples)
    result = ctx.execute(pipeline)
    logger.info("[zephyr-scan] done: counters=%s", getattr(result, "counters", None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
