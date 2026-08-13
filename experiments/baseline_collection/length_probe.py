# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-token length of the 3/3 docs hq dropped — length-limit artifact or spec issue?

hq's extractor is a Qwen3 LLM with a 32k-token context. Hypothesis: the docs all three
independent filters kept but hq dropped (verifier_score>=3) were dropped MECHANICALLY for
exceeding the context, not by the "useful content" spec. Test: tokenize each such doc's
full text (from the resiliparse raw extraction) with the Qwen3 tokenizer and compare the
length distribution against an hq-kept baseline.

NB the resiliparse EXTRACTED text is a lower bound on the HTML the extractor actually
ingests — so if even the extracted text clears 32k, the drop is definitively length-driven;
if it's short, the length story is weaker (but HTML markup could still push it over).

Reads resiliparse source in-region, tokenizes only the target URLs. Writes a small parquet.
"""

from __future__ import annotations

import os
import sys

import fsspec
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
RES_SRC = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"
SAMPLE = f"{WORKSPACE}/residual_sample.parquet"
MEMBERSHIP = f"{WORKSPACE}/membership/*.parquet"
OUT = f"{WORKSPACE}/length_probe/lp-{{shard:05d}}-of-{{total:05d}}.parquet"
SHARED = "target_urls"
TOK_GCS = f"{WORKSPACE}/qwen3_tok"  # tokenizer staged in-region so 200 workers don't rate-limit HF

_TOK = None
_TARGETS: dict | None = None


def probe(rec: dict) -> dict | None:
    global _TOK, _TARGETS
    url, text = rec.get("url"), rec.get("text")
    if not url or not text:
        return None
    if _TARGETS is None:
        from zephyr.execution import zephyr_worker_ctx

        _TARGETS = zephyr_worker_ctx().get_shared(SHARED)
    label = _TARGETS.get(url)
    if label is None:
        return None
    if _TOK is None:
        from transformers import AutoTokenizer

        local = "/tmp/qwen3_tok"
        if not os.path.exists(os.path.join(local, "tokenizer.json")):
            os.makedirs(local, exist_ok=True)
            fs = fsspec.filesystem("gcs")
            for path in fs.ls(TOK_GCS.removeprefix("gs://")):
                name = path.rsplit("/", 1)[-1]
                if name:
                    fs.get(path, os.path.join(local, name))
        _TOK = AutoTokenizer.from_pretrained(local)
    n = len(_TOK(text, add_special_tokens=False, truncation=False)["input_ids"])
    return {"url": url, "label": label, "n_qwen3_tokens": n, "n_chars": len(text)}


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    a = con.execute(f"SELECT DISTINCT url FROM read_parquet('{SAMPLE}') WHERE verifier_score>=3").fetchall()
    c = con.execute(f"SELECT DISTINCT url FROM read_parquet('{SAMPLE}') WHERE verifier_score=0").fetchall()
    b = con.execute(f"SELECT url FROM read_parquet('{MEMBERSHIP}') WHERE kept_hq USING SAMPLE 3000 ROWS").fetchall()
    targets: dict[str, str] = {}
    for (u,) in a:
        targets[u] = "dropped_3of3"
    for (u,) in c:
        targets.setdefault(u, "dropped_0vote")
    for (u,) in b:
        targets.setdefault(u, "hq_kept")
    print(f"[length-probe] targets: 3of3={len(a)} 0vote={len(c)} hq_kept={len(b)} (deduped total {len(targets)})")

    workers = int(os.environ.get("HQKW_WORKERS", "200"))
    ctx = ZephyrContext(
        name="length-probe",
        max_workers=workers,
        resources=ResourceConfig(cpu=1, ram="6g", regions=["us-central2"], preemptible=True),
    )
    ctx.put(SHARED, targets)
    ds = Dataset.from_files(RES_SRC).load_jsonl().map(probe).filter(lambda x: x is not None).reshard(16)
    ctx.execute(ds.write_parquet(OUT, skip_existing=True))
    print("[length-probe] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
