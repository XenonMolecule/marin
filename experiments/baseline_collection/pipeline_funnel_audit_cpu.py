# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only slice of the pipeline funnel (stages 0,1,3,4 — everything except BERT).

Same raw sample as pipeline_funnel_audit.py, but skips the TPU ModernBERT gate so it
schedules instantly and finishes in minutes. Produces:
  stage 0 rules    : 5-rule HTML keep/filter on raw HTML
  stage 1 fastText : P(useful) >= --ft-threshold
  stage 3 router   : ft-survivors -> jusText vs 1.7B   (computed on ft-survivors, an
                     upper bound vs post-BERT; BERT later trims this subset)
  stage 4 context  : of 1.7B-routed, Qwen3-1.7B input token length vs context windows

Records one row per ft-survivor {source, ft_prob, router_prob, tok_len} so the post-BERT
view can be reconstructed once BERT scores exist. BERT (stage 2) comes from the TPU job.
"""

import argparse
import gzip
import json
import logging
import os
import re

import fasttext
import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.extraction_specs import get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

USEFUL_TMPL = "{base}/data/data-{i:05d}-of-03000.parquet"
NOUSE_TMPL = "{base}/data_no_useful/data-{i:05d}-of-03000.parquet"
LABEL_USEFUL = "__label__useful"
ROUTER_POS = "__label__extractable"
QWEN_TOKENIZER = "Qwen/Qwen3-1.7B"
CTX_THRESHOLDS = (8192, 16384, 32768, 40960)
TOK_HIST_BIN = 2048
TOK_HIST_MAX = 131072

_SPEC = get_spec("high_quality")
SYSTEM_MESSAGE = _SPEC.system_message
USER_TEMPLATE = _SPEC.extraction_template

_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")

# stage-0 rules (vendored from classifier.py / rules_spec.json)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENT = re.compile(r"[.!?]")
_RULES_MAX_CHARS = 200_000
WPS_MAX, LWR_MAX, TTM_MIN, SENT_MIN, VW_MIN = 115.0, 0.923, 0.085, 4, 84


def body_strip(html: str) -> str:
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def fasttext_text(bs: str) -> str:
    return _WS_RE.sub(" ", bs).strip().lower()


def rules_keep(html: str) -> bool:
    h = html[:_RULES_MAX_CHARS]
    if not h:
        return False
    visible = _TAG.sub(" ", _COMMENT.sub(" ", _SCRIPT_STYLE.sub(" ", h)))
    words = visible.split()
    n_words = len(words)
    if n_words < VW_MIN:
        return False
    if len(visible.strip()) / len(h) < TTM_MIN:
        return False
    n_sent = len(_SENT.findall(visible))
    if n_sent < SENT_MIN:
        return False
    if n_words / n_sent > WPS_MAX:
        return False
    if sum(1 for w in words if len(w) >= 3) / n_words > LWR_MAX:
        return False
    return True


def _prob(model, text, pos_label):
    for prob, lab in model.f.predict(text, -1, 0.0, "strict"):
        if lab == pos_label:
            return float(prob)
    return 0.0


def iter_html(path):
    try:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["raw_html"], batch_size=1024):
                for html in batch.column("raw_html").to_pylist():
                    if html:
                        yield html
    except FileNotFoundError:
        return


def new_counts():
    return {"n_total": 0, "n_rules_kept": 0, "n_ft_kept": 0}


def local_copy(gs, local):
    with fsspec.open(gs, "rb") as s, open(local, "wb") as d:
        d.write(s.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-base", default="gs://marin-us-central2/datasets/high_quality_3000_distill")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--router-model", required=True)
    ap.add_argument("--ft-threshold", type=float, default=0.0121)
    ap.add_argument("--router-threshold", type=float, default=0.355820894241333)
    ap.add_argument("--html-cap", type=int, default=1_000_000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-warcs", type=int, default=20)
    ap.add_argument("--warc-seed", type=int, default=7)
    ap.add_argument("--subsample-stride", type=int, default=4)
    ap.add_argument("--warc-ids", default=None, help="explicit comma WARC indices (overrides split/shuffle/num-warcs)")
    ap.add_argument("--counts-only", action="store_true", help="stages 0,1 only (rules+ft); skip router+tokenizer")
    args = ap.parse_args()

    import random as _r

    if args.warc_ids:
        warcs = [int(x) for x in args.warc_ids.split(",")]
    else:
        with fsspec.open(args.manifest, "rt", encoding="utf-8") as f:
            manifest = json.load(f)
        warcs = list(manifest[args.split])
        _r.Random(args.warc_seed).shuffle(warcs)
        warcs = warcs[: args.num_warcs]
    logging.info(f"sampling {len(warcs)} WARCs split={args.split} stride={args.subsample_stride}: {warcs}")

    local_copy(args.ft_model, "/app/_ft.bin")
    ftmodel = fasttext.load_model("/app/_ft.bin")
    router = qwen_tok = None
    if not args.counts_only:
        from transformers import AutoTokenizer

        local_copy(args.router_model, "/app/_rt.bin")
        router = fasttext.load_model("/app/_rt.bin")
        qwen_tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    logging.info(f"models ready (counts_only={args.counts_only})")

    counts = {"all": new_counts(), "useful": new_counts(), "no_useful": new_counts()}
    router_hist = [0] * 101
    n_justext = {"all": 0, "useful": 0, "no_useful": 0}
    n_1p7b = {"all": 0, "useful": 0, "no_useful": 0}
    tok_hist = [0] * (TOK_HIST_MAX // TOK_HIST_BIN + 1)
    ctx_over = {str(t): 0 for t in CTX_THRESHOLDS}
    n_1p7b_tok = 0

    surv_tmp = "/app/_surv.jsonl.gz"
    surv_f = gzip.open(surv_tmp, "wt", encoding="utf-8")

    def bump(src, key):
        counts["all"][key] += 1
        counts[src][key] += 1

    gidx = -1
    for wi, warc in enumerate(warcs):
        for tmpl, src in ((USEFUL_TMPL, "useful"), (NOUSE_TMPL, "no_useful")):
            for html in iter_html(tmpl.format(base=args.data_base, i=warc)):
                gidx += 1
                if gidx % args.subsample_stride != 0:
                    continue
                bump(src, "n_total")
                if not rules_keep(html[: args.html_cap]):
                    continue
                bump(src, "n_rules_kept")
                bs = body_strip(html[: args.html_cap])
                ftt = fasttext_text(bs)
                ftp = _prob(ftmodel, ftt, LABEL_USEFUL)
                if ftp < args.ft_threshold:
                    continue
                bump(src, "n_ft_kept")
                if args.counts_only:
                    continue
                rp = _prob(router, ftt, ROUTER_POS)
                router_hist[min(100, int(rp * 100))] += 1
                to_justext = rp >= args.router_threshold
                (n_justext if to_justext else n_1p7b)["all"] += 1
                (n_justext if to_justext else n_1p7b)[src] += 1
                tok_len = None
                if not to_justext:
                    msgs = [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": USER_TEMPLATE.format(example=bs)},
                    ]
                    tok_len = len(qwen_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
                    tok_hist[min(len(tok_hist) - 1, tok_len // TOK_HIST_BIN)] += 1
                    for t in CTX_THRESHOLDS:
                        if tok_len > t:
                            ctx_over[str(t)] += 1
                    n_1p7b_tok += 1
                surv_f.write(
                    json.dumps(
                        {"source": src, "ft_prob": round(ftp, 5), "router_prob": round(rp, 5), "tok_len": tok_len}
                    )
                    + "\n"
                )
        logging.info(f"WARC {warc:05d} ({wi+1}/{len(warcs)}) totals all={counts['all']}")

    surv_f.close()
    with open(surv_tmp, "rb") as s, fsspec.open(f"{args.out_root}/survivors_cpu.jsonl.gz", "wb") as d:
        d.write(s.read())
    os.remove(surv_tmp)

    summary = {
        "split": args.split,
        "warcs": warcs,
        "subsample_stride": args.subsample_stride,
        "thresholds": {"ft": args.ft_threshold, "router": args.router_threshold},
        "counts": counts,
        "n_justext": n_justext,
        "n_1p7b": n_1p7b,
        "router_hist": router_hist,
        "tok_hist_bin": TOK_HIST_BIN,
        "tok_hist": tok_hist,
        "ctx_over": ctx_over,
        "n_1p7b_tok": n_1p7b_tok,
        "note": "CPU-only: BERT (stage 2) NOT applied; router/context on ft-survivors",
    }
    with fsspec.open(f"{args.out_root}/funnel_cpu.json", "wt", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(
        f"DONE: {json.dumps(counts['all'])} justext={n_justext['all']} 1.7b={n_1p7b['all']} over_ctx={ctx_over}"
    )


if __name__ == "__main__":
    main()
