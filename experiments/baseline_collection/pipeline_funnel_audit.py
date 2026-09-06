# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the full curation pipeline funnel on a RANDOM RAW sample.

For a random sample of the raw input population (high_quality_3000_distill =
``data/`` useful ∪ ``data_no_useful/`` abstentions, the actual pre-filter pool),
push every doc through the deployment cascade and count what each stage does:

  stage 0  rules     : the 5-rule HTML keep/filter classifier (rules_spec.json) on raw HTML
  stage 1  fastText  : P(useful) >= --ft-threshold      (0.0121, R0.99 gate)
  stage 2  ModernBERT: P(useful) >= --bert-threshold    (0.01364, R0.98 gate)  [TPU]
  stage 3  router    : survivors -> jusText (P(extractable) >= thr) vs the 1.7B model
  stage 4  context   : of 1.7B-routed docs, how many exceed the model context length

Each stage operates only on the survivors of the previous one (a true funnel). We
emit per-rank counts plus one tiny record per SURVIVOR ``{source, router_prob,
tok_len}`` so routing-threshold and context-length cuts can be re-done post-hoc
without re-running BERT.

Single-host v4-8 (world=4), doc-level sharding, length-bucketed BERT, persistent
GCS XLA cache — all reused from cascade_chat_filter (the proven, deadlock-free path).
"""

import argparse
import gzip
import json
import logging
import os
import re
import threading

import fsspec
import pyarrow.parquet as pq
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from transformers import AutoTokenizer

from experiments.baseline_collection.cascade_chat_filter import (
    LOCAL_XLA_CACHE,
    MODEL_ID,
    SYSTEM_MESSAGE,
    USER_TEMPLATE,
    _cache_sync_down,
    _cache_uploader,
    _ft_useful_prob,
    body_strip,
    bucket_for,
    fasttext_text,
    load_bert,
    pad_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

USEFUL_TMPL = "{base}/data/data-{i:05d}-of-03000.parquet"
NOUSE_TMPL = "{base}/data_no_useful/data-{i:05d}-of-03000.parquet"
ROUTER_POS = "__label__extractable"
QWEN_TOKENIZER = "Qwen/Qwen3-1.7B"
# Context lengths we report the "over-context" fraction at (the 1.7B's window is a config knob).
CTX_THRESHOLDS = (8192, 16384, 32768, 40960)
TOK_HIST_BIN = 2048
TOK_HIST_MAX = 131072

# --- stage 0: the 5 HTML drop-rules, vendored verbatim from the shipped
# classifier.py (data lives in rules_spec.json). KEEP unless any rule fires. ----
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENT = re.compile(r"[.!?]")
_RULES_MAX_CHARS = 200_000
WPS_MAX, LWR_MAX, TTM_MIN, SENT_MIN, VW_MIN = 115.0, 0.923, 0.085, 4, 84


def rules_keep(html: str) -> bool:
    """True = KEEP (predict useful); False = FILTER. Mirrors classifier.py::keep."""
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


def _router_prob(model, text: str) -> float:
    for prob, lab in model.f.predict(text, -1, 0.0, "strict"):
        if lab == ROUTER_POS:
            return float(prob)
    return 0.0


def iter_html(path):
    """Stream raw_html from a parquet shard; skip empty html. Missing file -> nothing."""
    try:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["raw_html"], batch_size=1024):
                for html in batch.column("raw_html").to_pylist():
                    if html:
                        yield html
    except FileNotFoundError:
        return


def new_counts() -> dict:
    return {"n_total": 0, "n_rules_kept": 0, "n_ft_kept": 0, "n_bert_kept": 0}


def _mp_fn(index):
    import fasttext

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-base", default="gs://marin-us-central2/datasets/high_quality_3000_distill")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--bert-ckpt", required=True)
    ap.add_argument("--router-model", required=True)
    ap.add_argument("--ft-threshold", type=float, default=0.0121)
    ap.add_argument("--bert-threshold", type=float, default=0.01364)
    ap.add_argument("--router-threshold", type=float, default=0.355820894241333)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--html-cap", type=int, default=1_000_000)
    ap.add_argument("--split", default="test", help="manifest split to sample WARCs from")
    ap.add_argument("--num-warcs", type=int, default=20, help="how many WARCs from the split to sample")
    ap.add_argument("--warc-seed", type=int, default=7)
    ap.add_argument("--subsample-stride", type=int, default=4, help="keep 1 of every K docs (across both classes)")
    ap.add_argument("--xla-cache", default="gs://marin-us-central2/tmp/cascade_xla_cache_v4")
    args = ap.parse_args()

    rank, world = xr.global_ordinal(), xr.world_size()
    is_main = rank == 0

    def log(msg):
        print(f"[funnel r{rank}/{world}] {msg}", flush=True)

    if args.xla_cache:
        xr.initialize_cache(LOCAL_XLA_CACHE, readonly=False)
        if is_main:
            threading.Thread(target=_cache_uploader, args=(args.xla_cache, log), daemon=True).start()

    device = xm.xla_device()

    with fsspec.open(args.manifest, "rt", encoding="utf-8") as f:
        manifest = json.load(f)
    warcs = list(manifest[args.split])
    import random as _r

    _r.Random(args.warc_seed).shuffle(warcs)
    warcs = warcs[: args.num_warcs]
    log(f"sampling {len(warcs)} WARCs from split={args.split}, stride={args.subsample_stride}; warcs={warcs}")

    # models: fastText + router on CPU (per-rank), BERT on TPU (rank-0 read + broadcast),
    # Qwen tokenizer for the 1.7B input-length stage.
    with fsspec.open(args.ft_model, "rb") as s, open(f"/app/_ft_{rank}.bin", "wb") as d:
        d.write(s.read())
    ftmodel = fasttext.load_model(f"/app/_ft_{rank}.bin")
    with fsspec.open(args.router_model, "rb") as s, open(f"/app/_rt_{rank}.bin", "wb") as d:
        d.write(s.read())
    router = fasttext.load_model(f"/app/_rt_{rank}.bin")
    bert_tok = AutoTokenizer.from_pretrained(MODEL_ID)
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    bert = load_bert(args.bert_ckpt, device, "sdpa", is_main)
    log("models ready; starting funnel")

    counts = {"all": new_counts(), "useful": new_counts(), "no_useful": new_counts()}
    router_hist = [0] * 101  # P(extractable)*100 over BERT survivors
    n_justext = {"all": 0, "useful": 0, "no_useful": 0}
    n_1p7b = {"all": 0, "useful": 0, "no_useful": 0}
    tok_hist = [0] * (TOK_HIST_MAX // TOK_HIST_BIN + 1)  # 1.7B-routed input token lengths
    ctx_over = {str(t): 0 for t in CTX_THRESHOLDS}
    n_1p7b_tok = 0

    pad_id = bert_tok.pad_token_id
    fs_out, _ = fsspec.core.url_to_fs(args.out_root)
    surv_path = f"{args.out_root}/survivors_r{rank:02d}.jsonl.gz"
    surv_tmp = f"/app/_surv_r{rank:02d}.jsonl.gz"
    surv_f = gzip.open(surv_tmp, "wt", encoding="utf-8")

    def bump(src, key):
        counts["all"][key] += 1
        counts[src][key] += 1

    gidx = -1  # global doc index across all sampled WARCs+classes (for subsample + sharding)

    def run_bert_batch(survivors):
        """survivors: list[(src, bs, ids, ftp)] -> score, route, tokenize the BERT-survivors."""
        nonlocal n_1p7b_tok
        survivors.sort(key=lambda s: len(s[2]))
        for start in range(0, len(survivors), args.batch_size):
            batch = survivors[start : start + args.batch_size]
            bucket = bucket_for(max(len(s[2]) for s in batch), args.max_length)
            input_ids, attn = pad_batch([s[2] for s in batch], pad_id, bucket, args.batch_size)
            with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
                logits = bert(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            xm.mark_step()
            probs = probs.cpu().tolist()[: len(batch)]
            for (src, bs, _ids, _ftp), pr in zip(batch, probs):
                if pr < args.bert_threshold:
                    continue
                bump(src, "n_bert_kept")
                rp = _router_prob(router, fasttext_text(bs))
                router_hist[min(100, int(rp * 100))] += 1
                to_justext = rp >= args.router_threshold
                (n_justext if to_justext else n_1p7b)["all"] += 1
                (n_justext if to_justext else n_1p7b)[src] += 1
                tok_len = None
                if not to_justext:  # 1.7B input-length stage
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
                surv_f.write(json.dumps({"source": src, "router_prob": round(rp, 5), "tok_len": tok_len}) + "\n")

    for wi, warc in enumerate(warcs):
        survivors = []
        for tmpl, src in ((USEFUL_TMPL, "useful"), (NOUSE_TMPL, "no_useful")):
            for html in iter_html(tmpl.format(base=args.data_base, i=warc)):
                gidx += 1
                if gidx % args.subsample_stride != 0:  # uniform doc subsample
                    continue
                if gidx % world != rank:  # doc-level sharding across ranks
                    continue
                bump(src, "n_total")
                if not rules_keep(html[: args.html_cap]):
                    continue
                bump(src, "n_rules_kept")
                bs = body_strip(html[: args.html_cap])
                ftp = _ft_useful_prob(ftmodel, fasttext_text(bs))
                if ftp < args.ft_threshold:
                    continue
                bump(src, "n_ft_kept")
                ids = bert_tok(fasttext_text(bs), truncation=True, max_length=args.max_length)["input_ids"]
                survivors.append((src, bs, ids, ftp))
        run_bert_batch(survivors)
        log(
            f"WARC {warc:05d} ({wi+1}/{len(warcs)}) done; "
            f"totals all={counts['all']['n_total']} rules={counts['all']['n_rules_kept']} "
            f"ft={counts['all']['n_ft_kept']} bert={counts['all']['n_bert_kept']}"
        )

    surv_f.close()
    with open(surv_tmp, "rb") as s, fsspec.open(surv_path, "wb") as d:
        d.write(s.read())
    os.remove(surv_tmp)

    summary = {
        "rank": rank,
        "world": world,
        "split": args.split,
        "warcs": warcs,
        "subsample_stride": args.subsample_stride,
        "thresholds": {"ft": args.ft_threshold, "bert": args.bert_threshold, "router": args.router_threshold},
        "counts": counts,
        "n_justext": n_justext,
        "n_1p7b": n_1p7b,
        "router_hist": router_hist,
        "tok_hist_bin": TOK_HIST_BIN,
        "tok_hist": tok_hist,
        "ctx_over": ctx_over,
        "n_1p7b_tok": n_1p7b_tok,
    }
    with fsspec.open(f"{args.out_root}/funnel_r{rank:02d}.json", "wt", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"DONE rank {rank}: {json.dumps(counts['all'])} justext={n_justext['all']} 1.7b={n_1p7b['all']}")


if __name__ == "__main__":
    # Pre-spawn (single process, no race): warm the local XLA cache from GCS so every
    # rank starts with the cascade's already-compiled seq-8192 programs (no recompile).
    _pre = argparse.ArgumentParser()
    _pre.add_argument("--xla-cache", default="gs://marin-us-central2/tmp/cascade_xla_cache_v4")
    _xc = _pre.parse_known_args()[0].xla_cache
    if _xc:
        try:
            n = _cache_sync_down(_xc, LOCAL_XLA_CACHE)
            print(f"xla-cache: downloaded {n} cached programs from {_xc}", flush=True)
        except Exception as e:
            print(f"xla-cache download error: {e}", flush=True)
    xmp.spawn(_mp_fn)
