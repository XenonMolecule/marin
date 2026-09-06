# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cascade-filter the high_quality_3000 distill corpus through BOTH classifiers
(fastText stage-1 -> survivor-BERT stage-2) and emit a distillation **chat**
dataset of the survivors, in the natural class distribution.

A doc is kept iff it passes BOTH gates, scored on the SAME preprocessed text the
classifiers were trained on (``to_fasttext_text`` = body_strip + whitespace-collapse
+ lowercase):

  * stage-1 fastText  P(useful) >= --ft-threshold   (default 0.0121, the R0.99 gate)
  * stage-2 BERT      P(useful) >= --bert-threshold  (default 0.01364, the R0.98 point)

For each survivor we emit the SAME 3-turn chat row as build_hq_distill_chat
(system = high_quality DSPy signature, user = template over case-preserved
body_strip, assistant = empty think + teacher ``final_output``). BOTH classes are
emitted as they survive — real useful docs AND ``[NO_USEFUL_CONTENT]`` abstentions
that the cascade nonetheless scored as useful (the hard look-useful negatives).

Runs on TPU in us-central2 (same region as the parquet + both models). fastText
runs on the host CPU (cheap, prunes ~54%); only the survivors hit the BERT on TPU.
Each rank processes a DISJOINT set of WARCs and writes one durable gzipped JSONL
shard per WARC (``{out}/{split}/data-{warc:05d}.jsonl.gz``), atomic + skip-existing,
so the run is fully resumable and needs NO cross-rank collectives after the initial
weight broadcast (so it can't deadlock mid-run).

The 350k TRAIN cap is applied later at assembly (concat train shards in the frozen
shuffled order, take the first N) so this job just emits all survivors per WARC.
"""

import argparse
import gzip
import json
import logging
import os
import random
import re
import threading
import time

import fsspec
import pyarrow.parquet as pq
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.baseline_collection.extraction_specs import get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = "answerdotai/ModernBERT-base"
LABEL_USEFUL = "__label__useful"
USEFUL_TMPL = "{base}/data/data-{i:05d}-of-03000.parquet"
NOUSE_TMPL = "{base}/data_no_useful/data-{i:05d}-of-03000.parquet"

# Chat scaffold (identical to build_hq_distill_chat.py).
EMPTY_THINK = "<think>\n\n</think>\n\n"
_SPEC = get_spec("high_quality")
SYSTEM_MESSAGE = _SPEC.system_message
USER_TEMPLATE = _SPEC.extraction_template

_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def body_strip(html: str) -> str:
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def fasttext_text(bs: str) -> str:
    """body_strip text -> the classifier input (collapse whitespace + lowercase)."""
    return _WS_RE.sub(" ", bs).strip().lower()


def chat_row(bs: str, final_output: str) -> dict:
    """3-turn chat from case-preserved body_strip text + teacher final_output."""
    user = USER_TEMPLATE.format(example=bs)
    assistant = f"{EMPTY_THINK}[[ ## text ## ]]\n{final_output}\n\n[[ ## completed ## ]]"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# Length buckets: pad each batch up to the smallest bucket >= its longest member.
# Pad tokens are masked out by attention_mask, so bucketed padding yields BIT-IDENTICAL
# logits to fixed 8192 padding while making short-doc batches (the common case) far
# cheaper. The batch dim is held fixed (= --batch-size) so XLA only compiles once per
# bucket (~7 programs total) instead of recompiling per shape.
BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


def bucket_for(length: int, max_length: int) -> int:
    for b in BUCKETS:
        if b >= length:
            return min(b, max_length)
    return max_length


def pad_batch(id_lists, pad_id: int, bucket: int, target_n: int):
    """Pad pre-tokenized id lists to (target_n, bucket). Dummy rows (beyond len(id_lists))
    get a single attended token to avoid all-masked NaN; their outputs are discarded."""
    input_ids = torch.full((target_n, bucket), pad_id, dtype=torch.long)
    attn = torch.zeros((target_n, bucket), dtype=torch.long)
    for k in range(target_n):
        if k < len(id_lists) and id_lists[k]:
            ids = id_lists[k]
            input_ids[k, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[k, : len(ids)] = 1
        else:
            attn[k, 0] = 1
    return input_ids, attn


def iter_docs(path):
    """Stream (raw_html, final_output) from a parquet shard; skip empty html."""
    try:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["raw_html", "final_output"], batch_size=1024):
                h = batch.column("raw_html").to_pylist()
                o = batch.column("final_output").to_pylist()
                for html, final in zip(h, o):
                    if html:
                        yield html, final
    except FileNotFoundError:
        return


def load_bert(ckpt, device, attn, is_main):
    """Build ModernBERT + load survivor weights (rank-0 read, broadcast) — the
    proven path from modernbert_tpu_smoke."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, attn_implementation=attn).to(
        device
    )
    sd = None
    if is_main:
        with fsspec.open(f"{ckpt}/latest.pt", "rb") as f:
            data = f.read()
        local = "/app/_ckpt_load.pt"
        with open(local, "wb") as fh:
            fh.write(data)
        sd = torch.load(local, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(sd)
    xm.broadcast_master_param(model)  # rank-0 weights -> all replicas (the ONLY collective)
    model.eval()
    return model


def process_warc(i, split, rank, world, ftmodel, bert, tokenizer, device, args, log):
    """fastText+BERT cascade this rank's slice of one WARC (both classes) -> chat part.

    DOC-LEVEL SHARDING: every rank processes every WARC but only documents whose global
    index (across the useful then no_useful files) satisfies ``idx % world == rank``, and
    writes a rank-suffixed part ``data-{warc}-r{rank}.jsonl.gz``. This parallelizes a single
    (expensive, ~16k-survivor) WARC across the whole slice instead of pinning it to one rank.
    Skip-existing per part keeps it resumable across preemptions.

    fastText prunes on CPU; survivors are tokenized, sorted by length, and scored in
    length-bucketed batches (masked pads -> identical scores to fixed-8192).
    """
    prefix = f"{args.out_root}/{split}/data-{i:05d}-r{rank:02d}"
    done = f"{prefix}.done"
    fs, done_rpath = fsspec.core.url_to_fs(done)
    if fs.exists(done_rpath):  # WARC-rank fully scored in a prior run
        return None

    def chunk_path(c):
        return f"{prefix}-c{c:04d}.jsonl.gz"

    def chunk_exists(c):
        _fs, rp = fsspec.core.url_to_fs(chunk_path(c))
        return _fs.exists(rp)

    def flush_chunk(cidx, lines):
        # Atomic checkpoint: write a gzip chunk locally then copy to GCS. Each chunk covers
        # CHUNK batches, so a mid-WARC preemption loses at most ~CHUNK batches, not the WARC.
        tmp = f"/app/_warc_{i:05d}_r{rank:02d}_c{cidx:04d}.jsonl.gz"
        with gzip.open(tmp, "wt", encoding="utf-8") as fo:
            fo.writelines(lines)
        with open(tmp, "rb") as src, fsspec.open(chunk_path(cidx), "wb") as dst:
            dst.write(src.read())
        os.remove(tmp)

    pad_id = tokenizer.pad_token_id
    pos_seen = neg_seen = 0
    survivors = []  # (bs_text, final_output, input_ids, ft_prob, is_pos)
    doc_idx = -1
    for tmpl, is_pos in ((USEFUL_TMPL, True), (NOUSE_TMPL, False)):
        for html, final in iter_docs(tmpl.format(base=args.data_base, i=i)):
            doc_idx += 1
            if doc_idx % world != rank:  # this rank owns only its 1/world of docs
                continue
            if is_pos:
                pos_seen += 1
            else:
                neg_seen += 1
            bs = body_strip(html[: args.html_cap])
            ft = fasttext_text(bs)
            ftp = _ft_useful_prob(ftmodel, ft)
            if ftp < args.ft_threshold:
                continue
            ids = tokenizer(ft, truncation=True, max_length=args.max_length)["input_ids"]
            survivors.append((bs, final, ids, ftp, is_pos))
    survivors.sort(key=lambda s: len(s[2]))  # group similar lengths -> small buckets

    n_batches = (len(survivors) + args.batch_size - 1) // args.batch_size
    chunk = args.chunk_batches
    # Resume: count consecutive already-written chunks; restart scoring after them.
    resume_chunks = 0
    while chunk_exists(resume_chunks):
        resume_chunks += 1
    resume_bi = resume_chunks * chunk
    log(
        f"WARC {i:05d} [{split}] read pos={pos_seen} neg={neg_seen} surv={len(survivors)} "
        f"-> {n_batches} batches, resume@chunk {resume_chunks} (batch {resume_bi})"
    )

    buf = []
    kept = 0
    for bi in range(resume_bi, n_batches):
        start = bi * args.batch_size
        batch = survivors[start : start + args.batch_size]
        bucket = bucket_for(max(len(s[2]) for s in batch), args.max_length)
        input_ids, attn = pad_batch([s[2] for s in batch], pad_id, bucket, args.batch_size)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
            logits = bert(input_ids=input_ids, attention_mask=attn).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1]
        xm.mark_step()
        if bi % 25 == 0:
            log(f"WARC {i:05d} [{split}] batch {bi}/{n_batches} bucket={bucket} kept(run)={kept}")
        probs = p.cpu().tolist()[: len(batch)]
        for (bs, final, _ids, ftp, is_pos), pr in zip(batch, probs):
            if pr >= args.bert_threshold:
                row = chat_row(bs, final)
                row["bert_prob"] = pr
                row["ft_prob"] = ftp
                row["source"] = "useful" if is_pos else "no_useful"
                row["warc"] = i
                buf.append(json.dumps(row) + "\n")
                kept += 1
        if (bi + 1) % chunk == 0:  # chunk boundary -> checkpoint
            flush_chunk(bi // chunk, buf)
            buf = []
    if buf:  # final partial chunk
        flush_chunk((n_batches - 1) // chunk, buf)
    with fsspec.open(done, "wt", encoding="utf-8") as d:  # mark WARC-rank complete
        d.write("done\n")
    log(f"WARC {i:05d} [{split}] DONE surv={len(survivors)} kept(this run)={kept}")
    return kept


def _ft_useful_prob(model, text):
    for prob, lab in model.f.predict(text, -1, 0.0, "strict"):
        if lab == LABEL_USEFUL:
            return float(prob)
    return 0.0


# --- Persistent XLA compile cache (survives preemption) -----------------------
# The BERT compile at seq 8192 takes ~15-20min on v4 and is re-paid on every
# preemption (hourly /held/ swarm evals outrank reserved-interactive). We back the
# torch_xla persistent cache with GCS: download before spawn, then rank-0 uploads
# newly-compiled programs every couple minutes. Across a few preemption cycles the
# cache fills and WARCs run without recompiling — preemption-independent progress.
LOCAL_XLA_CACHE = "/tmp/xla_cache"


def _cache_sync_down(gcs_dir: str, local_dir: str) -> int:
    os.makedirs(local_dir, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(gcs_dir)
    if not fs.exists(root):
        return 0
    n = 0
    for f in fs.find(root):
        rel = f[len(root) :].lstrip("/")
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst) or local_dir, exist_ok=True)
        with fs.open(f, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        n += 1
    return n


def _cache_sync_up(local_dir: str, gcs_dir: str) -> int:
    fs, root = fsspec.core.url_to_fs(gcs_dir)
    n = 0
    for dirpath, _, filenames in os.walk(local_dir):
        for fn in filenames:
            lp = os.path.join(dirpath, fn)
            rel = os.path.relpath(lp, local_dir)
            rp = f"{root}/{rel}"
            if fs.exists(rp) and fs.info(rp).get("size") == os.path.getsize(lp):
                continue  # already uploaded (cache files are immutable, keyed by hash)
            with open(lp, "rb") as s, fs.open(rp, "wb") as d:
                d.write(s.read())
            n += 1
    return n


def _cache_uploader(gcs_dir: str, log) -> None:
    while True:
        time.sleep(120)
        try:
            n = _cache_sync_up(LOCAL_XLA_CACHE, gcs_dir)
            if n:
                log(f"xla-cache: uploaded {n} new programs")
        except Exception as e:  # uploader must never kill the run
            log(f"xla-cache upload error: {e}")


def _mp_fn(index):
    import fasttext

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-base", default="gs://marin-us-central2/datasets/high_quality_3000_distill")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ft-model", required=True)
    ap.add_argument("--bert-ckpt", required=True)
    ap.add_argument("--ft-threshold", type=float, default=0.0121)
    ap.add_argument("--bert-threshold", type=float, default=0.01364)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--html-cap", type=int, default=1_000_000)
    ap.add_argument("--train-limit", type=int, default=None, help="Process only the first N (shuffled) train WARCs.")
    ap.add_argument("--val-limit", type=int, default=None, help="Process only the first N val WARCs.")
    ap.add_argument("--test-limit", type=int, default=None, help="Process only the first N test WARCs.")
    ap.add_argument(
        "--chunk-batches", type=int, default=100, help="Checkpoint a sub-part every N batches (preemption granularity)."
    )
    ap.add_argument("--warc-shard", type=int, default=0, help="This job's index in a multi-job WARC fan-out.")
    ap.add_argument("--warc-shards", type=int, default=1, help="Total jobs in the WARC fan-out (disjoint WARC subsets).")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--splits", default="val,test,train", help="Which splits to process this run.")
    ap.add_argument("--xla-cache", default="gs://marin-us-central2/tmp/cascade_xla_cache_v4")
    args = ap.parse_args()

    rank, world = xr.global_ordinal(), xr.world_size()
    is_main = rank == 0

    def log(msg):
        print(f"[casc r{rank}/{world}] {msg}", flush=True)

    # Persistent compile cache must be initialized before any XLA compilation.
    if args.xla_cache:
        xr.initialize_cache(LOCAL_XLA_CACHE, readonly=False)
        if is_main:
            threading.Thread(target=_cache_uploader, args=(args.xla_cache, log), daemon=True).start()

    device = xm.xla_device()

    with fsspec.open(args.manifest, "rt", encoding="utf-8") as f:
        manifest = json.load(f)
    want = args.splits.split(",")
    # Build the full (warc_index, split) worklist in a deterministic order. Every rank
    # processes EVERY WARC; doc-level sharding inside process_warc splits the work.
    work = []
    if "val" in want:
        v = list(manifest["val"])
        if args.val_limit is not None:
            v = v[: args.val_limit]
        work += [(i, "val") for i in v]
    if "test" in want:
        t = list(manifest["test"])
        if args.test_limit is not None:
            t = t[: args.test_limit]
        work += [(i, "test") for i in t]
    if "train" in want:
        tr = list(manifest["train"])
        random.Random(args.shuffle_seed).shuffle(tr)
        if args.train_limit is not None:
            tr = tr[: args.train_limit]
        work += [(i, "train") for i in tr]
    # WARC fan-out: independent jobs take disjoint WARC subsets (avoids fragile multi-host
    # gangs). Within each job, docs are still split across this job's ranks.
    mine = work[args.warc_shard :: args.warc_shards] if args.warc_shards > 1 else work
    log(
        f"loading models; shard {args.warc_shard}/{args.warc_shards}: {len(mine)}/{len(work)} WARCs, doc-sharded across {world} ranks"
    )

    # every rank needs the fastText model locally (CPU). Download per-rank.
    with fsspec.open(args.ft_model, "rb") as src, open(f"/app/_ft_{rank}.bin", "wb") as dst:
        dst.write(src.read())
    ftmodel = fasttext.load_model(f"/app/_ft_{rank}.bin")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bert = load_bert(args.bert_ckpt, device, "sdpa", is_main)
    log("models ready; starting cascade")

    total = 0
    for n, (i, split) in enumerate(mine):
        kept = process_warc(i, split, rank, world, ftmodel, bert, tokenizer, device, args, log)
        if kept is not None:
            total += kept
        if (n + 1) % 5 == 0:
            log(f"progress {n+1}/{len(mine)} WARCs done, cumulative kept(this rank)={total}")
    log(f"DONE rank {rank}: {len(mine)} WARCs, kept(this rank)={total}")


if __name__ == "__main__":
    # Pre-spawn (single process, no race): download the shared persistent XLA cache so
    # every spawned rank starts with the already-compiled programs from prior cycles.
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
