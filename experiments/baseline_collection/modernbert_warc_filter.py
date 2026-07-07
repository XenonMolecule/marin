# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 of the BERT WARC pipeline: score EVERY decoded doc with ModernBERT on TPU,
keep what BERT keeps, tombstone the rest, and record per-WARC timing for the scaling study.

This is the cascade's stage-2 machinery (``cascade_chat_filter.py``) with the fastText
stage-1 gate REMOVED — here BERT scores every document, not just the ~46% fastText
survivors. Input is the clean per-WARC parquet from ``decode_warcs_clean.py``
(``{doc_id, warc_hash, url, snapshot, html, text_body}``); ``text_body`` is already the
exact classifier preprocessing (body_strip + whitespace-collapse + lowercase), so we
just tokenize and score it.

Proven idioms kept verbatim from the cascade:
  * length-bucketed bf16 batching — pad each batch to the smallest bucket >= its longest
    member; masked pads give BIT-IDENTICAL logits to fixed-8192 while keeping short-doc
    batches cheap and XLA compiles ~7 programs total instead of per-shape.
  * rank-0 weight load + ``xm.broadcast_master_param`` — the ONLY cross-rank collective,
    so the run can't deadlock mid-WARC (the multi-host hang lesson).
  * persistent XLA compile cache on GCS — the 8192 compile is ~15-20 min and otherwise
    re-paid on every preemption; rank 0 uploads new programs every 2 min.
  * atomic per-chunk parquet writes + ``.done`` sentinel per (WARC, rank) — resumable,
    loses at most ``--chunk-batches`` batches on preemption.
  * two-level fan-out: independent single-host jobs take disjoint WARC subsets
    (``--warc-shard i --warc-shards N``); within a job, docs are split across ranks by
    ``doc_idx % world == rank`` so one big WARC parallelizes across the whole slice.

Keep rule: a doc is retained iff ``softmax(logits)[:,1] >= --bert-threshold`` (default
0.26 = the general 1M ModernBERT best-F1 point). Kept docs are written as parquet with a
``bert_prob`` column; dropped doc_ids go to a lightweight tombstone jsonl.gz.
"""

import argparse
import gzip
import io
import json
import logging
import os
import threading
import time

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = "answerdotai/ModernBERT-base"

# On-disk scratch for the rank-0 checkpoint download, tombstone staging, and the local
# XLA compile cache. The cascade hardcoded "/app" (the Iris gVisor container path); this
# is configurable via MBWARC_SCRATCH so the same script runs on a real dev TPU VM (where
# /app does not exist). Must be a WRITABLE, exec-mounted dir on the worker.
SCRATCH = os.environ.get("MBWARC_SCRATCH", "/tmp")

# Pad each batch up to the smallest bucket >= its longest member. Masked pads yield
# logits bit-identical to fixed-8192. Batch dim held at --batch-size so XLA compiles once
# per bucket (~7 programs) rather than per shape.
BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)

# Columns carried through from the decoded parquet into the kept output (+ bert_prob).
PASSTHROUGH_COLUMNS = ("doc_id", "url", "snapshot", "html", "text_body")


def bucket_for(length: int, max_length: int) -> int:
    for b in BUCKETS:
        if b >= length:
            return min(b, max_length)
    return max_length


def pad_batch(id_lists, pad_id: int, bucket: int, target_n: int):
    """Pad pre-tokenized id lists to (target_n, bucket). Dummy rows (beyond len(id_lists))
    get one attended token to avoid all-masked NaN; their outputs are discarded."""
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
    """Stream decoded-parquet rows (dicts of PASSTHROUGH_COLUMNS). Missing file -> empty."""
    try:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=list(PASSTHROUGH_COLUMNS), batch_size=1024):
                cols = {c: batch.column(c).to_pylist() for c in PASSTHROUGH_COLUMNS}
                for i in range(len(cols["text_body"])):
                    yield {c: cols[c][i] for c in PASSTHROUGH_COLUMNS}
    except FileNotFoundError:
        return


def load_bert(ckpt, device, attn, is_main):
    """Build ModernBERT + load classifier weights (rank-0 read, broadcast to replicas).

    ``reference_compile=False`` is REQUIRED on torch_xla: ModernBERT defaults it to True,
    which wraps layers in ``torch.compile``; Dynamo then traces with symbolic shapes and
    torch_xla's lazy ``layer_norm`` dies with "Cannot call numel() on tensor with symbolic
    sizes". The LazyTensor backend already fuses/compiles the graph, so the HF compile is
    both redundant and incompatible here.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, attn_implementation=attn, reference_compile=False
    ).to(device)
    if is_main:
        with fsspec.open(f"{ckpt}/latest.pt", "rb") as f:
            data = f.read()
        local = f"{SCRATCH}/_ckpt_load.pt"
        with open(local, "wb") as fh:
            fh.write(data)
        sd = torch.load(local, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(sd)
    xm.broadcast_master_param(model)  # rank-0 weights -> all replicas (the ONLY collective)
    model.eval()
    return model


def _exists(path: str) -> bool:
    fs, rp = fsspec.core.url_to_fs(path)
    return fs.exists(rp)


def _write_parquet(rows: list[dict], path: str) -> None:
    """Atomic parquet write: build locally then copy to GCS (no partial files visible)."""
    table = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)
    with fsspec.open(path, "wb") as dst:
        dst.write(buf.read())


def process_warc(warc_hash, rank, world, bert, tokenizer, device, args, log):
    """Score this rank's 1/world slice of one WARC's decoded docs; write kept parquet +
    tombstone. Returns (n_docs, n_kept, n_tokens, tokenize_s, bert_s, wall_s) for this rank,
    or None if already done."""
    t_wall0 = time.monotonic()
    kept_prefix = f"{args.out_root}/kept/data-{warc_hash}-r{rank:02d}"
    done_path = f"{args.out_root}/done/data-{warc_hash}-r{rank:02d}.done"
    tomb_path = f"{args.out_root}/tombstones/data-{warc_hash}-r{rank:02d}.jsonl.gz"
    timing_path = f"{args.out_root}/timing/data-{warc_hash}-r{rank:02d}.timing.json"
    if _exists(done_path):
        return None

    def chunk_path(c):
        return f"{kept_prefix}-c{c:04d}.parquet"

    in_path = f"{args.decoded_root}/data-{warc_hash}.parquet"
    pad_id = tokenizer.pad_token_id

    # Read this rank's docs and tokenize. text_body is the classifier input already.
    t_tok0 = time.monotonic()
    docs = []  # (row_dict, input_ids)
    n_docs = n_tokens = 0
    doc_idx = -1
    for row in iter_docs(in_path):
        doc_idx += 1
        if doc_idx % world != rank:  # this rank owns only its 1/world of docs
            continue
        text = row["text_body"] or ""
        if not text:
            continue
        n_docs += 1
        ids = tokenizer(text, truncation=True, max_length=args.max_length)["input_ids"]
        n_tokens += len(ids)
        docs.append((row, ids))
    docs.sort(key=lambda d: len(d[1]))  # group similar lengths -> small buckets
    tokenize_s = time.monotonic() - t_tok0

    n_batches = (len(docs) + args.batch_size - 1) // args.batch_size
    chunk = args.chunk_batches
    resume_chunks = 0
    while _exists(chunk_path(resume_chunks)):
        resume_chunks += 1
    resume_bi = resume_chunks * chunk
    log(f"WARC {warc_hash} read docs={n_docs} -> {n_batches} batches, resume@chunk {resume_chunks}")

    kept_buf = []
    tomb_buf = []  # tombstone lines for dropped docs (all docs, not just per-chunk)
    kept = 0
    bert_s = 0.0
    for bi in range(resume_bi, n_batches):
        start = bi * args.batch_size
        batch = docs[start : start + args.batch_size]
        bucket = bucket_for(max(len(d[1]) for d in batch), args.max_length)
        input_ids, attn = pad_batch([d[1] for d in batch], pad_id, bucket, args.batch_size)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        t_b0 = time.monotonic()
        with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
            logits = bert(input_ids=input_ids, attention_mask=attn).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1]
        xm.mark_step()
        probs = p.cpu().tolist()[: len(batch)]
        bert_s += time.monotonic() - t_b0
        if bi % 25 == 0:
            log(f"WARC {warc_hash} batch {bi}/{n_batches} bucket={bucket} kept(run)={kept}")
        for (row, _ids), pr in zip(batch, probs, strict=False):
            if pr >= args.bert_threshold:
                out = {c: row[c] for c in PASSTHROUGH_COLUMNS}
                out["warc_hash"] = warc_hash
                out["bert_prob"] = float(pr)
                kept_buf.append(out)
                kept += 1
            else:
                tomb_buf.append(json.dumps({"doc_id": row["doc_id"], "bert_prob": float(pr)}) + "\n")
        if (bi + 1) % chunk == 0:  # chunk boundary -> checkpoint kept docs
            if kept_buf:
                _write_parquet(kept_buf, chunk_path(bi // chunk))
            kept_buf = []
    if kept_buf:  # final partial chunk
        _write_parquet(kept_buf, chunk_path((n_batches - 1) // chunk))

    # Tombstone (dropped doc_ids) — written once at the end (small, append-free).
    tmp = f"{SCRATCH}/_tomb_{warc_hash}_r{rank:02d}.jsonl.gz"
    with gzip.open(tmp, "wt", encoding="utf-8") as fo:
        fo.writelines(tomb_buf)
    with open(tmp, "rb") as src, fsspec.open(tomb_path, "wb") as dst:
        dst.write(src.read())
    os.remove(tmp)

    wall_s = time.monotonic() - t_wall0
    timing = {
        "warc_hash": warc_hash,
        "rank": rank,
        "world": world,
        "n_docs": n_docs,
        "n_kept": kept,
        "n_tokens": n_tokens,
        "tokenize_s": round(tokenize_s, 2),
        "bert_s": round(bert_s, 2),
        "wall_s": round(wall_s, 2),
        "tpu_type": args.tpu_type,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    with fsspec.open(timing_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(timing) + "\n")
    with fsspec.open(done_path, "wt", encoding="utf-8") as d:
        d.write("done\n")
    log(f"WARC {warc_hash} DONE docs={n_docs} kept={kept} bert_s={bert_s:.1f} wall_s={wall_s:.1f}")
    return n_docs, kept, n_tokens, tokenize_s, bert_s, wall_s


# --- Persistent XLA compile cache (survives preemption) -----------------------
LOCAL_XLA_CACHE = f"{SCRATCH}/xla_cache"


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
                continue  # cache files are immutable (keyed by hash)
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


def _load_manifest(manifest_path: str) -> list[str]:
    with fsspec.open(manifest_path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _warc_path_hash(warc_path: str) -> str:
    import hashlib

    return hashlib.sha256(warc_path.encode()).hexdigest()[:12]


def _mp_fn(index):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="WARC manifest (same one the decode stage used).")
    ap.add_argument("--decoded-root", required=True, help="gs:// dir of decode_warcs_clean.py parquet output.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bert-ckpt", required=True)
    ap.add_argument("--bert-threshold", type=float, default=0.26)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-batches", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N (manifest-order) WARCs.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warc-shard", type=int, default=0)
    ap.add_argument("--warc-shards", type=int, default=1)
    ap.add_argument("--tpu-type", default="unknown", help="For the timing record only.")
    ap.add_argument("--xla-cache", default="gs://marin-us-east5/tmp/ttl=30d/modernbert_warc_xla_cache_v1")
    args = ap.parse_args()

    rank, world = xr.global_ordinal(), xr.world_size()
    is_main = rank == 0

    def log(msg):
        print(f"[mbwarc r{rank}/{world}] {msg}", flush=True)

    if args.xla_cache:
        xr.initialize_cache(LOCAL_XLA_CACHE, readonly=False)
        if is_main:
            threading.Thread(target=_cache_uploader, args=(args.xla_cache, log), daemon=True).start()

    device = xm.xla_device()

    warc_paths = _load_manifest(args.manifest)[args.start :]
    if args.limit is not None:
        warc_paths = warc_paths[: args.limit]
    warc_hashes = [_warc_path_hash(p) for p in warc_paths]
    # WARC fan-out: independent jobs take disjoint WARC subsets (no fragile multi-host gang).
    mine = warc_hashes[args.warc_shard :: args.warc_shards] if args.warc_shards > 1 else warc_hashes
    log(f"shard {args.warc_shard}/{args.warc_shards}: {len(mine)}/{len(warc_hashes)} WARCs, doc-sharded across {world}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bert = load_bert(args.bert_ckpt, device, "sdpa", is_main)
    log("model ready; scoring all docs (no fastText gate)")

    tot_docs = tot_kept = 0
    for n, wh in enumerate(mine):
        r = process_warc(wh, rank, world, bert, tokenizer, device, args, log)
        if r is not None:
            tot_docs += r[0]
            tot_kept += r[1]
        if (n + 1) % 5 == 0:
            log(f"progress {n + 1}/{len(mine)} WARCs; cumulative docs={tot_docs} kept={tot_kept}")
    log(f"DONE rank {rank}: {len(mine)} WARCs, docs={tot_docs} kept={tot_kept}")


if __name__ == "__main__":
    # Pre-spawn (single process, no race): download the shared XLA cache so every rank
    # starts with programs compiled in prior cycles.
    _pre = argparse.ArgumentParser()
    _pre.add_argument("--xla-cache", default="gs://marin-us-east5/tmp/ttl=30d/modernbert_warc_xla_cache_v1")
    _xc = _pre.parse_known_args()[0].xla_cache
    if _xc:
        try:
            n = _cache_sync_down(_xc, LOCAL_XLA_CACHE)
            print(f"xla-cache: downloaded {n} cached programs from {_xc}", flush=True)
        except Exception as e:
            print(f"xla-cache download error: {e}", flush=True)
    xmp.spawn(_mp_fn)
