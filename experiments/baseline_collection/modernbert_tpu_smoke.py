# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: train a ModernBERT sequence classifier on TPU (PyTorch/XLA) over the
same useful-vs-[NO_USEFUL_CONTENT] body_strip data the fastText classifier used.

ModernBERT has no JAX port and its reference path needs FlashAttention (CUDA-only),
so on TPU we load with attn_implementation="sdpa" and pad to a fixed length (static
shapes keep XLA from recompiling every step). This is intentionally single-device and
small — the goal is a real F1 number proving the encoder stage trains on TPU.

Input files are fastText-format gzipped text: "__label__useful <text>" per line.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import fsspec
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("modernbert_smoke")

LABEL_USEFUL = "__label__useful"
MODEL_ID = "answerdotai/ModernBERT-base"


def read_fasttext(path: str, max_rows: int) -> tuple[list[str], list[int]]:
    """Parse a gzipped fastText file into (texts, labels) with label 1 = useful."""
    texts, labels = [], []
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            if not text:
                continue
            labels.append(1 if label == LABEL_USEFUL else 0)
            texts.append(text)
            if len(texts) >= max_rows:
                break
    return texts, labels


def read_test(path: str, max_rows: int) -> tuple[list[str], list[int]]:
    """Read a test set. If ``path`` is a glob (35 frozen test shards = 35 snapshots),
    sample evenly from each shard so the eval is stratified across snapshots — the
    representative stand-in for the full 1.7M-doc test that's too slow to score whole."""
    if "*" not in path:
        return read_fasttext(path, max_rows)
    fs = fsspec.filesystem("gcs")
    shards = sorted("gs://" + p for p in fs.glob(path))
    per = max(1, max_rows // len(shards))
    rng = random.Random(0)
    texts, labels = [], []
    for shard in shards:
        # Read the whole shard then random-sample — shards are class-ordered (useful first),
        # so first-N would grab all-useful. Random within shard preserves the natural ratio.
        st, sl = read_fasttext(shard, 10**9)
        order = list(range(len(st)))
        rng.shuffle(order)
        for j in order[:per]:
            texts.append(st[j])
            labels.append(sl[j])
    return texts, labels


def read_fasttext_sharded(
    path: str, max_rows: int, rank: int, world: int, balance: bool = False
) -> tuple[list[str], list[int]]:
    """Each data-parallel rank reads a disjoint 1/world stride of the data. Default: the first
    ~max_rows rows (preserves the file's natural ~12:1 ratio). balance=True: equal positives and
    negatives (max_rows/2 each), striding within each class — a 1:1 training set."""
    if balance:
        per_class = max(1, (max_rows // 2) // world)
        pos, neg, pi, ni = [], [], 0, 0
        with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
            for line in f:
                label, _, text = line.rstrip("\n").partition(" ")
                if not text:
                    continue
                if label == LABEL_USEFUL:
                    if pi % world == rank and len(pos) < per_class:
                        pos.append(text)
                    pi += 1
                else:
                    if ni % world == rank and len(neg) < per_class:
                        neg.append(text)
                    ni += 1
                if len(pos) >= per_class and len(neg) >= per_class:
                    break
        return pos + neg, [1] * len(pos) + [0] * len(neg)
    target = max(1, max_rows // world)
    texts, labels, gi = [], [], 0
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            if not text:
                continue
            if gi % world == rank:
                labels.append(1 if label == LABEL_USEFUL else 0)
                texts.append(text)
                if len(texts) >= target:
                    break
            gi += 1
    return texts, labels


def read_presharded(prefix: str, rank: int) -> tuple[list[str], list[int]]:
    """Rank r reads its own pre-materialized shard ``{prefix}_{rank:02d}.txt.gz`` in full.

    At 1M+ rows the striding reader has every rank re-stream the whole train prefix
    (~5 GB ×world, and again on every preemption/resume) — the scaling bottleneck.
    Pre-sharding (see preshard_train.py) round-robins the same rows into per-rank
    files up front, so each rank reads only its ~1/world slice and resume is cheap.
    """
    path = f"{prefix}_{rank:02d}.txt.gz"
    texts, labels = [], []
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            if not text:
                continue
            labels.append(1 if label == LABEL_USEFUL else 0)
            texts.append(text)
    return texts, labels


def encode(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    # Fixed-length padding -> static shapes -> XLA compiles once.
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def f1_sweep(probs: list[float], truth: list[int]) -> tuple[float, float]:
    """Best F1 over a threshold sweep on P(useful); returns (best_f1, threshold)."""
    best_f1, best_t = 0.0, 0.5
    for t in [i / 50 for i in range(1, 50)]:
        tp = sum(1 for p, y in zip(probs, truth) if p >= t and y == 1)
        fp = sum(1 for p, y in zip(probs, truth) if p >= t and y == 0)
        fn = sum(1 for p, y in zip(probs, truth) if p < t and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def pr_sweep(probs: list[float], truth: list[int]) -> list[dict]:
    """Full precision/recall/F1 over a dense threshold grid — saved so any operating
    curve (e.g. excluded-vs-retained vs fastText) recomputes offline."""
    fine = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    grid = sorted(set(fine + [i / 50 for i in range(51)]))
    rows = []
    for t in grid:
        tp = sum(1 for p, y in zip(probs, truth) if p >= t and y == 1)
        fp = sum(1 for p, y in zip(probs, truth) if p >= t and y == 0)
        fn = sum(1 for p, y in zip(probs, truth) if p < t and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": round(t, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    return rows


CKPT_EVERY_OPT_STEPS = (
    5  # every 5 opt-steps — banks fast enough for ~20min preempt windows, less GCS-write hang exposure than 2
)


def save_checkpoint(ckpt_dir: str, model, scheduler, epoch: int, micro_done: int, is_main: bool) -> None:
    """Save model + LR schedule + position to GCS so a preempted run resumes. Optimizer state is
    NOT saved (fresh AdamW on resume) — losing Adam momentum is a minor cost vs the complexity of
    round-tripping XLA optimizer tensors. xm.save has a rendezvous so ALL ranks must call it."""
    # Rank-0-ONLY save (no cross-rank rendezvous). DP replicas are identical after each all-reduce, so
    # rank 0's weights are authoritative; the other ranks return immediately and re-sync at the next
    # optimizer all-reduce. This removes the rendezvous deadlock where a slow/stuck GCS write on rank 0
    # hangs the whole gang — the failure mode that froze checkpoints (e.g. on v6e).
    if not is_main:
        return
    print(f"[mb r0] [ckptsave] state_dict -> CPU (micro {micro_done})...", flush=True)
    cpu_model = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    state = {"model": cpu_model, "sched": scheduler.state_dict(), "epoch": epoch, "micro_done": micro_done}
    print("[mb r0] [ckptsave] writing to GCS...", flush=True)
    with fsspec.open(f"{ckpt_dir}/latest.pt", "wb") as f:
        torch.save(state, f)
    print("[mb r0] [ckptsave] DONE", flush=True)


def load_checkpoint(ckpt_dir: str):
    """Return the saved state dict, or None if no checkpoint exists yet."""
    try:
        with fsspec.open(f"{ckpt_dir}/latest.pt", "rb") as f:
            data = f.read()
    except (FileNotFoundError, OSError):
        return None
    local = "/app/_ckpt_load.pt"
    with open(local, "wb") as f:
        f.write(data)
    return torch.load(local, map_location="cpu", weights_only=False)


def _mp_fn(index) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train-rows", type=int, default=6000)
    ap.add_argument("--test-rows", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"], help="XLA-safe attention impl.")
    ap.add_argument("--warmup-steps", type=int, default=60, help="Linear LR warmup — prevents early NaN.")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (ModernBERT's native precision; XLA-friendly).")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing — needed for long context.")
    ap.add_argument("--grad-accum", type=int, default=1, help="Accumulate grads over N micro-batches (effective batch).")
    ap.add_argument("--out", default="", help="GCS path to save a durable result JSON (config + F1 + sweep + preds).")
    ap.add_argument("--balance", action="store_true", help="Train on a 1:1 balanced set (vs the natural ~12:1).")
    ap.add_argument("--ckpt", default="", help="GCS dir for checkpoint/resume (survives preemption).")
    ap.add_argument(
        "--train-presharded",
        default="",
        help="Prefix of per-rank pre-sharded train files ({prefix}_NN.txt.gz). Rank r reads only its "
        "shard — avoids the 1M-scale full-prefix re-read. Must be pre-sharded for exactly `world` ranks.",
    )
    ap.add_argument(
        "--resume-skip-micro",
        type=int,
        default=0,
        help="On resume, fast-forward the data iterator this many extra micro-batches past the checkpoint "
        "position. Steps over a poison batch that deterministically hangs the forward pass (weights stay "
        "at the checkpoint; only the skipped examples go untrained).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training entirely; load --ckpt and run only the held-out eval + write --out. "
        "Used to recover the eval when the final training steps flakily hang on TPU.",
    )
    args = ap.parse_args()
    rank, world = xr.global_ordinal(), xr.world_size()
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(f"[mb r0/{world}] {msg}", flush=True)

    device = xm.xla_device()
    log(f"world_size={world} device={device} attn={args.attn} bf16={args.bf16}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    log(f"loading {MODEL_ID} (attn_implementation={args.attn}) ...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, attn_implementation=args.attn).to(
        device
    )
    if args.grad_checkpoint:
        # use_reentrant=False: the reentrant checkpointer mishandles autocast state on XLA
        # ("module 'torch' has no attribute 'xla'"); non-reentrant is the modern, XLA-safe path.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("gradient checkpointing enabled (use_reentrant=False)")

    # Resume: load checkpoint MODEL weights BEFORE the broadcast, so the resume path is byte-identical
    # to a fresh start (weights -> single broadcast -> first forward). Loading AFTER the broadcast (the
    # old order) left a tangled lazy graph that hung the world16 forward first-compile. Scheduler state
    # is restored later, once the scheduler is constructed.
    # RANK-0-ONLY checkpoint read: 4 ranks reading the ~600MB file simultaneously deadlocks (same
    # multi-rank contention that hung the save). Rank 0 loads the weights; broadcast_master_param
    # distributes them; the resume position is all-reduced to every rank.
    resume_ck = load_checkpoint(args.ckpt) if (args.ckpt and is_main) else None
    start_epoch, start_micro = 0, 0
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        start_epoch, start_micro = resume_ck["epoch"], resume_ck["micro_done"]
        log(f"RESUMED model weights (rank 0): epoch={start_epoch} micro_done={start_micro}")
    xm.broadcast_master_param(model)  # rank 0's loaded-or-fresh weights -> all replicas
    # All-reduce the resume position from rank 0 (other ranks contribute 0) so every rank fast-forwards equally.
    pos = torch.tensor([start_epoch, start_micro], dtype=torch.int64, device=device)
    pos = xm.all_reduce(xm.REDUCE_SUM, pos)
    xm.mark_step()
    start_epoch, start_micro = int(pos[0].item()), int(pos[1].item())

    if args.train_presharded:
        tr_text, tr_y = read_presharded(args.train_presharded, rank)
    else:
        tr_text, tr_y = read_fasttext_sharded(args.train, args.train_rows, rank, world, balance=args.balance)
    te_text, te_y = read_test(args.test, args.test_rows) if is_main else ([], [])
    log(
        f"per-rank train={len(tr_y)} (useful={sum(tr_y)}); global~{len(tr_y) * world}; test={len(te_y)} (useful={sum(te_y)})"
    )

    # Tokenize per-batch (streaming), NOT all upfront — pre-tokenizing 100k+ docs at long
    # context OOMs host RAM. padding="max_length" keeps batch shapes static so XLA still
    # compiles once. Raw text lists stay in RAM (fine to ~300-500k; GCS streaming needed beyond).
    tr_labels = torch.tensor(tr_y)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n = len(tr_y)
    accum = max(1, args.grad_accum)  # effective batch = batch_size * accum; gives the rare class signal at 12:1
    micro_per_epoch = (n + args.batch_size - 1) // args.batch_size
    opt_steps_total = max(1, args.epochs * (micro_per_epoch // accum))
    scheduler = get_linear_schedule_with_warmup(opt, args.warmup_steps, opt_steps_total)

    # Restore the LR-schedule position on ALL ranks deterministically: the schedule is a pure function of
    # opt-step count, so re-stepping matches the saved state without broadcasting scheduler internals — and
    # keeps lr identical across ranks (each rank applies its own optimizer step, so divergent lr would split
    # the replicas). resume_ck (rank 0 only) is no longer needed for this.
    for _ in range(start_micro // accum):
        scheduler.step()
    model.train()
    log(
        f"training — lr={args.lr} eff_batch={args.batch_size * accum * world} "
        f"(bs={args.batch_size}*accum={accum}*world={world}) warmup={args.warmup_steps}/{opt_steps_total}; FIRST compiles ..."
    )
    opt_step_count = 0
    did_first = False  # granular logging on the first COMPUTED step (pinpoints world16 resume hang)
    for epoch in ([] if args.eval_only else range(start_epoch, args.epochs)):
        t0 = time.time()
        torch.manual_seed(1000 + epoch)  # deterministic per-epoch shuffle so resume is reproducible
        perm = torch.randperm(n)
        total_loss, steps = 0.0, 0
        opt.zero_grad()
        for i in range(0, n, args.batch_size):
            # Fast-forward past already-completed micro-batches when resuming mid-epoch. The extra
            # resume_skip_micro steps over a deterministic poison batch that hangs the forward pass.
            if epoch == start_epoch and steps < start_micro + args.resume_skip_micro:
                steps += 1
                continue
            dbg = not did_first
            if dbg:
                log(f"[firststep] fast-forward done at steps={steps}; encoding...")
            idx = perm[i : i + args.batch_size]
            enc = encode(tokenizer, [tr_text[j] for j in idx.tolist()], args.max_length)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            labels = tr_labels[idx].to(device)
            if dbg:
                log("[firststep] encoded + moved to device; forward (FIRST COMPILE)...")
            with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
            if dbg:
                log("[firststep] forward done; backward...")
            (out.loss / accum).backward()  # scale so accumulated grad = mean over the effective batch
            if dbg:
                log("[firststep] backward done; mark_step...")
            xm.mark_step()
            if dbg:
                log("[firststep] mark_step done; loss.item() (device sync)...")
            lv = out.loss.item()
            if dbg:
                log("[firststep] DONE — first computed step fully executed")
                did_first = True
            total_loss += lv
            steps += 1
            if steps % accum == 0:  # one optimizer step per `accum` micro-batches
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                xm.optimizer_step(opt)
                scheduler.step()
                xm.mark_step()
                opt.zero_grad()
                opt_step_count += 1
                if args.ckpt and opt_step_count % CKPT_EVERY_OPT_STEPS == 0:
                    save_checkpoint(args.ckpt, model, scheduler, epoch, steps, is_main)
                    log(f"  checkpoint @ epoch {epoch + 1} micro {steps} (opt-step {opt_step_count})")
            if steps == 1 and epoch == 0:
                log(f"FIRST STEP done (XLA compiled) in {time.time() - t0:.1f}s | loss={lv:.4f}")
            if steps % 25 == 0:
                # running mean — per-micro-step loss is single-example noise at batch_size=1; the mean is the signal
                rl = total_loss / steps
                log(
                    f"  epoch {epoch + 1} step {steps}: running_loss={rl:.4f} (last={lv:.4f})"
                    f"{'  <<< NaN/Inf!' if rl != rl or rl == float('inf') else ''}"
                )
        log(
            f"epoch {epoch + 1}: mean_loss={total_loss / max(steps, 1):.4f} ({time.time() - t0:.1f}s, {steps} micro-steps)"
        )
        if args.ckpt:  # checkpoint at epoch end -> resume starts the next epoch fresh
            save_checkpoint(args.ckpt, model, scheduler, epoch + 1, 0, is_main)

    xm.rendezvous("train_done")  # all replicas finish training before rank 0 evaluates

    # Eval on rank 0 only (no collectives) -> P(useful) -> threshold sweep -> best F1.
    if is_main:
        model.eval()
        probs: list[float] = []
        with torch.no_grad():
            for i in range(0, len(te_y), args.batch_size):
                enc = encode(tokenizer, te_text[i : i + args.batch_size], args.max_length)
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=args.bf16):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                p = torch.softmax(logits.float(), dim=-1)[:, 1]
                xm.mark_step()
                probs.extend(p.cpu().tolist())
        best_f1, best_t = f1_sweep(probs, te_y)
        acc = sum(1 for p, y in zip(probs, te_y) if (p >= 0.5) == bool(y)) / len(te_y)
        log(f"P(useful): min={min(probs):.4f} mean={sum(probs) / len(probs):.4f} max={max(probs):.4f}")
        log(f"test label balance: useful={sum(te_y)} no_useful={len(te_y) - sum(te_y)}")
        print(
            f"MODERNBERT_TPU_SMOKE best_f1={best_f1:.4f} threshold={best_t:.2f} acc={acc:.4f} n={len(te_y)}", flush=True
        )
        if args.out:
            result = {
                "config": {
                    k: getattr(args, k)
                    for k in ("max_length", "train_rows", "batch_size", "grad_accum", "epochs", "lr", "attn")
                },
                "world_size": world,
                "best_f1": best_f1,
                "best_threshold": best_t,
                "acc": acc,
                "n_test": len(te_y),
                "n_useful": sum(te_y),
                "sweep": pr_sweep(probs, te_y),
                "preds": [[round(p, 5), y] for p, y in zip(probs, te_y)],
            }
            with fsspec.open(args.out, "w") as f:
                json.dump(result, f)
            log(f"saved result -> {args.out}")
    xm.rendezvous("eval_done")  # keep replicas alive until rank 0's eval completes


if __name__ == "__main__":
    xmp.spawn(_mp_fn)
