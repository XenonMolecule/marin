# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Timing benchmark for the Levanter/JAX ModernBERT classifier forward pass (inference).

Sweeps the full grid {base, large} x {1024, 2048, 4096, 8192} ctx x {requested backends} in a
single TPU job, so two jobs (one v6e, one v5litepod/v5e) produce the entire table. Forward-pass
timing depends only on architecture, shapes, and dtype — NOT on weight values — so the model is
randomly initialized (mirrors ``modernbert_splash_check.py``). That means no checkpoint load, no
cross-region read, no tokenizer/HF download, and no dataset: the job is fully self-contained.

Inference setup mirrors the real scoring path (``score_modernbert_useful.py``): bf16 compute,
batch sharded over a 1-D ``data`` mesh across all chips, weights replicated, a bidirectional
segment mask. We time only the steady state (after warmup compiles the graph) and report
ms/forward, docs/sec, and per-chip docs/sec.

VANILLA materializes the O(seq^2) score matrix and OOMs at long ctx / large batch; SPLASH (TPU
Pallas) makes 8192 feasible. Each cell is timed independently and an OOM/compile failure is
recorded (not fatal) so one job still fills the rest of the grid.

Launch (standalone TPU Iris job — see the commands at the bottom of this file)."""

from __future__ import annotations

import argparse
import json
import logging
import time

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis
from jax.sharding import Mesh
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.modernbert import ModernBertConfig, ModernBertForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("modernbert_bench")

VOCAB_SIZE = 50368  # ModernBERT tokenizer (shared by base + large)
PAD_TOKEN_ID = 50283
MODEL_ID = "answerdotai/ModernBERT-base"

# Dims verified against the HF configs (large = 395M). Mirrors launch_modernbert_levanter.MODEL_PRESETS.
MODEL_PRESETS = {
    "base": dict(
        reference_checkpoint="answerdotai/ModernBERT-base",
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    ),
    "large": dict(
        reference_checkpoint="answerdotai/ModernBERT-large",
        hidden_dim=1024,
        intermediate_dim=2624,
        num_layers=28,
        num_heads=16,
    ),
}


def build_model(size: str, ctx: int, backend: AttentionBackend, *, key):
    preset = MODEL_PRESETS[size]
    config = ModernBertConfig(
        max_seq_len=ctx,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
        attn_backend=backend,
        tokenizer=MODEL_ID,
        reference_checkpoint=preset["reference_checkpoint"],
        hidden_dim=preset["hidden_dim"],
        intermediate_dim=preset["intermediate_dim"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
    )
    Vocab = Axis("vocab", VOCAB_SIZE)
    model = ModernBertForSequenceClassification.init(Vocab, config, key=key)
    return config, Vocab, model


def make_input(Vocab: Axis, Batch: Axis, Pos: Axis, *, key):
    """All-real (no pad) random tokens + bidirectional segment mask — full-context worst case."""
    ids = np.asarray(hax.random.randint(key, (Batch, Pos), 1, Vocab.size).array).astype(np.int32)
    seg = np.zeros((Batch.size, Pos.size), dtype=np.int32)  # all real (segment 0)
    tokens = hax.named(ids, (Batch, Pos))
    seg_named = hax.named(seg, (Batch, Pos))
    mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
    return tokens, mask


def bench_cell(
    size: str,
    ctx: int,
    backend: AttentionBackend,
    batch_per_device: int,
    num_devices: int,
    mesh: Mesh,
    axis_mapping: dict,
    policy: jmp.Policy,
    *,
    warmup: int,
    iters: int,
) -> dict:
    total_batch = batch_per_device * num_devices
    config, Vocab, model = build_model(size, ctx, backend, key=jrandom.PRNGKey(0))
    model = policy.cast_to_compute(model)  # bf16 compute, matches the scoring path
    Batch = Axis("batch", total_batch)
    Pos = config.max_Pos
    tokens, mask = make_input(Vocab, Batch, Pos, key=jrandom.PRNGKey(1))

    fwd = hax.named_jit(lambda m, t, msk: m(t, msk).astype(jnp.float32), axis_resources=axis_mapping)

    with mesh:
        for _ in range(warmup):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        t0 = time.perf_counter()
        for _ in range(iters):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        ms = (time.perf_counter() - t0) / iters * 1e3

    docs_per_sec = total_batch / (ms / 1e3)
    return {
        "size": size,
        "ctx": ctx,
        "backend": backend.value,
        "batch_per_device": batch_per_device,
        "total_batch": total_batch,
        "ms_per_forward": round(ms, 2),
        "docs_per_sec": round(docs_per_sec, 1),
        "docs_per_sec_per_chip": round(docs_per_sec / num_devices, 2),
    }


def run(args):
    devices = jax.devices()
    num_devices = len(devices)
    device_kind = devices[0].device_kind
    logger.info("devices: %d x %s", num_devices, device_kind)

    mesh = Mesh(np.array(devices), ("data",))
    axis_mapping = {"batch": "data"}
    policy = jmp.get_policy("p=f32,c=bfloat16")

    sizes = args.sizes.split(",")
    ctxs = [int(c) for c in args.ctxs.split(",")]
    backends = [AttentionBackend(b) for b in args.backends.split(",")]

    results = []
    for size in sizes:
        for ctx in ctxs:
            for backend in backends:
                tag = f"{size}/ctx{ctx}/{backend.value}/bpd{args.batch_per_device}"
                try:
                    cell = bench_cell(
                        size,
                        ctx,
                        backend,
                        args.batch_per_device,
                        num_devices,
                        mesh,
                        axis_mapping,
                        policy,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    logger.info(
                        "[ok] %s: %.1f ms/fwd  %.1f docs/s  %.2f docs/s/chip (batch=%d)",
                        tag,
                        cell["ms_per_forward"],
                        cell["docs_per_sec"],
                        cell["docs_per_sec_per_chip"],
                        cell["total_batch"],
                    )
                except Exception as e:
                    msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
                    logger.warning("[FAIL] %s: %s", tag, msg)
                    cell = {
                        "size": size,
                        "ctx": ctx,
                        "backend": backend.value,
                        "batch_per_device": args.batch_per_device,
                        "total_batch": args.batch_per_device * num_devices,
                        "error": msg,
                    }
                results.append(cell)

    summary = {
        "device_kind": device_kind,
        "num_devices": num_devices,
        "batch_per_device": args.batch_per_device,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    # Human-readable table to stdout (iris job logs is the primary readout).
    logger.info("===== MODERNBERT_INFERENCE_BENCHMARK =====")
    logger.info("device=%s  chips=%d  batch_per_device=%d", device_kind, num_devices, args.batch_per_device)
    logger.info("%-6s %-6s %-8s %12s %12s %14s", "size", "ctx", "backend", "ms/fwd", "docs/s", "docs/s/chip")
    for r in results:
        if "error" in r:
            logger.info("%-6s %-6s %-8s %12s %12s %14s", r["size"], r["ctx"], r["backend"], "OOM/ERR", "-", "-")
        else:
            logger.info(
                "%-6s %-6s %-8s %12.1f %12.1f %14.2f",
                r["size"],
                r["ctx"],
                r["backend"],
                r["ms_per_forward"],
                r["docs_per_sec"],
                r["docs_per_sec_per_chip"],
            )
    logger.info("BENCH_JSON %s", json.dumps(summary))

    if args.out:
        with fsspec.open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("wrote results -> %s", args.out)


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="base,large", help="comma-separated subset of {base,large}")
    p.add_argument("--ctxs", default="1024,2048,4096,8192", help="comma-separated context lengths")
    p.add_argument(
        "--backends",
        default="splash",
        help="comma-separated attention backends to time per cell. splash is the deployable "
        "long-context path (works at all ctx); pass 'vanilla,splash' to also see the short-ctx "
        "dense kernel (vanilla OOMs at long ctx).",
    )
    p.add_argument(
        "--batch-per-device",
        type=int,
        default=8,
        help="per-chip batch; total batch = this * num_chips, sharded over the data mesh.",
    )
    p.add_argument("--warmup", type=int, default=3, help="warmup forwards (compile + steady-state warmup).")
    p.add_argument("--iters", type=int, default=10, help="timed forwards per cell.")
    p.add_argument("--out", default="", help="optional GCS path for the results JSON (write in-region).")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
