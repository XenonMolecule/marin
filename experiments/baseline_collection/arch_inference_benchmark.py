# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-architecture TPU inference benchmark for the classifier architecture sweep.

Compares candidate classifier architectures' forward-pass throughput against ModernBERT-base
(the gate: any arch not strictly faster than ``mb-base`` dies). All models are Levanter/JAX and
forward-pass speed depends only on architecture, shapes, and dtype — NOT weight values — so every
model is randomly initialized (mirrors ``modernbert_inference_benchmark.py``): no checkpoint load,
no network, fully self-contained.

Methodology (copied from ``modernbert_inference_benchmark.py``): bf16 compute (``p=f32,c=bfloat16``),
batch sharded over a 1-D ``data`` mesh across all chips, weights replicated, bidirectional
all-real segment mask, warmup then timed iters with ``jax.block_until_ready``. Each (arch, ctx)
cell is timed independently and an OOM/compile failure is recorded (not fatal) so one bad cell
doesn't kill the grid.

Per arch we report params, ms/forward at each sweep ctx, seqs/s/chip, and the headline
``effective docs/s/chip at deployment`` = seqs_per_sec_per_chip(deployment_ctx) / windows_per_doc,
which normalizes full-doc archs (8192 ctx, 1 window/doc) against chunked archs (e.g. a 512-ctx
BERT scoring begin/middle/end = 3 windows/doc). The final table is sorted by effective docs/s/chip
with a ``vs mb-base`` ratio and a PASS/FAIL gate column.

CPU smoke (run locally before launching)::

    .venv/bin/python -m experiments.baseline_collection.arch_inference_benchmark --smoke

Launch (standalone TPU Iris job, v6e-4 in us-east5)::

    iris job run --tpu v6e-4 --region us-east5 --extra tpu --enable-extra-resources \\
        --memory 64GB --priority interactive --no-wait \\
        -- python -m experiments.baseline_collection.arch_inference_benchmark \\
        --out gs://marin-us-east5/benchmarks/arch_inference/v6e-4.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis, NamedArray
from haliax.partitioning import set_mesh
from jax.sharding import Mesh
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.modernbert import ModernBertConfig, ModernBertForSequenceClassification
from levanter.utils.jax_utils import parameter_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("arch_bench")

VOCAB_SIZE = 50368  # ModernBERT tokenizer (shared by all ModernBERT-family archs incl. Ettin)
PAD_TOKEN_ID = 50283
MODEL_ID = "answerdotai/ModernBERT-base"
GATE_ARCH = "mb-base"

# SPLASH (TPU Pallas flash) is the deployable long-context path; VANILLA materializes the
# O(seq^2) score matrix and OOMs at long ctx. Rule copied from the scoring path: SPLASH at
# ctx >= 8192, VANILLA below (VANILLA also works on CPU, which SPLASH does not).
SPLASH_MIN_CTX = 8192

DEFAULT_SWEEP_CTXS = (1024, 2048, 4096, 8192)

MakeInput = Callable[[Axis, Axis, jax.Array], tuple[NamedArray, AttentionMask]]


@dataclass(frozen=True)
class ArchSpec:
    """One candidate architecture: how to build it, feed it, and deploy it.

    Attributes:
        build: ``(ctx, key) -> model`` — random init at the given context length, with the
            correct attention backend for that ctx (N/A for non-attention archs).
        make_input: ``(Batch, Pos, key) -> (tokens, attn_mask)`` for one forward pass.
        deployment: ``(ctx, windows_per_doc)`` — the config this arch would score real docs
            with. Full-doc archs: ``(8192, 1.0)``; chunked archs (e.g. a 512-ctx BERT scoring
            begin/middle/end): ``(512, 3.0)``.
        sweep_ctxs: informational ctx grid to time (deployment ctx is always included).
    """

    build: Callable[[int, jax.Array], object]
    make_input: MakeInput
    deployment: tuple[int, float]
    sweep_ctxs: tuple[int, ...] = DEFAULT_SWEEP_CTXS


def backend_for_ctx(ctx: int) -> AttentionBackend:
    return AttentionBackend.SPLASH if ctx >= SPLASH_MIN_CTX else AttentionBackend.VANILLA


def modernbert_build(**dims) -> Callable[[int, jax.Array], ModernBertForSequenceClassification]:
    """Builder for any ModernBERT-family arch; ``dims`` are ModernBertConfig overrides."""

    def build(ctx: int, key: jax.Array) -> ModernBertForSequenceClassification:
        config = ModernBertConfig(
            max_seq_len=ctx,
            num_labels=2,
            pad_token_id=PAD_TOKEN_ID,
            attn_backend=backend_for_ctx(ctx),
            tokenizer=MODEL_ID,
            **dims,
        )
        return ModernBertForSequenceClassification.init(Axis("vocab", VOCAB_SIZE), config, key=key)

    return build


def token_input(vocab_size: int) -> MakeInput:
    """All-real (no pad) random tokens + bidirectional segment mask — full-context worst case."""

    def make_input(Batch: Axis, Pos: Axis, key: jax.Array) -> tuple[NamedArray, AttentionMask]:
        ids = np.asarray(hax.random.randint(key, (Batch, Pos), 1, vocab_size).array).astype(np.int32)
        seg = hax.named(np.zeros((Batch.size, Pos.size), dtype=np.int32), (Batch, Pos))  # all real
        tokens = hax.named(ids, (Batch, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg, seg)
        return tokens, mask

    return make_input


# Adding a new arch = ONE dict entry. Dims mirror launch_modernbert_levanter.MODEL_PRESETS
# (verified against the HF configs there). Ettin rope thetas / mean pooling are kept for
# fidelity but don't materially affect speed — dims do.
ARCHS: dict[str, ArchSpec] = {
    "mb-base": ArchSpec(  # THE GATE (h768 L22, 149M)
        build=modernbert_build(hidden_dim=768, intermediate_dim=1152, num_layers=22, num_heads=12),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "ettin17": ArchSpec(  # h256 L7
        build=modernbert_build(
            hidden_dim=256,
            intermediate_dim=384,
            num_layers=7,
            num_heads=4,
            local_rope_theta=160000.0,
            classifier_pooling="mean",
        ),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "ettin32": ArchSpec(  # h384 L10
        build=modernbert_build(
            hidden_dim=384,
            intermediate_dim=576,
            num_layers=10,
            num_heads=6,
            local_rope_theta=160000.0,
            classifier_pooling="mean",
        ),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "ettin68": ArchSpec(  # h512 L19
        build=modernbert_build(
            hidden_dim=512,
            intermediate_dim=768,
            num_layers=19,
            num_heads=8,
            local_rope_theta=160000.0,
            classifier_pooling="mean",
        ),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "pruned8": ArchSpec(  # layer-pruned ModernBERT-base, h768 L8
        build=modernbert_build(hidden_dim=768, intermediate_dim=1152, num_layers=8, num_heads=12),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "tiny4": ArchSpec(  # from-scratch tiny, h384 L4
        build=modernbert_build(hidden_dim=384, intermediate_dim=576, num_layers=4, num_heads=6),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
    "tiny2": ArchSpec(  # from-scratch tiny, h256 L2
        build=modernbert_build(hidden_dim=256, intermediate_dim=384, num_layers=2, num_heads=4),
        make_input=token_input(VOCAB_SIZE),
        deployment=(8192, 1.0),
    ),
}


def _funnelbert_build(ctx: int, key: jax.Array):
    from levanter.models.funnelbert import FunnelBertConfig, FunnelBertForSequenceClassification

    config = FunnelBertConfig(
        max_seq_len=ctx,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
        attn_backend=backend_for_ctx(ctx),
        tokenizer=MODEL_ID,
    )
    return FunnelBertForSequenceClassification.init(Axis("vocab", VOCAB_SIZE), config, key=key)


ARCHS["funnelbert"] = ArchSpec(  # 4 full ModernBERT layers + 8x pool + 4 global layers, 79M
    build=_funnelbert_build,
    make_input=token_input(VOCAB_SIZE),
    deployment=(8192, 1.0),
)


def _pooled_transformer_build(ctx: int, key: jax.Array):
    from levanter.models.pooled_transformer import PooledTransformerClassifier, PooledTransformerConfig

    config = PooledTransformerConfig(max_seq_len=ctx)
    return PooledTransformerClassifier.init(Axis("vocab", VOCAB_SIZE), config, key=key)


# Deployment (8192, 1.0): pooled ended up TRAINING and scoring full-doc at 8192 (one forward per
# doc), so charge it one window — the earlier (4096, 2.0) entry assumed a chunked deployment that
# was dropped when pooled moved to the 8192 TreeCache.
ARCHS["pooled_transformer"] = ArchSpec(  # 64x window pooling + 4 super-token layers, 26M
    build=_pooled_transformer_build,
    make_input=token_input(VOCAB_SIZE),
    deployment=(8192, 1.0),
)


def _pooled_big_build(ctx: int, key: jax.Array):
    from levanter.models.pooled_transformer import PooledTransformerClassifier, PooledTransformerConfig

    config = PooledTransformerConfig(max_seq_len=ctx, embed_dim=384, hidden_dim=768, num_layers=6, num_heads=12)
    return PooledTransformerClassifier.init(Axis("vocab", VOCAB_SIZE), config, key=key)


# Capacity-test variant (~63M vs pooled's 26M). Deployment is (8192, 1.0): both pooled runs train
# and score full-doc at 8192, so windows_per_doc is 1 — the 4096x2 entry above predates that.
ARCHS["pooled_big"] = ArchSpec(
    build=_pooled_big_build,
    make_input=token_input(VOCAB_SIZE),
    deployment=(8192, 1.0),
)


def _bert_build(ctx: int, key: jax.Array):
    from levanter.models.bert import BertConfig, BertForSequenceClassification

    config = BertConfig(max_seq_len=ctx, num_labels=2)  # MiniLM-L6-H384 dims by default
    return BertForSequenceClassification.init(Axis("vocab", 30522), config, key=key)


# Deployment (512, 3.0): begin/middle/end 512-token windows per doc, aggregated — the bme recipe.
ARCHS["bert"] = ArchSpec(  # MiniLM-L6-H384, 22.7M, 512-ctx chunked scoring
    build=_bert_build,
    make_input=token_input(30522),
    deployment=(512, 3.0),
    sweep_ctxs=(128, 256, 512),
)


def _bigdn_build(ctx: int, key: jax.Array):
    from levanter.models.bigdn import BigdnConfig, BigdnForSequenceClassification

    config = BigdnConfig(max_seq_len=ctx)
    return BigdnForSequenceClassification.init(Axis("vocab", VOCAB_SIZE), config, key=key)


ARCHS["bigdn"] = ArchSpec(  # bidirectional GatedDeltaNet encoder, h512 L12, 86M, from scratch
    build=_bigdn_build,
    make_input=token_input(VOCAB_SIZE),
    deployment=(8192, 1.0),
)


@dataclass
class CellResult:
    arch: str
    ctx: int
    backend: str
    total_batch: int
    params: int | None = None
    ms_per_forward: float | None = None
    seqs_per_sec: float | None = None
    seqs_per_sec_per_chip: float | None = None
    error: str | None = None


@dataclass
class ArchResult:
    arch: str
    deployment_ctx: int
    windows_per_doc: float
    cells: list[CellResult] = field(default_factory=list)
    params: int | None = None
    # ctx the headline was measured at (== deployment_ctx normally; a proxy in --smoke / custom --ctxs)
    headline_ctx: int | None = None
    effective_docs_per_sec_per_chip: float | None = None
    vs_gate: float | None = None
    gate: str | None = None


def bench_cell(
    name: str,
    spec: ArchSpec,
    ctx: int,
    batch_per_device: int,
    num_devices: int,
    mesh: Mesh,
    axis_mapping: dict,
    policy: jmp.Policy,
    *,
    warmup: int,
    iters: int,
) -> CellResult:
    total_batch = batch_per_device * num_devices
    backend = backend_for_ctx(ctx).value
    Batch = Axis("batch", total_batch)
    Pos = Axis("position", ctx)

    fwd = hax.named_jit(lambda m, t, msk: m(t, msk).astype(jnp.float32), axis_resources=axis_mapping)

    # haliax set_mesh (not the legacy `with mesh:`) — newer sharding paths (e.g. bert's init)
    # require the jax.set_mesh-style context; matches the deployed scorer (score_modernbert_useful).
    with set_mesh(mesh):
        # Build inside the mesh: some archs' init paths use sharding constraints that require an
        # active mesh (harmless for the rest).
        model = spec.build(ctx, jrandom.PRNGKey(0))
        params = int(parameter_count(model))
        model = policy.cast_to_compute(model)  # bf16 compute, matches the scoring path
        tokens, mask = spec.make_input(Batch, Pos, jrandom.PRNGKey(1))
        for _ in range(warmup):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        t0 = time.perf_counter()
        for _ in range(iters):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        ms = (time.perf_counter() - t0) / iters * 1e3

    seqs_per_sec = total_batch / (ms / 1e3)
    return CellResult(
        arch=name,
        ctx=ctx,
        backend=backend,
        total_batch=total_batch,
        params=params,
        ms_per_forward=round(ms, 2),
        seqs_per_sec=round(seqs_per_sec, 1),
        seqs_per_sec_per_chip=round(seqs_per_sec / num_devices, 2),
    )


def bench_arch(
    name: str,
    spec: ArchSpec,
    ctxs: list[int] | None,
    batch_per_device: int,
    num_devices: int,
    mesh: Mesh,
    axis_mapping: dict,
    policy: jmp.Policy,
    *,
    warmup: int,
    iters: int,
) -> ArchResult:
    deployment_ctx, windows_per_doc = spec.deployment
    if ctxs is None:
        # Per-arch informational grid; the deployment ctx is always measured.
        arch_ctxs = sorted(set(spec.sweep_ctxs) | {deployment_ctx})
    else:
        arch_ctxs = ctxs  # explicit --ctxs / --smoke override (deployment ctx may be absent)

    result = ArchResult(arch=name, deployment_ctx=deployment_ctx, windows_per_doc=windows_per_doc)
    for ctx in arch_ctxs:
        tag = f"{name}/ctx{ctx}/{backend_for_ctx(ctx).value}/bpd{batch_per_device}"
        try:
            cell = bench_cell(
                name,
                spec,
                ctx,
                batch_per_device,
                num_devices,
                mesh,
                axis_mapping,
                policy,
                warmup=warmup,
                iters=iters,
            )
            logger.info(
                "[ok] %s: %.1f ms/fwd  %.1f seqs/s  %.2f seqs/s/chip (batch=%d, %.1fM params)",
                tag,
                cell.ms_per_forward,
                cell.seqs_per_sec,
                cell.seqs_per_sec_per_chip,
                cell.total_batch,
                (cell.params or 0) / 1e6,
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            logger.warning("[FAIL] %s: %s", tag, msg)
            cell = CellResult(
                arch=name,
                ctx=ctx,
                backend=backend_for_ctx(ctx).value,
                total_batch=batch_per_device * num_devices,
                error=msg,
            )
        result.cells.append(cell)

    ok_cells = [c for c in result.cells if c.error is None]
    if ok_cells:
        result.params = ok_cells[0].params
        # Headline cell: the deployment ctx if measured, else the largest measured ctx as a proxy
        # (happens in --smoke / custom --ctxs; the JSON records which ctx was actually used).
        headline = next((c for c in ok_cells if c.ctx == deployment_ctx), None)
        if headline is None:
            headline = max(ok_cells, key=lambda c: c.ctx)
            logger.warning(
                "%s: deployment ctx %d not measured; using ctx %d as proxy for the headline",
                name,
                deployment_ctx,
                headline.ctx,
            )
        result.headline_ctx = headline.ctx
        assert headline.seqs_per_sec_per_chip is not None
        result.effective_docs_per_sec_per_chip = round(headline.seqs_per_sec_per_chip / windows_per_doc, 2)
    return result


def apply_gate(arch_results: list[ArchResult]) -> None:
    """Set vs_gate + gate in place: PASS = strictly faster (effective docs/s/chip) than mb-base."""
    gate_result = next((r for r in arch_results if r.arch == GATE_ARCH), None)
    gate_eff = gate_result.effective_docs_per_sec_per_chip if gate_result else None
    for ar in arch_results:
        if ar.effective_docs_per_sec_per_chip is None:
            ar.gate = "FAIL"
        elif gate_eff is None:
            ar.gate = "-"  # no gate reference in this run
        elif ar.arch == GATE_ARCH:
            ar.vs_gate = 1.0
            ar.gate = "GATE"
        else:
            ar.vs_gate = round(ar.effective_docs_per_sec_per_chip / gate_eff, 2)
            ar.gate = "PASS" if ar.effective_docs_per_sec_per_chip > gate_eff else "FAIL"


def _fmt_params(params: int | None) -> str:
    return f"{params / 1e6:.1f}M" if params is not None else "-"


def print_tables(arch_results: list[ArchResult], device_kind: str, num_devices: int, batch_per_device: int) -> None:
    logger.info("===== ARCH_INFERENCE_BENCHMARK =====")
    logger.info("device=%s  chips=%d  batch_per_device=%d", device_kind, num_devices, batch_per_device)

    # Informational per-ctx grid.
    logger.info("%-10s %-8s %-6s %-8s %12s %14s", "arch", "params", "ctx", "backend", "ms/fwd", "seqs/s/chip")
    for ar in arch_results:
        for c in ar.cells:
            if c.error is not None:
                logger.info(
                    "%-10s %-8s %-6d %-8s %12s %14s", ar.arch, _fmt_params(ar.params), c.ctx, c.backend, "OOM/ERR", "-"
                )
            else:
                logger.info(
                    "%-10s %-8s %-6d %-8s %12.1f %14.2f",
                    ar.arch,
                    _fmt_params(ar.params),
                    c.ctx,
                    c.backend,
                    c.ms_per_forward,
                    c.seqs_per_sec_per_chip,
                )

    # Headline table: effective docs/s/chip at deployment, sorted, with the mb-base gate.
    logger.info("----- effective docs/s/chip at deployment (sorted; gate = strictly faster than %s) -----", GATE_ARCH)
    logger.info(
        "%-10s %-8s %-12s %-8s %16s %12s %6s",
        "arch",
        "params",
        "deploy",
        "meas@",
        "eff docs/s/chip",
        "vs mb-base",
        "gate",
    )
    for ar in arch_results:
        deploy = f"{ar.deployment_ctx}x{ar.windows_per_doc:g}"
        if ar.effective_docs_per_sec_per_chip is None:
            logger.info(
                "%-10s %-8s %-12s %-8s %16s %12s %6s",
                ar.arch,
                _fmt_params(ar.params),
                deploy,
                "-",
                "OOM/ERR",
                "-",
                "FAIL",
            )
            continue
        vs = f"{ar.vs_gate:.2f}x" if ar.vs_gate is not None else "-"
        logger.info(
            "%-10s %-8s %-12s %-8d %16.2f %12s %6s",
            ar.arch,
            _fmt_params(ar.params),
            deploy,
            ar.headline_ctx,
            ar.effective_docs_per_sec_per_chip,
            vs,
            ar.gate or "-",
        )


def run(args) -> None:
    devices = jax.devices()
    num_devices = len(devices)
    device_kind = devices[0].device_kind
    logger.info("devices: %d x %s", num_devices, device_kind)

    mesh = Mesh(np.array(devices), ("data",))
    axis_mapping = {"batch": "data"}
    policy = jmp.get_policy("p=f32,c=bfloat16")

    arch_names = list(ARCHS) if args.archs == "all" else args.archs.split(",")
    unknown = [a for a in arch_names if a not in ARCHS]
    if unknown:
        raise ValueError(f"unknown archs {unknown}; available: {sorted(ARCHS)}")
    ctxs = [int(c) for c in args.ctxs.split(",")] if args.ctxs else None

    arch_results = [
        bench_arch(
            name,
            ARCHS[name],
            ctxs,
            args.batch_per_device,
            num_devices,
            mesh,
            axis_mapping,
            policy,
            warmup=args.warmup,
            iters=args.iters,
        )
        for name in arch_names
    ]

    apply_gate(arch_results)
    arch_results.sort(key=lambda r: r.effective_docs_per_sec_per_chip or 0.0, reverse=True)
    print_tables(arch_results, device_kind, num_devices, args.batch_per_device)

    summary = {
        "device_kind": device_kind,
        "num_devices": num_devices,
        "batch_per_device": args.batch_per_device,
        "warmup": args.warmup,
        "iters": args.iters,
        "smoke": args.smoke,
        "gate_arch": GATE_ARCH,
        "archs": [
            {
                "arch": ar.arch,
                "params": ar.params,
                "deployment_ctx": ar.deployment_ctx,
                "windows_per_doc": ar.windows_per_doc,
                "headline_ctx": ar.headline_ctx,
                "effective_docs_per_sec_per_chip": ar.effective_docs_per_sec_per_chip,
                "vs_gate": ar.vs_gate,
                "gate": ar.gate,
                "cells": [{k: v for k, v in vars(c).items() if k != "arch" and v is not None} for c in ar.cells],
            }
            for ar in arch_results
        ],
    }
    logger.info("BENCH_JSON %s", json.dumps(summary))

    if args.out:
        with fsspec.open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("wrote results -> %s", args.out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archs", default="all", help=f"comma-separated subset of {sorted(ARCHS)}, or 'all'.")
    p.add_argument(
        "--ctxs",
        default="",
        help="comma-separated context lengths overriding every arch's sweep grid. Default: each "
        "arch's own sweep_ctxs (plus its deployment ctx).",
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
    p.add_argument(
        "--smoke",
        action="store_true",
        help="CPU plumbing check: ctx 128 only, batch 2, 1 warmup + 1 iter, all requested archs. "
        "Timings are meaningless; the headline uses ctx 128 as a proxy for deployment.",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.smoke:
        args.ctxs = "128"
        args.batch_per_device = 2
        args.warmup = 1
        args.iters = 1
    run(args)


if __name__ == "__main__":
    main()
