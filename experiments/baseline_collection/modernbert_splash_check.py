# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M4: validate the TPU SPLASH attention backend for ModernBERT and benchmark it vs VANILLA.

Two things, on the same randomly-initialized ModernBERT-base-shaped model and the same input
(with a realistic pad/segment mask so the symmetric local window + segment masking are exercised):

  1. CORRECTNESS — run a forward under two attention backends (default ``vanilla`` vs ``splash``)
     and assert the logits match. ``vanilla`` materializes the full O(seq^2) scores; ``splash`` is
     the Pallas flash kernel with the symmetric ``bidirectional_window`` LocalMask. They must agree
     (masked pads are bit-equivalent per the port). This is the prerequisite for trusting any
     SPLASH-trained model (M5's sweep runs entirely on SPLASH).

  2. THROUGHPUT — time N forwards per backend at the given seq len and report ms/forward and the
     SPLASH speedup. At 8192 with alternating global/local attention, SPLASH should be markedly
     faster (local layers become O(seq*window)); VANILLA may even OOM at 8192 — that itself is the
     point (SPLASH is what makes 8192 feasible).

SPLASH is TPU-only (Pallas), so launch this on a TPU worker. On CPU only ``vanilla`` is available
(use ``--backend-b vanilla`` for a self-consistency sanity check).
"""

import argparse
import logging
import time

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.modernbert import ModernBertConfig, ModernBertForSequenceClassification

logger = logging.getLogger(__name__)


def _build(config: ModernBertConfig, Vocab: Axis, backend: AttentionBackend, key):
    import dataclasses

    return ModernBertForSequenceClassification.init(Vocab, dataclasses.replace(config, attn_backend=backend), key=key)


def _make_input(Vocab: Axis, Batch: Axis, Pos: Axis, *, pad_frac: float, key):
    """Random tokens with a per-row pad tail; segment mask marks real=0 / pad=-1."""
    ids = np.asarray(hax.random.randint(key, (Batch, Pos), 1, Vocab.size).array)
    seg = np.zeros((Batch.size, Pos.size), dtype=np.int32)
    n_real = max(1, int(Pos.size * (1.0 - pad_frac)))
    seg[:, n_real:] = -1  # pad tail excluded from attention
    tokens = hax.named(ids.astype(np.int32), (Batch, Pos))
    seg_named = hax.named(seg, (Batch, Pos))
    mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
    return tokens, mask


def run(args):
    import numpy as _np
    from jax.sharding import Mesh

    logging.basicConfig(level=logging.INFO)
    logger.info("devices: %s", jax.devices())

    config = ModernBertConfig(max_seq_len=args.max_seq_len, num_labels=2, pad_token_id=50283)
    Vocab = Axis("vocab", 50368)
    Batch = Axis("batch", args.batch)
    Pos = config.max_Pos

    key = jrandom.PRNGKey(0)
    backend_a = AttentionBackend(args.backend_a)
    backend_b = AttentionBackend(args.backend_b)
    # Same key => identical weights; only the attention backend differs.
    model_a = _build(config, Vocab, backend_a, key)
    model_b = _build(config, Vocab, backend_b, key)

    tokens, mask = _make_input(Vocab, Batch, Pos, pad_frac=args.pad_frac, key=jrandom.PRNGKey(1))

    # SPLASH requires a non-empty mesh; shard the batch over a 1-D "data" mesh over all chips.
    mesh = Mesh(_np.array(jax.devices()), ("data",))
    axis_mapping = {"batch": "data"}
    fwd = hax.named_jit(lambda m, t, msk: m(t, msk).astype(jnp.float32), axis_resources=axis_mapping)

    def _bench(model, label):
        for _ in range(args.warmup):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            jax.block_until_ready(fwd(model, tokens, mask).array)
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        logger.info("[throughput] %s: %.1f ms/forward (batch=%d, seq=%d)", label, ms, Batch.size, Pos.size)
        return ms

    # Compare at MATCHED precision. The splash kernel computes in bf16; comparing it to fp32-vanilla
    # conflates a masking bug with mere bf16 rounding. So: (1) bf16 floor = |vanilla_bf16 - vanilla_fp32|
    # (the unavoidable bf16 error), (2) backend diff = |splash_bf16 - vanilla_bf16| (the real question).
    # SPLASH is correct iff the backend diff is within a small multiple of the bf16 floor (i.e. splash
    # adds no error beyond bf16 flash-vs-dense reduction order). A masking bug would be orders larger.
    policy = jmp.get_policy("p=f32,c=bfloat16")
    model_a_bf16 = policy.cast_to_compute(model_a)
    model_b_bf16 = policy.cast_to_compute(model_b)

    with mesh:
        ref_fp32 = np.asarray(fwd(model_a, tokens, mask).array)  # vanilla, fp32
        a_bf16 = np.asarray(fwd(model_a_bf16, tokens, mask).array)  # vanilla, bf16
        b_bf16 = np.asarray(fwd(model_b_bf16, tokens, mask).array)  # splash,  bf16

        bf16_floor = float(np.max(np.abs(a_bf16 - ref_fp32)))
        backend_diff = float(np.max(np.abs(b_bf16 - a_bf16)))
        # PASS if splash is within ~3x the bf16 floor (or both tiny). 3x allows flash-vs-dense order.
        close = backend_diff <= max(3.0 * bf16_floor, 5e-3)
        logger.info(
            "[correctness] %s vs %s @ seq=%d batch=%d: backend_diff(bf16)=%.3e  bf16_floor=%.3e  " "ratio=%.2f  PASS=%s",
            backend_a.value,
            backend_b.value,
            Pos.size,
            Batch.size,
            backend_diff,
            bf16_floor,
            backend_diff / bf16_floor if bf16_floor > 0 else float("inf"),
            close,
        )
        ms_a = _bench(model_a_bf16, backend_a.value)
        ms_b = _bench(model_b_bf16, backend_b.value)

    if ms_b > 0:
        logger.info("[throughput] %s is %.2fx vs %s (bf16)", backend_b.value, ms_a / ms_b, backend_a.value)
    if not close:
        raise SystemExit(
            f"FAIL: {backend_b.value} differs from {backend_a.value} beyond bf16 floor "
            f"(backend_diff={backend_diff:.3e}, floor={bf16_floor:.3e})"
        )
    logger.info("PASS: SPLASH agrees with VANILLA within bf16 flash-vs-dense tolerance — trustworthy for training.")


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--backend-a", default="vanilla")
    p.add_argument("--backend-b", default="splash")
    p.add_argument("--pad-frac", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
