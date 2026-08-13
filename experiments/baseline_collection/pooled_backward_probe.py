# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin down which PooledTransformer op SIGSEGVs the TPU XLA compiler in the BACKWARD pass.

The forward pass runs fine on TPU at ctx 8192 (2.1 ms/fwd in arch_inference_benchmark) and
forward+backward is clean on CPU, but the training step SIGSEGVs (exit 139) at ctx 8192 for both
per-device batch 8 and 2 — i.e. it is a compiler lowering bug, not a memory or math problem. The
prime suspect is the windowed pooling (reshape ``[b, t, e] -> [b, s, w, e]`` then reduce over the
window axis); this repo has a precedent of a TPU ``SpatialMajorConvolution`` lowering SIGSEGV.

A segfault kills the process and (with the log plane flaky) takes stdout with it, so each variant
writes a ``<name>_OK`` marker to GCS as soon as its gradient step completes. Whichever marker is
MISSING is the variant that crashed.

Variants, cheapest-hypothesis first:
  meanmaxmin  - current default (mean + max + min over each window)
  mean        - mean only (is max/min the trigger?)
  mean_minor  - mean, but reduce over the MINOR-most axis after a transpose (is the reduce-over-
                a-major-axis lowering the trigger?)
  matmul      - mean via einsum against a block indicator (pure MXU; no windowed reduce at all)

Launch (v6e-4, us-east5)::

    iris job run --tpu v6e-4 --region us-east5 --extra tpu --enable-extra-resources \\
        --memory 64GB --priority interactive --no-wait --job-name pooled-bwd-probe \\
        -- python -m experiments.baseline_collection.pooled_backward_probe
"""

import argparse
import logging

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import Mesh
from levanter.layers.attention import AttentionMask
from levanter.models.classification import ClassificationExample
from levanter.models.pooled_transformer import PooledTransformerClassifier, PooledTransformerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pooled_probe")

MARKER_ROOT = "gs://marin-us-east5/benchmarks/pooled_backward_probe"
VOCAB = 50368
PAD = 50283


def _example(batch: int, ctx: int) -> tuple[ClassificationExample, Axis, Axis]:
    B, P = Axis("batch", batch), Axis("position", ctx)
    ids = np.random.default_rng(0).integers(1, VOCAB, (batch, ctx)).astype(np.int32)
    seg = np.zeros((batch, ctx), dtype=np.int32)
    seg[:, int(ctx * 0.75) :] = -1  # realistic pad tail
    sg = hax.named(seg, (B, P))
    mask = AttentionMask(is_causal=False).with_segment_ids(sg, sg)
    label = hax.named(np.arange(batch, dtype=np.int32) % 2, (B,))
    return ClassificationExample.init(tokens=hax.named(ids, (B, P)), label=label, attn_mask=mask), B, P


def _grad_step(model, ex) -> float:
    # .scalar(): compute_loss returns a scalar-valued NamedArray, but value_and_grad needs a bare
    # scalar ("Gradient only defined for scalar-output functions").
    fn = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, e: m.compute_loss(e).scalar()))
    _, grads = fn(model, ex)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    return float(jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in leaves)))


def _mark(name: str, payload: str) -> None:
    with fsspec.open(f"{MARKER_ROOT}/{name}_OK", "w") as f:
        f.write(payload)
    logger.info("[OK] %s -> marker written (%s)", name, payload)


def run(ctx: int, batch: int) -> None:
    logger.info("devices: %s", jax.devices())
    mesh = Mesh(np.array(jax.devices()), ("data",))
    ex, _, _ = _example(batch, ctx)

    variants = [
        ("meanmaxmin", dict(pool_kind="meanmaxmin")),
        ("mean", dict(pool_kind="mean")),
        ("mean_minor", dict(pool_kind="mean_minor")),
        ("matmul", dict(pool_kind="matmul")),
    ]
    with set_mesh(mesh):
        for name, overrides in variants:
            try:
                cfg = PooledTransformerConfig(max_seq_len=ctx, **overrides)
            except ValueError as e:  # variant not implemented in this build of the model
                logger.warning("[SKIP] %s: %s", name, e)
                continue
            logger.info("=== probing %s (ctx=%d batch=%d) — if this is the last line, it segfaulted", name, ctx, batch)
            model = PooledTransformerClassifier.init(Axis("vocab", VOCAB), cfg, key=jax.random.PRNGKey(0))
            gnorm = _grad_step(model, ex)
            _mark(name, f"grad_norm={gnorm:.4f} ctx={ctx} batch={batch}")
    logger.info("PROBE COMPLETE — all variants that wrote markers compiled + ran backward on TPU")


def run_trainer_path(ctx: int, batch: int) -> None:
    """Stage 2 bisect: pooling is EXONERATED (all 4 variants passed backward on TPU), so walk the
    remaining differences between the bare gradient step and the real training step, one at a time.

    Order matters — each stage adds exactly one ingredient, so the first MISSING marker names the
    culprit: bf16 compute cast -> optimizer update -> gradient accumulation (the scan) -> the real
    Trainer. ``mb-clf-lpv11-pooled-1M`` used batch 256 with per-device 2 = 32 accumulated microsteps.
    """
    import jmp
    import optax
    from levanter.data.dataset import ListAsyncDataset
    from levanter.distributed import DistributedConfig
    from levanter.main.train_classifier import train_classifier
    from levanter.optim import AdamConfig
    from levanter.tracker import NoopConfig
    from levanter.trainer import TrainerConfig

    logger.info("devices: %s", jax.devices())
    mesh = Mesh(np.array(jax.devices()), ("data",))
    cfg = PooledTransformerConfig(max_seq_len=ctx)
    ex, _, _ = _example(batch, ctx)
    policy = jmp.get_policy("p=f32,c=bfloat16")

    with set_mesh(mesh):
        model = PooledTransformerClassifier.init(Axis("vocab", VOCAB), cfg, key=jax.random.PRNGKey(0))

        logger.info("=== stage bf16_cast — grad step with the Trainer's compute policy")
        gnorm = _grad_step(policy.cast_to_compute(model), ex)
        _mark("bf16_cast", f"grad_norm={gnorm:.4f}")

        logger.info("=== stage optimizer — grad step + an Adam update")
        opt = optax.adam(1e-4)
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = opt.init(params)

        @eqx.filter_jit
        def _opt_step(m, o, e):
            loss, grads = eqx.filter_value_and_grad(lambda mm, ee: mm.compute_loss(ee).scalar())(m, e)
            updates, o2 = opt.update(eqx.filter(grads, eqx.is_inexact_array), o, params)
            return eqx.apply_updates(m, updates), o2, loss

        _, _, loss = _opt_step(model, opt_state, ex)
        _mark("optimizer", f"loss={float(loss):.4f}")

        logger.info("=== stage grad_accum — 8 microbatches accumulated through lax.scan")

        @eqx.filter_jit
        def _accum_step(m, e):
            def body(carry, _):
                loss, grads = eqx.filter_value_and_grad(lambda mm, ee: mm.compute_loss(ee).scalar())(m, e)
                return jax.tree_util.tree_map(jnp.add, carry, eqx.filter(grads, eqx.is_inexact_array)), loss

            zeros = jax.tree_util.tree_map(jnp.zeros_like, eqx.filter(m, eqx.is_inexact_array))
            acc, losses = jax.lax.scan(body, zeros, None, length=8)
            return jnp.mean(losses), acc

        loss, _ = _accum_step(model, ex)
        _mark("grad_accum", f"loss={float(loss):.4f}")

    logger.info("=== stage trainer — the REAL Trainer for 2 steps (accumulation on)")
    Pos = cfg.max_Pos
    data = []
    rng = np.random.default_rng(0)
    for k in range(batch * 4):
        ids = np.full((Pos.size,), PAD, dtype=np.int32)
        n = Pos.size // 2 if k % 2 == 0 else Pos.size // 4
        ids[:n] = rng.integers(1, VOCAB, n)
        seg = np.full((Pos.size,), -1, dtype=np.int32)
        seg[:n] = 0
        sg = hax.named(seg, Pos)
        data.append(
            ClassificationExample.init(
                tokens=hax.named(ids, Pos),
                label=hax.named(np.int32(k % 2), ()),
                attn_mask=AttentionMask(is_causal=False).with_segment_ids(sg, sg),
            )
        )
    trainer_config = TrainerConfig(
        id="pooled-probe-trainer",
        num_train_steps=2,
        train_batch_size=batch * 2,
        # -1 = NO gradient accumulation, which raw-array models require (see the pooled preset in
        # launch_modernbert_levanter.py). Set a positive value here to reproduce the reshape crash.
        per_device_parallelism=-1,
        max_eval_batches=1,
        tracker=NoopConfig(),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
    )

    class _Tok:
        def __len__(self):
            return VOCAB

    train_classifier(
        trainer_config=trainer_config,
        model_config=cfg,
        optimizer_config=AdamConfig(learning_rate=3e-4, warmup=0),
        train_dataset=ListAsyncDataset(data),
        tokenizer=_Tok(),
        warm_start=False,
    )
    _mark("trainer", "2 steps completed")
    logger.info("TRAINER-PATH PROBE COMPLETE — no stage reproduced the SIGSEGV")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument(
        "--mode",
        default="pooling",
        choices=["pooling", "trainer"],
        help="pooling = the 4 pool_kind variants; trainer = bf16/optimizer/accumulation/Trainer stages.",
    )
    args = p.parse_args()
    if args.mode == "pooling":
        run(args.ctx, args.batch)
    else:
        run_trainer_path(args.ctx, args.batch)


if __name__ == "__main__":
    main()
