# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch ModernBERT useful-classifier training on TPU via Levanter (the (b) path).

Builds a ``TrainClassifierConfig`` (levanter) + ``TrainClassifierOnPodConfig`` (marin) and submits
through ``run_levanter_train_classifier`` — i.e. the same Fray/Iris machinery as train_lm/train_dpo,
so we inherit region-local checkpoints, preemption auto-resume, wandb, and the TPU splash kernel.

Run this module itself as a (CPU) Iris coordinator job; it submits the TPU training job and waits.
Use ``--smoke`` for a tiny validation run, ``--no-submit`` to build + print the config locally.

Data (train + eval) is in us-east5, in-region for the TPU. ``assert_data_in_region`` fails fast on
launch if any data path is not in the run region — cross-region reads are the project's #1 cost
driver, and marin's own region check intentionally skips train/validation URLs, so we guard here.
"""

import argparse
import logging
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.main.train_classifier import ClassificationDataConfig, TrainClassifierConfig
from levanter.models.modernbert import ModernBertConfig, PrunedModernBertConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.remote import remote
from marin.training.training import TrainClassifierOnPodConfig, run_levanter_train_classifier
from rigging.filesystem import region_from_prefix


def assert_data_in_region(paths: dict[str, str], region: str) -> None:
    """Fail fast if ANY data path is not in ``region``.

    Cross-region reads are this project's #1 cost driver, and marin's own region check deliberately
    SKIPS train_urls/validation_urls — so we guard the data paths explicitly here. Every gs:// data
    path the TPU worker reads (train + eval) MUST be in the run's region.
    """
    bad = []
    for name, path in paths.items():
        if not path.startswith("gs://"):
            continue
        path_region = region_from_prefix(path)
        if path_region != region:
            bad.append(f"{name}={path} is in region {path_region!r}, not {region!r}")
    if bad:
        raise ValueError(
            "Refusing to launch: data paths would be read CROSS-REGION (the #1 cost driver). "
            "Mirror them into the run region first.\n  " + "\n  ".join(bad)
        )


logger = logging.getLogger(__name__)

# SURVIVOR data: docs that pass the fastText stage-1 filter — the operationally relevant
# distribution for stage-2. Use the PRE-SHUFFLED, IN-REGION (us-east5) 1M sample that the torch
# 0.705 model trained on. CRITICAL: the raw `useful_cascade_survivors/parts` shards are
# CLASS-ORDERED (useful block then no_useful block); reading them in order yields class-homogeneous
# microbatches → degenerate training (loss→0). The preshard is mixed/shuffled (~20% useful) AND
# in-region (no cross-region egress). We also .shuffle() the dataset in train_classifier for safety.
# Eval stays the general frozen 7000-doc test so F1 compares to 0.705 + the cascade operating curves.
# BOTH in us-east5 (in-region for the TPU) — NO cross-region reads. The test set has an existing
# us-east5 mirror (identical 35 shards); the frozen-7000 sample is deterministic so F1 still compares
# to the 0.705 baseline. (Earlier TEST_GLOB pointed at us-central2 → every run read 10.2GB
# cross-region. Cross-region egress is the project's #1 cost driver — never read the eval set
# cross-region again.)
TRAIN_GLOB = "gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_1M_w4/train_shard_*.txt.gz"
TEST_GLOB = "gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz"
MODEL_ID = "answerdotai/ModernBERT-base"
PAD_TOKEN_ID = 50283  # ModernBERT pad token (shared by base + large — same 50368-token tokenizer)

# All presets share the ModernBERT tokenizer/vocab/pad token (Ettin verified identical: 50368
# vocab, pad 50283, local_attention 128, global-every-3) — so they also share the existing
# tokenized TreeCache. Dims verified against each checkpoint's HF config. `config_cls` picks the
# levanter config type (pruned variants use PrunedModernBertConfig surgery); `warm_start=False`
# presets train from scratch (tiny archs have no checkpoint).
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
    # Ettin (JHU 2025, arXiv 2507.11412): ModernBERT architecture at small scale, MLM-pretrained
    # 2T tokens. Both rope thetas are 160k (unlike base's 10k local); classifier_pooling=mean per
    # the released configs.
    "ettin17": dict(
        reference_checkpoint="jhu-clsp/ettin-encoder-17m",
        hidden_dim=256,
        intermediate_dim=384,
        num_layers=7,
        num_heads=4,
        local_rope_theta=160000.0,
        classifier_pooling="mean",
    ),
    "ettin32": dict(
        reference_checkpoint="jhu-clsp/ettin-encoder-32m",
        hidden_dim=384,
        intermediate_dim=576,
        num_layers=10,
        num_heads=6,
        local_rope_theta=160000.0,
        classifier_pooling="mean",
    ),
    "ettin68": dict(
        reference_checkpoint="jhu-clsp/ettin-encoder-68m",
        hidden_dim=512,
        intermediate_dim=768,
        num_layers=19,
        num_heads=8,
        local_rope_theta=160000.0,
        classifier_pooling="mean",
    ),
    # Layer-pruned ModernBERT-base: warm-start from base's BOTTOM 8 of 22 layers ("Poor Man's
    # BERT"); prefix-keep preserves each layer's global/local assignment.
    "pruned8": dict(
        config_cls="pruned",
        reference_checkpoint="answerdotai/ModernBERT-base",
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=8,
        num_heads=12,
    ),
    # From-scratch tiny ModernBERT (the "2-4 layer transformer" full-context take). No pretrain.
    "tiny4": dict(
        warm_start=False,
        hidden_dim=384,
        intermediate_dim=576,
        num_layers=4,
        num_heads=6,
    ),
    "tiny2": dict(
        warm_start=False,
        hidden_dim=256,
        intermediate_dim=384,
        num_layers=2,
        num_heads=4,
    ),
}


def _funnelbert_factory(args):
    from levanter.models.funnelbert import FunnelBertConfig

    return FunnelBertConfig(
        max_seq_len=args.max_seq_len,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
        attn_backend=args.attn_backend,
        tokenizer=MODEL_ID,
    )


def _pooled_transformer_factory(args):
    from levanter.models.pooled_transformer import PooledTransformerConfig

    return PooledTransformerConfig(max_seq_len=args.max_seq_len)


def _pooled_big_factory(args):
    """Capacity test for pooled: ~2.4x the params (26M → ~63M), same 64-token pooling.

    pooled is half embedding table, so "is it data-limited or capacity-limited?" is answered by
    widening the super-token transformer (hidden 512→768, layers 4→6, embed 256→384) while keeping
    the pooling window fixed. Its cost over the 8192-token input scales with embed_dim (1.5x) and
    the per-super-token layers scale ~3.4x, so it MUST be re-benchmarked before it earns a slot.
    """
    from levanter.models.pooled_transformer import PooledTransformerConfig

    return PooledTransformerConfig(
        max_seq_len=args.max_seq_len,
        embed_dim=384,
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
    )


def _bert_factory(args):
    from levanter.models.bert import BertConfig

    return BertConfig(max_seq_len=args.max_seq_len, num_labels=2)


def _bigdn_factory(args):
    from levanter.models.bigdn import BigdnConfig

    return BigdnConfig(max_seq_len=args.max_seq_len)


# Non-ModernBERT-config archs: built via factory(args) instead of ModernBertConfig(**dims).
# `data_tokenizer` overrides the ClassificationDataConfig tokenizer (bert uses MiniLM WordPiece —
# its own token cache namespace; everything else shares the ModernBERT tokenizer + TreeCache).
# `defaults` pre-sets CLI knobs (bert wants ctx 512 chunked; pooled_transformer wants ctx 4096
# chunked per its design) — explicit flags still win.
MODEL_FACTORIES = {
    "funnelbert": dict(factory=_funnelbert_factory, warm_start=True),
    # ctx 8192 truncated (NOT chunked): pooled's deployment recipe is 2x4096-token windows = the
    # same 8192 tokens of coverage, and truncated@8192 reuses the finished 8192-cap TreeCache
    # (memory-bounded streaming) instead of needing a 32768-cap chunk cache — the lpv11 corpus
    # averages ~54 KB/doc, so any in-RAM 1M-doc cache OOMs. Also makes pooled directly comparable
    # to every other arch in the sweep, which all train truncated@8192.
    # per_device_parallelism MUST be -1 (no gradient accumulation). pooled_transformer's params are
    # RAW jnp arrays, not NamedArrays: levanter's microbatching reshapes/scans every plain-array leaf
    # of its args as if it carried the batch (fn is called as fn(model, *batch)), so the embedding
    # table hits "cannot reshape (50368, 256) into (4, 1, 256)". NamedArray models are immune (they
    # are skipped for lacking a Batch axis). pdp=-1 makes microbatch_size None → no accumulation, no
    # reshape. Diagnosed by pooled_backward_probe --mode trainer (2026-08-09).
    "pooled": dict(
        factory=_pooled_transformer_factory,
        warm_start=False,
        defaults=dict(max_seq_len=8192, use_cache=True, per_device_parallelism=-1),
    ),
    "bert": dict(
        factory=_bert_factory,
        warm_start=True,
        data_tokenizer="nreimers/MiniLM-L6-H384-uncased",
        defaults=dict(max_seq_len=512, chunked=True),
    ),
    "pooled_big": dict(
        factory=_pooled_big_factory,
        warm_start=False,
        defaults=dict(max_seq_len=8192, use_cache=True, per_device_parallelism=-1),
    ),
    "bigdn": dict(factory=_bigdn_factory, warm_start=False),
}

# region -> region-local checkpoint bucket
REGION_BUCKETS = {
    "us-east5": "gs://marin-us-east5",
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "europe-west4": "gs://marin-eu-west4",
}


def build_config(args) -> TrainClassifierConfig:
    factory_entry = MODEL_FACTORIES.get(args.model_size)
    data_tokenizer = MODEL_ID if factory_entry is None else factory_entry.get("data_tokenizer", MODEL_ID)
    data = ClassificationDataConfig(
        train_urls=[args.train_glob],
        validation_urls=[args.test_glob],
        tokenizer=data_tokenizer,
        max_train_rows=args.train_rows,
        eval_rows=args.test_rows,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        chunked=args.chunked,
        max_chunks_per_doc=args.max_chunks_per_doc,
        chunk_overlap=args.chunk_overlap,
    )
    # For chunked runs the real step count = #chunks // batch and is computed worker-side (depends on
    # the doc-length distribution); this doc-based estimate is just a placeholder + LR-warmup anchor.
    steps_per_epoch = max(1, args.train_rows // args.batch_size)
    num_train_steps = steps_per_epoch * args.epochs

    trainer = TrainerConfig(
        id=args.run_id,
        tracker=WandbConfig(
            project="modernbert-useful", tags=["modernbert", "classifier", "levanter", f"arch-{args.model_size}"]
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=args.batch_size,
        # Per-device microbatch. MUST be small at 8192 ctx: -1 (=batch/num_devices=64 on v6e-4)
        # OOMs HBM (QKV/MLP activations at batch 64 x 8192 need ~186 GB > 32 GB/chip). 2 matches
        # the torch micro-batch; effective batch stays args.batch_size via grad accumulation.
        per_device_parallelism=args.per_device_parallelism,
        num_train_steps=num_train_steps,
        steps_per_eval=num_train_steps,  # F1 eval is done post-hoc on the frozen test
        checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=15), keep=[]),
    )
    if factory_entry is not None:
        model = factory_entry["factory"](args)
        preset_warm_start = factory_entry["warm_start"]
    else:
        preset = dict(MODEL_PRESETS[args.model_size])
        config_cls = ModernBertConfig if preset.pop("config_cls", None) != "pruned" else PrunedModernBertConfig
        preset_warm_start = preset.pop("warm_start", True)
        model = config_cls(
            max_seq_len=args.max_seq_len,
            num_labels=2,
            pad_token_id=PAD_TOKEN_ID,
            attn_backend=args.attn_backend,
            tokenizer=MODEL_ID,
            **preset,
        )
    return TrainClassifierConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=AdamConfig(learning_rate=args.lr, warmup=args.warmup, max_grad_norm=1.0),
        warm_start=preset_warm_start and not args.no_warm_start,
        hf_save_path=None,  # set by the marin out-path machinery to {output_path}/hf
    )


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--model-size",
        default="base",
        choices=sorted(set(MODEL_PRESETS) | set(MODEL_FACTORIES)),
        help="Arch preset: ModernBERT-config presets (base/large/ettin*/pruned8/tiny*) or factory "
        "archs (funnelbert/pooled/bert/bigdn). Sets dims + warm-start; factory archs may pre-set "
        "ctx/chunked defaults. large@8192 is HBM-heavy — use per-device-parallelism 1.",
    )
    p.add_argument("--train-glob", default=TRAIN_GLOB)
    p.add_argument("--test-glob", default=TEST_GLOB)
    p.add_argument("--train-rows", type=int, default=200_000)
    p.add_argument("--test-rows", type=int, default=7000)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--per-device-parallelism",
        type=int,
        default=2,
        help="Per-device microbatch (grad-accum to batch-size). Keep small at 8192 ctx; -1 OOMs.",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--attn-backend", type=AttentionBackend, default=AttentionBackend.VANILLA)
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Explicit TreeCache dir. REQUIRED for non-default tokenizers with --use-cache: the "
        "default dir derivation is tokenizer-blind and a mismatched cache only WARNS (silent "
        "corruption). e.g. .../_clf_token_cache_chunk32768_minilm for bert.",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Stream training data from a prebuilt tokenized TreeCache (memory-bounded; required for "
        "5M/10M which OOM the in-memory default). Cache must be prebuilt to avoid build races.",
    )
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument(
        "--chunked",
        action="store_true",
        help="MIL chunked training: split each doc into max-seq-len-sized chunks, each chunk inherits "
        "the doc label. In-memory only. Step count is recomputed worker-side from the chunk count.",
    )
    p.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=16,
        help="Training cap: docs with more chunks contribute a seeded random subset (inference uses all).",
    )
    p.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.0,
        help="0.0 = non-overlapping tiles; 0.5 = 50%% overlap (stride = ctx/2). Swept as a hyperparameter.",
    )
    p.add_argument("--tpu-type", default="v6e-4")
    p.add_argument(
        "--memory-gb",
        type=int,
        default=None,
        help="Container host RAM (GB). Default (None) → with_tpu's 128g. The dataset materializes "
        "all --train-rows texts in host RAM (~25KB/doc), so 5M needs ~256, 10M ~400 (v5p host=448).",
    )
    p.add_argument("--region", default="us-east5", choices=sorted(REGION_BUCKETS))
    p.add_argument(
        "--non-preemptible",
        action="store_true",
        help="Request a NON-preemptible TPU (preemptible=False). Use for short jobs (e.g. a re-eval) that "
        "must run to completion without getting bumped off the churny preemptible pool mid-eval.",
    )
    p.add_argument("--no-submit", action="store_true", help="Build + print config; do not submit.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny validation run: 1024 rows, 1024 ctx (VANILLA-safe), warm-start — validates the path.",
    )
    args = p.parse_args()

    # Factory archs carry their own ctx/chunked defaults (e.g. bert = 512 chunked); they apply only
    # where the user left the parser default, so explicit flags still win.
    factory_entry = MODEL_FACTORIES.get(args.model_size)
    if factory_entry is not None:
        for name, value in factory_entry.get("defaults", {}).items():
            if getattr(args, name) == p.get_default(name):
                setattr(args, name, value)
                logger.info("preset %s: defaulting --%s to %r", args.model_size, name.replace("_", "-"), value)

    # Fail fast (on the laptop, before submitting) if any data path would be read cross-region.
    assert_data_in_region({"train_glob": args.train_glob, "test_glob": args.test_glob}, args.region)

    if args.smoke:
        args.train_rows = min(args.train_rows, 1024)
        args.batch_size = min(args.batch_size, 32)
        args.max_seq_len = min(args.max_seq_len, 1024)  # VANILLA is O(seq^2); keep the smoke cheap
        args.warmup = 0

    # The chunked end-of-run eval at ctx>=4096 with VANILLA attention SIGSEGVs the TPU XLA compiler
    # (SpatialMajorConvolution lowering) on the eval's batch-16 forward; SPLASH (Pallas flash) compiles
    # cleanly (proven at c8192). Training is fine with vanilla, but force splash so the eval survives.
    if args.chunked and args.max_seq_len >= 4096 and args.attn_backend == AttentionBackend.VANILLA:
        logger.warning("chunked + ctx>=4096: forcing SPLASH (vanilla chunked eval crashes the XLA compiler)")
        args.attn_backend = AttentionBackend.SPLASH

    config = build_config(args)
    output_path = f"{REGION_BUCKETS[args.region]}/checkpoints/modernbert-useful/{args.run_id}"
    logger.info("output_path: %s", output_path)
    logger.info(
        "run_id=%s rows=%d batch=%d steps=%d max_seq_len=%d attn=%s warm_start=%s tpu=%s region=%s " "chunked=%s%s",
        args.run_id,
        args.train_rows,
        args.batch_size,
        config.trainer.num_train_steps,
        args.max_seq_len,
        args.attn_backend,
        config.warm_start,
        args.tpu_type,
        args.region,
        args.chunked,
        (
            f" (max_chunks={args.max_chunks_per_doc} overlap={args.chunk_overlap}, steps recomputed worker-side)"
            if args.chunked
            else ""
        ),
    )

    if args.no_submit:
        logger.info("--no-submit: built config OK, not submitting.")
        return

    tpu_kwargs = {"ram": f"{args.memory_gb}g"} if args.memory_gb else {}
    if args.non_preemptible:
        tpu_kwargs["preemptible"] = False  # hold the slot through the eval; don't get bumped
    pod_config = TrainClassifierOnPodConfig(
        train_config=config,
        resources=ResourceConfig.with_tpu(args.tpu_type, **tpu_kwargs),
        output_path=output_path,
        env_vars={
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "HF_TOKEN": os.environ["HF_TOKEN"],
        },
    )
    # run_levanter_train_* now runs the Levanter main in-process (upstream restructure);
    # submit it to the TPU pod as its own Fray job, mirroring marin.experiment.train._train_job.
    remote(run_levanter_train_classifier, name="train_classifier", resources=pod_config.resources)(pod_config)


if __name__ == "__main__":
    main()
