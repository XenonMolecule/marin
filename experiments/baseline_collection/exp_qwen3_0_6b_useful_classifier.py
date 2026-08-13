# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a Qwen3-0.6B *discriminative* useful-vs-[NO_USEFUL_CONTENT] classifier on TPU.

The generative high_quality extractor reframed as binary classification: stock ``Qwen/Qwen3-0.6B``
(base, NOT the distilled checkpoint) + a 2-way head over the last token of the teacher prompt,
trained to predict keep (``useful``) vs discard (``[NO_USEFUL_CONTENT]``). Same data + frozen test
as the ModernBERT classifier (F1 ~= 0.70), so the F1 is directly comparable.

Each example is wrapped in the EXACT teacher chat prompt (system signature + user template incl. the
full high_quality spec + the assistant ``<think></think>[[ ## text ## ]]\\n[`` prefix); the pooled
position is the trailing ``[``, where the generative model would commit to content vs the abstention
marker. The head is warm-started from the backbone's unembedding row for the ``NO`` token (8996).

Mirrors ``launch_modernbert_levanter.py``: builds a ``TrainDecoderClassifierConfig`` (levanter) +
``TrainClassifierOnPodConfig`` (marin) and submits through ``run_levanter_train_decoder_classifier``.
Run this module as a (CPU) Iris coordinator job; it submits the TPU job and blocks (keep-alive).

Data is in us-east5, in-region for the TPU; ``assert_data_in_region`` fails fast on cross-region.
"""

import argparse
import dataclasses
import logging
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.train_decoder_classifier import DecoderClassificationDataConfig, TrainDecoderClassifierConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.remote import remote
from marin.training.training import TrainClassifierOnPodConfig, run_levanter_train_decoder_classifier
from rigging.filesystem import region_from_prefix

from experiments.baseline_collection.extraction_specs import get_spec
from experiments.qwen3 import qwen3_0_6b_hd128

logger = logging.getLogger(__name__)

# Reuse the ModernBERT classifier's exact shards (in us-east5, in-region for the TPU). SURVIVOR train
# = docs passing the fastText stage-1 filter (the operationally relevant stage-2 distribution),
# pre-shuffled + class-mixed. TEST = frozen ~7000-doc natural-ratio set, so F1 compares to 0.70.
TRAIN_GLOB = "gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_1M_w4/train_shard_*.txt.gz"
TEST_GLOB = "gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz"

TOKENIZER = "Qwen/Qwen3-0.6B"
BASE_CHECKPOINT = "Qwen/Qwen3-0.6B"
# "NO" of [NO_USEFUL_CONTENT]: '[' (58) 'NO' (8996) '_USE' ... — 58 is generic, 8996 is the distinctive
# decision token. Verified against the Qwen3 tokenizer.
DECISION_TOKEN_ID = 8996

REGION_BUCKETS = {
    "us-east5": "gs://marin-us-east5",
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "europe-west4": "gs://marin-eu-west4",
}

EMPTY_THINK = "<think>\n\n</think>\n\n"
TEXT_HEADER = "[[ ## text ## ]]\n"


def build_prompt_scaffolding(spec_id: str) -> tuple[str, str]:
    """``(prompt_head, prompt_tail)`` reproducing the teacher chat around the document HTML.

    ``prompt_head`` ends right before the HTML; ``prompt_tail`` resumes after it and ends at the
    pooled decision token (the trailing ``[``). The Qwen3 chat markers are written literally — the
    tokenizer maps them to their special-token ids.
    """
    spec = get_spec(spec_id)
    if "{example}" not in spec.extraction_template:
        raise ValueError(f"spec {spec_id!r} template missing {{example}}")
    user_pre, user_post = spec.extraction_template.split("{example}")
    head = f"<|im_start|>system\n{spec.system_message}<|im_end|>\n<|im_start|>user\n{user_pre}"
    tail = f"{user_post}<|im_end|>\n<|im_start|>assistant\n{EMPTY_THINK}{TEXT_HEADER}["
    return head, tail


def assert_data_in_region(paths: dict[str, str], region: str) -> None:
    bad = []
    for name, path in paths.items():
        if path.startswith("gs://") and region_from_prefix(path) != region:
            bad.append(f"{name}={path} is in {region_from_prefix(path)!r}, not {region!r}")
    if bad:
        raise ValueError(
            "Refusing to launch: data paths would be read CROSS-REGION (the #1 cost driver).\n  " + "\n  ".join(bad)
        )


def build_config(args) -> TrainDecoderClassifierConfig:
    head, tail = build_prompt_scaffolding(args.spec_id)
    data = DecoderClassificationDataConfig(
        train_urls=[args.train_glob],
        validation_urls=[args.test_glob],
        tokenizer=TOKENIZER,
        prompt_head=head,
        prompt_tail=tail,
        max_train_rows=args.train_rows,
        eval_rows=args.test_rows,
    )
    steps_per_epoch = max(1, args.train_rows // args.batch_size)
    num_train_steps = steps_per_epoch * args.epochs

    trainer = TrainerConfig(
        id=args.run_id,
        tracker=WandbConfig(project="qwen3-useful", tags=["qwen3", "0.6b", "classifier", "discriminative", "levanter"]),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=args.batch_size,
        per_device_parallelism=args.per_device_parallelism,
        num_train_steps=num_train_steps,
        steps_per_eval=num_train_steps,  # F1 eval is post-hoc on the frozen test
        checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=15), keep=[]),
    )
    model = dataclasses.replace(
        qwen3_0_6b_hd128,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),  # match pretrained Qwen3 RoPE
        max_seq_len=args.max_seq_len,
        tokenizer=TOKENIZER,  # explicit -> avoid the HF-fetch thrash (qwen3 tokenizer thrash memory)
        reference_checkpoint=BASE_CHECKPOINT,
    )
    return TrainDecoderClassifierConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=AdamConfig(
            learning_rate=args.lr, warmup=args.warmup, weight_decay=args.weight_decay, max_grad_norm=1.0
        ),
        warm_start=not args.no_warm_start,
        warm_head=not args.no_warm_head,
        decision_token_id=DECISION_TOKEN_ID,
        hf_save_path=None,  # set by the marin out-path machinery to {output_path}/hf
    )


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--spec-id", default="high_quality")
    p.add_argument("--train-glob", default=TRAIN_GLOB)
    p.add_argument("--test-glob", default=TEST_GLOB)
    p.add_argument("--train-rows", type=int, default=200_000)
    p.add_argument("--test-rows", type=int, default=7000)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--per-device-parallelism",
        type=int,
        default=2,
        help="Per-device microbatch (grad-accum to batch-size). Keep small at long ctx to fit HBM.",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--no-warm-start", action="store_true", help="Random backbone instead of pretrained Qwen3.")
    p.add_argument("--no-warm-head", action="store_true", help="Random head instead of the NO-token warm-start.")
    p.add_argument("--tpu-type", default="v6e-4")
    p.add_argument("--region", default="us-east5", choices=sorted(REGION_BUCKETS))
    p.add_argument("--no-submit", action="store_true", help="Build + print config; do not submit.")
    p.add_argument("--smoke", action="store_true", help="Tiny validation run: 1024 rows, 4096 ctx, warm-start.")
    args = p.parse_args()

    assert_data_in_region({"train_glob": args.train_glob, "test_glob": args.test_glob}, args.region)

    if args.smoke:
        args.train_rows = min(args.train_rows, 1024)
        args.batch_size = min(args.batch_size, 16)
        args.max_seq_len = min(args.max_seq_len, 4096)  # must stay >= ~1810 to fit the spec prompt
        args.test_rows = min(args.test_rows, 512)
        args.warmup = 0

    config = build_config(args)
    output_path = f"{REGION_BUCKETS[args.region]}/checkpoints/qwen3-useful/{args.run_id}"
    logger.info("output_path: %s", output_path)
    logger.info(
        "run_id=%s rows=%d batch=%d steps=%d ctx=%d lr=%g warm_start=%s warm_head=%s tpu=%s region=%s",
        args.run_id,
        args.train_rows,
        args.batch_size,
        config.trainer.num_train_steps,
        args.max_seq_len,
        args.lr,
        not args.no_warm_start,
        not args.no_warm_head,
        args.tpu_type,
        args.region,
    )

    if args.no_submit:
        logger.info("--no-submit: built config OK, not submitting.")
        return

    pod_config = TrainClassifierOnPodConfig(
        train_config=config,
        resources=ResourceConfig.with_tpu(args.tpu_type),
        output_path=output_path,
        env_vars={
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "HF_TOKEN": os.environ["HF_TOKEN"],
        },
    )
    # run_levanter_train_* now runs the Levanter main in-process (upstream restructure);
    # submit it to the TPU pod as its own Fray job, mirroring marin.experiment.train._train_job.
    remote(run_levanter_train_decoder_classifier, name="train_decoder_classifier", resources=pod_config.resources)(
        pod_config
    )


if __name__ == "__main__":
    main()
