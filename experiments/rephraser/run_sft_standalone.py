# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native standalone SFT child — generalized for any Qwen3 size + domain.

Generalization of `run_medical_sft_standalone.py`. The medical-specific
file is kept (frozen) for the in-flight 0.6B-Base re-run; this one is the
forward-looking implementation that supports 0.6B / 8B / 14B Base across
all three domains (code / math / medical). Used by all the 8B-Base
coordinators (and eventually by the 0.6B-Base coordinators once we
deprecate the medical-specific child).

WHY THIS FILE EXISTS
--------------------
The same SFT logic — initialize from HF Base, single-epoch from a
pre-tokenized cache, write one HF export at the end — applies to every
domain x size combination in this paper's revised plan:

  - Medical 0.6B-Base re-run (the bug-fix that started this whole branch)
  - Code 8B-Base, Math 8B-Base, Medical 8B-Base (the new sweep)
  - eventually 14B-Base re-runs if we ever need them on Iris

Rather than clone the medical standalone six times, this version takes
`--model-size` (0.6b / 8b / 14b) and `--domain` (code / math / medical) on
the CLI and resolves the right Qwen3Config + run-name prefix at boot.

1:1 PARITY WITH THE EXISTING SFT RECIPE
---------------------------------------
The TrainLmConfig built here is field-by-field identical to what
`extraction_sft_recipe._run_single_epoch_sft:340-381` would produce for
the same (model, lr, bs, wd, warmup) cell — same tags shape, same
WandbConfig (project=marin, no entity/group), same checkpointer policy
(rolling 10-min, keep=[]), same hf_save_steps, same `pad_tokenizer_to_match_model=True`.
The ONLY intentional divergences are:
  - `initialize_from_hf` is set to a `*-Base` reference (whole point)
  - the local `tokenizer` reference is updated to the padded converter's
    tokenizer post-pad (Levanter patch in `train_lm.main`, see comments there)

PATH HYGIENE
------------
No `gs://` paths are hardcoded for the cache or the model. Region is
detected at boot via `region_tracker.detect_current_region()` (reads
MARIN_PREFIX / MARIN_REGION / GCP metadata fallback). Cache + checkpoint
paths are composed from `REGION_TO_BUCKET[region]` + the rel-path passed
on the CLI. `_assert_cache_local` HARD-fails on any cross-region read.

USAGE
-----
Normally invoked by a coordinator (e.g. `code_extraction_sft_v3_base_8b.py`)
but can be run by hand for debug:

    python experiments/rephraser/run_sft_standalone.py \\
        --domain code --branch extraction \\
        --model-size 8b \\
        --cache-rel-path tokenized/code_extract-commented_qwen3-0.6b-base_sft-e52621 \\
        --cache-step-name code_extract-commented_qwen3-0.6b-base_sft \\
        --hf-model-name Qwen/Qwen3-8B-Base \\
        --learning-rate 2e-6 --batch-size 32 \\
        --weight-decay 0.05 --warmup 0.0 \\
        --config-name lr2e-6_bs32_wd0.05_wu0
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import math
import os
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import TrainLmOnPodConfig, _prepare_training_run

from experiments.qwen3 import qwen3_0_6b_hd128, qwen3_8b_base, qwen3_14b
from experiments.scaling_law_sweeps import region_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/sft_base_results/"
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/sft_base/"


# Per-size base config + RoPE/seq_len overrides. The convention (matching
# every other 0.6B/14B SFT file in this repo) is to take the canonical
# Qwen3Config from `experiments/qwen3.py` and apply:
#   - rope = DefaultRotaryEmbeddingsConfig(theta=1e6, factor=1.0)  # HF default
#   - max_seq_len = 4096                                            # SFT context
#   - hf_max_position_embeddings = 32768                            # preserve full RoPE
# capacity for HF re-export). The `_with_rope` variants below match
# `code_extraction_sft_v3_base.qwen3_0_6b_hd128_with_rope` and
# `code_extraction_sft_v3_14b_sweep.qwen3_14b_with_rope` exactly.
_QWEN3_0_6B_BASE_CONFIG = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)
_QWEN3_8B_BASE_CONFIG = dataclasses.replace(
    qwen3_8b_base,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)
_QWEN3_14B_BASE_CONFIG = dataclasses.replace(
    qwen3_14b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

_MODEL_SIZE_TO_CONFIG = {
    "0.6b": _QWEN3_0_6B_BASE_CONFIG,
    "8b": _QWEN3_8B_BASE_CONFIG,
    "14b": _QWEN3_14B_BASE_CONFIG,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Identity
    parser.add_argument(
        "--domain",
        choices=["code", "math", "medical"],
        required=True,
        help="Used in the run name + tags + output_path prefix.",
    )
    parser.add_argument(
        "--branch",
        choices=["extraction", "resiliparse"],
        required=True,
    )
    parser.add_argument(
        "--model-size",
        choices=list(_MODEL_SIZE_TO_CONFIG),
        required=True,
        help="Maps to the canonical Qwen3Config in experiments/qwen3.py with SFT-appropriate RoPE overrides.",
    )
    parser.add_argument("--config-name", required=True)
    # Data
    parser.add_argument("--cache-rel-path", required=True)
    parser.add_argument("--cache-step-name", required=True)
    parser.add_argument(
        "--cache-tokenizer",
        default="Qwen/Qwen3-0.6B-Base",
        help=(
            "Tokenizer to use at training time. Default Qwen/Qwen3-0.6B-Base, which "
            "is byte-identical to all other Qwen3 *-Base tokenizer.json files (sha "
            "c0382117ea329cdf). Pad_tokenizer_to_match_model brings its len up to "
            "the model's vocab_size."
        ),
    )
    # Model
    parser.add_argument(
        "--hf-model-name",
        required=True,
        help="HF reference checkpoint, e.g. Qwen/Qwen3-8B-Base.",
    )
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("--lr-schedule", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=4096)
    # Bookkeeping
    parser.add_argument(
        "--run-suffix",
        default="qwen3-base-rerun",
        help="Suffix on every run name. Distinguishes new runs from prior buggy ones.",
    )
    parser.add_argument("--wandb-project", default="marin")
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--tracker-prefix", default=DEFAULT_TRACKER_PREFIX)
    parser.add_argument("--tpu-type", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Local-path safety
# ---------------------------------------------------------------------------
def _assert_cache_local(cache_dir: str, region: str) -> None:
    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    if not cache_dir.startswith(expected_prefix):
        raise ValueError(
            f"cache_dir={cache_dir!r} is not in the local region's bucket "
            f"(expected prefix {expected_prefix!r}). Cross-region reads forbidden."
        )


def _read_token_count(cache_dir: str) -> int:
    stats_path = f"{cache_dir.rstrip('/')}/train/.stats.json"
    with fsspec.open(stats_path, "r") as f:
        stats = json.load(f)
    if not stats.get("total_tokens"):
        raise ValueError(f"No total_tokens in {stats_path}")
    return int(stats["total_tokens"])


def _resolve_local_cache_dir(cache_rel_path: str, region: str) -> str:
    bucket = region_tracker.REGION_TO_BUCKET[region]
    return f"{bucket}/{cache_rel_path.strip('/')}"


# ---------------------------------------------------------------------------
# Data + train config builders
# ---------------------------------------------------------------------------
def _build_data_config(cache_dir: str, cache_step_name: str, cache_tokenizer: str) -> LmDataConfig:
    """Reproduce `lm_data_config(training_set=tokenized, validation_sets={})` exactly.

    See run_medical_sft_standalone for the longer explanation. Field-by-field
    matches what `step_to_lm_mixture_component` would produce for a single
    training cache + no validation sets.
    """
    cache_str = cache_dir.rstrip("/") + "/"
    source = UrlDatasetSourceConfig(
        tags=[],
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_str,
        format=TextLmDatasetFormat(),
    )
    component = DatasetComponent(
        source=source,
        cache_dir=source.cache_dir,
        format=source.format,
        tags=source.tags,
    )
    return LmDataConfig(
        components={cache_step_name: component},
        train_weights={cache_step_name: 1.0},
        tokenizer=cache_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
        shuffle_before_trainval_split=True,
    )


def _build_train_lm_config(args: argparse.Namespace, data_config: LmDataConfig, num_train_steps: int) -> TrainLmConfig:
    """Build the TrainLmConfig 1:1 with `_run_single_epoch_sft`."""
    model_config = _MODEL_SIZE_TO_CONFIG[args.model_size]

    tags = [
        args.domain,
        f"{args.domain}-{args.branch}-base-rerun-p1",
        args.config_name,
        "sft",
        f"qwen3-{args.model_size}-base",
        "rerun=base-fix",
        f"branch={args.branch}",
        f"lr={args.learning_rate}",
        f"bs={args.batch_size}",
        f"wd={args.weight_decay}",
        f"warmup={args.warmup}",
    ]

    inner = TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(project=args.wandb_project, tags=tags),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=args.batch_size,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10), keep=[]),
            allow_nondivisible_batch_size=True,
            initialize_from=None,
        ),
        train_seq_len=args.seq_len,
        model=model_config,
        optimizer=AdamConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup=args.warmup,
            decay=args.decay,
            lr_schedule=args.lr_schedule,
            max_grad_norm=args.max_grad_norm,
        ),
        hf_save_steps=num_train_steps,
    )
    inner = dataclasses.replace(
        inner,
        initialize_from_hf=args.hf_model_name,
        pad_tokenizer_to_match_model=True,
    )
    return inner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_local_tpu_type(override: str | None = None) -> str:
    if override:
        return override
    for env_var in ("IRIS_DEVICE_VARIANT", "TPU_TYPE", "ACCELERATOR_TYPE"):
        if env_var in os.environ:
            return os.environ[env_var]
    logger.warning("Could not detect TPU type; defaulting to v5p-8")
    return "v5p-8"


def _build_run_name(args: argparse.Namespace) -> str:
    parts = [args.domain, args.branch, args.config_name, f"qwen3-{args.model_size}-base"]
    if args.run_suffix.strip():
        parts.append(args.run_suffix.strip())
    return "-".join(parts)


def _write_summary(
    *,
    args: argparse.Namespace,
    region: str,
    run_name: str,
    output_path: str,
    num_train_steps: int,
    total_tokens: int,
) -> None:
    payload = {
        "run_name": run_name,
        "domain": args.domain,
        "branch": args.branch,
        "config_name": args.config_name,
        "model_size": args.model_size,
        "model_variant": f"Qwen3-{args.model_size}-Base",
        "hf_model_name": args.hf_model_name,
        "cache_rel_path": args.cache_rel_path,
        "cache_tokenizer": args.cache_tokenizer,
        "region": region,
        "output_path": output_path,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup": args.warmup,
            "decay": args.decay,
            "lr_schedule": args.lr_schedule,
            "max_grad_norm": args.max_grad_norm,
            "seq_len": args.seq_len,
        },
        "tokens": {"total_tokens": total_tokens, "num_train_steps": num_train_steps},
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    path = f"{args.results_prefix.rstrip('/')}/{run_name}.json"
    try:
        with fsspec.open(path, "w") as f:
            f.write(json.dumps(payload, indent=2))
        logger.info("Wrote run summary: %s", path)
    except Exception as e:
        logger.warning("Failed to write summary at %s: %s", path, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    region = region_tracker.detect_current_region()
    run_name = _build_run_name(args)
    logger.info("SFT child boot — region=%s, run=%s", region, run_name)

    bucket = region_tracker.resolve_checkpoint_prefix(
        method_name=f"{args.domain}_{args.branch}",
        experiment_tag=f"base_rerun_{args.model_size}",
        run_name=run_name,
        local_region=region,
        tracker_prefix=args.tracker_prefix,
    )
    output_path = f"{bucket}/checkpoints/sft-base/{run_name}"
    logger.info("Checkpoint output_path: %s", output_path)

    cache_dir = _resolve_local_cache_dir(args.cache_rel_path, region)
    _assert_cache_local(cache_dir, region)
    logger.info("Local cache: %s", cache_dir)

    total_tokens = _read_token_count(cache_dir)
    num_train_steps = math.ceil(total_tokens / (args.batch_size * args.seq_len))
    logger.info(
        "Single-epoch SFT: tokens=%d, batch=%d, seq=%d -> %d steps",
        total_tokens,
        args.batch_size,
        args.seq_len,
        num_train_steps,
    )

    data_config = _build_data_config(
        cache_dir,
        cache_step_name=args.cache_step_name,
        cache_tokenizer=args.cache_tokenizer,
    )
    train_lm_config = _build_train_lm_config(args, data_config, num_train_steps)

    tpu_type = _detect_local_tpu_type(override=args.tpu_type)
    pod_config = TrainLmOnPodConfig(
        train_config=train_lm_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=output_path,
        env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=16000"},
    )

    _prepared, train_config_ready, env, _extras = _prepare_training_run(pod_config)
    for k, v in env.items():
        os.environ[k] = v

    try:
        import jax as _jax

        _jax.distributed.initialize()
        logger.info(
            "jax.distributed initialized (process_count=%d, process_index=%d)",
            _jax.process_count(),
            _jax.process_index(),
        )
    except Exception as exc:
        logger.warning("jax.distributed.initialize() raised: %s. Continuing.", exc)

    import importlib

    train_lm_module = importlib.import_module("levanter.main.train_lm")
    logger.info("Launching levanter.main.train_lm.main() in-process")
    train_lm_module.main(train_config_ready)
    logger.info("Training finished cleanly.")

    _write_summary(
        args=args,
        region=region,
        run_name=run_name,
        output_path=output_path,
        num_train_steps=num_train_steps,
        total_tokens=total_tokens,
    )
    done_marker = f"{output_path}/.sft_base_DONE"
    try:
        with fsspec.open(done_marker, "w") as f:
            f.write(
                json.dumps(
                    {
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "run_name": run_name,
                        "domain": args.domain,
                        "branch": args.branch,
                        "model_size": args.model_size,
                        "config_name": args.config_name,
                        "region": region,
                    }
                )
            )
        logger.info("Wrote DONE marker: %s", done_marker)
    except Exception as e:
        logger.warning("Failed to write DONE marker: %s", e)


if __name__ == "__main__":
    main()
