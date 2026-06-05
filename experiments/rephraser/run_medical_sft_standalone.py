# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native standalone SFT child for the medical Qwen3-0.6B-Base re-runs.

Runs **inside one TPU iris job** (submitted by either
`medical_extraction_sft_v2_base.py` or `medical_resiliparse_v2_base.py`).
Receives one (lr, bs, wd, warmup) cell of the Phase-1 sweep via CLI args,
detects the worker's region from MARIN_PREFIX/MARIN_REGION/GCP metadata,
resolves the tokenized cache path to the LOCAL region bucket, asserts the
cache is local (HARD fail otherwise — no silent cross-region egress), then
single-epoch fine-tunes Qwen3-0.6B-Base in-process using Levanter.

This is the analog of `experiments/scaling_law_sweeps/run_curation_train_standalone.py`
for SFT. Each child is fully self-contained — no Marin executor, no Ray, no
nested iris submit.

WHY THIS FILE EXISTS
--------------------
Background: the existing `medical_extraction_sft_v2.py` /
`medical_resiliparse_sweep.py` / `medical_extraction_v2_sweep.py` scripts ALL
hardcode `Qwen/Qwen3-0.6B` (the post-trained instruct model), even though
their docstrings and the published `code_math_medical.md` table claim
`Qwen3-0.6B-Base`. The 51.22 MMLU-Medical baseline + the ±0.3 / +2.4 deltas
in the published table are therefore on the WRONG model variant. This script
plus its two coordinator launchers re-runs the medical 0.6B sweep on the
correct `Qwen3-0.6B-Base` weights, so the size-scan story (0.6B-Base →
14B-Base) becomes internally consistent.

DATA REUSE NOTE
---------------
The TOKENIZED caches are reused as-is from the existing instruct runs because
`Qwen/Qwen3-0.6B` and `Qwen/Qwen3-0.6B-Base` ship identical `tokenizer.json`
(same vocab + BPE merges). So the bytes in the cache are bit-identical for
either model. Only the `--hf-model-name` differs at training time.

PATH HYGIENE
------------
No `gs://` paths are hardcoded in this file. The local bucket is resolved at
boot via `region_tracker.detect_current_region()` + `REGION_TO_BUCKET[region]`,
both of which read MARIN_PREFIX (or fall back to the GCP metadata server).
The only constants here are the cache directory NAMES (e74e0d / 2061a9
suffixes) — those are content-addressed artifact identifiers, not regional
paths.

Usage (normally invoked by the coordinator, but can be run by hand for debug):

    python experiments/rephraser/run_medical_sft_standalone.py \\
        --branch extraction \\
        --cache-rel-path tokenized/medical_extract-v2_qwen3-0.6b_sft-e74e0d \\
        --hf-model-name Qwen/Qwen3-0.6B-Base \\
        --learning-rate 5e-6 --batch-size 32 --weight-decay 0.01 \\
        --warmup 0.03 --config-name lr5e-6-bs32-wd0.01-wu0.03 \\
        --run-suffix qwen3-0.6b-base-rerun
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

from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.scaling_law_sweeps import region_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Where per-run summary JSONs land. Matches the curation sweep's flat layout
# (one file per completed run). Reading all results for the report rewrite is
# then `gcloud storage cat <prefix>/*.json | jq -s`. Lives in us-central1
# because it's tiny (~2 KB/run) and centrally queryable from any region.
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/medical_sft_base_results/"

# Region-lock tracker prefix. Same neutral home as the curation sweep — tiny
# files (~50 B/run), accessed once per launch, cross-region read is negligible.
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/medical_sft_base/"

# Fixed Qwen3-0.6B-Base model config. Same shape used by every other
# 0.6B-Base SFT script in this repo (`code_extraction_sft_v3_base.py`,
# `math_top3_hp_sweep.py` etc.). theta=1e6 / factor=1.0 reproduces the HF
# RoPE config; max_seq_len=4096 matches the SFT sequence length; the actual
# RoPE capacity at 32768 tokens is preserved via `hf_max_position_embeddings`
# so HF inference at long context still works after re-export.
QWEN3_0_6B_BASE_CONFIG = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Identity
    parser.add_argument(
        "--branch",
        choices=["extraction", "resiliparse"],
        required=True,
        help=(
            "Which data branch this run trains on. Affects the run name + tags only; "
            "the actual cache path is set by --cache-rel-path."
        ),
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help=(
            "HP cell identifier, e.g. 'lr5e-6-bs32-wd0.01-wu0.03'. Used in the " "run name / W&B id / checkpoint dir."
        ),
    )
    # Data
    parser.add_argument(
        "--cache-rel-path",
        required=True,
        help="Tokenized cache path RELATIVE to the region bucket, e.g. "
        "'tokenized/medical_extract-v2_qwen3-0.6b_sft-e74e0d'. Resolved at "
        "boot to '<REGION_TO_BUCKET[region]>/<cache-rel-path>/'.",
    )
    parser.add_argument(
        "--cache-step-name",
        required=True,
        help="The original tokenize-step NAME (e.g. 'medical_extract-v2_qwen3-0.6b_sft' "
        "or 'medical_resiliparse_qwen3-0.6b_sft'). Used as the components dict key "
        "in LmDataConfig so this matches what `lm_data_config(training_set=tokenized, "
        "validation_sets={})` produced for the original instruct runs (see "
        "`extraction_sft_recipe._build_sft_branch:481-488` and "
        "`marin.processing.tokenize.data_configs.step_to_lm_mixture_component:36-52`).",
    )
    parser.add_argument(
        "--cache-tokenizer",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer name BAKED INTO THE CACHE (not the model variant). Both medical "
        "caches were tokenized with 'Qwen/Qwen3-0.6B' (the cache's `.executor_info` "
        "config field). Qwen3-0.6B and Qwen3-0.6B-Base ship identical tokenizer.json so "
        "the bytes are bit-equivalent — but for 1:1 LmDataConfig parity with the prior "
        "(buggy instruct) runs we keep this string identical.",
    )
    # Model
    parser.add_argument(
        "--hf-model-name",
        default="Qwen/Qwen3-0.6B-Base",
        help="HF model name to initialize from. Default Qwen3-0.6B-Base (the whole point of these re-runs).",
    )
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument(
        "--decay",
        type=float,
        default=0.97,
        help=("LR decay floor (cosine end-LR fraction). 0.97 matches every " "other 0.6B/14B sweep in this repo."),
    )
    parser.add_argument("--lr-schedule", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=4096)
    # Bookkeeping
    parser.add_argument(
        "--run-suffix",
        default="qwen3-0.6b-base-rerun",
        help=(
            "Suffix appended to the run name to distinguish from the broken instruct runs "
            "in W&B. Always include 'base' to make the model variant unmistakable."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="marin",
        help="W&B project. We deliberately do NOT plumb entity/group — the original "
        "recipe (`extraction_sft_recipe._run_single_epoch_sft:341-344`) sets only "
        "project + tags. We match that to keep the W&B-side surface 1:1; rely on "
        "tag filtering to separate the rerun from the buggy instruct runs.",
    )
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--tracker-prefix", default=DEFAULT_TRACKER_PREFIX)
    parser.add_argument("--tpu-type", default=None, help="Override TPU type. If unset, infer from IRIS_DEVICE_VARIANT.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Local-path safety
# ---------------------------------------------------------------------------
def _assert_cache_local(cache_dir: str, region: str) -> None:
    """HARD invariant: the cache_dir must live in the local region's bucket.

    Mirrors `_assert_all_components_local` from `run_curation_train_standalone.py`.
    Runs BEFORE any tensorstore open so a misconfiguration is caught loudly
    instead of silently paying egress.
    """
    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    if not cache_dir.startswith(expected_prefix):
        raise ValueError(
            f"cache_dir={cache_dir!r} is not in the local region's bucket "
            f"(expected prefix {expected_prefix!r}). Cross-region reads forbidden — "
            f"pre-copy the cache into {expected_prefix} first."
        )


# ---------------------------------------------------------------------------
# Token count
# ---------------------------------------------------------------------------
def _read_token_count(cache_dir: str) -> int:
    """Read total_tokens from `<cache_dir>/train/.stats.json`.

    Same convention as `extraction_sft_recipe._read_token_count`. We require
    the top-level `total_tokens` key to be set; the per-shard fallback is
    not necessary here because all of our medical caches were written by a
    single `default_tokenize` step that writes the aggregate stats.
    """
    stats_path = f"{cache_dir.rstrip('/')}/train/.stats.json"
    with fsspec.open(stats_path, "r") as f:
        stats = json.load(f)
    if not stats.get("total_tokens"):
        raise ValueError(f"No total_tokens in {stats_path}")
    return int(stats["total_tokens"])


# ---------------------------------------------------------------------------
# Data config
# ---------------------------------------------------------------------------
def _build_data_config(cache_dir: str, cache_step_name: str, cache_tokenizer: str) -> LmDataConfig:
    """Build a single-source LmDataConfig that is 1:1 with what the existing
    instruct runs produced via `lm_data_config(training_set=tokenized, validation_sets={})`.

    We replicate, rather than call, that helper because the helper requires a
    Marin `ExecutorStep` and we are deliberately bypassing the Marin executor
    (Iris-native, no Ray). Field-by-field this matches:

      - `marin.processing.tokenize.data_configs.lm_data_config:55-98`
        (which ultimately calls `lm_mixture_data_config:101-156`)
      - `marin.processing.tokenize.data_configs.step_to_lm_mixture_component:36-52`
        (which produces the inner `DatasetComponent`)
      - `marin.processing.tokenize.tokenize.TokenizeConfig.as_lm_dataset_source_config:115-135`
        (which produces the `UrlDatasetSourceConfig` inside the `DatasetComponent`)

    Concretely the original instruct run's LmDataConfig had:
      - components = {"<cache-step-name>": DatasetComponent(source=UrlDatasetSourceConfig(...), cache_dir, format, tags)}
      - train_weights = {"<cache-step-name>": 1.0}
      - tokenizer = "Qwen/Qwen3-0.6B" (the tokenize step's tokenizer string)
      - shuffle=True, permutation_type="feistel", block_cross_document_attention=True,
        shuffle_before_trainval_split=True, cache_dir=None
      - train_urls left empty here because (a) the cache is already built so
        Levanter never re-tokenizes, (b) the original train_urls point at
        us-central1 raw paths and we want zero risk of cross-region read from a
        worker landed in us-east5. `_assert_cache_local` enforces the latter
        regardless.
    """
    cache_str = cache_dir.rstrip("/") + "/"

    # Step 1: the source — same shape as TokenizeConfig.as_lm_dataset_source_config
    # would have produced, with empty raw URLs (see above for rationale).
    source = UrlDatasetSourceConfig(
        tags=[],  # original TokenizeConfig.tags was [] per the cache's .executor_info
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_str,
        format=TextLmDatasetFormat(),
    )

    # Step 2: the DatasetComponent — same shape as step_to_lm_mixture_component output.
    component = DatasetComponent(
        source=source,
        cache_dir=source.cache_dir,
        format=source.format,
        tags=source.tags,
    )

    # Step 3: the LmDataConfig — same shape as lm_mixture_data_config output for
    # a single training component with weight 1.0 and no validation sets.
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


# ---------------------------------------------------------------------------
# Train config
# ---------------------------------------------------------------------------
def _build_train_lm_config(args: argparse.Namespace, data_config: LmDataConfig, num_train_steps: int) -> TrainLmConfig:
    """Build the inner Levanter TrainLmConfig with `initialize_from_hf` set.

    1:1 PARITY with `extraction_sft_recipe._run_single_epoch_sft` (lines
    338-381). Field-by-field equivalent — the ONLY thing the new runs do
    differently is point at `Qwen/Qwen3-0.6B-Base` for `initialize_from_hf`
    instead of `Qwen/Qwen3-0.6B`. Every knob below is intentionally identical:

      - `mp=jmp.get_policy("p=f32,c=bfloat16")` — same precision policy
      - `steps_per_eval=min(50, num_train_steps)` — same eval cadence
      - `save_interval=timedelta(minutes=10)`, `keep=[]` — same checkpoint
        policy (rolling 10-min restart point, NO permanent intermediate
        checkpoints saved to GCS)
      - `hf_save_steps=num_train_steps` — exactly one HF export at the final step
      - `allow_nondivisible_batch_size=True` — same; required for some HP cells
        where bs doesn't divide the dataset cleanly
      - `pad_tokenizer_to_match_model=True` — same; safe-no-op when the cache
        tokenizer matches the model (which it does — Qwen3-0.6B and -0.6B-Base
        share tokenizer.json)
      - WandbConfig: project="marin" + tags only, NO entity/group — matches
        the original recipe at extraction_sft_recipe.py:341-344 exactly. We
        rely on tag filtering in the W&B UI to separate the rerun from the
        buggy instruct runs (the `rerun=base-fix` and `model=qwen3-0.6b-base`
        tags below).

    If you change anything in this function, mirror the change in
    `extraction_sft_recipe._run_single_epoch_sft` — they MUST stay in sync to
    preserve the apples-to-apples comparison.
    """
    # Tags follow the existing instruct-run convention
    # (`medical_extraction_v2_sweep.py:114`: ("medical", "extractv2-sweep-p1",
    # config_name, "sft", "qwen3-0.6b")) plus two new tags that make the
    # bug-fix re-run unmistakable in W&B: `model=qwen3-0.6b-base` and
    # `rerun=base-fix`. Cell-level HP is encoded in the tags so dashboard
    # filtering works without an entity/group.
    tags = [
        "medical",
        f"medical-{args.branch}-base-rerun-p1",
        args.config_name,
        "sft",
        "qwen3-0.6b-base",
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
            tracker=WandbConfig(
                project=args.wandb_project,
                tags=tags,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=args.batch_size,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[],
            ),
            allow_nondivisible_batch_size=True,
            initialize_from=None,
        ),
        train_seq_len=args.seq_len,
        model=QWEN3_0_6B_BASE_CONFIG,
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
    # `initialize_from_hf` is what differentiates this re-run from the buggy
    # instruct runs — the only intentional divergence in this whole file.
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
    """Detect TPU type from iris's `IRIS_DEVICE_VARIANT` env var.

    Single-host shapes only for medical 0.6B-Base — v5p-8 / v4-8 / v6e-4 are
    all vm_count=1 so no multi-host JAX init is needed.
    """
    if override:
        return override
    for env_var in ("IRIS_DEVICE_VARIANT", "TPU_TYPE", "ACCELERATOR_TYPE"):
        if env_var in os.environ:
            return os.environ[env_var]
    logger.warning("Could not detect TPU type from env; defaulting to v5p-8")
    return "v5p-8"


def _resolve_local_cache_dir(cache_rel_path: str, region: str) -> str:
    """Compose the LOCAL gs:// cache path from the region's bucket + rel-path.

    No hardcoded `gs://marin-{region}` literals — `REGION_TO_BUCKET` is the
    canonical map (sourced from `rigging.filesystem.REGION_TO_DATA_BUCKET`)
    and handles the `europe-west4 → marin-eu-west4` short-form correctly.
    """
    bucket = region_tracker.REGION_TO_BUCKET[region]  # e.g. "gs://marin-us-east5"
    return f"{bucket}/{cache_rel_path.strip('/')}"


def _build_run_name(args: argparse.Namespace) -> str:
    """Compose the run name from branch + config + suffix.

    The suffix in particular keeps the buggy instruct runs and the corrected
    Base runs cleanly separated in both W&B and GCS.

    Example: 'medical-extraction-lr5e-6-bs32-wd0.01-wu0.03-qwen3-0.6b-base-rerun'
    """
    parts = [f"medical-{args.branch}", args.config_name]
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
    """Write a flat per-run summary JSON to the central results prefix.

    Cheap atomic record of which (lr, bs, wd, warmup, branch, model_variant)
    combination was actually run, where it landed, and how many steps/tokens
    it saw. The report-rewrite step (`task #8`) reads these to repopulate the
    medical 0.6B numbers in the table without scraping W&B by hand.
    """
    payload = {
        "run_name": run_name,
        "branch": args.branch,
        "config_name": args.config_name,
        "model_variant": "Qwen3-0.6B-Base",
        "hf_model_name": args.hf_model_name,
        "cache_rel_path": args.cache_rel_path,
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
    except Exception as e:  # non-fatal — training succeeded, summary write is best-effort
        logger.warning("Failed to write summary at %s: %s", path, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    # 1. Region detection (parses MARIN_PREFIX/MARIN_REGION/GCP metadata).
    region = region_tracker.detect_current_region()
    run_name = _build_run_name(args)
    logger.info("Standalone medical SFT child boot — region=%s, run=%s", region, run_name)

    # 2. Region-lock the run. Pins this run to whichever region first wrote
    #    the tracker file. Subsequent re-launches resume in the same region
    #    so checkpoint reads stay local. Returns the local bucket as a string.
    bucket = region_tracker.resolve_checkpoint_prefix(
        method_name=f"medical_{args.branch}",
        experiment_tag="base_rerun",
        run_name=run_name,
        local_region=region,
        tracker_prefix=args.tracker_prefix,
    )
    logger.info("Region-locked: %s -> %s", run_name, bucket)
    output_path = f"{bucket}/checkpoints/medical-sft-base/{run_name}"
    logger.info("Checkpoint output_path: %s", output_path)

    # 3. Resolve cache path locally and assert no cross-region.
    cache_dir = _resolve_local_cache_dir(args.cache_rel_path, region)
    _assert_cache_local(cache_dir, region)
    logger.info("Local cache: %s", cache_dir)

    # 4. Determine num_train_steps from the cache's token count (single epoch).
    total_tokens = _read_token_count(cache_dir)
    num_train_steps = math.ceil(total_tokens / (args.batch_size * args.seq_len))
    logger.info(
        "Single-epoch SFT: tokens=%d, batch=%d, seq=%d -> %d steps",
        total_tokens,
        args.batch_size,
        args.seq_len,
        num_train_steps,
    )

    # 5. Build data + train configs.
    data_config = _build_data_config(
        cache_dir,
        cache_step_name=args.cache_step_name,
        cache_tokenizer=args.cache_tokenizer,
    )
    train_lm_config = _build_train_lm_config(args, data_config, num_train_steps)

    # 6. Wrap in TrainLmOnPodConfig — needed only for `_prepare_training_run`'s
    #    env-var setup; we do NOT call run_levanter_train_lm (that submits a
    #    nested Ray job, which is exactly what we're moving away from).
    tpu_type = _detect_local_tpu_type(override=args.tpu_type)
    pod_config = TrainLmOnPodConfig(
        train_config=train_lm_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=output_path,
        env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=16000"},
    )

    # 7. Prepare env and run Levanter in-process.
    _prepared, train_config_ready, env, _extras = _prepare_training_run(pod_config)
    for k, v in env.items():
        os.environ[k] = v

    # Single-host (vm_count=1) so libtpu's PJRT mesh is enough — but we still
    # call jax.distributed.initialize() with no args because Levanter's
    # WandbConfig.init uses `multihost_broadcast_sync` which needs the
    # distributed client to exist. The no-args form is a no-op when no
    # MEGASCALE_* env vars are present, so it's safe on single-host too.
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

    # 8. Write per-run summary + DONE marker.
    _write_summary(
        args=args,
        region=region,
        run_name=run_name,
        output_path=output_path,
        num_train_steps=num_train_steps,
        total_tokens=total_tokens,
    )
    done_marker = f"{output_path}/.medical_sft_base_DONE"
    try:
        with fsspec.open(done_marker, "w") as f:
            f.write(
                json.dumps(
                    {
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "run_name": run_name,
                        "branch": args.branch,
                        "config_name": args.config_name,
                        "region": region,
                        "model_variant": "Qwen3-0.6B-Base",
                    }
                )
            )
        logger.info("Wrote completion marker: %s", done_marker)
    except Exception as e:
        logger.warning("Failed to write completion marker at %s: %s", done_marker, e)


if __name__ == "__main__":
    main()
