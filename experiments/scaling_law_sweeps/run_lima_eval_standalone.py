# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone LIMA eval runner: load a trained checkpoint, compute LIMA loss.

Given a run_name + region, this script:

1. Parses the run_name to recover the Llama model config (hidden, layers, heads,
   d_ff) using the same `completed_adamh_heuristic._build_model_config` helper
   that training uses. Exact structural match to the saved checkpoint.

2. Discovers the final checkpoint under
   `gs://marin-<region>/checkpoints/isoflop-curation/<run_name>/checkpoints/step-*`
   (picks the highest step number — typically one final-checkpoint per run).

3. Loads the LIMA validation cache at
   `gs://marin-<region>/tokenized/lima_text-<LIMA_HASH>/`.

4. Runs a single forward pass over LIMA using Levanter's TaggedEvaluator and
   writes the resulting loss to
   `<results-prefix>/<run_name>_lima_eval.json`.

The script is intentionally self-contained so it can be launched as a one-shot
Iris child per run — no Marin Executor, no Zephyr coordinator. Just boot,
load, eval, write.

Usage (inside an Iris TPU job, region=checkpoint's region):

    python experiments/scaling_law_sweeps/run_lima_eval_standalone.py \\
        --run-name curation-dclm-expFM_natural-3e+20-d1536-L16-B256 \\
        --region us-central1

Output file (merged/side-car pattern so we don't touch the training summary):

    gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/
        curation-dclm-expFM_natural-3e+20-d1536-L16-B256.json
      -> {
           "run_name": "...",
           "region": "us-central1",
           "checkpoint_step": 47517,
           "lima_cache_hash": "41ca0d",
           "eval/lima/loss": 3.14,
           "eval/lima/bpb": 0.72,
           "total_eval_tokens": 652669,
           "completed_at": "2026-04-21T20:00:00Z",
         }

The plotter loads both the training summary and this side-car, merges eval
dicts, so `eval/lima/loss` becomes plottable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

# Canonical LIMA tokenized cache hash — must match
# `_LIMA_CACHE_HASH` in data_curation_math.py. Update both together.
LIMA_CACHE_HASH: str = "41ca0d"

# Where to land the per-run eval result. Kept separate from the training
# summaries so we don't race with training-time summary writes.
DEFAULT_OUTPUT_PREFIX: str = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/"


def parse_run_name(run_name: str) -> dict[str, Any]:
    """Recover plan fields from a run_name.

    Accepted shapes (produced by `curation_plan.PlannedRun.run_name_core`):
        curation-<method>-expFM_natural-<budget>-d<hidden>-L<layers>-B<batch>
        curation-<method>-expFM_natural-<budget>-d<hidden>-L<layers>-B<batch>-canonical

    The method name itself can contain underscores (e.g. "nemotron_full",
    "llm_curated_bos_fixed"), so we don't split greedily — we regex-match the
    trailing `-d<>-L<>-B<>` structure and anchor from there.
    """
    m = re.match(
        r"^curation-(?P<method>.+?)-expFM_natural-(?P<budget>[0-9.e+-]+)"
        r"-d(?P<hidden>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)(?P<suffix>-canonical)?$",
        run_name,
    )
    if not m:
        raise ValueError(f"Could not parse run_name={run_name!r}; unexpected shape.")
    return {
        "method": m.group("method"),
        "budget": float(m.group("budget")),
        "hidden": int(m.group("hidden")),
        "layers": int(m.group("layers")),
        "batch": int(m.group("batch")),
        "suffix": m.group("suffix") or "",
    }


def discover_checkpoint(prefix: str, run_name: str) -> tuple[str, int]:
    """Return (checkpoint_dir, step_number) for the final checkpoint.

    `prefix` is the region-local bucket (resolved from `MARIN_PREFIX` at the
    caller). We read the checkpoint under `{prefix}/checkpoints/isoflop-curation/<run>`.

    Raises FileNotFoundError if no step-* subdir exists.
    """
    root = f"{prefix.rstrip('/')}/checkpoints/isoflop-curation/{run_name}/checkpoints/"
    fs, _ = fsspec.core.url_to_fs(root)
    if not fs.exists(root):
        raise FileNotFoundError(
            f"No checkpoint root at {root}. MARIN_PREFIX={prefix!r} does NOT contain "
            f"the checkpoint for run {run_name!r}. This usually means the worker landed "
            f"in a different region than where the checkpoint was saved. "
            f"Fix the launcher's region constraint; aborting to avoid cross-region reads."
        )
    entries = [e for e in fs.ls(root) if "/step-" in e]
    if not entries:
        raise FileNotFoundError(f"No step-* checkpoint under {root}")
    steps = []
    for e in entries:
        mm = re.search(r"step-(\d+)/?$", e)
        if mm:
            steps.append((int(mm.group(1)), e))
    steps.sort()
    final_step, final_path = steps[-1]
    # fsspec.ls strips scheme for gs:// sometimes; re-prefix from the caller's bucket.
    if "://" in prefix:
        scheme = prefix.split("://", 1)[0]
        if not final_path.startswith(scheme + "://"):
            final_path = f"{scheme}://{final_path.lstrip('/')}"
    return final_path.rstrip("/"), final_step


def _build_eval_config(
    *,
    run_name: str,
    meta: dict[str, Any],
    checkpoint_path: str,
    lima_cache: str,
) -> Any:
    """Assemble a Levanter `EvalLmConfig` for LIMA eval.

    This mirrors the relevant pieces of `run_curation_train_standalone.py`
    but strips the training parts: no optimizer, no training data, just the
    LIMA cache as a single tagged eval set.
    """
    from levanter.data.text import (
        DatasetComponent,
        LMMixtureDatasetConfig,
        TextLmDatasetFormat,
        UrlDatasetSourceConfig,
    )
    from levanter.main.eval_lm import EvalLmConfig
    from levanter.trainer import TrainerConfig

    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    # Build the Llama model config exactly how training built it.
    model_config = completed_adamh_heuristic._build_model_config(meta["hidden"], seq_len=4096)

    lima_source = UrlDatasetSourceConfig(
        cache_dir=lima_cache,
        train_urls=[],
        validation_urls=[],
        format=TextLmDatasetFormat(),
        tags=["lima"],
    )
    components = {
        "lima": DatasetComponent(
            source=lima_source,
            cache_dir=lima_cache,
            format=TextLmDatasetFormat(),
            tags=["lima"],
        )
    }
    mixture = LMMixtureDatasetConfig(
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        components=components,
        # Training weight 0.0 → Marin convention that zero-weight components are
        # validation-only; Levanter's `_has_nonzero_weight` excludes from train.
        train_weights={"lima": 0.0},
        shuffle=False,
        permutation_type="feistel",
    )

    # Keep the eval batch small; LIMA is ~1330 docs. TrainerConfig defaults
    # are fine for single-host.
    trainer = TrainerConfig(
        checkpointer=None,
        num_train_steps=1,  # required field, not used
        train_batch_size=8,
        per_device_eval_parallelism=-1,  # auto
    )

    return EvalLmConfig(
        checkpoint_path=checkpoint_path,
        trainer=trainer,
        data=mixture,
        max_eval_length=4096,
        model=model_config,
    )


def _write_result(output_prefix: str, run_name: str, payload: dict) -> str:
    out_dir = output_prefix.rstrip("/")
    out_path = f"{out_dir}/{run_name}.json"
    with fsspec.open(out_path, "w") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _enforce_local_read(prefix: str, path: str, *, what: str) -> None:
    """Fail fast if `path` points outside the region-local bucket.

    `prefix` is the worker's MARIN_PREFIX (e.g. `gs://marin-us-central1`),
    which Iris sets automatically based on where the job was scheduled.
    Every path we read must live under that prefix — reading from another
    bucket would incur cross-region egress.
    """
    if not path.startswith(prefix.rstrip("/") + "/"):
        raise AssertionError(
            f"CROSS-REGION SAFEGUARD TRIPPED: {what} path {path!r} is not under the "
            f"worker's MARIN_PREFIX {prefix!r}. Aborting to avoid egress. "
            f"This is a bug in the launcher/eval script."
        )


def _preflight(prefix: str, checkpoint_path: str, lima_cache: str) -> None:
    """Verify both required inputs exist under MARIN_PREFIX before loading
    the model. Fails fast with a clear message if anything is off."""
    import fsspec

    _enforce_local_read(prefix, checkpoint_path, what="checkpoint")
    _enforce_local_read(prefix, lima_cache, what="LIMA cache")

    fs_ckpt, _ = fsspec.core.url_to_fs(checkpoint_path)
    if not fs_ckpt.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    fs_lima, _ = fsspec.core.url_to_fs(lima_cache)
    lima_stats = f"{lima_cache.rstrip('/')}/validation/.stats.json"
    if not fs_lima.exists(lima_stats):
        raise FileNotFoundError(
            f"LIMA validation stats not found at {lima_stats}. "
            f"LIMA cache must be tokenized under MARIN_PREFIX={prefix!r} before this eval can run."
        )
    logger.info("Preflight OK: checkpoint + LIMA cache both present under %s.", prefix)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--lima-hash", default=LIMA_CACHE_HASH)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    # MARIN_PREFIX is auto-set by Iris to the local region's bucket based on
    # where the worker landed. All reads must stay under it; the only
    # cross-region write allowed is the final result JSON to `--output-prefix`.
    from rigging.filesystem import marin_prefix

    prefix = marin_prefix()
    meta = parse_run_name(args.run_name)
    lima_cache = f"{prefix.rstrip('/')}/tokenized/lima_text-{args.lima_hash}/"
    checkpoint_path, final_step = discover_checkpoint(prefix, args.run_name)

    logger.info("Eval plan:")
    logger.info("  run_name:      %s", args.run_name)
    logger.info("  parsed meta:   %s", meta)
    logger.info("  MARIN_PREFIX:  %s", prefix)
    logger.info("  checkpoint:    %s  (step %d)", checkpoint_path, final_step)
    logger.info("  lima cache:    %s", lima_cache)
    logger.info("  output prefix: %s  (ONLY allowed cross-region write)", args.output_prefix)

    # Safeguards: block cross-region reads before we spend TPU time.
    _preflight(prefix, checkpoint_path, lima_cache)

    if args.dry_run:
        logger.info("--dry-run: stopping before model load.")
        return

    # Build config and delegate to Levanter's eval_lm.main.
    config = _build_eval_config(
        run_name=args.run_name,
        meta=meta,
        checkpoint_path=checkpoint_path,
        lima_cache=lima_cache,
    )

    # Capture loss via tracker → we run eval_lm.main and harvest the per-tag
    # loss from the evaluator's log_dict. To keep the surface small we write
    # directly from within a thin fork of eval_lm.main: load model, evaluate,
    # write sidecar.
    # Rather than redo everything, defer to eval_lm's main but also capture
    # its tracker output by using an in-memory tracker.

    import jax
    import haliax as hax
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning
    from levanter.checkpoint import load_checkpoint
    from levanter.eval import LossFnOutput, TaggedEvaluator, eval_model
    from levanter.models.lm_model import LmHeadModel, LmExample
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode
    import equinox as eqx
    import jmp
    import jax.numpy as jnp
    import levanter

    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = config.trainer.EvalBatch
    Pos = config.model.max_Pos.resize(config.max_eval_length)

    datasets = config.data.tagged_eval_sets(Pos)
    if not datasets:
        raise ValueError("LIMA eval set is empty — check the cache path!")

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.use_device_mesh():
        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        mp: jmp.Policy = config.trainer.mp

        def eval_loss_fn(model: LmHeadModel, batch: LmExample) -> LossFnOutput:
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            per_pos_loss = model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array
            per_pos_weight = batch.loss_weight.array
            per_pos_token_id = jnp.roll(batch.tokens.array, -1, axis=-1)
            return per_pos_loss, per_pos_weight, per_pos_token_id

        evaluator = TaggedEvaluator(
            EvalBatch=Batch,
            tagged_eval_sets=datasets,
            loss_fn=eval_loss_fn,
            tokenizer=tokenizer,
            axis_mapping=compute_axis_mapping,
            max_examples_per_dataset=None,
        )

        # Load checkpoint.
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

        log_dict = eval_model(evaluator, model, prefix="eval")

    logger.info("Eval complete. log_dict keys: %s", sorted(log_dict.keys()))

    # Harvest the LIMA loss/bpb. Keys follow Levanter convention
    # `eval/<tag>/loss`, plus an overall `eval/loss`.
    lima_loss = log_dict.get("eval/lima/loss")
    lima_bpb = log_dict.get("eval/lima/bpb")
    overall_loss = log_dict.get("eval/loss")
    total_eval_tokens = log_dict.get("eval/total_eval_tokens")

    payload = {
        "run_name": args.run_name,
        "marin_prefix": prefix,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": final_step,
        "lima_cache_hash": args.lima_hash,
        "lima_cache": lima_cache,
        "meta": meta,
        "eval/lima/loss": float(lima_loss) if lima_loss is not None else None,
        "eval/lima/bpb": float(lima_bpb) if lima_bpb is not None else None,
        "eval/loss": float(overall_loss) if overall_loss is not None else None,
        "total_eval_tokens": int(total_eval_tokens) if total_eval_tokens is not None else None,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    out_path = _write_result(args.output_prefix, args.run_name, payload)
    logger.info("Wrote LIMA eval result → %s", out_path)
    logger.info("  loss=%.4f bpb=%.4f", payload["eval/lima/loss"], payload["eval/lima/bpb"] or -1)

    # Flush tracker to be safe.
    try:
        levanter.tracker.current_tracker().finish()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("LIMA eval failed: %s", e)
        sys.exit(1)
