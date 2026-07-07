# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the canonical CORE_TASKS pretraining eval suite over a batch of checkpoints.

Reads one checkpoint path per line from a manifest file and fans each out as an
independent ``default_eval`` ExecutorStep (TPU-native Levanter LM-eval harness,
CORE_TASKS by default). ``executor_main`` dedupes by content hash, so reruns skip
checkpoints that already have results.

Each path must point at an HF-format export (the harness looks for ``config.json``;
see ``lib/marin/src/marin/evaluation/run.py:_normalize_model_path``). Two path shapes
are supported via ``DISCOVER_LATEST_CHECKPOINT``:

  * Leaf ``.../hf/step-N/`` dirs (config.json directly inside) -> set it False.
  * Run/output dirs containing ``hf/step-N/`` subdirs -> set it True to auto-pick
    the latest step-N (``discover_hf_checkpoints(path)[-1]``, sorted by mtime).

Launch via iris (pin --zone to the region the checkpoints live in to avoid
cross-region reads):

    uv run iris --cluster=marin job run --no-wait \\
        --zone us-central1 \\
        --memory 8GB --enable-extra-resources \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HF_TOKEN "$(cat ~/.cache/huggingface/token)" \\
        -- python experiments/exp_core_eval_245.py \\
        --manifest experiments/core_eval_245_checkpoints.txt
"""

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import InputName, executor_main

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS

# --- Knobs ---------------------------------------------------------------------
# Manifests hold `hf_dir` paths (a `.../hf/` dir with multiple step-N inside), so
# discover the latest checkpoint under each (harness picks most-recent by mtime,
# which is the largest step-N).
DISCOVER_LATEST_CHECKPOINT: bool = True
DEFAULT_TPU: str = "v5p-8"  # override per-region with --tpu; v4-8 only in us-central2.
# -------------------------------------------------------------------------------


def read_manifest(manifest_path: str) -> list[str]:
    """Return non-empty, non-comment checkpoint paths from a newline-delimited manifest."""
    with open(manifest_path) as f:
        paths = [line.strip() for line in f]
    return [p for p in paths if p and not p.startswith("#")]


def build_steps(manifest_path: str, tpu: str):
    checkpoints = read_manifest(manifest_path)
    resource_config = ResourceConfig.with_tpu(tpu)
    return [
        default_eval(
            InputName.hardcoded(path) if not path.startswith("gs://") else path,
            resource_config=resource_config,
            evals=CORE_TASKS,
            discover_latest_checkpoint=DISCOVER_LATEST_CHECKPOINT,
        )
        for path in checkpoints
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Newline-delimited file of checkpoint hf_dir paths (one per line; # comments allowed).",
    )
    parser.add_argument("--tpu", default=DEFAULT_TPU, help="TPU slice, e.g. v5p-8 / v6e-8 / v4-8.")
    args, remaining = parser.parse_known_args()
    # executor_main is @draccus.wrap()'d and parses sys.argv itself; hand it only
    # the flags it owns (e.g. --dry_run), not our --manifest/--tpu.
    sys.argv = [sys.argv[0], *remaining]
    executor_main(steps=build_steps(args.manifest, args.tpu))
