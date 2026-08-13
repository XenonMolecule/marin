# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone training child for the data-curation IsoFLOP sweep.

Runs **inside one TPU iris job** (submitted by `launch_curation_sweep.py`).
Takes a `PlannedRun`'s CLI args, detects its region, region-locks the run via
the tracker, computes a region-local checkpoint output_path, builds the
LMMixtureDatasetConfig (with paloma + uncheatable_eval validation, all pointing
at LOCAL `gs://marin-{region}/...` paths), and calls Levanter directly
in-process.

This is the analog of `experiments/baseline_collection/run_extract_standalone.py`
for training. Each child is fully self-contained -- no executor coordination,
no MARIN_PREFIX inheritance from the parent.

Usage (normally invoked by the coordinator, but can be run by hand for debug):

    python experiments/scaling_law_sweeps/run_curation_train_standalone.py \\
        --method dclm --experiment-tag expA_natural \\
        --budget 3e18 --hidden-dim 512 --num-layers 6 --num-heads 4 \\
        --intermediate-dim 2048 --batch-size 32 --train-steps 6406 \\
        --learning-rate 0.0063 --adam-lr 0.000656 --epsilon 1.85e-8 \\
        --beta1 0.9 --beta2 0.9999 --t-exp 8.4e8 --t-target 2.22e12 \\
        --seq-len 4096
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib
import json
import logging
import os
import threading
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import TrainLmOnPodConfig, _prepare_training_run

from experiments.scaling_law_sweeps import region_tracker
from experiments.scaling_law_sweeps.curation_plan import (
    METHODS,
    PlannedRun,
)

logger = logging.getLogger(__name__)

# Where region-lock tracker files live (single neutral home bucket -- tiny files,
# rare cross-region reads are negligible cost).
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/data_curation_isoflop/"

# Where per-run summary JSONs land (one file per completed run, flat layout).
# Reading all results for scaling-law plots is then:
#     gcloud storage cat <prefix>/*.json | jq -s '.'
# The file is written once by the standalone child just before the DONE marker,
# so a run's summary appears atomically on training success.
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_isoflop_results/"

# A single central JSONL that lists runs whose WandB data was cached locally
# (in offline mode) and needs later `wandb sync`. One line per affected run.
# Lives in us-central1 because it's tiny (~200 B/line) and centrally queryable.
DEFAULT_SYNC_PENDING_PATH = "gs://marin-us-central1/metadata/data_curation_isoflop_sync_pending.jsonl"


def _assert_all_components_local(tokenized, region: str) -> None:
    """Hard invariant: every component's cache_dir must be the region-local bucket.

    Uses `region_tracker.REGION_TO_BUCKET` rather than naive `f"gs://marin-{region}"`
    because some regions have non-obvious bucket names (region `europe-west4`
    has bucket `gs://marin-eu-west4`).

    Raises on any path that would trigger cross-region reads or unsupported
    URI schemes. Runs BEFORE we open tensorstore, so a misconfiguration is
    caught loudly instead of silently paying egress.
    """
    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    for name, comp in tokenized.components.items():
        for field_name, path in (
            ("cache_dir", comp.cache_dir),
            ("source.cache_dir", getattr(comp.source, "cache_dir", None)),
        ):
            if path is None:
                continue
            if not path.startswith(expected_prefix):
                raise ValueError(
                    f"component {name!r} has non-local {field_name}={path!r}; "
                    f"expected prefix {expected_prefix!r}. Cross-region reads "
                    f"forbidden -- pre-copy the cache into {expected_prefix} first."
                )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    PlannedRun.add_cli_args(parser)
    parser.add_argument(
        "--tracker-prefix",
        default=DEFAULT_TRACKER_PREFIX,
        help="GCS prefix where region-lock tracker files live.",
    )
    parser.add_argument(
        "--tpu-type",
        default=None,
        help="Override the local TPU type for ResourceConfig. If unset, infer from env.",
    )
    parser.add_argument(
        "--wandb-project",
        default="marin",
    )
    parser.add_argument(
        "--wandb-entity",
        default="marin-community",
    )
    parser.add_argument(
        "--wandb-group",
        default="data-curation-isoflop",
        help="WandB group so all 762 runs land in one comparison view.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to the run name (affects both output_path and "
        "WandB run id). Use this to isolate repeated smoke/debug runs of the "
        "same plan into fresh WandB runs and checkpoint dirs.",
    )
    parser.add_argument(
        "--results-prefix",
        default=DEFAULT_RESULTS_PREFIX,
        help="GCS prefix where the per-run summary.json is written on training completion.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "offline_no_sync"],
        default="auto",
        help="'auto' probes api.wandb.ai and uses offline mode on probe failure. "
        "'online' or 'offline' skip the probe and force the mode. "
        "'offline_no_sync' behaves like 'offline' but does NOT download/upload "
        "the wandb cache or append to the sync-pending log -- the run's wandb data "
        "dies with the container. Use this to skip WandB egress entirely once you "
        "trust the pipeline and only care about the eval_metrics.jsonl / summary.json.",
    )
    parser.add_argument(
        "--sync-pending-path",
        default=DEFAULT_SYNC_PENDING_PATH,
        help="GCS path of the JSONL log listing offline WandB runs that need later sync.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Levanter TrainerConfig.seed. The default 0 is what the whole sweep uses; "
        "the data-shuffle permutation, model init, and loader keys all split from "
        "PRNGKey(seed), so a different value gives a fully independent draw (different "
        "data order AND init). Use with --run-suffix to isolate a seed-noise probe of a "
        "single cell into a fresh run name / output dir / WandB run.",
    )
    return parser.parse_args(argv)


def _detect_local_tpu_type(override: str | None = None) -> str:
    """Best-effort detection of the TPU slice this worker is on.

    Falls back to override or 'v4-8' if nothing else is available. Iris's
    runtime sets `IRIS_DEVICE_VARIANT` (e.g., 'v5p-8' or 'v4-32') on the
    worker; that's the canonical source.
    """
    import os

    if override:
        return override
    for env_var in ("IRIS_DEVICE_VARIANT", "TPU_TYPE", "ACCELERATOR_TYPE"):
        if env_var in os.environ:
            return os.environ[env_var]
    logger.warning("Could not detect TPU type from env; defaulting to v4-8")
    return "v4-8"


def _build_model_config(plan: PlannedRun) -> Qwen3Config:
    """Build the Qwen3 model config from the plan's primitive args.

    Mirrors `CompletedAdamHHeuristic._build_model_config` so the child produces
    bit-identical configs to what the heuristic would on the coordinator side.
    """
    return Qwen3Config(
        hidden_dim=plan.hidden_dim,
        intermediate_dim=plan.intermediate_dim,
        num_layers=plan.num_layers,
        num_heads=plan.num_heads,
        num_kv_heads=plan.num_heads,
        max_seq_len=plan.seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _build_optimizer_config(plan: PlannedRun) -> AdamHConfig:
    """Build the AdamH optimizer config from the plan's primitive args."""
    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    h = completed_adamh_heuristic
    return AdamHConfig(
        learning_rate=plan.learning_rate,
        adam_lr=plan.adam_lr,
        min_lr_ratio=h.min_lr_ratio,
        warmup=h.warmup,
        beta1=plan.beta1,
        beta2=plan.beta2,
        epsilon=plan.epsilon,
        max_grad_norm=h.max_grad_norm,
        lr_schedule=h.lr_schedule,
        decay=h.decay,
        nesterov=h.nesterov,
    )


def _build_tags(plan: PlannedRun, method) -> list[str]:
    """WandB tags for filtering/grouping runs in the dashboard.

    Design: coarse grouping tags first (experiment=A, method=dclm) so the
    WandB Runs sidebar lets you filter with one click. Architecture + budget
    details follow as `key=value` tags for precise slicing.
    """
    # Short experiment label ("A" or "B") -- easiest filter in the UI.
    short_exp = "A" if plan.experiment_tag.startswith("expA") else "B"
    # Params in "156M" / "1.2B" form.
    # Model config → total param count (see `_build_summary` for the same calc).
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.models.qwen import Qwen3Config

    mc = Qwen3Config(
        hidden_dim=plan.hidden_dim,
        intermediate_dim=plan.intermediate_dim,
        num_layers=plan.num_layers,
        num_heads=plan.num_heads,
        num_kv_heads=plan.num_heads,
        max_seq_len=plan.seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    params = mc.total_trainable_params(128256)
    params_short = f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
    # T_target in a readable form.
    t_target_short = f"{plan.t_target / 1e12:.0f}T" if plan.t_target >= 1e12 else f"{plan.t_target:.1e}"
    return [
        # --- coarse grouping (one-click filters) ---
        f"experiment={short_exp}",
        f"method={plan.method_name}",
        # --- budget / scale ---
        f"budget={plan.budget:.0e}",
        f"params={params_short}",
        f"T_target={t_target_short}",
        # --- architecture ---
        f"d_model={plan.hidden_dim}",
        f"num_layers={plan.num_layers}",
        f"num_heads={plan.num_heads}",
        f"batch_size={plan.batch_size}",
        f"train_steps={plan.train_steps}",
        # --- method details ---
        f"d_obs={method.d_obs_tokens:.2e}",
        f"d_proj={method.d_proj:.2e}",
        f"s={method.s:.1f}",
        f"sampled_warcs={method.sampled_warcs}",
        # --- full experiment tag (precise) ---
        f"exp_full={plan.experiment_tag}",
        # --- optimizer ---
        "optimizer=completed-adamh",
        f"tokenizer={method.tokenizer}",
    ]


def _build_train_lm_config(
    plan: PlannedRun, tokenized, tags: list[str], wandb_project: str, wandb_entity: str, wandb_group: str, seed: int = 0
):
    """Build the inner Levanter TrainLmConfig.

    Mirrors what `simulated_epoching_train` -> `default_train` would have
    constructed but without the ExecutorStep wrapper. We're skipping the
    executor entirely since this script IS the worker.
    """
    from levanter.main import train_lm

    model_config = _build_model_config(plan)
    optimizer_config = _build_optimizer_config(plan)

    return train_lm.TrainLmConfig(
        data=tokenized,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                tags=tags,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            seed=seed,
            train_batch_size=plan.batch_size,
            per_device_parallelism=-1,
            num_train_steps=plan.train_steps,
            # Eval every 5000 steps (was 1000): each eval runs the FULL eval set
            # (max_eval_batches=None), ~35s, so 1000 was ~17% wall-clock overhead
            # on long runs and dominated short/fast cells. A guaranteed final eval
            # (trainer.train force=True) always captures the terminal loss, so the
            # scaling-law data point is unaffected — only intermediate curve
            # markers get sparser. Loss VALUES are unchanged (full eval set kept).
            steps_per_eval=5000,
            # Mesh: plumb tensor_parallel into the "model" axis. Without this,
            # Levanter's default mesh has model=1 and data=total_chips, so any
            # multi-host plan with batch_size < total_chips hits ZeroDivisionError
            # in _validate_and_set_defaults (per_device_parallelism = 0).
            # Matches Marin's experiments/defaults.py:452 canonical pattern.
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": plan.tensor_parallel},
            ),
            allow_nondivisible_batch_size=True,
            # Checkpoint policy: rolling 15-min time-based for preemption recovery
            # (auto-deleted on next save within an attempt) + one permanent final
            # checkpoint (via trainer.py's `force=True` save at end of training).
            # NO intermediate step checkpoints -- `keep=[]` disables them, saving
            # ~2 TB across the 508-run sweep (3 permanent intermediates x 1.2 GB
            # x 508 runs).
            #
            # delete_old_temp_checkpoints=False preserves the rolling temp across
            # Iris-level child retries so a fresh attempt can resume from the
            # crashed attempt's last 15-min checkpoint instead of training from
            # scratch. Net storage cost: still exactly 1 rolling temp per run
            # (~12 GB for d=2432) — same as the default; the only difference is
            # cross-attempt persistence. HF exports (under hf/) and the final
            # force-saved Levanter checkpoint are unaffected by this flag.
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=15),
                keep=[],
                delete_old_temp_checkpoints=False,
            ),
        ),
        train_seq_len=plan.seq_len,
        model=model_config,
        optimizer=optimizer_config,
        z_loss_weight=plan.z_loss_weight,
    )


def _probe_wandb_healthy(timeout_seconds: float = 10.0) -> bool:
    """Probe WandB's GraphQL endpoint with the smallest real query.

    Hits `POST https://api.wandb.ai/graphql` with the introspection query
    `{ __typename }` -- the minimum valid GraphQL request. WandB answers with:

      - 200 + JSON body `{"data": {"__typename": "Query"}}` when the API is
        healthy and the server can parse and execute a trivial query.
      - 401 Unauthorized when the API is healthy but we didn't auth -- still
        means WandB is up and responding, just rejecting our request.

    Anything else (5xx, connection refused, DNS fail, timeout) → WandB is
    actually broken or unreachable; fall back to offline mode.

    Why not HEAD on `/` : WandB's root returns 404 (no app served there),
    which is uninformative about API health. GraphQL is what `wandb.init()`
    actually talks to, so this is the probe most directly correlated with
    whether init will succeed.
    """
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=b'{"query":"{ __typename }"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout_seconds)
        return True
    except urllib.error.HTTPError as e:
        # API responded -- HTTP error means alive-but-rejecting. 401 is the
        # expected unauthenticated answer; treat any 4xx similarly. 5xx means
        # the API server itself is unhealthy → mark as bad.
        if 400 <= e.code < 500:
            logger.info("WandB probe: HTTP %s (expected for unauthed GraphQL) -- healthy", e.code)
            return True
        logger.warning("WandB probe: HTTP %s (server error) -- marking unhealthy", e.code)
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("WandB probe: connection failure (%s) -- marking unhealthy", e)
        return False
    except Exception as e:
        logger.warning("WandB probe: unexpected error (%s) -- marking unhealthy", e)
        return False


def _decide_wandb_mode(user_pref: str) -> str:
    """Pick the WandB mode to run under.

    `user_pref` is one of 'auto', 'online', 'offline', 'offline_no_sync':
      - 'auto' (default): probe api.wandb.ai; use 'online' if healthy, 'offline' otherwise.
      - 'online' / 'offline' / 'offline_no_sync': use literally, skip the probe.

    'offline' and 'offline_no_sync' both run Levanter with WANDB_MODE=offline,
    but 'offline_no_sync' disables the GCS cache download/upload so the run's
    WandB data dies with the container (no egress, at the cost of no later sync).
    """
    if user_pref in ("online", "offline", "offline_no_sync"):
        return user_pref
    # 'auto'
    return "online" if _probe_wandb_healthy() else "offline"


def _download_wandb_cache_if_exists(src_gcs: str, local_dir: str) -> bool:
    """Fetch an existing wandb/ cache from GCS to local so WandB SDK can extend it.

    Called on startup for offline-mode runs: if a prior attempt uploaded a
    wandb cache to `{output_path}/wandb/`, we grab it before Levanter starts
    so the SDK sees existing run state and appends rather than starting fresh.

    Returns True if we downloaded something, False if nothing was there.
    """
    import pathlib

    try:
        fs, urlpath = fsspec.core.url_to_fs(src_gcs)
        if not fs.exists(urlpath):
            return False
        pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
        fs.get(urlpath.rstrip("/") + "/", local_dir.rstrip("/") + "/", recursive=True)
        logger.info("Downloaded existing WandB cache: %s → %s", src_gcs, local_dir)
        return True
    except Exception as e:
        logger.warning("Failed to download WandB cache from %s: %s", src_gcs, e)
        return False


def _start_wandb_cache_uploader_thread(
    local_dir: str,
    dest_gcs: str,
    interval_seconds: int = 15 * 60,
) -> tuple[threading.Thread, threading.Event]:
    """Periodically mirror local wandb/ to GCS so preempt-resume loses at most
    `interval_seconds` of wandb logs. Cadence matches the Levanter checkpointer
    (15 min default) so model state and wandb state are preserved at the same
    granularity.

    Returns (thread, stop_event). Caller must set the stop event and join at
    shutdown so the final upload isn't raced.
    """
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            # Wait first -- we don't want to double-upload immediately after startup.
            if stop.wait(interval_seconds):
                return
            _upload_wandb_cache(local_dir, dest_gcs)

    thread = threading.Thread(target=loop, name="wandb-cache-uploader", daemon=True)
    thread.start()
    return thread, stop


def _upload_wandb_cache(local_dir: str, dest_gcs: str) -> bool:
    """Copy a local `wandb/` dir to GCS. Non-fatal on error. Returns True on success.

    Destination must be in the SAME region as the checkpoint output_path so we
    don't pay cross-region egress on this ~100 MB dir. Caller composes
    `dest_gcs = {output_path}/wandb/` where output_path is already region-local.
    """
    import pathlib

    src = pathlib.Path(local_dir)
    if not src.exists() or not any(src.iterdir()):
        logger.info("No WandB offline cache at %s -- nothing to upload", src)
        return False

    try:
        fs, urlpath = fsspec.core.url_to_fs(dest_gcs)
        # fsspec.put copies recursively; trailing slash on remote is important.
        fs.put(str(src) + "/", urlpath.rstrip("/") + "/", recursive=True)
        logger.info("Uploaded WandB offline cache: %s → %s", src, dest_gcs)
        return True
    except Exception as e:
        logger.warning("Failed to upload WandB cache from %s to %s: %s", src, dest_gcs, e)
        return False


def _append_sync_pending_row(row: dict, jsonl_path: str) -> None:
    """Append one JSONL line to the central sync-pending log. Non-fatal.

    JSONL append is safe (1-line-per-row, atomic at GCS write granularity for
    the coordination volumes we see). Even if two workers race, lines don't
    corrupt each other -- we only ever write one row per run per child.
    """
    try:
        fs, urlpath = fsspec.core.url_to_fs(jsonl_path)
        line = json.dumps(row) + "\n"
        # Read-modify-write. Not atomic, but acceptable for our throughput.
        existing = b""
        if fs.exists(urlpath):
            with fs.open(urlpath, "rb") as f:
                existing = f.read()
        with fs.open(urlpath, "wb") as f:
            f.write(existing + line.encode())
        logger.info("Appended sync-pending row for %s", row.get("run_name"))
    except Exception as e:
        logger.warning("Failed to append sync-pending row to %s: %s", jsonl_path, e)


# NOTE: a former `_init_jax_distributed_for_multihost_tpu()` helper lived here.
# It manually called jax.distributed.initialize() for multi-host TPU because
# iris's initialize_jax() used to skip TPU. iris now initializes multi-host TPU
# itself (lib/iris/src/iris/runtime/jax_init.py), and Levanter routes through it,
# so the manual helper was dead code and is removed. Do NOT reintroduce a manual
# init in this file: a second initialize() after Levanter's crashes with
# "jax.distributed.initialize() must be called before any JAX calls ...".


# Step slack for sourcing final-eval metrics. Levanter's final eval normally
# logs at step = train_steps - 1, so a slack of 10 absorbs any off-by-N
# discrepancy across Levanter versions/configs while still catching the
# preempt-resume race that has shipped multiple wrong summaries (see
# `audit_summary_vs_wandb.py` for context).
_FINAL_EVAL_STEP_SLACK = 10


def _read_last_eval_metrics(output_path: str) -> dict | None:
    """Read the last row of `{output_path}/checkpoints/eval_metrics.jsonl`.

    Returns the dict of final-eval metrics, or None if the file is missing or
    empty. Non-fatal -- a failure here should not tank the training run.
    """
    path = f"{output_path.rstrip('/')}/checkpoints/eval_metrics.jsonl"
    try:
        with fsspec.open(path, "r") as f:
            lines = [line for line in f if line.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception as e:
        logger.warning("Failed to read eval_metrics.jsonl at %s: %s", path, e)
        return None


def _read_last_eval_metrics_checked(output_path: str, expected_step: int) -> dict | None:
    """`_read_last_eval_metrics` plus a step-consistency check.

    Rejects (returns None) if the captured row's `step` is more than
    `_FINAL_EVAL_STEP_SLACK` behind `expected_step`. The checked-too-early
    failure mode -- where eval_metrics.jsonl read returns a row from earlier
    in training while the run is actually complete -- has produced multiple
    wrong summaries in this sweep; refusing the row forces the caller to fall
    back to W&B (or fail loudly) rather than write a corrupted summary.
    """
    last = _read_last_eval_metrics(output_path)
    if last is None:
        return None
    captured = last.get("step")
    if captured is None:
        logger.warning("eval_metrics.jsonl last row has no 'step' field; treating as untrusted.")
        return None
    if int(captured) < expected_step - _FINAL_EVAL_STEP_SLACK:
        logger.error(
            "eval_metrics.jsonl last row step=%d is %d steps behind expected_step=%d (slack=%d). "
            "This is the partial-write race documented in audit_summary_vs_wandb.py; "
            "refusing to use this row.",
            int(captured),
            expected_step - int(captured),
            expected_step,
            _FINAL_EVAL_STEP_SLACK,
        )
        return None
    return last


def _fetch_final_eval_from_wandb(run_name: str, project: str, entity: str | None, expected_step: int) -> dict | None:
    """Pull eval/* keys from W&B's run.summary as the canonical final-eval source.

    Returns None (caller falls back to file) if:
      - the wandb library / API call fails (e.g., offline mode, no creds);
      - the run isn't `finished`;
      - run.summary's `_step` is more than `_FINAL_EVAL_STEP_SLACK` behind
        `expected_step` (W&B run is truncated / from a partial incarnation);
      - the run summary has no eval/* keys at all.

    On success returns a dict of all eval/* keys with numeric values cast to
    float (matching the on-disk eval_metrics.jsonl shape).
    """
    try:
        import wandb
    except Exception as e:
        logger.warning("wandb import failed (%s); cannot fetch final-eval from W&B.", e)
        return None
    try:
        api = wandb.Api()
        run_path = f"{entity}/{project}/{run_name}" if entity else f"{project}/{run_name}"
        run = api.run(run_path)
    except Exception as e:
        logger.warning("W&B run lookup failed for %s: %s", run_name, e)
        return None
    if run.state != "finished":
        logger.warning("W&B run %s state=%s (not finished); skipping W&B summary source.", run_name, run.state)
        return None
    last_step = run.summary.get("_step")
    if last_step is None or int(last_step) < expected_step - _FINAL_EVAL_STEP_SLACK:
        logger.warning(
            "W&B run %s _step=%s is behind expected_step=%d (slack=%d); skipping.",
            run_name,
            last_step,
            expected_step,
            _FINAL_EVAL_STEP_SLACK,
        )
        return None

    # `_step` is NOT evidence that the eval/* keys are current. It advances on every training
    # log; eval/* only refreshes when an eval cycle runs (every few thousand steps). A child
    # querying its own run right after training -- before the final eval has synced -- sees
    # `_step == expected_step` while eval/* still holds the PREVIOUS eval. Observed on
    # dclm_10k_mix 3e+17-d512: summary reported the step-10000 eval (uncheatable 1.4265) as
    # final, when the true step-13137 value was 1.3166. That is a 0.11 BPB error, and it
    # silently affected ~40% of runs -- enough to fabricate spikes in the scaling curves.
    #
    # So validate the eval's OWN step: take the last history row that actually carries eval
    # keys and require IT to be at the final step.
    try:
        eval_rows = [
            row
            for row in run.scan_history(page_size=10000)
            if any(k.startswith("eval/") for k in row) and row.get("_step") is not None
        ]
    except Exception as e:
        logger.warning("W&B history scan failed for %s (%s); skipping W&B summary source.", run_name, e)
        return None
    if not eval_rows:
        logger.warning("W&B run %s has no history rows with eval/* keys; skipping.", run_name)
        return None
    final_row = max(eval_rows, key=lambda r: int(r["_step"]))
    eval_step = int(final_row["_step"])
    if eval_step < expected_step - _FINAL_EVAL_STEP_SLACK:
        logger.warning(
            "W&B run %s: latest EVAL is at step=%d but expected_step=%d (slack=%d) -- the final "
            "eval has not synced yet. Falling back to eval_metrics.jsonl.",
            run_name,
            eval_step,
            expected_step,
            _FINAL_EVAL_STEP_SLACK,
        )
        return None

    # Read the values OUT OF the validated row, NOT out of `run.summary`. Validating that
    # history contains a final-step eval row and then reading `run.summary` was the incomplete
    # version of this fix: the guard passes while summary still holds an earlier cycle's values.
    # Observed AFTER that fix shipped, on the first cell of the lambda=0.01 sweep -- all 60
    # metric keys stale, `eval/paloma/4chan/loss` off by 8.74. One consistent cycle or nothing.
    eval_dict: dict = {}
    for k, v in final_row.items():
        if not k.startswith("eval/") or v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            eval_dict[k] = float(v)
        else:
            eval_dict[k] = v
    if not eval_dict:
        logger.warning("W&B run %s: final eval row carries no eval/* keys; skipping.", run_name)
        return None
    return eval_dict


def _resolve_final_eval(
    run_name: str,
    output_path: str,
    plan: PlannedRun,
    wandb_project: str,
    wandb_entity: str | None,
) -> dict | None:
    """Source final-eval metrics with W&B as primary, eval_metrics.jsonl fallback.

    The fallback is itself step-checked, so a partial eval_metrics.jsonl read
    won't silently produce a wrong summary. Returns None when neither source
    is trustworthy -- callers should treat that as "do not write summary or
    DONE marker", letting the coordinator retry the run.
    """
    expected_step = plan.train_steps - 1  # Levanter logs final eval at train_steps - 1
    via_wandb = _fetch_final_eval_from_wandb(run_name, wandb_project, wandb_entity, expected_step)
    if via_wandb is not None:
        logger.info("Final-eval sourced from W&B run.summary (%d keys).", len(via_wandb))
        return via_wandb
    via_file = _read_last_eval_metrics_checked(output_path, expected_step)
    if via_file is not None:
        logger.info(
            "Final-eval sourced from eval_metrics.jsonl (%d keys, step=%d).",
            len(via_file),
            int(via_file.get("step", -1)),
        )
        return via_file
    return None


def _build_summary(
    plan: PlannedRun,
    method,
    region: str,
    run_name: str,
    output_path: str,
    final_eval: dict | None,
) -> dict:
    """Compose a flat per-run summary JSON with everything scaling-law plots need.

    Fields are grouped into: plan (identity + hyperparams), method (D_obs, s,
    tokenizer), model (params), tokens (trained + slice + epochs), run
    (completion metadata), and eval (final eval metrics from the last eval
    cycle -- per-dataset bpb/loss + macro/micro aggregates).

    For ExpA / expFM_natural: effective slice is D_obs (no Levanter slicing).
    For ExpB / ExpC sliced regime (target_epochs >= 1): effective slice is
    D_obs * T_exp / (T_target/s), matching Levanter's slice formula.
    For ExpC data-rich regime (target_epochs < 1): no slicing applied — the
    runner trains naturally on D_obs and effective slice = D_obs.
    """
    model_config = _build_model_config(plan)
    vocab_size = 128256  # meta-llama/Meta-Llama-3.1-8B tokenizer
    total_params = model_config.total_trainable_params(vocab_size)

    # Slicing semantics: gate on tag (expB/expC opt in to simulated epoching) AND
    # on the regime (target_epochs >= 1, i.e. T_target >= D_proj). Data-rich
    # ExpC methods (LC, Resiliparse) have target_epochs < 1 so they fall through
    # to the natural-D_obs path even though their tag is expC_*.
    needs_slicing = plan.experiment_tag.startswith(("expB", "expC")) and plan.t_target >= method.d_proj
    tokens_trained = plan.batch_size * plan.seq_len * plan.train_steps
    if needs_slicing:
        slice_tokens = int(method.d_obs_tokens * plan.t_exp / (plan.t_target / method.s))
    else:
        slice_tokens = method.d_obs_tokens
    effective_epochs = tokens_trained / slice_tokens if slice_tokens else 0.0

    return {
        "plan": {
            "method_name": plan.method_name,
            "experiment_tag": plan.experiment_tag,
            "run_name": run_name,
            "run_name_core": plan.run_name_core,
            "budget_flops": plan.budget,
            "hidden_dim": plan.hidden_dim,
            "num_layers": plan.num_layers,
            "num_heads": plan.num_heads,
            "intermediate_dim": plan.intermediate_dim,
            "batch_size": plan.batch_size,
            "train_steps": plan.train_steps,
            "seq_len": plan.seq_len,
            "learning_rate": plan.learning_rate,
            "adam_lr": plan.adam_lr,
            "beta1": plan.beta1,
            "beta2": plan.beta2,
            "epsilon": plan.epsilon,
            "z_loss_weight": plan.z_loss_weight,
            "t_exp": plan.t_exp,
            "t_target": plan.t_target,
        },
        "method": {
            "name": method.name,
            "tokenized_rel_path": method.tokenized_rel_path,
            "d_obs_tokens": method.d_obs_tokens,
            "d_proj_tokens": method.d_proj,
            "s_scale_factor": method.s,
            "sampled_warcs": method.sampled_warcs,
            "total_warcs": method.total_warcs,
            "tokenizer": method.tokenizer,
        },
        "model": {
            "total_trainable_params": total_params,
            "vocab_size": vocab_size,
        },
        "tokens": {
            "tokens_trained": tokens_trained,
            "slice_tokens": slice_tokens,
            "effective_epochs": effective_epochs,
        },
        "run": {
            "region": region,
            "output_path": output_path,
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "eval": final_eval,
    }


def _write_summary(summary: dict, results_prefix: str, run_name: str) -> None:
    """Write summary.json to `{results_prefix}/{run_name}.json`. Non-fatal on error."""
    path = f"{results_prefix.rstrip('/')}/{run_name}.json"
    try:
        with fsspec.open(path, "w") as f:
            f.write(json.dumps(summary, indent=2, default=str))
        logger.info("Wrote run summary: %s", path)
    except Exception as e:
        logger.warning("Failed to write summary at %s: %s", path, e)


def _early_is_process_0() -> bool:
    """Pre-JAX-init rank check via TPU_WORKER_ID.

    For multi-host TPU (replicas>1, coscheduled by tpu-name), iris sets
    `TPU_WORKER_ID=0..N-1` per VM before our entrypoint runs. Single-host
    runs leave it unset, which we treat as rank 0. We need this BEFORE
    `jax.distributed.initialize` fires so we can gate side effects (wandb
    cache I/O, summary writes, DONE markers) to one VM.
    """
    return os.environ.get("TPU_WORKER_ID", "0") == "0"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    plan = PlannedRun.from_namespace(args)
    is_process_0 = _early_is_process_0()

    # Apply optional run suffix. This isolates checkpoints + WandB run ids +
    # region-lock tracker entries for repeated smoke/debug launches of the same
    # plan without accidentally resuming from a stale broken checkpoint or
    # polluting an existing WandB run.
    suffix = args.run_suffix.strip()
    run_name = plan.run_name_core + (f"-{suffix}" if suffix else "")

    # 1. Detect region (parses MARIN_REGION or MARIN_PREFIX env var).
    region = region_tracker.detect_current_region()
    logger.info("Standalone training child boot -- region=%s, run=%s", region, run_name)

    # 2. Region-lock: claim if first run, verify match if not.
    #    Raises RegionMismatch (non-zero exit) if a different region claimed it earlier.
    #    Include the suffix in the tracker lookup so suffixed runs don't collide
    #    with the base plan's tracker entry.
    bucket = region_tracker.resolve_checkpoint_prefix(
        plan.method_name,
        plan.experiment_tag,
        run_name,
        local_region=region,
        tracker_prefix=args.tracker_prefix,
        # Liveness gate: on a region mismatch, refuse to migrate (would spawn a
        # concurrent duplicate) while the claimed region's checkpoint is still
        # fresh. Matches the output_path layout computed just below.
        checkpoint_rel_path=f"checkpoints/isoflop-curation/{run_name}",
    )
    logger.info("Region-locked: %s → %s", run_name, bucket)

    # 3. Compute output_path in the pinned region's bucket.
    output_path = f"{bucket}/checkpoints/isoflop-curation/{run_name}"
    logger.info("Checkpoint output_path: %s", output_path)

    # 3b. Decide WandB mode BEFORE Levanter starts. On startup for offline mode,
    #     download any prior attempt's wandb cache so the SDK sees continuity
    #     across preempt-resume boundaries. A 15-min periodic uploader keeps
    #     GCS in sync with the local cache at the same cadence as Levanter's
    #     time-based checkpointer -- so model state and wandb state survive
    #     preemption at the same granularity.
    wandb_mode = _decide_wandb_mode(args.wandb_mode)
    # Both "offline" and "offline_no_sync" translate to WANDB_MODE=offline for
    # the SDK -- they differ only in our GCS cache+sync behavior below.
    os.environ["WANDB_MODE"] = "offline" if wandb_mode in ("offline", "offline_no_sync") else wandb_mode
    logger.info("WandB mode: %s (user-pref: %s)", wandb_mode, args.wandb_mode)
    wandb_gcs_dir = f"{output_path}/wandb"
    wandb_local_dir = "/app/wandb"
    wandb_cache_thread = None
    wandb_cache_stop = None
    # Multi-VM gating: only rank 0 manages the wandb cache. Other VMs would
    # race on the same GCS upload path (last-writer-wins, partial uploads).
    # WandB itself only emits from process 0 inside Levanter, so non-rank-0
    # VMs have nothing to cache anyway.
    if wandb_mode == "offline" and is_process_0:
        _download_wandb_cache_if_exists(wandb_gcs_dir, wandb_local_dir)
        wandb_cache_thread, wandb_cache_stop = _start_wandb_cache_uploader_thread(
            local_dir=wandb_local_dir,
            dest_gcs=wandb_gcs_dir,
            interval_seconds=15 * 60,  # matches checkpointer save_interval
        )
    elif wandb_mode == "offline_no_sync":
        logger.info("Wandb offline_no_sync: skipping GCS cache download/upload entirely.")

    # 4. Build mixture with LOCAL gs:// paths directly (no mirror:// abstraction).
    #    Data must be pre-copied into gs://marin-{region}/tokenized/... ahead of time.
    method = METHODS[plan.method_name]
    tokenized = method.as_lm_mixture_config(
        region=region,
        with_uncheatable_eval=True,
        with_paloma=True,
    )
    _assert_all_components_local(tokenized, region)
    # Simulated epoching semantics:
    #
    #   Experiment A (expA_natural): no slicing. The model trains naturally on
    #   D_obs, looping/shuffling the full pool as many times as the compute
    #   budget allows. Leave target_budget/experiment_budget as None so
    #   Levanter skips its slicing branch entirely.
    #
    #   Experiment B (expB_*): simulate training on a T_target regime with
    #   fewer experiment tokens (T_exp). Intended slice is
    #       slice = T_exp * D_proj / T_target
    #   but Levanter's slicing uses D_obs (not D_proj):
    #       slice_L = D_obs * (experiment_budget / target_budget)
    #   Equating: target_budget = T_target / s, experiment_budget = T_exp.
    #
    # Without this scaling, ExpA previously set target_budget = T_exp * s,
    # which made Levanter slice the dataset to D_obs/s (~1M tokens), causing
    # the model to over-epoch ~s times and memorize the slice -- exactly the
    # divergence symptom observed in the smoke run.
    #
    # ExpC adds a third regime: data-rich methods (LC, Resiliparse) have
    # T_target < D_proj (target_epochs < 1) so the model would only see a
    # fraction of D_proj at the target scale. Under uniformity the cache is
    # i.i.d. with D_proj, so training NATIVELY on the cache for T_exp tokens
    # is statistically equivalent to drawing T_exp i.i.d. tokens from D_proj
    # — no slicing needed. We gate on (tag in {expB, expC}) AND
    # (target_epochs >= 1, i.e. T_target >= D_proj) so ExpC's data-rich runs
    # fall through to the natural-D_obs path.
    needs_slicing = plan.experiment_tag.startswith(("expB", "expC")) and plan.t_target >= method.d_proj
    if needs_slicing:
        tokenized = dataclasses.replace(
            tokenized,
            target_budget=int(plan.t_target / method.s),
            experiment_budget=int(plan.t_exp),
        )
        logger.info(
            "Mixture: train=%s (weight 1.0), validation=%d datasets (weight 0.0); "
            "%s slicing: target_budget=%d (= T_target %.2e / s %.1f), experiment_budget=%d",
            plan.method_name,
            len(tokenized.components) - 1,
            plan.experiment_tag,
            tokenized.target_budget,
            plan.t_target,
            method.s,
            tokenized.experiment_budget,
        )
    elif plan.experiment_tag.startswith(("expB", "expC")):
        logger.info(
            "Mixture: train=%s (weight 1.0), validation=%d datasets (weight 0.0); "
            "%s data-rich (T_target=%.2e < D_proj=%.2e, target_epochs=%.3f) "
            "-- no Levanter slicing, training naturally on D_obs",
            plan.method_name,
            len(tokenized.components) - 1,
            plan.experiment_tag,
            plan.t_target,
            method.d_proj,
            plan.t_target / method.d_proj,
        )
    else:
        logger.info(
            "Mixture: train=%s (weight 1.0), validation=%d datasets (weight 0.0); "
            "ExpA natural epoching -- no Levanter slicing (target/experiment budgets unset)",
            plan.method_name,
            len(tokenized.components) - 1,
        )

    # 5. Build TrainLmConfig + TrainLmOnPodConfig.
    tags = _build_tags(plan, method)
    train_lm_config = _build_train_lm_config(
        plan,
        tokenized,
        tags,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        seed=args.seed,
    )
    tpu_type = _detect_local_tpu_type(override=args.tpu_type)
    pod_config = TrainLmOnPodConfig(
        train_config=train_lm_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=output_path,
        env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=16000"},
    )

    # 6. IN-PROCESS Levanter call. We are already running on the TPU iris
    #    worker that the coordinator allocated. Calling `run_levanter_train_lm`
    #    would submit a NESTED iris job for training (consuming a 2nd TPU
    #    worker -- wasteful, and capacity-prone). Instead we apply the env vars
    #    `_prepare_training_run` would have set, then invoke `train_lm.main`
    #    directly in this process.
    _prepared_config, train_config_ready, env, _extras = _prepare_training_run(pod_config)
    for k, v in env.items():
        os.environ[k] = v

    # NOTE: we deliberately do NOT call jax.distributed.initialize() here.
    # Levanter's trainer.initialize() does it for us, via
    # levanter.distributed.DistributedConfig.initialize() ->
    # iris.runtime.jax_init.initialize_jax(), which auto-detects the multi-host
    # TPU pod topology (no-op on single-host). Crucially, Levanter calls it
    # BEFORE any jax.device_count()/devices() touch and BEFORE the wandb tracker
    # (trainer.py: distributed.initialize() precedes _validate_and_set_defaults()
    # and the tracker), so the distributed client exists when WandbConfig.init
    # needs it. A second manual initialize() here would re-init after the backend
    # is already up and crash with "jax.distributed.initialize() must be called
    # before any JAX calls that might initialise the XLA backend" -- which is
    # exactly what happened once iris's init stopped skipping TPU. Rank-gating for
    # our own wandb side effects uses TPU_WORKER_ID (`_early_is_process_0`), not
    # jax, so it does not depend on init order.
    train_lm_module = importlib.import_module("levanter.main.train_lm")
    logger.info("Launching levanter.main.train_lm.main() in-process (no nested submit)")
    try:
        train_lm_module.main(train_config_ready)
        logger.info("Training finished cleanly.")
    finally:
        # Stop the periodic wandb cache uploader + do a final upload in offline
        # mode. Append a sync-pending row so a later helper script can
        # `wandb sync` these runs back online. Rank-0 only -- non-rank-0 VMs
        # have no wandb cache to upload (Levanter's wandb is process-0 only).
        if wandb_mode == "offline" and is_process_0:
            if wandb_cache_stop is not None:
                wandb_cache_stop.set()
            if wandb_cache_thread is not None:
                wandb_cache_thread.join(timeout=30)
            uploaded = _upload_wandb_cache(wandb_local_dir, wandb_gcs_dir)
            if uploaded:
                _append_sync_pending_row(
                    {
                        "run_name": run_name,
                        "wandb_gcs_path": wandb_gcs_dir,
                        "output_path": output_path,
                        "method": plan.method_name,
                        "experiment_tag": plan.experiment_tag,
                        "region": region,
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                    },
                    jsonl_path=args.sync_pending_path,
                )

    # 7. Write a per-run summary JSON + DONE marker (rank-0 only). N replicas
    #    racing on the same GCS object is correctness-safe (last-writer-wins,
    #    same content) but pointless. Non-rank-0 VMs exit cleanly here.
    if not is_process_0:
        logger.info(
            "Non-rank-0 VM (TPU_WORKER_ID=%s); exiting without writing summary/DONE.", os.environ.get("TPU_WORKER_ID")
        )
        return

    # Per-run summary JSON to the central results prefix. This is the canonical
    # feed for scaling-law plots -- one flat file per completed run with plan,
    # method, model, tokens, final-eval metrics. See `_build_summary`.
    #
    # Source of truth for `final_eval` is W&B's `run.summary` (canonical record),
    # with eval_metrics.jsonl as fallback. Both paths assert the captured step
    # matches plan.train_steps to catch the preempt-resume / partial-write race
    # that has produced wrong summaries in this sweep before. If neither source
    # is trustworthy we skip the summary AND DONE marker so the coordinator
    # retries the run on next launch instead of silently shipping bad numbers.
    final_eval = _resolve_final_eval(
        run_name=run_name,
        output_path=output_path,
        plan=plan,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    if final_eval is None:
        logger.error(
            "Could not source trustworthy final-eval for %s from either W&B or "
            "eval_metrics.jsonl. Skipping summary + DONE marker; the coordinator "
            "will retry this run on next launch.",
            run_name,
        )
        return
    summary = _build_summary(
        plan=plan,
        method=method,
        region=region,
        run_name=run_name,
        output_path=output_path,
        final_eval=final_eval,
    )
    _write_summary(summary, args.results_prefix, run_name)
    # When a run-suffix is in play (rescue/retry runs), also write the summary
    # under the canonical (no-suffix) name so the dashboard + plotter pick it
    # up without manual copy. The suffixed copy stays for provenance.
    if args.run_suffix.strip():
        _write_summary(summary, args.results_prefix, plan.run_name_core)

    # 8. Write a completion marker so the coordinator can skip this run on
    #    future launches. Same purpose as Marin's ExecutorStep STATUS_SUCCESS.
    done_marker_path = f"{output_path}/.data_curation_DONE"
    marker_payload = json.dumps(
        {
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "run_name_core": plan.run_name_core,
            "method": plan.method_name,
            "experiment_tag": plan.experiment_tag,
            "region": region,
            "train_steps": plan.train_steps,
        }
    )
    try:
        with fsspec.open(done_marker_path, "w") as f:
            f.write(marker_payload)
        logger.info("Wrote completion marker: %s", done_marker_path)
    except Exception as e:
        # Non-fatal -- training succeeded, we just couldn't write the marker.
        # Coordinator will retry this run next launch, but it'll be a cheap
        # resume (Levanter sees the existing checkpoints).
        logger.warning("Failed to write completion marker at %s: %s", done_marker_path, e)


if __name__ == "__main__":
    main()
