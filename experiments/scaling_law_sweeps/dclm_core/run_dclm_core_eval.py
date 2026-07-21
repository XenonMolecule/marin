# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the DCLM CORE 22-task eval against a single HuggingFace checkpoint.

This is the per-checkpoint runner that the sweep launcher invokes once per
model. It uses Levanter's lm-eval-harness adapter (TPU-native JAX inference)
to run the 22 CORE tasks, then applies DCLM's centering formula to produce
a CORE score.

Outputs a JSON of shape:
    {
        "checkpoint": "<gs:// or local path>",
        "run_name": "<derived from checkpoint or --run-name>",
        "lm_eval_raw": <full lm-eval-harness result dict>,
        "dclm": <compute_core(...) output, including 'Core', 'Core_v2', per-task centered, etc.>
    }

The mapping from DCLM task names to lm-eval-harness task names + per-task
metrics lives in `task_mapping.py`. That mapping is the dominant source of
divergence from DCLM-published numbers — calibrate against a published
DCLM reference model before trusting results.

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval \\
        --hf-checkpoint gs://marin-eu-west4/.../hf/step-56002 \\
        --output-json gs://marin-us-central1/metadata/data_curation_core_results/<run>.json \\
        --tokenizer meta-llama/Meta-Llama-3.1-8B \\
        [--limit 8]   # smoke-test: cap each task to 8 examples
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import time
import typing
from pathlib import Path

# NOTE: this eval runs the HF Hub ONLINE. A warmed-hub-cache + HF_HUB_OFFLINE=1 path
# was tried to dodge the cold-start Hub rate limit, but the shared core_tasks_hub_cache
# only covers the CORE_TASKS datasets, not the DCLM CORE 22-task set, so task loading
# failed offline ("run the library in offline mode"). The rate limit ("We had to rate
# limit you") is instead TRANSIENT — ridden out by _load_hf_with_retry (below) plus
# iris job retries, exactly as the 246-run 10k CORE sweep completed. Do not re-add a
# blanket offline flag without first warming a hub cache for THIS task set's datasets.

import haliax as hax
import jmp
import levanter.eval_harness as eval_harness
from haliax.partitioning import round_axis_for_partitioning
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.eval_harness import TaskConfig
from levanter.models.lm_model import LmHeadModel
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from rigging.filesystem import filesystem as marin_filesystem
from rigging.timing import ExponentialBackoff, retry_with_backoff

from experiments.scaling_law_sweeps.dclm_core.centering import compute_core
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP, TaskMapEntry

logger = logging.getLogger(__name__)


# In-task progress visibility: bridge lm-eval / Levanter's tqdm bars to wandb.
# Without this, a single bigbench_language_identification task can take 30-90
# min with zero visible progress in the wandb run; we only see metrics at
# task-completion boundaries. The bridge throttles to one wandb.log per 10s
# (tqdm refreshes ~10x/sec by default) and never raises if wandb is unavailable.
_LAST_WANDB_LOG_TS: dict[str, float] = {"t": 0.0}


def _install_tqdm_wandb_bridge(task_alias: str, throttle_seconds: float = 10.0) -> None:
    """Forward tqdm progress updates to the active wandb run.

    Idempotent: the underlying tqdm.refresh patch is installed once per
    process; subsequent calls only update the current task alias used as a
    metric prefix. Silent no-op on non-master processes (wandb.run is None).
    """
    import time

    import tqdm.std

    tqdm.std.tqdm._dclm_task_alias = task_alias  # type: ignore[attr-defined]

    if getattr(tqdm.std.tqdm.refresh, "_dclm_patched", False):
        return

    orig_refresh = tqdm.std.tqdm.refresh

    def patched_refresh(self, *args, **kwargs):
        ret = orig_refresh(self, *args, **kwargs)
        try:
            import wandb

            if wandb.run is None:
                return ret
            now = time.time()
            if now - _LAST_WANDB_LOG_TS["t"] < throttle_seconds:
                return ret
            _LAST_WANDB_LOG_TS["t"] = now
            alias = getattr(tqdm.std.tqdm, "_dclm_task_alias", "unknown_task")
            fmt = self.format_dict
            payload = {
                f"progress/{alias}/n": self.n,
                f"progress/{alias}/total": self.total or 0,
                f"progress/{alias}/rate": fmt.get("rate") or 0,
                f"progress/{alias}/elapsed_s": fmt.get("elapsed") or 0,
            }
            if self.total:
                payload[f"progress/{alias}/pct"] = 100.0 * self.n / self.total
            wandb.log(payload)
        except Exception:
            # Monitoring must never break the eval.
            pass
        return ret

    patched_refresh._dclm_patched = True  # type: ignore[attr-defined]
    tqdm.std.tqdm.refresh = patched_refresh  # type: ignore[method-assign]


_CUSTOM_TASKS_DIR = Path(__file__).parent / "custom_tasks"


def _materialize_resolved_custom_tasks() -> str:
    """Stage our custom_tasks/ to a temp dir, rewriting relative `data_files`
    in any YAML so it resolves correctly regardless of CWD.

    Why: lm-eval's TaskManager passes `dataset_kwargs.data_files` straight to
    HuggingFace datasets, which resolves relative paths against CWD. Our
    jeopardy.yaml lives next to its jeopardy_all.jsonl, so we rewrite the
    YAML to use the absolute path before lm-eval loads it.
    """
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="dclm_core_tasks_"))
    shutil.copytree(_CUSTOM_TASKS_DIR, tmp, dirs_exist_ok=True)

    # Resolve relative data_files paths in every YAML we just copied.
    for yaml_file in tmp.rglob("*.yaml"):
        text = yaml_file.read_text()
        orig = text
        # Match `data_files: <relative-path>` (no leading slash) and rewrite
        # to absolute path adjacent to the (already-staged) YAML.
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("data_files:"):
                rel = stripped.split(":", 1)[1].strip()
                if rel and not rel.startswith("/") and not rel.startswith("hf://"):
                    abs_path = yaml_file.parent / rel
                    new_line = line.replace(stripped, f"data_files: {abs_path}")
                    text = text.replace(line, new_line)
        if text != orig:
            yaml_file.write_text(text)

    return str(tmp)


def _install_custom_task_path():
    """Patch lm_eval.tasks.TaskManager so calls without include_path pick up
    our (path-resolved) custom_tasks/ dir. Levanter's eval harness constructs
    TaskManager() with no args; we need our DCLM-CORE custom YAMLs (e.g.
    jeopardy) findable.
    """
    resolved_path = _materialize_resolved_custom_tasks()

    import lm_eval.tasks as tasks

    original_init = tasks.TaskManager.__init__

    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            include_path = resolved_path
        original_init(self, *args, include_path=include_path, **kwargs)

    tasks.TaskManager.__init__ = patched_init


def build_task_configs(limit: int | None) -> tuple[list[TaskConfig], dict[str, TaskMapEntry]]:
    """Build the 22 TaskConfigs and an alias→entry map for result extraction.

    The alias is `<dclm_name>_<num_fewshot>shot` to disambiguate cases where
    the same lm-eval task is run at multiple shot counts (e.g. hellaswag 0
    vs 10 shot). Levanter uses task_alias as the result key.
    """
    tasks: list[TaskConfig] = []
    alias_to_entry: dict[str, TaskMapEntry] = {}
    for entry in CORE_TASK_MAP:
        alias = f"{entry.dclm}_{entry.num_fewshot}shot"
        tasks.append(
            TaskConfig(
                task=entry.lm_eval,
                task_alias=alias,
                num_fewshot=entry.num_fewshot,
            )
        )
        alias_to_entry[alias] = entry
    return tasks, alias_to_entry


def extract_dclm_results(
    lm_eval_results: dict,
    alias_to_entry: dict[str, TaskMapEntry],
) -> tuple[dict[str, float], dict[str, str]]:
    """Pull per-task metric values from lm-eval's nested result dict.

    Returns (dclm_named_raw_results, extraction_log).
    """
    results_section = lm_eval_results.get("results", {})
    raw: dict[str, float] = {}
    log: dict[str, str] = {}
    for alias, entry in alias_to_entry.items():
        # lm-eval's results dict keys by the alias we set. Metric keys are
        # suffixed with ",stderr" / ",none" / similar — pick the plain key.
        task_results = results_section.get(alias)
        if task_results is None:
            log[entry.dclm] = f"MISSING_TASK_RESULTS (looked for alias {alias!r})"
            continue
        wanted = entry.metric
        # lm-eval's per-task results dict has keys like "acc,none", "acc_stderr,none",
        # "acc_norm,none". Pick the exact metric (no stderr).
        candidates = [k for k in task_results if k == wanted or k.startswith(f"{wanted},")]
        candidates = [k for k in candidates if "stderr" not in k]
        if not candidates:
            log[entry.dclm] = f"MISSING_METRIC: wanted {wanted!r}; available: {sorted(task_results)}"
            continue
        value = task_results[candidates[0]]
        if not isinstance(value, (int, float)):
            log[entry.dclm] = f"NON_NUMERIC_METRIC: {value!r}"
            continue
        raw[entry.dclm] = float(value)
        log[entry.dclm] = f"OK from {candidates[0]} = {value}"
    return raw, log


def _open_for_write(path: str):
    """Return a file-like object for either local or gs:// paths."""
    if path.startswith("gs://"):
        return marin_filesystem("gcs").open(path, "w")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w")


def _open_for_read(path: str):
    """Return a file-like object for either local or gs:// paths. Raises FileNotFoundError."""
    if path.startswith("gs://"):
        fs = marin_filesystem("gcs")
        if not fs.exists(path):
            raise FileNotFoundError(path)
        return fs.open(path, "r")
    return open(path, "r")


def _path_exists(path: str) -> bool:
    """Best-effort existence check across gs:// and local paths.

    fsspec's gcsfs is required because iris worker containers don't ship the
    `gcloud` CLI. We already use marin_filesystem("gcs") elsewhere for writes.
    """
    if path.startswith("gs://"):
        try:
            return marin_filesystem("gcs").exists(path)
        except Exception as e:
            logger.warning("gcs exists() check failed for %s: %s — assuming missing.", path, e)
            return False
    return Path(path).exists()


def _resolve_partial_dir(output_json: str, run_name: str) -> str:
    """Where to stash per-task partials.

    e.g. gs://marin-us-central1/metadata/data_curation_core_results/<run>.json
         → gs://marin-us-central1/metadata/data_curation_core_results/partial/<run>
    """
    parent = output_json.rsplit("/", 1)[0]
    return f"{parent}/partial/{run_name}"


def _partial_path(partial_dir: str, task_alias: str) -> str:
    return f"{partial_dir.rstrip('/')}/{task_alias}.json"


def _read_partial(path: str) -> dict | None:
    """Try to load a per-task partial. Returns None if missing or unreadable."""
    if not _path_exists(path):
        return None
    try:
        with _open_for_read(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read partial %s: %s", path, e)
        return None


def _write_partial(path: str, data: dict) -> None:
    with _open_for_write(path) as f:
        json.dump(data, f, indent=2, default=_json_default)


def _merge_lm_eval_outputs(partials: list[dict]) -> dict:
    """Merge per-task lm-eval output dicts into one combined output.

    lm-eval's output dict has top-level keys 'results', 'configs', 'versions',
    'samples', etc., each keyed by the task alias. Since we ran each task in
    isolation, the per-alias entries are disjoint and we can concatenate them.

    For top-level scalar keys (no task aliasing), we take the value from any
    partial (they should match across single-task runs).
    """
    merged: dict = {}
    aliased_keys = {"results", "configs", "versions", "samples", "n-shot", "higher_is_better"}
    for partial in partials:
        for key, value in partial.items():
            if key in aliased_keys and isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged.setdefault(key, value)
    return merged


def _is_retryable_hf_error(exc: Exception) -> bool:
    """True for the transient HF Hub failures a cold-start herd produces.

    Covers the raw connection errors AND the `expected str, bytes or os.PathLike
    object, not NoneType` TypeError: levanter's `from_hf` probes every model type
    to find the match, and building the llama probe fetches a tokenizer from the
    Hub. Under contention that fetch fails and propagates as None into a path op,
    surfacing as a TypeError rather than a clean connection error. Both retry
    cleanly once the herd thins (retry re-runs the whole from_hf).
    """
    msg = str(exc).lower()
    return (
        "couldn't connect to" in msg
        or "huggingface.co" in msg
        or "connection" in msg
        or "timed out" in msg
        or "not nonetype" in msg
        # The 1000-req/5min Hub quota under a cold-start herd. Transient — the
        # message names the quota explicitly; match it directly rather than lean
        # on the incidental hf.co URL so a reworded error still retries.
        or "rate limit" in msg
        or "you hit the quota" in msg
        # A rate-limited/failed HF fetch inside from_hf's model-type probe returns
        # None, which then blows up as a None-attribute error (`… has no attribute
        # 'endswith'` / `'init'`). Scoped to the wrapped model load, so retrying is
        # safe. Retry re-runs the whole from_hf once the herd thins.
        or "nonetype' object has no attribute" in msg
    )


def _load_hf_with_retry(what: str, fn):
    """Run an HF-touching load, retrying through transient Hub connection errors.

    Many evals cold-starting together overwhelm the HF Hub on config/tokenizer
    resolution; individual jobs then fail with "couldn't connect to
    huggingface.co" or the 1000-req/5min rate limit ("We had to rate limit you").
    Retry with jittered exponential backoff (up to ~40 min) so a large-fleet herd
    fully thins and each job eventually resolves rather than exhausting retries
    while the Hub is still throttling. Non-connection errors (a real bad
    checkpoint) are NOT retried — they re-raise immediately.
    """
    return retry_with_backoff(
        fn,
        retryable=_is_retryable_hf_error,
        max_attempts=24,
        max_elapsed=2400.0,
        backoff=ExponentialBackoff(initial=5.0, maximum=120.0, factor=2.0, jitter=0.5),
        on_retry=lambda e, i: logger.warning("HF load %s failed (attempt %d), backing off: %s", what, i, e),
        operation=what,
    )


_DATASET_CACHE_MARKER = "/eval_datasets/dclm_core_hf_cache/"
# The warmed HF hub-config cache (gpt2 + the other reference model configs levanter's
# from_hf model-type probe loads). Shared with the CORE_TASKS eval — the probe is
# checkpoint-independent (it iterates every registered Levanter model), so the same
# cache resolves from_hf for any checkpoint. Built by
# experiments/scaling_law_sweeps/core_tasks/build_core_tasks_dataset_cache.py.
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"


def _prepare_hub_cache(gcs_cache: str) -> int:
    """Sync the warmed HF hub-config cache from in-region GCS into HF_HOME so
    from_hf's reference-config probe (gpt2, ...) resolves OFFLINE — zero Hub
    requests, immune to the shared-token rate limit. HF_HUB_OFFLINE/HF_HOME are
    already set at module import; this only populates HF_HOME/hub before from_hf runs.
    """
    local_dir = os.environ.get("HF_HOME", "/tmp/dclm_core_hf_home")
    base = gcs_cache.rstrip("/")
    fs = marin_filesystem("gcs")
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for remote in fs.find(base):  # gcsfs returns scheme-less bucket/key paths
        key = remote.split(_HUB_CACHE_MARKER, 1)[-1]
        dest = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = remote if remote.startswith("gs://") else f"gs://{remote}"
        with fs.open(src, "rb") as r, open(dest, "wb") as w:
            w.write(r.read())
        n += 1
    logger.info("Synced %d hub-cache files %s -> %s (HF_HOME, offline)", n, base, local_dir)
    return n


def _prepare_offline_dataset_cache(gcs_cache: str, local_dir: str = "/tmp/dclm_core_hf_cache") -> int:
    """Sync the canonical HF datasets cache from in-region GCS to local disk and
    switch `datasets` to OFFLINE mode.

    Eval task loading otherwise pulls all 22 CORE datasets from the HF Hub during
    the run — the dominant, sustained source of HF API requests that trips the
    1000-req/5min rate limit across a large fleet. Reading from a pre-mirrored,
    byte-identical in-region cache removes those calls entirely. Must run before
    any `datasets`/lm-eval import reads the cache env. Returns files synced.
    """
    base = gcs_cache.rstrip("/")
    fs = marin_filesystem("gcs")
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for remote in fs.find(base):  # gcsfs returns scheme-less bucket/key paths
        key = remote.split(_DATASET_CACHE_MARKER, 1)[-1]
        dest = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = remote if remote.startswith("gs://") else f"gs://{remote}"
        with fs.open(src, "rb") as r, open(dest, "wb") as w:
            w.write(r.read())
        n += 1
    os.environ["HF_DATASETS_CACHE"] = local_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    logger.info("Synced %d dataset-cache files %s -> %s; HF_DATASETS_OFFLINE=1", n, base, local_dir)
    return n


def _resilient_lev_config(checkpoint: str):
    """LevConfig class for a checkpoint, resilient to from_hf's offline registry scan.

    from_hf probes EVERY registered config's default HF reference; under HF_HUB_OFFLINE a
    NON-matching config whose default repo isn't resolvable offline (gpt2/gemma/... in the
    eval container) crashes the probe DETERMINISTICALLY (retries don't help). Fall back to
    resolving the LevConfig directly from the checkpoint's own model_type via the registry.
    """
    try:
        return HFCheckpointConverter.from_hf(checkpoint).LevConfigClass
    except Exception:
        import json

        import fsspec
        from levanter.models.lm_model import LmConfig

        with fsspec.open(f"{checkpoint.rstrip('/')}/config.json", "r") as f:
            model_type = json.load(f)["model_type"]
        return LmConfig.get_known_choices()[model_type]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-checkpoint",
        required=True,
        help="Path to HF checkpoint (gs:// or local). Must contain config.json + model.safetensors + tokenizer files.",
    )
    p.add_argument("--output-json", required=True, help="Where to write the final result JSON (gs:// or local).")
    p.add_argument(
        "--summary-json",
        default=None,
        help="Where to write the small scores-only summary sibling (no samples). Defaults next to "
        "--output-json. Point this at a PERMANENT prefix while --output-json goes to a ttl= prefix, "
        "so scores persist forever while sample-heavy finals/partials auto-expire.",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer override. Defaults to the checkpoint path (Levanter picks it up from there).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to embed in the output JSON (otherwise derived from checkpoint).",
    )
    p.add_argument("--limit", type=int, default=None, help="Smoke test: cap each task to N examples. None = full eval.")
    p.add_argument("--max-length", type=int, default=2048, help="Max sequence length for eval. DCLM uses 2048.")
    p.add_argument("--wandb-project", default="marin-dclm-core", help="WandB project (set empty string to disable).")
    p.add_argument("--wandb-tags", action="append", default=[], help="WandB tag (can be repeated).")
    p.add_argument(
        "--log-samples",
        action="store_true",
        help="If set, persist per-example prompts/generations/scores in each task's "
        "partial JSON under `samples`. Useful for debugging score divergence vs "
        "DCLM reference. Costs more storage (~MB per task) and a bit of wall time.",
    )
    p.add_argument(
        "--task-filter",
        default=None,
        help="Comma-separated DCLM task names. If set, only these tasks run (and partials "
        "for other tasks are ignored, not skipped). Use to debug a single task "
        "(e.g. --task-filter squad,arc_easy).",
    )
    p.add_argument(
        "--dataset-cache-gcs",
        default=None,
        help="GCS prefix of the canonical HF datasets cache (in this checkpoint's region). "
        "If set, sync it to local disk and run datasets OFFLINE so task loading never hits "
        "the HF Hub API (which rate-limits at 1000 req/5min and blocks large eval fleets).",
    )
    p.add_argument(
        "--hub-cache-gcs",
        default=None,
        help="GCS prefix of the warmed HF hub-config cache (core_tasks_hub_cache, in this "
        "checkpoint's region). Synced into HF_HOME so from_hf's reference-config probe "
        "(gpt2, ...) resolves offline — the other half of making the eval hermetic (models, "
        "not just datasets). Without it a large fleet blows the Hub rate limit on cold start.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Must happen before any from_hf / `datasets` / lm-eval load. The hub cache
    # feeds from_hf's offline reference-config probe; the dataset cache feeds
    # offline task loading. Both make the eval hermetic vs the shared-token Hub limit.
    if args.hub_cache_gcs:
        _prepare_hub_cache(args.hub_cache_gcs)
    if args.dataset_cache_gcs:
        _prepare_offline_dataset_cache(args.dataset_cache_gcs)

    _install_custom_task_path()

    checkpoint_path: str = args.hf_checkpoint.rstrip("/")
    tokenizer_path: str = args.tokenizer or checkpoint_path
    run_name = args.run_name or checkpoint_path.split("/")[-3]  # parent of /hf/step-N
    partial_dir = _resolve_partial_dir(args.output_json, run_name)

    logger.info("DCLM CORE eval: checkpoint=%s run_name=%s", checkpoint_path, run_name)
    logger.info("Per-task partials: %s", partial_dir)

    tasks, alias_to_entry = build_task_configs(args.limit)
    logger.info("Built %d task configs", len(tasks))

    # --- Optional task filter (debug single-task runs) -----------------------
    if args.task_filter:
        keep = {t.strip() for t in args.task_filter.split(",")}
        tasks = [t for t in tasks if any(t.task_alias.startswith(f"{k}_") for k in keep)]
        logger.info("--task-filter narrowed to %d task(s): %s", len(tasks), [t.task_alias for t in tasks])

    # --- Skip-if-done: load any existing per-task partials -------------------
    done_partials: dict[str, dict] = {}
    missing_tasks: list[TaskConfig] = []
    for task in tasks:
        ppath = _partial_path(partial_dir, task.task_alias)
        existing = _read_partial(ppath)
        if existing is not None:
            done_partials[task.task_alias] = existing
            logger.info("  RESUME ✓ %s (partial exists)", task.task_alias)
        else:
            missing_tasks.append(task)
    logger.info("Per-task status: %d done, %d to run", len(done_partials), len(missing_tasks))

    # --- Run any missing tasks, checkpointing after each ---------------------
    if missing_tasks:
        wandb_cfg = (
            WandbConfig(project=args.wandb_project, tags=args.wandb_tags, name=run_name) if args.wandb_project else None
        )
        trainer_config = TrainerConfig(
            tracker=wandb_cfg,
            mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
            per_device_eval_parallelism=1,
        )
        # Stagger the cold-start HF Hub hit: when hundreds of evals launch together
        # they thundering-herd AutoConfig resolution and most fail with "couldn't
        # connect to huggingface.co". An upfront jitter spreads the first hit; the
        # retry wrapper rides out whatever contention remains.
        stagger = random.uniform(0, 60)
        logger.info("Staggering HF cold-start by %.1fs to avoid thundering-herd", stagger)
        time.sleep(stagger)
        model_config = _load_hf_with_retry(
            "HFCheckpointConverter.from_hf", lambda: _resilient_lev_config(checkpoint_path)()
        )

        trainer_config.initialize()
        tokenizer = _load_hf_with_retry("load_tokenizer", lambda: load_tokenizer(tokenizer_path))
        compute_axis_mapping = trainer_config.compute_axis_mapping
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        with trainer_config.use_device_mesh():
            vocab_size = len(tokenizer)
            Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)
            if vocab_size != Vocab.size:
                logger.info("Rounding vocab size from %d to %d for partitioning", vocab_size, Vocab.size)

            converter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=checkpoint_path, tokenizer=tokenizer)
            logger.info("Loading HF checkpoint from %s", checkpoint_path)
            model = converter.load_pretrained(
                model_config.model_type,
                ref=checkpoint_path,
                dtype=trainer_config.mp.compute_dtype,
                axis_mapping=parameter_axis_mapping,
            )
            model = typing.cast(LmHeadModel, inference_mode(model, True))
            logger.info("Model loaded; running %d tasks per-task with checkpointing", len(missing_tasks))

            for i, task in enumerate(missing_tasks):
                logger.info(
                    "[%d/%d] Evaluating task %s (alias=%s, num_fewshot=%d)",
                    i + 1,
                    len(missing_tasks),
                    task.task,
                    task.task_alias,
                    task.num_fewshot,
                )
                _install_tqdm_wandb_bridge(task.task_alias)
                single_config = eval_harness.LmEvalHarnessConfig(
                    task_spec=[task],
                    max_examples=args.limit,
                    max_length=args.max_length,
                    log_samples=args.log_samples,
                    confirm_run_unsafe_code=True,
                )
                outputs = eval_harness.run_lm_eval_harness(
                    single_config,
                    model,
                    tokenizer,
                    trainer_config.EvalBatch,
                    axis_resources=compute_axis_mapping,
                    mp=trainer_config.mp,
                )
                if outputs is None:
                    # Non-master process: skip writing but continue the loop so we stay in
                    # the device mesh context for subsequent tasks.
                    continue
                ppath = _partial_path(partial_dir, task.task_alias)
                _write_partial(ppath, outputs)
                done_partials[task.task_alias] = outputs
                logger.info("  ✓ wrote partial %s", ppath)

    # Non-master process bails before aggregation.
    if not done_partials:
        logger.info("Non-master process: nothing to aggregate.")
        return

    # --- Aggregate partials → final JSON ------------------------------------
    # CRITICAL: always read ALL partials from disk at aggregation time. A run
    # with --task-filter (e.g. tail3 jobs that only re-run squad / coqa /
    # bigbench_language_identification) has only those tasks in `done_partials`,
    # but the partial dir on GCS may also contain the other 19 written by a
    # sibling main job. Aggregating from `done_partials.values()` here would
    # produce a 3-task "final" and OVERWRITE the main's good final JSON. This
    # was the actual root cause of the 2026-05-23 N/A bug on 4 priority configs.
    all_partials_on_disk: list[dict] = []
    full_task_aliases, _ = build_task_configs(args.limit)
    for task in full_task_aliases:
        ppath = _partial_path(partial_dir, task.task_alias)
        loaded = _read_partial(ppath)
        if loaded is not None:
            all_partials_on_disk.append(loaded)
    logger.info(
        "Aggregating from %d partial JSONs on disk (in-memory done_partials had %d)",
        len(all_partials_on_disk),
        len(done_partials),
    )
    if len(all_partials_on_disk) < len(CORE_TASK_MAP):
        logger.warning(
            "Only %d / %d CORE partials on disk — final Core will be marked N/A. "
            "Skipping the final-JSON write to avoid overwriting a sibling's good final.",
            len(all_partials_on_disk),
            len(CORE_TASK_MAP),
        )
        return
    combined_results = _merge_lm_eval_outputs(all_partials_on_disk)
    raw, extraction_log = extract_dclm_results(combined_results, alias_to_entry)
    logger.info("Extracted %d / %d CORE task scores", len(raw), len(CORE_TASK_MAP))
    for dclm_task, status in extraction_log.items():
        logger.info("  %s: %s", dclm_task, status)

    dclm_aggregation = compute_core(raw)

    output = {
        "checkpoint": checkpoint_path,
        "run_name": run_name,
        "tokenizer": tokenizer_path,
        "args": {
            "limit": args.limit,
            "max_length": args.max_length,
        },
        "partial_dir": partial_dir,
        "extraction_log": extraction_log,
        "dclm": dclm_aggregation,
        "lm_eval_raw": combined_results,
    }

    with _open_for_write(args.output_json) as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Wrote results to %s", args.output_json)
    logger.info("CORE score: %s", dclm_aggregation.get("Core"))

    # Also write a small summary sibling so dashboards/analyses can read only
    # the score + extraction log without pulling the ~1 GB final (which is
    # dominated by `lm_eval_raw.samples` when log_samples=True).
    summary = {
        "run_name": run_name,
        "checkpoint": checkpoint_path,
        "tokenizer": tokenizer_path,
        "partial_dir": partial_dir,
        "dclm": dclm_aggregation,
        "extraction_log": extraction_log,
    }
    summary_path = args.summary_json or (args.output_json.rsplit(".json", 1)[0] + "_summary.json")
    with _open_for_write(summary_path) as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info("Wrote summary to %s", summary_path)


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    return repr(value)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # finelog/`iris job logs` is unreliable during outages, so persist the traceback to GCS
        # next to the summary path (<run>_error.txt) for offline diagnosis.
        import sys as _sys
        import traceback as _tb

        _tbtext = _tb.format_exc()
        try:
            _av = _sys.argv
            _sp = None
            for _i, _a in enumerate(_av):
                if _a == "--summary-json":
                    _sp = _av[_i + 1]
                elif _a == "--output-json" and _sp is None:
                    _sp = _av[_i + 1]
            if _sp:
                _errp = _sp.replace("_summary.json", "_error.txt")
                import fsspec as _fsspec

                with _fsspec.open(_errp, "w") as _f:
                    _f.write(_tbtext)
        except Exception:
            pass
        raise
