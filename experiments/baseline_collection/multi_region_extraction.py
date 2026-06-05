# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-region LLM extraction orchestrator.

Dispatches batches of WARC files to Iris jobs across multiple regions and TPU
types. Iris places each job wherever compute is available — no region pinning.
Each job downloads its assigned WARCs from CommonCrawl (free ingress) and writes
extracted text to the worker's local region bucket.

The orchestrator is **fully re-entrant**: all progress state lives in GCS output
files. Kill and restart at any time — it scans completed hashes and picks up
where it left off. In-flight Iris jobs continue running independently.

Multi-spec
----------
Each run targets exactly one ``--spec`` (key into the registry in
``extraction_specs.py``). Output is namespaced per-spec on GCS:

    gs://{regional_bucket}/documents/baseline_llm_extraction/{spec_id}/data-{warc_hash}/

Priority + manifest
-------------------
``--manifest`` selects the WARC list; ``--exclude-manifest`` (repeatable)
subtracts WARCs (by hash) before dispatch. ``--iris-priority`` plumbs the
priority band into ``iris job run --priority``. Run several instances of this
script in parallel for different (spec, priority-tier) pairs; they compose
through the existing GCS ``_done`` skip-list and the steal-mode batch claims.

Usage::

    # priority tier (interactive)
    python -m experiments.baseline_collection.multi_region_extraction \\
        --spec default_v1 \\
        --manifest experiments/distill/subsets/baseline_warcs_100.txt \\
        --iris-priority interactive

    # overflow tier (batch), excluding the priority WARCs while they're in-flight
    python -m experiments.baseline_collection.multi_region_extraction \\
        --spec default_v1 \\
        --manifest experiments/distill/baseline_warcs_3000.txt \\
        --exclude-manifest experiments/distill/subsets/baseline_warcs_100.txt \\
        --iris-priority batch
"""

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field

from iris.rpc import job_pb2
from rigging.filesystem import REGION_TO_DATA_BUCKET

from experiments.baseline_collection.download_warcs import (
    _load_manifest,
    _warc_path_hash,
)
from experiments.baseline_collection.extraction_specs import (
    LEGACY_SPEC_ID,
    SPECS,
    ExtractionSpec,
    get_spec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fleet configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TpuFleetEntry:
    """A TPU type to dispatch work to.

    The TPU variant is requested by Zephyr worker tasks (not the entrypoint
    job, which runs on CPU). See ``download_and_extract.py`` — it reads
    ``tpu_variant`` from the worker config to build ``ResourceConfig``.

    ``region`` pins the iris ``--region`` flag for the parent so it lands in
    a region that actually has the variant. Iris auto-inherits this region
    constraint to Zephyr child tasks; without it, the parent may land in a
    region (e.g. us-east1 for CPU jobs) where the requested TPU variant
    isn't deployed and the children fail unschedulable.
    """

    tpu_type: str
    """TPU variant requested by Zephyr workers (e.g. ``v5p-8``, ``v6e-8``)."""

    region: str = ""
    """Optional region pin for iris ``--region``. See class docstring."""

    num_workers: int = 16
    """Zephyr workers per job."""

    batch_size: int = 50
    """WARCs per dispatch batch."""


# All TPU types to use — Iris places them wherever available.
# Add/remove entries as compute opens up.
FLEET: list[TpuFleetEntry] = [
    TpuFleetEntry(tpu_type="v5p-8", num_workers=16, batch_size=50),
    TpuFleetEntry(tpu_type="v6e-8", num_workers=16, batch_size=50),
    # TpuFleetEntry(tpu_type="v6e-4", num_workers=32, batch_size=50),
]


# ---------------------------------------------------------------------------
# Extraction configuration
# ---------------------------------------------------------------------------

# Model weights pre-copied to each regional bucket.
# Keys are canonical GCP region names (matching REGION_TO_DATA_BUCKET).
# Specs in the registry currently share this checkpoint; if you need a
# spec-specific model, add a model_ckpt_subpath field to ExtractionSpec.
_REPHRASER_CKPT = "checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

MODEL_BY_REGION: dict[str, str] = {
    region: f"gs://{bucket}/{_REPHRASER_CKPT}" for region, bucket in REGION_TO_DATA_BUCKET.items()
}

# Base output subdirectory. Per-spec outputs live at f"{OUTPUT_SUBDIR}/{spec_id}".
OUTPUT_SUBDIR = "documents/baseline_llm_extraction"

# Master WARC manifest (default for --manifest)
DEFAULT_WARC_MANIFEST = "experiments/distill/baseline_warcs_3000.txt"

# Batch manifests are written here (tiny files, write-once)
MANIFEST_STAGING_BUCKET = "gs://marin-us-central2"
MANIFEST_STAGING_DIR = f"{MANIFEST_STAGING_BUCKET}/tmp/extraction_manifests"

# Iris cluster config — passed as a top-level flag to the iris CLI.
# Override at the CLI with --iris-config or via IRIS_CONFIG env var.
DEFAULT_IRIS_CONFIG = "lib/iris/examples/marin.yaml"


# Iris priority band names accepted on the CLI. Maps to job_pb2 enum values
# only when validating; the names themselves are forwarded to ``iris job run``.
_PRIORITY_BAND_NAMES = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def _output_subdir_for(spec_id: str) -> str:
    """GCS subdirectory for a spec's outputs, relative to the regional bucket.

    The legacy spec writes to the unprefixed path (where pre-registry data
    already lives); all others nest under their id.
    """
    if spec_id == LEGACY_SPEC_ID:
        return OUTPUT_SUBDIR
    return f"{OUTPUT_SUBDIR}/{spec_id}"


def _all_output_dirs(spec_id: str) -> list[str]:
    """All regional output URIs to scan for completed files for a spec."""
    sub = _output_subdir_for(spec_id)
    return [f"gs://{bucket}/{sub}" for bucket in REGION_TO_DATA_BUCKET.values()]


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _scan_completed_hashes(output_dirs: list[str]) -> set[str]:
    """Scan all regional output directories for completed WARC hashes.

    Returns the set of 12-char hex hashes found in filenames like
    ``data-{hash}.jsonl.gz``.

    Uses google-cloud-storage directly (not fsspec/gcsfs) to dodge SSL
    cert issues with aiohttp on some local environments.
    """
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    completed: set[str] = set()
    for output_dir in output_dirs:
        # output_dir is gs://{bucket}/{subdir}
        if not output_dir.startswith("gs://"):
            continue
        rest = output_dir[len("gs://") :]
        bucket_name, _, prefix = rest.partition("/")
        if not bucket_name:
            continue
        prefix = prefix.rstrip("/") + "/"
        try:
            bucket = client.bucket(bucket_name)
            for blob in bucket.list_blobs(prefix=prefix + "data-"):
                # Only count files at the immediate level — nested specs
                # (e.g. ``low_quality`` scanning the unprefixed path would
                # otherwise pick up ``med_quality/data-*.jsonl.gz``) have
                # additional path segments and belong to a different spec.
                rel = blob.name[len(prefix) :]
                if "/" in rel:
                    continue
                if rel.startswith("data-") and rel.endswith(".jsonl.gz"):
                    h = rel[len("data-") : -len(".jsonl.gz")]
                    if len(h) == 12:
                        completed.add(h)
        except Exception as e:
            # Bucket may not exist or be inaccessible — skip but warn.
            logger.warning("Scan failed for %s: %s", output_dir, e)
            continue
    return completed


# ---------------------------------------------------------------------------
# Batch manifest management
# ---------------------------------------------------------------------------


def _gcs_write_text(gs_uri: str, text: str) -> None:
    """Upload a small text blob to GCS. Uses google-cloud-storage to dodge
    the gcsfs/aiohttp SSL issue that hits some local environments."""
    from google.cloud import storage as gcs_storage

    if not gs_uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {gs_uri!r}")
    bucket_name, _, blob_path = gs_uri[len("gs://") :].partition("/")
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_string(text)


def _write_batch_manifest(warc_paths: list[str], batch_id: str) -> str:
    """Write a batch manifest to GCS. Returns the GCS path."""
    manifest_path = f"{MANIFEST_STAGING_DIR}/batch_{batch_id}.txt"
    content = "\n".join(warc_paths) + "\n"
    _gcs_write_text(manifest_path, content)
    logger.info("Wrote batch manifest: %s (%d WARCs)", manifest_path, len(warc_paths))
    return manifest_path


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def _build_config_json(spec: ExtractionSpec, entry: TpuFleetEntry, manifest_path: str) -> dict:
    """Build the DownloadAndExtractConfig as a JSON-serializable dict.

    ``tpu_variant`` is consumed by the worker to construct the Zephyr
    ResourceConfig — the entrypoint itself runs on CPU under the new Iris
    pattern (entrypoint = coordinator, workers carry accelerators).
    """
    return {
        "warc_manifest_path": manifest_path,
        "output_subdir": _output_subdir_for(spec.spec_id),
        "model_name_by_region": MODEL_BY_REGION,
        "template": spec.extraction_template,
        "system_message": spec.system_message,
        "prompt_column": "html",
        "generated_text_column": "generated_text",
        "apply_chat_template": True,
        "max_doc_tokens": 28672,
        "engine_kwargs": {"max_model_len": 32768, "enable_prefix_caching": True},
        "generation_kwargs": {"temperature": 0.0, "max_tokens": 4096},
        "strip_thinking": True,
        "filter_patterns": [r"\[NO_USEFUL_CONTENT\]"],
        "min_output_chars": 50,
        "num_workers": entry.num_workers,
        "max_records_per_generate": 500,
        "http_timeout": 600,
        "max_retries": 5,
        "tpu_variant": entry.tpu_type,
    }


def _build_iris_command(
    entry: TpuFleetEntry,
    config_gcs: str,
    job_name: str,
    iris_priority: str | None,
    iris_config: str,
) -> list[str]:
    """Build the ``iris job run`` argv. Pure function for testability.

    ``--config`` is a top-level Iris flag; it must precede the ``job run`` subcommand.
    The entrypoint runs on CPU; the TPU variant is requested by Zephyr workers
    (see ``DownloadAndExtractConfig.tpu_variant``).
    """
    cmd = [
        "iris",
        "--config",
        iris_config,
        "job",
        "run",
        "--cpu",
        "2",
        "--memory",
        "2GB",
        "--no-wait",
        "--job-name",
        job_name,
        "--max-retries",
        "3",
        "--extra",
        "vllm",
    ]
    if entry.region:
        cmd += ["--region", entry.region]
    if iris_priority:
        cmd += ["--priority", iris_priority]
    cmd += [
        "--",
        "python",
        "-m",
        "experiments.baseline_collection.download_and_extract",
        "--config_path",
        config_gcs,
    ]
    return cmd


def _submit_iris_job(
    spec: ExtractionSpec,
    entry: TpuFleetEntry,
    manifest_path: str,
    batch_id: str,
    iris_priority: str | None,
    iris_config: str,
    dry_run: bool,
) -> str:
    """Submit an Iris job via CLI. Returns the job name for tracking."""
    config_dict = _build_config_json(spec, entry, manifest_path)

    # Write config to a temp GCS path
    config_gcs = f"{MANIFEST_STAGING_DIR}/config_{batch_id}.json"
    if not dry_run:
        _gcs_write_text(config_gcs, json.dumps(config_dict, indent=2))

    job_name = f"extract-{spec.spec_id}-{batch_id}"

    cmd = _build_iris_command(entry, config_gcs, job_name, iris_priority, iris_config)

    logger.info(
        "Submitting Iris job: %s (tpu=%s, %d WARCs, priority=%s)",
        job_name,
        entry.tpu_type,
        entry.batch_size,
        iris_priority or "<default>",
    )
    logger.info("Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("DRY-RUN: skipping actual submission")
        return job_name

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        # Iris emits info logs to stderr; the actual error lives near the end.
        # Show the tail (post-tunnel-setup chatter is at the top) and bump the
        # cap so messages aren't truncated mid-sentence.
        err_tail = result.stderr[-2000:]
        logger.error("Job submission failed (rc=%d):\n%s", result.returncode, err_tail)
        raise RuntimeError(
            f"iris job run failed (rc={result.returncode}): {err_tail.splitlines()[-1] if err_tail else '<no stderr>'}"
        )

    logger.info("Submitted job %s: %s", job_name, result.stdout.strip()[:200])
    return job_name


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DispatchState:
    """Tracks active jobs within a single orchestrator process.

    ``submitted_hashes`` is the per-process in-flight set: WARC hashes that
    THIS orchestrator already dispatched. Subsequent iterations subtract them
    from the working set so the orchestrator does not re-dispatch the same
    WARCs while their first job is still running. Multiple orchestrator
    processes don't share this set; cross-process dedup relies on GCS
    completion markers and Zephyr ``skip_existing``.
    """

    active_jobs: list[str] = field(default_factory=list)
    submitted_hashes: set[str] = field(default_factory=set)
    jobs_submitted: int = 0


def _load_excluded_hashes(exclude_manifests: list[str]) -> set[str]:
    """Load WARC hashes from manifests to subtract from the working set."""
    excluded: set[str] = set()
    for path in exclude_manifests:
        warcs = _load_manifest(path)
        for w in warcs:
            excluded.add(_warc_path_hash(w))
        logger.info("Exclusion manifest %s: %d WARCs (cumulative %d)", path, len(warcs), len(excluded))
    return excluded


def dispatch_loop(
    spec: ExtractionSpec,
    manifest_path: str,
    exclude_manifests: list[str],
    fleet: list[TpuFleetEntry],
    iris_priority: str | None,
    iris_config: str,
    poll_interval_seconds: int,
    max_concurrent_jobs: int,
    dry_run: bool,
) -> None:
    """Main orchestrator loop. Runs until all WARCs are extracted.

    Fully re-entrant: scans GCS for completed hashes on each iteration.
    Kill and restart at any time.
    """
    all_warcs_full = _load_manifest(manifest_path)
    excluded_hashes = _load_excluded_hashes(exclude_manifests)
    all_warcs = [w for w in all_warcs_full if _warc_path_hash(w) not in excluded_hashes]

    output_dirs = _all_output_dirs(spec.spec_id)

    logger.info("Spec: %s — %s", spec.spec_id, spec.description or "(no description)")
    logger.info(
        "Manifest %s: %d WARCs total, %d after exclusion (%d excluded)",
        manifest_path,
        len(all_warcs_full),
        len(all_warcs),
        len(all_warcs_full) - len(all_warcs),
    )
    logger.info("Fleet: %s", [(e.tpu_type, e.batch_size) for e in fleet])
    logger.info("Output namespace: %s — scanning %d regional dirs", _output_subdir_for(spec.spec_id), len(output_dirs))
    logger.info("Iris priority: %s", iris_priority or "<default>")
    if dry_run:
        logger.info("DRY-RUN mode: will not write configs or submit jobs.")

    if not all_warcs:
        logger.warning("Working set is empty after exclusion. Nothing to do.")
        return

    iteration = 0
    state = DispatchState()

    while True:
        iteration += 1
        logger.info("--- Dispatch iteration %d ---", iteration)

        # 1. Scan ALL regional buckets for completed output files. Skip the
        #    scan in dry-run so the planning preview works without GCS auth.
        if dry_run:
            logger.info("DRY-RUN: skipping GCS scan; assuming nothing completed yet.")
            completed_hashes: set[str] = set()
        else:
            completed_hashes = _scan_completed_hashes(output_dirs)

        # Subtract hashes we already dispatched in this process so we don't
        # double-submit while a previous round's job is still running.
        skip_hashes = completed_hashes | state.submitted_hashes
        remaining = [w for w in all_warcs if _warc_path_hash(w) not in skip_hashes]
        all_warc_hashes = {_warc_path_hash(w) for w in all_warcs}
        done_hashes = completed_hashes & all_warc_hashes
        in_flight = len(state.submitted_hashes - completed_hashes)
        logger.info(
            "Progress: %d/%d done (%.1f%%), %d in-flight (this process), %d remaining to dispatch",
            len(done_hashes),
            len(all_warcs),
            100 * len(done_hashes) / max(len(all_warcs), 1),
            in_flight,
            len(remaining),
        )

        # Exit only when GCS confirms all WARCs are complete. "Nothing left
        # to dispatch" is not enough — in-flight Iris jobs may still die and
        # need re-dispatch. Keep polling until completion is observed.
        if done_hashes >= all_warc_hashes:
            logger.info("All %d WARCs extracted (verified in GCS). Orchestrator complete.", len(all_warcs))
            break

        if not remaining:
            logger.info(
                "Nothing new to dispatch (%d in-flight). Sleeping until next scan.",
                in_flight,
            )
            time.sleep(poll_interval_seconds)
            continue

        # 2. Dispatch batches to fleet TPU types (round-robin)
        offset = 0
        jobs_this_round = 0

        for entry in fleet:
            if offset >= len(remaining):
                break
            if jobs_this_round >= max_concurrent_jobs:
                break

            batch = remaining[offset : offset + entry.batch_size]
            offset += len(batch)
            batch_hashes = {_warc_path_hash(w) for w in batch}

            batch_id = f"{entry.tpu_type}_{int(time.time())}_{state.jobs_submitted}"
            if dry_run:
                manifest_gcs = f"{MANIFEST_STAGING_DIR}/batch_{batch_id}.txt  [DRY-RUN — not written]"
            else:
                manifest_gcs = _write_batch_manifest(batch, batch_id)

            try:
                job_name = _submit_iris_job(
                    spec=spec,
                    entry=entry,
                    manifest_path=manifest_gcs,
                    batch_id=batch_id,
                    iris_priority=iris_priority,
                    iris_config=iris_config,
                    dry_run=dry_run,
                )
                state.active_jobs.append(job_name)
                state.submitted_hashes.update(batch_hashes)
                state.jobs_submitted += 1
                jobs_this_round += 1
            except Exception as e:
                logger.error("Failed to submit job for %s: %s", entry.tpu_type, e)
                continue

        if dry_run:
            logger.info("DRY-RUN: dispatched %d simulated jobs this round; exiting.", jobs_this_round)
            break

        if jobs_this_round == 0 and remaining:
            logger.warning(
                "No jobs submitted this round but %d WARCs remain. In-flight jobs may still be working. Waiting...",
                len(remaining),
            )

        # 3. Wait before next scan
        logger.info(
            "Waiting %ds before next scan (%d jobs submitted this round)...",
            poll_interval_seconds,
            jobs_this_round,
        )
        time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--spec",
        required=True,
        choices=sorted(SPECS),
        help="Extraction spec id (key in extraction_specs.SPECS).",
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_WARC_MANIFEST,
        help=f"Primary WARC manifest. Default: {DEFAULT_WARC_MANIFEST}",
    )
    p.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="WARC manifest(s) to subtract from --manifest before dispatch. Repeatable.",
    )
    p.add_argument(
        "--iris-priority",
        choices=sorted(_PRIORITY_BAND_NAMES),
        default=None,
        help="Iris priority band for submitted jobs (default: Iris's default, typically interactive).",
    )
    p.add_argument("--max-concurrent-jobs", type=int, default=6)
    p.add_argument("--poll-interval", type=int, default=120)
    p.add_argument(
        "--iris-config",
        default=os.environ.get("IRIS_CONFIG", DEFAULT_IRIS_CONFIG),
        help=(
            "Path to the iris cluster config (top-level --config flag for the iris CLI). "
            f"Default: $IRIS_CONFIG or {DEFAULT_IRIS_CONFIG}"
        ),
    )
    p.add_argument(
        "--fleet",
        default=None,
        help=(
            "Override the default FLEET with a comma-separated list of TPU types "
            "(e.g. 'v6e-4' or 'v5p-8,v6e-8'). Useful for one-off smoke tests."
        ),
    )
    p.add_argument(
        "--num-workers-override",
        type=int,
        default=None,
        help=(
            "Override Zephyr ``num_workers`` for every fleet entry. Each worker "
            "carries its own TPU under the new Iris pattern, so a high value "
            "fans out to many concurrent slices. Use 1-2 for cheap smoke tests."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one round of dispatch decisions and exit; do not write configs or submit jobs.",
    )
    return p.parse_args()


def _parse_fleet_entry_token(token: str) -> TpuFleetEntry:
    """Parse a single ``--fleet`` token. Format: ``<tpu>[:<region>]``.

    Examples: ``v5p-8`` (no region pin), ``v5p-8:us-east5``.
    """
    parts = [p.strip() for p in token.split(":", 1)]
    tpu = parts[0]
    region = parts[1] if len(parts) == 2 else ""
    return TpuFleetEntry(tpu_type=tpu, region=region)


def _resolve_fleet(fleet_arg: str | None, num_workers_override: int | None) -> list[TpuFleetEntry]:
    """Parse the --fleet override or return the module default, optionally
    forcing a uniform ``num_workers`` across all entries."""
    base = FLEET if not fleet_arg else [_parse_fleet_entry_token(t) for t in fleet_arg.split(",") if t.strip()]
    if num_workers_override is None:
        return base
    return [
        TpuFleetEntry(
            tpu_type=e.tpu_type,
            region=e.region,
            num_workers=num_workers_override,
            batch_size=e.batch_size,
        )
        for e in base
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    spec = get_spec(args.spec)

    dispatch_loop(
        spec=spec,
        manifest_path=args.manifest,
        exclude_manifests=args.exclude_manifest,
        fleet=_resolve_fleet(args.fleet, args.num_workers_override),
        iris_priority=args.iris_priority,
        iris_config=args.iris_config,
        poll_interval_seconds=args.poll_interval,
        max_concurrent_jobs=args.max_concurrent_jobs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
