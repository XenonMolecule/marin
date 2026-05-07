# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill `.data_curation_DONE` markers for completed warc-scaling runs.

The standalone runner writes the per-run summary.json BEFORE the marker:

    _write_summary(...)         # GCS write — usually succeeds
    ...
    fsspec.open(done_marker_path, "w").write(marker_payload)  # GCS write — sometimes fails silently

When the marker write fails, the coordinator's `skip_if_done` check returns
False on next launch and re-submits a completed run. The deep-scan dashboard
also displayed those as "100% IN-PROGRESS" because there's no marker.

This script:
  1. Lists summaries in the results bucket.
  2. For each, reads the summary's `run.region` and `plan.train_steps`.
  3. Checks if the marker exists at `{region_bucket}/checkpoints/isoflop-curation/{run_name}/.data_curation_DONE`.
  4. If missing, writes a fresh marker.

**Safety invariants**:
  - Only operates on runs with a summary file (i.e., training completed).
  - Region is sourced from the summary itself, never the tracker.
  - Skip-if-exists semantics — never overwrites a real marker.
  - Marker payload is informational only; training data is untouched.

Usage::

    # Dry run — list what would be written, no GCS writes:
    uv run python -m experiments.scaling_law_sweeps.backfill_done_markers --dry-run

    # Actually backfill:
    uv run python -m experiments.scaling_law_sweeps.backfill_done_markers
"""

from __future__ import annotations

import argparse
import json
import logging

from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)

RESULTS_BUCKET = "marin-us-central1"
RESULTS_PREFIX = "metadata/data_curation_warc_scaling_results/"
CHECKPOINT_PREFIX = "checkpoints/isoflop-curation/"
REGION_TO_BUCKET = {
    "us-central1": "marin-us-central1",
    "us-central2": "marin-us-central2",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "europe-west4": "marin-eu-west4",
}


def _build_marker_payload(summary: dict) -> dict:
    plan = summary.get("plan", {})
    run = summary.get("run", {})
    return {
        "completed_at": run.get("completed_at"),
        "run_name_core": plan.get("run_name_core") or plan.get("run_name"),
        "method": plan.get("method_name"),
        "experiment_tag": plan.get("experiment_tag"),
        "region": run.get("region"),
        "train_steps": plan.get("train_steps"),
        "_backfilled_marker": True,
    }


def _marker_path(region: str, run_name: str) -> tuple[str, str]:
    """Return (bucket_name, object_path) for the marker file."""
    bucket = REGION_TO_BUCKET[region]
    obj = f"{CHECKPOINT_PREFIX}{run_name}/.data_curation_DONE"
    return bucket, obj


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="List missing markers but do not write.")
    args = parser.parse_args(argv)

    client = gcs_storage.Client()
    results_bucket = client.bucket(RESULTS_BUCKET)

    summaries = list(results_bucket.list_blobs(prefix=RESULTS_PREFIX))
    logger.info("Scanning %d summary files...", len(summaries))

    written = 0
    already_present = 0
    skipped_unknown_region = 0
    errors = 0

    for blob in summaries:
        fname = blob.name.rsplit("/", 1)[-1]
        if not (fname.startswith("curation-") and fname.endswith(".json")):
            continue
        run_name = fname[: -len(".json")]
        # Skip any -smoke or other suffixed runs from prior testing — they do
        # not need a marker for the production sweep's skip_if_done logic.
        if run_name.endswith("-smoke"):
            continue
        try:
            summary = json.loads(blob.download_as_text())
        except Exception as e:
            logger.warning("failed to read %s: %s", fname, e)
            errors += 1
            continue
        region = (summary.get("run") or {}).get("region")
        if not region or region not in REGION_TO_BUCKET:
            logger.warning("summary %s has unknown region %r — skipping", run_name, region)
            skipped_unknown_region += 1
            continue
        bucket_name, obj = _marker_path(region, run_name)
        bucket = client.bucket(bucket_name)
        marker_blob = bucket.blob(obj)
        if marker_blob.exists():
            already_present += 1
            continue
        # Marker missing — write it.
        payload = _build_marker_payload(summary)
        if args.dry_run:
            logger.info("[DRY-RUN] would write gs://%s/%s", bucket_name, obj)
            written += 1
            continue
        try:
            marker_blob.upload_from_string(json.dumps(payload), content_type="application/json")
            logger.info("wrote gs://%s/%s", bucket_name, obj)
            written += 1
        except Exception as e:
            logger.error("write failed for gs://%s/%s: %s", bucket_name, obj, e)
            errors += 1

    logger.info(
        "Summary: %d wrote, %d already-present, %d unknown-region, %d errors (dry_run=%s)",
        written,
        already_present,
        skipped_unknown_region,
        errors,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
