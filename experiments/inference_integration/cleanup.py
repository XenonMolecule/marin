# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual cleanup of integration-test outputs.

Every test deletes its own results on success. This helper exists for the
case where a test crashed mid-run and left orphan files behind. Usage:

    python experiments/inference_integration/cleanup.py
    python experiments/inference_integration/cleanup.py --region us-central1
    python experiments/inference_integration/cleanup.py --prefix inf-integ-

The script lists every directory under ``gs://marin-{region}/`` whose name
starts with ``--prefix`` and deletes them after printing what it will do.
It only touches integration-test prefixes; production paths under
``checkpoints/``, ``documents/``, etc. are safe.
"""
from __future__ import annotations

import argparse
import sys

import fsspec
from rigging.filesystem import REGION_TO_DATA_BUCKET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=None,
        help="Specific marin region to clean (default: every known region).",
    )
    parser.add_argument(
        "--prefix",
        default="inf-integ-",
        help="Only delete directories whose name starts with this prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without deleting anything.",
    )
    args = parser.parse_args()

    if args.region:
        regions = [args.region]
    else:
        regions = sorted(REGION_TO_DATA_BUCKET)

    fs = fsspec.filesystem("gcs")

    total_deleted = 0
    for region in regions:
        bucket = REGION_TO_DATA_BUCKET[region]
        bucket_path = bucket  # `marin-{region}` (no `gs://` for fsspec gcs)
        try:
            entries = fs.ls(bucket_path)
        except FileNotFoundError:
            print(f"[{region}] bucket not accessible; skipping", file=sys.stderr)
            continue

        targets = [entry for entry in entries if entry.rsplit("/", 1)[-1].startswith(args.prefix)]
        if not targets:
            print(f"[{region}] no integration-test dirs matching {args.prefix!r}")
            continue

        for target in targets:
            label = f"gs://{target}"
            if args.dry_run:
                print(f"[{region}] DRY RUN: would delete {label}")
                continue
            try:
                fs.delete(target, recursive=True)
                print(f"[{region}] deleted {label}")
                total_deleted += 1
            except Exception as exc:
                print(f"[{region}] FAILED to delete {label}: {exc}", file=sys.stderr)

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to actually delete.")
    else:
        print(f"\nDeleted {total_deleted} integration-test prefix(es).")


if __name__ == "__main__":
    main()
