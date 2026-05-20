# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the inference integration tests.

Centralizes: logging setup, sample-record printing, output cleanup, and a
small wrapper that asserts the standard completeness invariants every test
relies on.
"""
from __future__ import annotations

import logging
import sys

import fsspec
from marin.inference.distributed import InferenceResult, ResponseRecord


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def assert_complete(result: InferenceResult, expected_n: int, expected_ids: set[str]) -> list[ResponseRecord]:
    """Assert standard completeness invariants. Returns the materialized records."""
    if not result.is_complete:
        raise AssertionError(f"missing shards: {result.missing_shards}")
    records = result.to_list()
    if len(records) != expected_n:
        raise AssertionError(f"expected {expected_n} response records, got {len(records)}")
    got_ids = {r.id for r in records}
    if got_ids != expected_ids:
        missing = expected_ids - got_ids
        extra = got_ids - expected_ids
        raise AssertionError(f"id mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    empty = [r.id for r in records if not r.response]
    if empty:
        raise AssertionError(f"{len(empty)} record(s) had empty response text; first ids: {empty[:5]}")
    return records


def print_sample(records: list[ResponseRecord], n: int = 3, max_chars: int = 200) -> None:
    print(f"\n--- Sample of {min(n, len(records))} outputs ---")
    for record in records[:n]:
        text = record.response
        if len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, full length {len(record.response)})"
        print(f"id={record.id!r}, shard={record.shard}, extras={dict(record.extra) or {}}")
        print(f"  response: {text!r}")
    print("---")


def cleanup_run(result: InferenceResult) -> None:
    """Delete the run's output + input directories. Best-effort, never raises.

    The deleted prefix is ``<results_uri>/..`` — the parent dir containing
    both ``outputs/`` (the shard files) and ``inputs/`` (materialized prompts).
    """
    run_prefix = result.results_uri.removesuffix("/outputs")
    fs, fs_path = fsspec.core.url_to_fs(run_prefix)
    try:
        if fs.exists(fs_path):
            fs.delete(fs_path, recursive=True)
            print(f"Cleaned up: {run_prefix}")
        else:
            print(f"Nothing to clean (already gone): {run_prefix}")
    except Exception as exc:
        print(f"WARNING: cleanup of {run_prefix} failed: {exc}", file=sys.stderr)


def exit_pass(message: str = "PASS") -> None:
    print(f"\n{message}")
    sys.exit(0)
