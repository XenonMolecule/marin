# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality-stratified subset filter for the existing Nemotron-CC filtered JSONL.

This runs DOWNSTREAM of `filter_nemotron_full` (which produces JSONL shards
named `{snapshot}-{shard:05d}-of-{total:05d}.jsonl.gz` with `nemotron_quality`
on each row). Here we take those shards and emit a subset keeping only rows
whose `nemotron_quality` is in a configured allow-list.

Two initial variants, registered in `pipeline_nemotron_quality.py`:

- ``baseline_nemotron_qhigh``: quality=high only (both kind=actual AND kind=synthetic)
- ``baseline_nemotron_qmedplus``: quality in {high, medium-high, medium}
  (both kind=actual AND kind=synthetic)

The existing `baseline_nemotron_full` covers all 5 quality levels, so together
the three variants let us study the "fewer-better" vs "more-mixed" tradeoff at
fixed compute in the scaling-law sweep.

Determinism
-----------
To enable identical output across regions (so we can run in-region and avoid
cross-region egress without trusting anything opaque), the filter:

1. Lists input shards via `fsspec.glob` sorted lexicographically.
2. Processes each shard sequentially; within a shard, preserves input row
   order (no sort, no shuffle).
3. Writes one output shard per input shard with the same filename suffix.
4. Gzips with a fixed mtime so shard bytes are reproducible.

Verification: after running in two regions, `sha256` of each output shard
must match across regions. If they don't, determinism is broken — investigate
before proceeding.
"""

from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)

# Canonical quality ordering from Nemotron-CC's classifier.
QUALITY_LEVELS = ("high", "medium-high", "medium", "medium-low", "low")

# Preset allow-lists used by the pipeline registrations.
QUALITY_HIGH = ("high",)
QUALITY_MEDPLUS = ("high", "medium-high", "medium")


@dataclass(frozen=True)
class FilterNemotronQualityConfig:
    """Config for the quality-stratified downstream filter.

    - ``input_dir``: GCS path containing the JSONL shards produced by
      ``filter_nemotron_full`` (or the organic-only variant). Shard filenames
      are ``{snapshot}-{shard:05d}-of-{total:05d}.jsonl.gz``.
    - ``output_path``: where to write the filtered JSONL. The Executor's
      content hash makes the full directory name; callers typically pass
      ``this_output_path()``.
    - ``allowed_quality``: tuple of quality levels to keep. Rows with
      ``nemotron_quality`` outside this set are dropped. Must be a subset of
      ``QUALITY_LEVELS``.
    """

    input_dir: str
    output_path: str
    allowed_quality: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.allowed_quality) - set(QUALITY_LEVELS)
        if unknown:
            raise ValueError(f"allowed_quality contains unknown levels {unknown}; must be subset of {QUALITY_LEVELS}")
        if not self.allowed_quality:
            raise ValueError("allowed_quality must not be empty")


def _list_input_shards(input_dir: str) -> list[str]:
    """Return sorted list of input shard paths. Sorting is what makes the filter deterministic."""
    fs, _ = fsspec.core.url_to_fs(input_dir)
    # Input shards are named "{snapshot}-{shard:05d}-of-{total:05d}.jsonl.gz"
    pattern = f"{input_dir.rstrip('/')}/*.jsonl.gz"
    matches = fs.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No JSONL shards found under {pattern}")
    # fsspec returns bare paths without the scheme; re-prefix to full URIs.
    scheme = input_dir.split("://", 1)[0] if "://" in input_dir else ""
    if scheme:
        matches = [m if m.startswith(scheme + "://") else f"{scheme}://{m}" for m in matches]
    return sorted(matches)


def _filter_shard(
    src: str,
    dst: str,
    allowed: frozenset[str],
) -> tuple[int, int, dict[str, int]]:
    """Copy rows whose nemotron_quality is in `allowed` from src to dst.

    Returns (rows_in, rows_out, breakdown) where breakdown is per-quality counts
    of kept rows.
    """
    rows_in = 0
    rows_out = 0
    breakdown: dict[str, int] = {}

    # mtime=0 makes the gzip stream reproducible across runs/regions.
    with fsspec.open(src, "rb") as src_fh, fsspec.open(dst, "wb") as dst_fh:
        with (
            gzip.open(src_fh, "rt", encoding="utf-8") as src_gz,
            gzip.GzipFile(fileobj=dst_fh, mode="wb", mtime=0) as dst_gz,
        ):
            for line in src_gz:
                if not line.strip():
                    continue
                rows_in += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSONL line in %s; skipping", src)
                    continue
                q = row.get("nemotron_quality")
                if q in allowed:
                    dst_gz.write(line.encode("utf-8"))
                    rows_out += 1
                    breakdown[q] = breakdown.get(q, 0) + 1

    return rows_in, rows_out, breakdown


def filter_nemotron_quality(config: FilterNemotronQualityConfig, max_workers: int = 32) -> None:
    """Emit a quality-filtered subset of the Nemotron-CC filtered JSONL.

    Shards are processed in parallel (I/O-bound, safe for determinism
    because each output shard depends only on its own input shard and the
    allow-list). Per-shard order sortedness of the input list is preserved
    in the stats log so we can diff two runs.

    Output shard filenames mirror input shard filenames so callers can diff
    the two dirs. Writes a small ``_quality_filter_stats.json`` under
    ``output_path`` with per-quality counts + totals for downstream
    verification.
    """
    allowed = frozenset(config.allowed_quality)
    output_dir = config.output_path.rstrip("/")

    shards = _list_input_shards(config.input_dir)
    logger.info(
        "quality-filter: input=%s output=%s allowed=%s shards=%d workers=%d",
        config.input_dir,
        output_dir,
        sorted(allowed),
        len(shards),
        max_workers,
    )

    total_in = 0
    total_out = 0
    total_breakdown: dict[str, int] = {}

    def _task(idx_src: tuple[int, str]) -> tuple[int, str, int, int, dict[str, int]]:
        idx, src = idx_src
        fname = src.rsplit("/", 1)[-1]
        dst = f"{output_dir}/{fname}"
        rows_in, rows_out, breakdown = _filter_shard(src, dst, allowed)
        return idx, fname, rows_in, rows_out, breakdown

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_task, (i, s)) for i, s in enumerate(shards)]
        completed = 0
        for fut in as_completed(futures):
            _idx, fname, rows_in, rows_out, breakdown = fut.result()
            total_in += rows_in
            total_out += rows_out
            for k, v in breakdown.items():
                total_breakdown[k] = total_breakdown.get(k, 0) + v
            completed += 1
            # Log every 50 shards to keep output manageable on 5621-shard runs.
            if completed % 50 == 0 or completed == len(shards):
                logger.info(
                    "  [%d/%d] %s: %d/%d kept -- running totals in=%d out=%d",
                    completed,
                    len(shards),
                    fname,
                    rows_out,
                    rows_in,
                    total_in,
                    total_out,
                )

    stats = {
        "input_dir": config.input_dir,
        "output_dir": output_dir,
        "allowed_quality": list(config.allowed_quality),
        "total_input_rows": total_in,
        "total_output_rows": total_out,
        "breakdown_by_quality": total_breakdown,
        "shard_count": len(shards),
    }
    stats_path = f"{output_dir}/_quality_filter_stats.json"
    with fsspec.open(stats_path, "w") as f:
        f.write(json.dumps(stats, indent=2, sort_keys=True))
    logger.info(
        "quality-filter DONE: %d/%d kept (%.1f%%) breakdown=%s",
        total_out,
        total_in,
        100.0 * total_out / max(total_in, 1),
        total_breakdown,
    )


def config_from_preset(
    preset: str,
    input_dir: str,
    output_path: str,
) -> FilterNemotronQualityConfig:
    """Lookup helper so pipeline.py stays declarative.

    Presets: ``"high"`` → `QUALITY_HIGH`; ``"medplus"`` → `QUALITY_MEDPLUS`.
    """
    presets = {
        "high": QUALITY_HIGH,
        "medplus": QUALITY_MEDPLUS,
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; known: {sorted(presets)}")
    return FilterNemotronQualityConfig(
        input_dir=input_dir,
        output_path=output_path,
        allowed_quality=presets[preset],
    )


# Re-exported so the ExecutorStep hash is stable against minor reshuffles.
__all__ = [
    "QUALITY_HIGH",
    "QUALITY_LEVELS",
    "QUALITY_MEDPLUS",
    "FilterNemotronQualityConfig",
    "config_from_preset",
    "filter_nemotron_quality",
]
