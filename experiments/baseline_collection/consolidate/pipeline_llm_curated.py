# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reshape + tokenize the consolidated LLM extraction output.

Consumes ``resolved.jsonl.gz`` (post-transfer; paths point at
``by_region/{region}/data-{hash}/batch_*.jsonl.gz`` in us-central1), filters
out empty batches (``num_records == 0``), reshards kept records into ~200
flat shards, and tokenizes with the llama3 tokenizer to match every other
curation baseline in ``pipeline.py``.

Outputs::

    documents/baseline_llm_curated/data-{shard:05d}-of-00200.jsonl.gz
        Reshaped canonical records, text field only.

    tokenized/baseline_llm_curated-{hash}/
        Levanter cache for the scaling-law sweep.

Usage::

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/consolidate/pipeline_llm_curated.py
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass

import fsspec
from fray.v2.types import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_jsonl

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

logger = logging.getLogger(__name__)

RESOLVED_MANIFEST = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/" "resolved/resolved.jsonl.gz"
NUM_OUTPUT_SHARDS = 200

# Source → consolidated archive path rewrite. resolved.jsonl.gz was written
# with paths pointing at the original regional buckets (e.g.
# ``gs://marin-eu-west4/documents/baseline_llm_extraction/data-.../batch_....jsonl.gz``).
# After the Phase D transfer, every canonical batch lives in us-central1 under
# ``by_region/{region}/data-.../batch_....jsonl.gz`` and we want the tokenize
# step to read from those (intra-region, free). This dict maps source-bucket
# prefix → consolidated-archive prefix.
SOURCE_TO_ARCHIVE_PREFIX: dict[str, str] = {
    "gs://marin-us-central1/documents/baseline_llm_extraction": (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/us-central1"
    ),
    "gs://marin-us-east1/documents/baseline_llm_extraction": (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/us-east1"
    ),
    "gs://marin-us-east5/documents/baseline_llm_extraction": (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/us-east5"
    ),
    "gs://marin-us-west4/documents/baseline_llm_extraction": (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/us-west4"
    ),
    "gs://marin-eu-west4/documents/baseline_llm_extraction": (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/europe-west4"
    ),
}


def _remap_to_archive(path: str) -> str:
    """Rewrite a regional-source path to its us-central1 consolidated-archive path."""
    for src_prefix, dst_prefix in SOURCE_TO_ARCHIVE_PREFIX.items():
        if path.startswith(src_prefix + "/"):
            return dst_prefix + path[len(src_prefix) :]
    raise ValueError(f"path {path!r} does not match any known source prefix")


@dataclass(frozen=True)
class ReshapeLlmCuratedConfig:
    """Config for the reshape step.

    ``resolved_manifest_path`` is a GCS URI to the resolved.jsonl.gz produced
    by the resolver. ``output_path`` is filled in by the executor via
    ``this_output_path()``. ``num_shards`` controls the reshard fanout; if the
    tokenizer later chokes on 200 shards we can lower this without changing
    upstream data.
    """

    resolved_manifest_path: str
    output_path: str
    num_shards: int = NUM_OUTPUT_SHARDS


def _load_nonempty_canonical_paths(manifest_path: str) -> list[str]:
    """Read resolved.jsonl.gz, return canonical paths for batches with >0 records.

    Source paths in the manifest point at the original regional buckets; we
    remap each to its us-central1 ``by_region/{region}/`` location so the
    downstream Zephyr reads are intra-region (free).
    """
    paths: list[str] = []
    skipped = 0
    with fsspec.open(manifest_path, "rb") as f, gzip.open(f, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("num_records") or 0) > 0:
                paths.append(_remap_to_archive(row["path"]))
            else:
                skipped += 1
    logger.info(
        "Loaded %d non-empty canonical paths (remapped to consolidated archive); skipped %d empty/invalid batches",
        len(paths),
        skipped,
    )
    return paths


def reshape_llm_curated(config: ReshapeLlmCuratedConfig) -> None:
    """Reshape all canonical batches into flat shards (filters empties).

    Uses Zephyr's ``Dataset.from_iterable`` + ``flat_map(load_jsonl)`` so we
    can feed an explicit list of paths (the canonical subset) instead of a
    glob over ``by_region/`` — which would include the losing duplicates.
    """
    paths = _load_nonempty_canonical_paths(config.resolved_manifest_path)
    if not paths:
        raise RuntimeError(f"No non-empty canonical paths in {config.resolved_manifest_path}")

    n = config.num_shards
    template = f"{config.output_path}/data-{{shard:05d}}-of-{n:05d}.jsonl.gz"
    pipeline = Dataset.from_iterable(paths).flat_map(load_jsonl).reshard(n).write_jsonl(template, skip_existing=True)
    ctx = ZephyrContext(name="reshape-llm-curated", max_workers=200)
    ctx.put("config", config)
    ctx.execute(pipeline)


reshape_step = ExecutorStep(
    name="documents/baseline_llm_curated",
    description=(
        "Reshape deduplicated LLM extraction batches (via resolved.jsonl.gz) "
        f"into {NUM_OUTPUT_SHARDS} flat shards. Filters out empty batches."
    ),
    fn=remote(reshape_llm_curated, resources=ResourceConfig(cpu=4, ram="32g")),
    config=ReshapeLlmCuratedConfig(
        resolved_manifest_path=RESOLVED_MANIFEST,
        output_path=this_output_path(),
    ),
)

tokenize_step = default_tokenize(
    name="baseline_llm_curated",
    dataset=reshape_step / "*.jsonl.gz",
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(
        steps=[tokenize_step],
        description="LLM extraction: reshape canonical batches + tokenize with llama3.",
    )
