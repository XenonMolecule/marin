# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dedupe the llm_curated extraction (us-central1) and produce stats + final dataset.

Same pipeline as ``dedup_resiliparse.py``; key differences:
  - region: us-central1
  - source: ``gs://marin-us-central1/documents/baseline_llm_curated-050243``
  - dedupe column: ``text`` (the cleaned post-processed output that's tokenized
    in training; ``generated_text`` is the raw LLM output preserved for debug)

See ``dedup_resiliparse.py`` for the full pipeline shape and the note on how
marin defaults compare to DCLM/Nemotron-CC published settings.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

import fsspec
from fray.v2 import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.exact import dedup_exact_document
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from rigging.filesystem import (
    check_path_in_region,
    marin_temp_bucket,
    region_from_metadata,
)
from rigging.log_setup import configure_logging

from experiments.baseline_collection.dedup.apply_dedup import (
    ApplyDedupConfig,
    apply_dedup,
    cluster_size_histogram,
)
from experiments.baseline_collection.dedup.prep_with_id import PrepConfig, prep_with_id

logger = logging.getLogger(__name__)
T = TypeVar("T")

RAW_PATH = "gs://marin-us-central1/documents/baseline_llm_curated-050243"
PREFIX = "michaelryan/dedup_llm_curated"


def _assert_not_none(value: T | None) -> T:
    assert value is not None
    return value


def _write_stats(stats_path: str, payload: dict) -> dict:
    with fsspec.open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Stats → %s", stats_path)
    logger.info("%s", json.dumps(payload, indent=2))
    return {"success": True, "path": stats_path}


def build_steps() -> list[StepSpec]:
    region = _assert_not_none(region_from_metadata())

    raw = StepSpec(name="raw_llm_curated", override_output_path=RAW_PATH)
    check_path_in_region(raw.name, raw.output_path, region)

    bucket = marin_temp_bucket(ttl_days=14, prefix=PREFIX)

    prep = StepSpec(
        name="prep",
        output_path_prefix=bucket,
        deps=[raw],
        fn=lambda op: prep_with_id(
            PrepConfig(
                input_path=raw.output_path,
                output_path=op,
                text_field="text",
                max_parallelism=200,
            )
        ),
    )

    exact = StepSpec(
        name="exact",
        output_path_prefix=bucket,
        deps=[prep],
        fn=lambda op: dedup_exact_document(
            input_paths=prep.output_path,
            output_path=op,
            text_field="text",
            max_parallelism=512,
        ),
    )

    fuzzy = StepSpec(
        name="fuzzy",
        output_path_prefix=bucket,
        deps=[prep],
        fn=lambda op: dedup_fuzzy_document(
            input_paths=prep.output_path,
            output_path=op,
            text_field="text",
            max_parallelism=1024,
            worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
        ),
    )

    deduped = StepSpec(
        name="deduped",
        output_path_prefix=bucket,
        deps=[prep, exact, fuzzy],
        fn=lambda op: apply_dedup(
            ApplyDedupConfig(
                prepped_path=prep.output_path,
                sidecar_paths=[exact.output_path, fuzzy.output_path],
                output_path=op,
                max_parallelism=512,
            )
        ),
    )

    stats = StepSpec(
        name="stats",
        output_path_prefix=bucket,
        deps=[exact, fuzzy, deduped],
        fn=lambda op: _write_stats(
            f"{op}/dedup_stats.json",
            {
                "source": "llm_curated",
                "raw_path": raw.output_path,
                "prep_path": prep.output_path,
                "exact_path": exact.output_path,
                "fuzzy_path": fuzzy.output_path,
                "deduped_path": deduped.output_path,
                "cluster_size_histogram": cluster_size_histogram(fuzzy.output_path),
                "fuzzy_params": {
                    "num_perms": 286,
                    "num_bands": 26,
                    "ngram_size_chars": 5,
                    "seed": 42,
                    "approx_jaccard_threshold": 0.75,
                },
            },
        ),
    )

    return [raw, prep, exact, fuzzy, deduped, stats]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
