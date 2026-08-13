# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rescue: re-run dclm_filter for the single missing shard 00079.

The main filter run completed 199/200 shards before the parent crashed with
``IndexError: list index out of range`` (likely a transient post-completion
finalization bug). This script feeds dclm_filter the full input glob with
skip_existing=True — Zephyr will skip the 199 already-committed shards and
only re-process 00079.
"""

from __future__ import annotations

import logging

import marin.transform.dclm_filter as df
from fray.types import ResourceConfig
from marin.transform.dclm_filter import DclmFilterConfig, dclm_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

LLM_CURATED_DOCS = "gs://marin-us-central1/documents/baseline_llm_curated-050243"
LID_DIR = "gs://marin-us-central1/resources/dclm/lid_176-951189"
QUALITY_DIR = "gs://marin-us-central1/resources/dclm/fasttext_oh_eli5-d828e9"
BANLISTS_DIR = "gs://marin-us-central2/resources/dclm/banlists"
FILTER_OUT = "gs://marin-us-central1/filtered/dclm_filter_llm_curated_dclm_filtered_v1"

WORKER_REGIONS = ["us-central1", "us-central2", "us-east1", "us-east5", "us-west1", "us-west4"]

# Pin worker resources via the same monkey-patch trick the standalone runner uses.
_original_ctx = df.ZephyrContext


def _make_ctx(name: str, **kwargs):
    import dataclasses

    kwargs.setdefault("max_workers", 5)
    kwargs.setdefault(
        "resources",
        ResourceConfig(cpu=2, ram="12g", regions=WORKER_REGIONS, preemptible=False),
    )
    if isinstance(kwargs.get("resources"), ResourceConfig):
        kwargs["resources"] = dataclasses.replace(kwargs["resources"], regions=WORKER_REGIONS, preemptible=False)
    return _original_ctx(name=name, **kwargs)


df.ZephyrContext = _make_ctx  # type: ignore[assignment]


def main() -> None:
    dclm_filter(
        DclmFilterConfig(
            input_path=f"{LLM_CURATED_DOCS}/*.jsonl.gz",
            output_path=FILTER_OUT,
            lid_model_path=LID_DIR,
            quality_model_path=QUALITY_DIR,
            banlists_path=BANLISTS_DIR,
        )
    )


if __name__ == "__main__":
    main()
