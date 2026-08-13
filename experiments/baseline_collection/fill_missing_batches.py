# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directly fill the missing batches of specific WARCs — bypasses steal/claim.

The steal/claim endgame can leave a giant WARC one-or-two batches short forever:
a stealer dies mid-batch and its orphaned ``_stealing`` marker permanently blocks
that batch (no stale override), or every fresh worker dies re-downloading the
giant before reaching the gap. This tool sidesteps all of that: it downloads each
WARC once, recomputes its batches, and writes ONLY the currently-missing ones
directly (idempotent at temperature 0), then writes ``_done`` if that completes
the WARC. No claims, no stealing — deterministic gap-fill for the pathological
long tail.

Usage (on a TPU node)::

    python -m experiments.baseline_collection.fill_missing_batches \
        --pipeline llm_pipeline_v1 \
        --manifest experiments/distill/random_subsets/lpv1_last2.txt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import Any

from experiments.baseline_collection.download_warcs import _download_one_warc, _load_manifest, _warc_path_hash
from experiments.baseline_collection.run_extract_standalone import (
    MAX_DOC_TOKENS,
    _batch_output_path,
    _count_records_in_all_batches,
    _filter_by_length,
    _find_completed_batches_all_regions,
    _is_warc_done_any_region,
    _process_batch_pipeline,
    _register_completed_warc,
    _registry_prefix_for,
    _write_batch_output,
    _write_batch_profile,
    _write_done_marker,
    _write_token_stats,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="Pipeline id (llm_pipeline_v1 / llm_simple_v1)")
    parser.add_argument("--manifest", required=True, help="Manifest of WARCs whose gaps to fill")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=250, help="MUST match the run's group-size so indices align")
    parser.add_argument("--num-shards", type=int, default=1, help="Split missing batches across N parallel jobs")
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This job's shard index [0, num_shards); does missing batches where i %% num_shards == shard",
    )
    args = parser.parse_args()

    from experiments.baseline_collection.pipelines.pipeline_specs import get_pipeline

    pipeline = get_pipeline(args.pipeline)
    output_subdir = f"documents/baseline_llm_extraction/{pipeline.pipeline_id}"
    registry_prefix = _registry_prefix_for(output_subdir)

    from rigging.filesystem import marin_prefix

    output_dir = f"{marin_prefix()}/{output_subdir}"
    model_name = args.model or f"{marin_prefix()}/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import jax

    devices = jax.devices()
    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)

    from vllm import LLM

    t0 = time.monotonic()
    llm: Any = LLM(
        model=model_name, tensor_parallel_size=tp, max_model_len=args.max_model_len, enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()
    logger.info("Engine loaded in %.1fs (tp=%d, pipeline=%s)", time.monotonic() - t0, tp, pipeline.pipeline_id)

    bs = args.batch_size
    for warc in _load_manifest(args.manifest):
        h = _warc_path_hash(warc)
        warc_dir = f"{output_dir}/data-{h}"
        if _is_warc_done_any_region(h, output_subdir):
            logger.info("%s already done — skip", h)
            continue
        records = _filter_by_length(_download_one_warc(warc), MAX_DOC_TOKENS)
        num_batches = math.ceil(len(records) / bs)
        completed = _find_completed_batches_all_regions(h, output_subdir)
        missing = [i for i in range(num_batches) if i not in completed]
        # Shard the missing batches across parallel jobs (disjoint by index modulo).
        # Each shard writes its own batches directly (no claims), and whichever shard
        # finishes last sees all_done >= num_batches and writes _done — idempotently.
        if args.num_shards > 1:
            missing = [i for i in missing if i % args.num_shards == args.shard]
        logger.info(
            "%s: %d records, %d batches, %d already done, filling %d missing: %s",
            h,
            len(records),
            num_batches,
            len(completed),
            len(missing),
            missing,
        )
        for i in missing:
            batch = records[i * bs : (i + 1) * bs]
            if not batch:
                continue
            out_records, kept, filtered, token_stats, profile = _process_batch_pipeline(batch, llm, tokenizer, pipeline)
            _write_batch_output(_batch_output_path(warc_dir, i), out_records)
            _write_token_stats(warc_dir, i, token_stats)
            _write_batch_profile(warc_dir, i, profile)
            logger.info("  filled batch %d (kept=%d filtered=%d)", i, kept, filtered)
        # Write _done if the WARC is now complete.
        all_done = _find_completed_batches_all_regions(h, output_subdir)
        if len(all_done) >= num_batches:
            final_kept = _count_records_in_all_batches(h, output_subdir)
            _write_done_marker(
                warc_dir,
                {
                    "warc": warc,
                    "total_records": len(records),
                    "total_kept": final_kept,
                    "num_batches": num_batches,
                    "filled_by": "fill_missing_batches",
                },
            )
            _register_completed_warc(h, registry_prefix)
            logger.info("WARC COMPLETE via gap-fill: %s (%d/%d batches)", h, len(all_done), num_batches)
        else:
            logger.warning("%s: still %d/%d after fill — gaps in other regions?", h, len(all_done), num_batches)


if __name__ == "__main__":
    main()
