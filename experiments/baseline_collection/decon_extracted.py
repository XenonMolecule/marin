#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mark eval-contamination spans on a corpus against a decontamination source.

Runs `marin.processing.classification.decon` in DECONTAMINATE mode: builds a
bloom filter of n-grams from ``--decon-source`` (rendered eval items, one
``text`` field per doc), then marks every corpus doc with the spans that
overlap the filter. **READ-ONLY** — it writes an attributes tree
(``{id, attributes: {<attribute_name>: [[start, end, score], ...]}}``) and the
bloom; it deletes nothing. Removing flagged docs is a deliberate downstream
step taken only after inspecting the flag rate.

General-purpose: point ``--input-path`` / ``--decon-source`` at any corpus and
any source to decontaminate other datasets later.

Bloom sizing: ``--estimated-ngrams`` must be >= the number of distinct n-grams
in the source or the false-positive rate degrades sharply. Oversizing only
costs memory, so the default is generous for a CORE-v2-scale eval source.

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --region us-central1 --job-name decon-high-quality-core-v2 \\
        -- python experiments/baseline_collection/decon_extracted.py \\
           --input-path gs://marin-us-central1/documents/baseline_high_quality_deduped/10364warcs/deduped/ \\
           --decon-source gs://marin-us-central1/decontamination/dclm_core_v2/ \\
           --output-path gs://marin-us-central1/documents/baseline_high_quality_decon/10364warcs_core_v2/ \\
           --ngram-length 13
"""

from __future__ import annotations

import argparse
import logging

from marin.processing.classification.decon import DeconConfig, DeconMode, NGramConfig, decontaminate

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="Corpus dir to mark (gs:// or local).")
    parser.add_argument("--decon-source", required=True, help="Eval-item source dir to build the filter from.")
    parser.add_argument("--output-path", required=True, help="Where to write the attributes tree + bloom.")
    parser.add_argument("--ngram-length", type=int, default=13, help="N-gram length (GPT-3 standard = 13).")
    parser.add_argument("--text-field", default="text", help="Text field in both corpus and source records.")
    parser.add_argument(
        "--attribute-name", default="eval_contamination", help="Key under which overlap spans are recorded."
    )
    parser.add_argument(
        "--estimated-ngrams",
        type=int,
        default=20_000_000,
        help="Bloom capacity — must exceed the source's distinct n-gram count.",
    )
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=1e-12,
        # The corpus has billions of n-grams queried against the filter, so a
        # loose per-lookup fp produces many false flags corpus-wide; keep it
        # tiny. At 1e-12 over ~8e9 corpus n-grams, expected false hits < 0.01.
        help="Bloom false-positive rate (keep tiny: corpus-wide hits scale with fp).",
    )
    parser.add_argument("--processes", type=int, default=16, help="Shards for the (small) filter-build fan-out.")
    parser.add_argument("--worker-ram", default="8g", help="RAM per bloom-build/mark Zephyr worker.")
    args = parser.parse_args()

    config = DeconConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        decontaminate_source=args.decon_source,
        mode=DeconMode.DECONTAMINATE,
        attribute_name=args.attribute_name,
        # stride=0 -> every n-gram position; overlap_threshold is unused in
        # DECONTAMINATE marking (any overlap > 0 is recorded with its score).
        ngram=NGramConfig(ngram_length=args.ngram_length, stride=0, overlap_threshold=0.0),
        estimated_doc_count=args.estimated_ngrams,
        false_positive_rate=args.false_positive_rate,
        processes=args.processes,
        text_field=args.text_field,
        worker_ram=args.worker_ram,
    )

    logger.info("Decon DECONTAMINATE config:\n%s", config)
    result = decontaminate(config)
    logger.info("Decon complete: %s", result)
    logger.info("Attributes + bloom written under: %s", args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
