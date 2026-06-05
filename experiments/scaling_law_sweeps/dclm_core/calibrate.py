# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibration harness: run DCLM CORE eval against a reference model whose
published CORE number we know, and compare.

We have two classes of reference candidates:

A. **Standard HF-format models DCLM published CORE for** (loadable directly
   in Levanter via HFCheckpointConverter.from_hf):
     - meta-llama/Llama-2-7b-hf   — DCLM Table 4 reports CORE ≈ 0.493 (v2)
     - allenai/OLMo-7B            — DCLM Table 4 reports CORE ≈ 0.461 (v2)
     - TinyLlama/TinyLlama-1.1B-Chat-v1.0 — no published CORE; not useful
   The 7B Llama models are gated (need HF_TOKEN) but otherwise straightforward.

B. **DCLM's own released models** (apple/DCLM-*) — these are OpenLM
   format, NOT standard HF Llama. Loading them in Levanter requires a
   format converter we don't have. Skip for calibration.

Calibration plan: run on a smaller standard-HF reference first (to verify
no crashes), then on Llama-2-7B for actual number-matching. If our number
matches DCLM-published within ~0.5%, the TPU lm-eval-harness path is
trusted and we proceed with Phase 3 pilot. Otherwise, we investigate
divergence task-by-task (see plan file).

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.calibrate \\
        --reference llama2-7b \\
        --output-json /tmp/calibrate_llama2_7b.json

The script just resolves a friendly name to an HF Hub ID and the published
CORE number, then dispatches to run_dclm_core_eval.main with the right args.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceModel:
    name: str
    hf_id: str
    published_core_v2: float | None  # None if not published
    notes: str = ""


REFERENCES: dict[str, ReferenceModel] = {
    "llama2-7b": ReferenceModel(
        name="llama2-7b",
        hf_id="meta-llama/Llama-2-7b-hf",
        published_core_v2=0.493,
        notes="DCLM paper Table 4 / Table 11 — gated model, requires HF_TOKEN."
        " The exact published number depends on which DCLM version (v1 vs v2 centering);"
        " confirm against the paper appendix before trusting.",
    ),
    "olmo-7b": ReferenceModel(
        name="olmo-7b",
        hf_id="allenai/OLMo-7B-hf",
        published_core_v2=0.461,
        notes="DCLM paper Table 4 — open weights, easy to fetch."
        " Note: there are several OLMo variants; this is the standard 7B"
        " HF-format release. Architecture is OLMo (custom), so verify"
        " Levanter's HFCheckpointConverter supports it.",
    ),
    # Smaller smoke-test option (no published CORE, but cheap to run):
    "tinyllama-1.1b": ReferenceModel(
        name="tinyllama-1.1b",
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        published_core_v2=None,
        notes="Smoke test only — DCLM doesn't publish CORE for TinyLlama."
        " Useful to verify the pipeline runs end-to-end before paying"
        " for Llama-2-7B inference.",
    ),
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reference", required=True, choices=sorted(REFERENCES), help="Which reference model to calibrate against."
    )
    ap.add_argument("--output-json", required=True, help="Where to write the CORE result.")
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: cap each task to N examples.")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    ref = REFERENCES[args.reference]
    logger.info("Calibration reference: %s (HF: %s)", ref.name, ref.hf_id)
    if ref.published_core_v2 is not None:
        logger.info("Expected published Core_v2: %.3f", ref.published_core_v2)
    else:
        logger.info("No published Core_v2 — smoke test only.")
    logger.info("Notes: %s", ref.notes)

    # Build the same args run_dclm_core_eval expects, in-process.
    from experiments.scaling_law_sweeps.dclm_core import run_dclm_core_eval

    sys.argv = [
        "run_dclm_core_eval",
        "--hf-checkpoint",
        ref.hf_id,
        "--output-json",
        args.output_json,
        "--run-name",
        f"calibrate-{ref.name}",
        "--tokenizer",
        ref.hf_id,
        "--max-length",
        str(args.max_length),
    ]
    if args.limit is not None:
        sys.argv += ["--limit", str(args.limit)]
    run_dclm_core_eval.main()

    if ref.published_core_v2 is not None:
        logger.info("=" * 70)
        logger.info(
            "CALIBRATION: compare the Core/Core_v2 in %s against published %.3f", args.output_json, ref.published_core_v2
        )
        logger.info("Acceptance: within ~0.5%% absolute.")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
