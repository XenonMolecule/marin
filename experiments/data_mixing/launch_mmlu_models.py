# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch mmlu_sl_verb (5-shot) over TRAINED sweep checkpoints (domain-target + baselines).

Counterpart of `data_mixing.launch_cultural_models` for the MMLU suite: results land under
`metadata/mmlu_target_models/{shots}shot/{run_name}/`, separate from the swarm namespace
(`olmix_swarm_mmlu`, which must stay a clean 363-per-corpus count for fit gating) and from
the curation sweeps' canonical `mmlu_sl_verb_results/`.

    iris --cluster marin job run --no-preemptible -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python -m experiments.data_mixing.launch_mmlu_models \\
               --manifest experiments/core_eval_manifests/<manifest>.txt --launch
"""

from __future__ import annotations

import dataclasses

from experiments.scaling_law_sweeps.mmlu.launch_mmlu_manifest import MMLU_SUITE, main

MMLU_MODELS_SUITE = dataclasses.replace(
    MMLU_SUITE,
    results_subpath="metadata/mmlu_target_models/{shots}shot/{run_name}/",
)


if __name__ == "__main__":
    main(MMLU_MODELS_SUITE)
