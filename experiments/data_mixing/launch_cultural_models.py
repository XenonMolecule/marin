# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch cultural_sl_verb (5-shot) over TRAINED cultural-target sweep checkpoints.

Same machinery as `data_mixing.launch_swarm_cultural`, but for the d512/d1536
`*_mix_cultural_clp_lambda0p01` training runs rather than swarm proxies. Results land
under `metadata/cultural_target_models/{shots}shot/{run_name}/` so they never mix with
the swarm namespace (which must stay a clean 363-per-corpus count for fit gating).

Build the manifest with `experiments.data_mixing.build_cultural_models_manifest`, then:

    iris --cluster marin job run --no-preemptible -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python -m experiments.data_mixing.launch_cultural_models \\
               --manifest experiments/core_eval_manifests/cultural_models_wave1.txt --launch
"""

from __future__ import annotations

import dataclasses

from experiments.scaling_law_sweeps.cultural.launch_cultural_manifest import CULTURAL_SUITE
from experiments.scaling_law_sweeps.mmlu.launch_mmlu_manifest import main

CULTURAL_MODELS_SUITE = dataclasses.replace(
    CULTURAL_SUITE,
    results_subpath="metadata/cultural_target_models/{shots}shot/{run_name}/",
)


if __name__ == "__main__":
    main(CULTURAL_MODELS_SUITE)
