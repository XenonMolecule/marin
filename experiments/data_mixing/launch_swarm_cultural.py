# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the cultural suite (cultural_sl_verb, 5-shot) over OLMIX swarm proxy checkpoints.

Same machinery as `scaling_law_sweeps.cultural.launch_cultural_manifest`, but results land
under `metadata/olmix_swarm_cultural/{shots}shot/{run_name}/` — the swarm side-sweep
namespace, sibling of `olmix_swarm_bpb` — so 363-run swarms never collide with the
curation sweeps' canonical `cultural_sl_verb_results/`. Build the manifest with
`experiments.data_mixing.build_olmix_bpb_manifest`.

    iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.data_mixing.launch_swarm_cultural \\
               --manifest experiments/core_eval_manifests/olmix_swarm_all.txt --launch
"""

from __future__ import annotations

import dataclasses

from experiments.scaling_law_sweeps.cultural.launch_cultural_manifest import CULTURAL_SUITE
from experiments.scaling_law_sweeps.mmlu.launch_mmlu_manifest import main

SWARM_CULTURAL_SUITE = dataclasses.replace(
    CULTURAL_SUITE,
    results_subpath="metadata/olmix_swarm_cultural/{shots}shot/{run_name}/",
)


if __name__ == "__main__":
    main(SWARM_CULTURAL_SUITE)
