# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch mmlu_sl_verb (5-shot) over OLMIX swarm proxy checkpoints.

Same machinery as `scaling_law_sweeps.mmlu.launch_mmlu_manifest`, but results land under
`metadata/olmix_swarm_mmlu/{shots}shot/{run_name}/` — the swarm side-sweep namespace,
sibling of `olmix_swarm_bpb` — so 363-run swarms never collide with the curation sweeps'
canonical `mmlu_sl_verb_results/`. Build the manifest with
`experiments.data_mixing.build_olmix_bpb_manifest` (its run_name,region,output_path CSV
is suite-agnostic).

    iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.data_mixing.launch_swarm_mmlu \\
               --manifest experiments/core_eval_manifests/olmix_swarm_all.txt --launch
"""

from __future__ import annotations

import dataclasses

from experiments.scaling_law_sweeps.mmlu.launch_mmlu_manifest import MMLU_SUITE, main

SWARM_MMLU_SUITE = dataclasses.replace(
    MMLU_SUITE,
    results_subpath="metadata/olmix_swarm_mmlu/{shots}shot/{run_name}/",
)


if __name__ == "__main__":
    main(SWARM_MMLU_SUITE)
