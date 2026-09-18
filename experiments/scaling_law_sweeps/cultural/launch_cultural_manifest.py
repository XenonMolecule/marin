# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the cultural suite (`cultural_sl_verb`, 5-shot) over a checkpoint manifest.

Reuses `mmlu.launch_mmlu_manifest` with the cultural `EvalSuite`: one region-pinned iris
child per checkpoint running `run_cultural_eval.py`, reading the staged suite from the
checkpoint's own bucket. Results land at
`gs://<bucket>/metadata/cultural_sl_verb_results/<shots>shot/<run_name>/results.json`.

Stage the data first (`build_cultural_tasks.py --upload-to` into every bucket the
manifest touches); a child whose bucket lacks `eval_datasets/cultural_sl_verb/` fails at
YAML generation with a missing-split error rather than silently scoring nothing.

    iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.scaling_law_sweeps.cultural.launch_cultural_manifest \\
               --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_dedup245.txt \\
               --launch
"""

from __future__ import annotations

from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import NUM_SHOTS, SUPPORTED_SHOTS
from experiments.scaling_law_sweeps.mmlu.launch_mmlu_manifest import EvalSuite, main

RESULTS_SUBPATH = "metadata/cultural_sl_verb_results/{shots}shot/{run_name}/"
CACHE_SUBPATH = "eval_datasets/cultural_sl_verb/"

CULTURAL_SUITE = EvalSuite(
    name="cultural",
    run_module="experiments.scaling_law_sweeps.cultural.run_cultural_eval",
    results_subpath=RESULTS_SUBPATH,
    cache_subpath=CACHE_SUBPATH,
    supported_shots=SUPPORTED_SHOTS,
    default_shots=(NUM_SHOTS,),
)


if __name__ == "__main__":
    main(CULTURAL_SUITE)
