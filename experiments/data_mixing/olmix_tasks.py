# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The BPB optimization target: OLMo Base-Easy minus everything in DCLM Core v2.

DCLM Core v2 is our held-out test set, so no task it contains may enter the mixture
objective -- otherwise the final "optimized mixture beats natural" comparison is
measured on data the optimizer was pointed at.

Construction, from marin's staged 56-task suite in
``experiments/scaling_law_sweeps/olmo_bpb/olmo_bpb_tasks_set.py``:

1. Start from ``OLMO_BASE_EASY_BPB`` (36 gold-continuation bpb tasks) +
   ``QA_RC_MC_BPB`` (20 multiple-choice rc:bpb tasks) = 56.
2. Drop the **5 tasks olmix itself drops**. Every fit config in the olmix repo sets
   ``filtering.drop_metrics`` to exclude ``qasper_yesno``, ``sciriff_yesno``,
   ``lab_bench_dbqa``, ``lab_bench_protocolqa``, ``medqa_en`` (plus two chat-ppl
   metrics we never had). The paper's §A.2 confirms these are the "5 additional
   metrics" it had originally included and removed.
3. Drop the **10 tasks that are in DCLM Core v2** (see :data:`CORE_V2_ALIASES`).
4. Drop membership olmix's suite does not use: the ``0shot`` code variants
   (``codex_humaneval``/``codex_mbpp`` are 3-shot there) and ``minerva_math_500``
   (their math family is the 7 Minerva subject splits). Note ``gsm8k`` is NOT in
   olmix's suite either -- see :func:`build_olmix_exact_tasks` -- but it is kept
   here because this objective is marin's, not a reproduction of theirs.
5. Add **MMLU** as 4 category-level tasks. MMLU is *not* among Core v2's 22
   ``low_variance_datasets``, so it is legal here, and it is the largest world-
   knowledge signal in olmix's own suite (8 of their 33 qa metrics).

That leaves 42 tasks: 8 math / 19 code / 15 QA. For comparison, olmix's own 51-task
suite is roughly 13% math / 37% code / 50% QA; dropping the Core v2 overlap removes
almost all of the easy commonsense QA, so ours is code-heavier at 45%. The objective
is still the **flat mean of per-task predictions** -- OlmixBase does not weight tasks
or families (``weights = np.ones(n)/n`` in ``LogLinearExactProposer``; the
``obj_weights`` path belongs to the per-family ablation, Table 3's middle row).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fsspec

from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    OLMO_BASE_EASY_BPB,
    QA_RC_MC_BPB,
)

# Where dclm_core defines Core v2. `Core_v2` is `low_variance_datasets_centered`
# (aggregated_metrics_upstream.py), so `low_variance_datasets` IS the Core v2 task list.
_AGGREGATION_JSON = (
    Path(__file__).resolve().parents[1] / "scaling_law_sweeps" / "dclm_core" / "additional_aggregation.json"
)
CORE_V2_AGGREGATION_KEY = "low_variance_datasets"

# Our task-dir names vs DCLM's eval-task names for the same underlying dataset.
# Only entries whose names differ need to be here; equal names match directly.
CORE_V2_ALIASES: dict[str, tuple[str, ...]] = {
    "lambada": ("lambada_openai",),
    "csqa": ("commonsense_qa",),
    "hellaswag": ("hellaswag", "hellaswag_zeroshot"),
    "openbookqa": ("openbook_qa",),
}

# The 5 metrics olmix drops in every fit config (olmix `filtering.drop_metrics`).
OLMIX_DROPPED_TASKS: frozenset[str] = frozenset(
    {
        "qasper_yesno",
        "sciriff_yesno",
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "medqa_en",
    }
)

# Variants marin stages that olmix's suite does not use.
_NON_OLMIX_VARIANTS: frozenset[str] = frozenset(
    {
        "codex_humaneval/gold_bpb_0shot",
        "codex_mbpp/gold_bpb_0shot",
        "minerva_math_500/gold_bpb_0shot",
    }
)

# MMLU, aggregated the way olmix does: the 4 categories, not the 57 subtasks. If only
# per-subtask requests can be staged, aggregate with olmix's example-count weights
# (their `aggregate_mmlu`) before fitting -- do NOT treat 57 subtasks as 57 tasks, or
# MMLU alone would outvote every other capability in the flat mean.
MMLU_BPB: tuple[str, ...] = (
    "mmlu_stem/rc_5shot",
    "mmlu_humanities/rc_5shot",
    "mmlu_social_sciences/rc_5shot",
    "mmlu_other/rc_5shot",
)

# Capability families, used for reporting only -- never to weight the objective.
# In olmix's suite but not marin's objective, and vice versa.
#
# GSM8K is marin's alone: the paper's Table 9 lists Minerva MATH's 7 subject splits as the
# entire Math family, and "GSM" appears nowhere in the paper. Keeping it would make our math
# family 8 tasks against their 7 and shift the flat 1/n mean.
GSM8K_VARIANT = "gsm8k/gold_bpb_5shot"

MATH_PREFIXES = ("gsm8k", "minerva_math")
CODE_PREFIXES = ("codex_", "mt_mbpp")


def task_name(task_variant: str) -> str:
    """``"arc_easy/rc_5shot"`` -> ``"arc_easy"``."""
    return task_variant.split("/", 1)[0]


def core_v2_task_names(aggregation_json: Path | str = _AGGREGATION_JSON) -> frozenset[str]:
    """The 22 DCLM Core v2 task names, read from dclm_core rather than hard-coded.

    Reading the file means adding a task to Core v2 automatically removes it from our
    objective, instead of silently contaminating the held-out set.
    """
    with open(aggregation_json) as f:
        aggregation = json.load(f)
    if CORE_V2_AGGREGATION_KEY not in aggregation:
        raise KeyError(
            f"{aggregation_json} has no {CORE_V2_AGGREGATION_KEY!r}; "
            f"Core v2's definition moved and the overlap filter is no longer valid"
        )
    return frozenset(aggregation[CORE_V2_AGGREGATION_KEY])


def overlaps_core_v2(task_variant: str, core_v2: frozenset[str]) -> bool:
    """Does this task's underlying dataset appear in Core v2?"""
    name = task_name(task_variant)
    candidates = CORE_V2_ALIASES.get(name, (name,))
    return any(alias in core_v2 for alias in candidates)


def build_target_tasks(
    *, include_mmlu: bool = True, aggregation_json: Path | str = _AGGREGATION_JSON
) -> tuple[str, ...]:
    """The mixture objective's task list, in a stable order.

    Args:
        include_mmlu: include the 4 MMLU category tasks. They must be staged into
            ``eval_datasets/olmo_in_loop_evals/oe_eval_tasks/`` first; set False to
            run the 38-task objective while staging is pending.
    """
    core_v2 = core_v2_task_names(aggregation_json)
    kept = [
        tv
        for tv in (*OLMO_BASE_EASY_BPB, *QA_RC_MC_BPB)
        if tv not in _NON_OLMIX_VARIANTS
        and task_name(tv) not in OLMIX_DROPPED_TASKS
        and not overlaps_core_v2(tv, core_v2)
    ]
    if include_mmlu:
        kept.extend(MMLU_BPB)
    return tuple(kept)


def build_olmix_exact_tasks(
    *, include_mmlu: bool = True, aggregation_json: Path | str = _AGGREGATION_JSON
) -> tuple[str, ...]:
    """olmix's own 51-task BPB suite, reproduced exactly (paper Table 9).

    Differs from :func:`build_target_tasks` in exactly two ways, both verified against
    the paper rather than inferred:

    1. **Core v2 overlap is NOT removed.** The 10 tasks marin holds out -- arc_easy,
       arc_challenge, csqa, hellaswag, winogrande, piqa, coqa, jeopardy, squad, lambada
       -- are all in Table 9, so an exact reproduction must fit on them.
    2. **GSM8K is dropped** (:data:`GSM8K_VARIANT`); Table 9's Math family is Minerva's
       7 subject splits alone.

    Minerva needs no change: the staged ``gold_bpb_0shot`` directories are misnamed. Their
    ``config.json`` is ``num_shots=4``, ``fewshot_source="Minerva:MATH:fixed"``,
    ``compute_gold_bpb=true``, alias ``minerva_math_*::bpb`` -- i.e. olmes'
    ``minerva_math_{subject}:bpb::olmes`` -- and every staged request context carries the 4
    hand-written exemplars. Verified across all 7 subjects.

    **This forfeits the held-out comparison.** DCLM Core v2 contains the 10 tasks re-added
    above, so a mixture optimised against this suite may not then be reported as beating
    natural *on Core v2*. Use :func:`build_target_tasks` for that claim and this one for
    comparability with the paper; both read the same evals, so running both is free.

    Args:
        include_mmlu: include the 4 MMLU category tasks (Table 9 averages the 57 subjects
            into 4 categories, which is what :data:`MMLU_BPB` stages).
        aggregation_json: unused for filtering; accepted so the signature matches
            :func:`build_target_tasks` and callers can swap one for the other.
    """
    kept = [
        tv
        for tv in (*OLMO_BASE_EASY_BPB, *QA_RC_MC_BPB)
        if tv not in _NON_OLMIX_VARIANTS and task_name(tv) not in OLMIX_DROPPED_TASKS and tv != GSM8K_VARIANT
    ]
    if include_mmlu:
        kept.extend(MMLU_BPB)
    return tuple(kept)


def task_family(task_variant: str) -> str:
    """``"math"`` / ``"code"`` / ``"qa"`` -- for reporting, not for weighting."""
    name = task_name(task_variant)
    if name.startswith(MATH_PREFIXES):
        return "math"
    if name.startswith(CODE_PREFIXES):
        return "code"
    return "qa"


def family_counts(tasks: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tv in tasks:
        counts[task_family(tv)] = counts.get(task_family(tv), 0) + 1
    return counts


def staged_task_variants(oe_eval_tasks_root: str) -> frozenset[str]:
    """Enumerate ``<task>/<variant>`` dirs actually present under an
    ``oe_eval_tasks/`` root, so a missing stage fails before 363 runs, not after."""
    fs, root = fsspec.core.url_to_fs(oe_eval_tasks_root.rstrip("/"))
    found: set[str] = set()
    for task_dir in fs.ls(root, detail=False):
        task = os.path.basename(task_dir.rstrip("/"))
        if not task:
            continue
        for variant_dir in fs.ls(task_dir, detail=False):
            variant = os.path.basename(variant_dir.rstrip("/"))
            if variant:
                found.add(f"{task}/{variant}")
    return frozenset(found)


def assert_tasks_staged(tasks: tuple[str, ...], oe_eval_tasks_root: str) -> None:
    """Raise unless every task in ``tasks`` has staged eval requests."""
    missing = sorted(set(tasks) - staged_task_variants(oe_eval_tasks_root))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} target task(s) are not staged under {oe_eval_tasks_root}: {missing}. "
            f"Stage them (see experiments/scaling_law_sweeps/dclm_core/build_eval_dataset_cache.py) "
            f"or pass include_mmlu=False if only MMLU is outstanding."
        )
