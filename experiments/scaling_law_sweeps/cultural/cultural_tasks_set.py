# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cultural eval suite (`cultural_sl_verb`): task table, shot count, YAML generation.

Four human-centered benchmarks rendered in the CRFM fork's `mmlu_sl_verb` shape
(verbalized multiple choice, lettered choices in the context and in the scored
continuation, 5-shot, the seven soft metrics) and run through the Levanter harness like
`experiments/scaling_law_sweeps/mmlu/`. Rules of record:
`.agents/projects/cultural_eval_suite.md`.

Tasks (all prefixed `cultural_sl_verb_` so Levanter's alias renaming, which appends the
group alias suffix to every subtask that shares the group's prefix, keys result rows
`cultural_sl_verb_<task>_<N>shot`):

  * `blend_<country>` x 16 — BLEnD MCQ, one task per country, MC-4.
  * `culturalbench`        — CulturalBench-Easy, 45 countries in one task, MC-4.
  * `fork`                 — FORK dining customs, 2-way.
  * `normad`               — NormAd yes/no etiquette judgements, 2-way.

The staged data is plain parquet (`<data_dir>/<task>/{dev,test}.parquet`), so the task
YAMLs are generated at eval time with absolute `data_files` paths — lm-eval resolves
relative `data_files` against the working directory, not the YAML — and registered via
`lm_eval_include_path`. No HF Hub access is needed to load them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import BLEND_COUNTRIES

GROUP_TASK_NAME = "cultural_sl_verb"
NUM_SHOTS = 5
SUPPORTED_SHOTS: tuple[int, ...] = (NUM_SHOTS,)
LETTERS = "ABCD"

TEST_SPLIT = "test"
DEV_SPLIT = "dev"
MANIFEST_FILENAME = "_manifest.json"


@dataclass(frozen=True)
class TaskSpec:
    """One lm-eval task of the suite.

    Exactly one of `description` (static preamble, MMLU-style) and `doc_description`
    (jinja fragment rendered at the top of every document, for tasks whose documents span
    countries) is set.
    """

    name: str
    benchmark: str
    num_choices: int
    description: str | None = None
    doc_description: str | None = None

    def __post_init__(self):
        if (self.description is None) == (self.doc_description is None):
            raise ValueError(f"{self.name}: set exactly one of description / doc_description")
        if not 2 <= self.num_choices <= len(LETTERS):
            raise ValueError(f"{self.name}: num_choices must be in [2, {len(LETTERS)}]")


def blend_task_name(country_key: str) -> str:
    return f"{GROUP_TASK_NAME}_blend_{country_key.lower()}"


def _blend_spec(country_key: str) -> TaskSpec:
    return TaskSpec(
        name=blend_task_name(country_key),
        benchmark="blend",
        num_choices=4,
        description=(
            "The following are multiple choice questions (with answers) about everyday life in "
            f"{BLEND_COUNTRIES[country_key]}.\n\n"
        ),
    )


CULTURALBENCH_TASK = TaskSpec(
    name=f"{GROUP_TASK_NAME}_culturalbench",
    benchmark="culturalbench",
    num_choices=4,
    doc_description="The following are multiple choice questions (with answers) about culture in {{country}}.\n\n",
)
FORK_TASK = TaskSpec(
    name=f"{GROUP_TASK_NAME}_fork",
    benchmark="fork",
    num_choices=2,
    description="The following are multiple choice questions (with answers) about dining customs around the world.\n\n",
)
NORMAD_TASK = TaskSpec(
    name=f"{GROUP_TASK_NAME}_normad",
    benchmark="normad",
    num_choices=2,
    doc_description="The following are questions (with answers) about social etiquette in {{country}}.\n\n",
)

TASK_SPECS: tuple[TaskSpec, ...] = (
    *(_blend_spec(k) for k in BLEND_COUNTRIES),
    CULTURALBENCH_TASK,
    FORK_TASK,
    NORMAD_TASK,
)
SPEC_BY_NAME: dict[str, TaskSpec] = {s.name: s for s in TASK_SPECS}
BENCHMARKS: tuple[str, ...] = ("blend", "culturalbench", "fork", "normad")
TASKS_BY_BENCHMARK: dict[str, tuple[str, ...]] = {
    b: tuple(s.name for s in TASK_SPECS if s.benchmark == b) for b in BENCHMARKS
}

CULTURAL_BY_SHOTS: dict[int, EvalTaskConfig] = {
    NUM_SHOTS: EvalTaskConfig(GROUP_TASK_NAME, NUM_SHOTS, task_alias=f"{GROUP_TASK_NAME}_{NUM_SHOTS}shot"),
}


def task_for_shots(num_fewshot: int) -> EvalTaskConfig:
    """The single group task config at `num_fewshot`; ValueError on an unblessed shot count."""
    if num_fewshot not in CULTURAL_BY_SHOTS:
        raise ValueError(f"Unsupported shot count {num_fewshot}; expected one of {SUPPORTED_SHOTS}.")
    return CULTURAL_BY_SHOTS[num_fewshot]


def result_key(task_name: str, shots: int) -> str:
    """The results.json row key Levanter writes for `task_name` under the aliased group."""
    return f"{task_name}_{shots}shot"


# Verbatim from the fork's `lm_eval/tasks/mmlu/sl_verb/_default_template_yaml`.
METRIC_LIST: list[dict] = [
    {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
    {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
    {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
    {"metric": "logprob", "aggregation": "mean", "higher_is_better": False},
    {"metric": "choice_logprob", "aggregation": "mean", "higher_is_better": False},
    {"metric": "choice_prob_norm", "aggregation": "mean", "higher_is_better": False},
    {"metric": "choice_logprob_norm", "aggregation": "mean", "higher_is_better": True},
]
METRIC_NAMES: tuple[str, ...] = tuple(m["metric"] for m in METRIC_LIST)


def doc_to_text(spec: TaskSpec) -> str:
    lines = [f"{LETTERS[i]}. {{{{choices[{i}]}}}}" for i in range(spec.num_choices)]
    body = "{{question.strip()}}\n" + "\n".join(lines) + "\nAnswer:"
    return (spec.doc_description or "") + body


def doc_to_choice(spec: TaskSpec) -> str:
    items = ", ".join(f"'{LETTERS[i]}. ' + choices[{i}]" for i in range(spec.num_choices))
    return "{{[" + items + "]}}"


def task_yaml(spec: TaskSpec, data_dir: str) -> dict:
    """The lm-eval task config for `spec`, pointing at its staged parquet under `data_dir`."""
    task_dir = os.path.join(data_dir, spec.name)
    files = {split: os.path.join(task_dir, f"{split}.parquet") for split in (TEST_SPLIT, DEV_SPLIT)}
    for path in files.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{spec.name}: staged split missing at {path}")
    cfg: dict = {
        "task": spec.name,
        "dataset_path": "parquet",
        "dataset_kwargs": {"data_files": files},
        "test_split": TEST_SPLIT,
        "fewshot_split": DEV_SPLIT,
        "fewshot_config": {"sampler": "first_n"},
        "output_type": "multiple_choice",
        "doc_to_text": doc_to_text(spec),
        "doc_to_choice": doc_to_choice(spec),
        "doc_to_target": "gold",
        "metric_list": METRIC_LIST,
        "metadata": {"version": 1.0},
    }
    if spec.description is not None:
        cfg["description"] = spec.description
    return cfg


def group_yaml() -> dict:
    """The group config; the aggregate row exists for completeness, the consolidator
    recomputes the benchmark-level macro average itself (BLEnD's 16 tasks would otherwise
    dominate a flat mean)."""
    return {
        "group": GROUP_TASK_NAME,
        "task": [s.name for s in TASK_SPECS],
        "aggregate_metric_list": [
            {"metric": "acc", "weight_by_size": True},
            {"metric": "acc_norm", "weight_by_size": True},
            *({"metric": m, "aggregation": "mean", "weight_by_size": False} for m in METRIC_NAMES[2:]),
        ],
        "metadata": {"version": 1.0},
    }


def write_task_yamls(data_dir: str, out_dir: str) -> list[str]:
    """Write one YAML per task plus the group YAML into `out_dir`; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for spec in TASK_SPECS:
        path = os.path.join(out_dir, f"{spec.name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(task_yaml(spec, data_dir), f, sort_keys=False, allow_unicode=True)
        written.append(path)
    group_path = os.path.join(out_dir, f"{GROUP_TASK_NAME}.yaml")
    with open(group_path, "w") as f:
        yaml.safe_dump(group_yaml(), f, sort_keys=False)
    written.append(group_path)
    return written
