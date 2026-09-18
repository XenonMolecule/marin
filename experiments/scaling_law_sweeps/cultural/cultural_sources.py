# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source pins and cleaning rules for the cultural suite.

Every rule here is decided in `.agents/projects/cultural_eval_suite.md` and pinned by
`tests/scaling_law_sweeps/test_cultural_sl_verb.py`:

  * BLEnD: one document per (country, question ID); cut the generation instruction; strip a
    trailing "(Provide ...)" format tail (737 short-answer-era instructions) but keep the
    "(e.g., ...)" answer-type hints (45); drop documents whose gold is `not-applicable`.
  * CulturalBench-Easy: every row, composite "Read the following statements" items included.
  * FORK: drop `Type == Underspecified` (US-default gold with no country named).
  * NormAd: keep `Gold Label in {yes, no}`; the story is the question, choices yes / no.

Functions take already-loaded rows (list of dicts) so the rules are testable without the
network; `build_cultural_tasks.py` does the downloading.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass

from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import (
    CULTURALBENCH_TASK,
    FORK_TASK,
    LETTERS,
    NORMAD_TASK,
    NUM_SHOTS,
    SPEC_BY_NAME,
    TaskSpec,
    blend_task_name,
)
from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import (
    BLEND_COUNTRIES,
    BLEND_HF_REPO,
    BLEND_MC_FILE,
    dedup_mc_rows,
    parse_mc_row,
)

BLEND_REVISION = "488f141cf0084b5835037ebac50f07bf09ddd68b"
BLEND_NOT_APPLICABLE = "not-applicable"

CULTURALBENCH_HF_REPO = "kellycyy/CulturalBench"
CULTURALBENCH_REVISION = "996d62a55f6a49c13596f0a31b772812a1aacd67"
CULTURALBENCH_FILE = "CulturalBench-Easy.csv"
CULTURALBENCH_COMPOSITE_MARKER = "Read the following statements"

FORK_GITHUB_REPO = "shramay-palta/FORK_ACL2023"
FORK_REVISION = "13e6f7c182b4e87876f10fec94444e0e67888b92"
FORK_FILE = "fork.jsonl"
FORK_URL = f"https://raw.githubusercontent.com/{FORK_GITHUB_REPO}/{FORK_REVISION}/{FORK_FILE}"
FORK_UNDERSPECIFIED = "Underspecified"

NORMAD_HF_REPO = "akhilayerukola/NormAd"
NORMAD_REVISION = "bb1a2997b6ab74b1f901493b15a67c2250a366c9"
NORMAD_FILE = "normad_etiquette_final_data.csv"
NORMAD_CHOICES: tuple[str, str] = ("yes", "no")

SOURCE_PINS: dict[str, dict[str, str]] = {
    "blend": {"repo": f"hf://datasets/{BLEND_HF_REPO}", "revision": BLEND_REVISION, "file": BLEND_MC_FILE},
    "culturalbench": {
        "repo": f"hf://datasets/{CULTURALBENCH_HF_REPO}",
        "revision": CULTURALBENCH_REVISION,
        "file": CULTURALBENCH_FILE,
    },
    "fork": {"repo": f"https://github.com/{FORK_GITHUB_REPO}", "revision": FORK_REVISION, "file": FORK_FILE},
    "normad": {"repo": f"hf://datasets/{NORMAD_HF_REPO}", "revision": NORMAD_REVISION, "file": NORMAD_FILE},
}

# A trailing parenthetical that starts with "Provide", allowing one level of nesting
# ("(Provide Arabic numerals (e.g., 12) only.)").
_BLEND_PROVIDE_TAIL = re.compile(r"\s*\(Provide[^()]*(?:\([^()]*\)[^()]*)*\)\s*$")


@dataclass(frozen=True)
class CulturalDoc:
    native_id: str
    country: str
    question: str
    choices: tuple[str, ...]
    gold: int
    category: str

    def as_record(self) -> dict:
        rec = asdict(self)
        rec["choices"] = list(self.choices)
        return rec


@dataclass(frozen=True)
class CulturalTask:
    spec: TaskSpec
    docs: tuple[CulturalDoc, ...]  # source order

    def __post_init__(self):
        for d in self.docs:
            if len(d.choices) != self.spec.num_choices:
                raise ValueError(f"{self.spec.name}: doc {d.native_id} has {len(d.choices)} choices")
            if not 0 <= d.gold < len(d.choices):
                raise ValueError(f"{self.spec.name}: doc {d.native_id} gold {d.gold} out of range")

    def split(self) -> tuple[tuple[CulturalDoc, ...], tuple[CulturalDoc, ...]]:
        """(dev, test): NUM_SHOTS documents prime every context and are held out of test.

        Shots are taken round-robin over the gold index in source order (the first gold-0
        document, then the first gold-1, ...), so the few-shot block never shows the model
        a one-sided answer prior: NormAd's source file is sorted by country and label
        (its first five rows are all Egypt / yes) and BLEnD's first rows lean on one
        letter. The dev split keeps that alternating order for lm-eval's `first_n`
        sampler; test keeps source order.
        """
        if len(self.docs) <= NUM_SHOTS:
            raise ValueError(f"{self.spec.name}: only {len(self.docs)} docs, need more than {NUM_SHOTS}")
        dev = select_balanced_shots(self.docs, NUM_SHOTS)
        chosen = {d.native_id for d in dev}
        return dev, tuple(d for d in self.docs if d.native_id not in chosen)


def select_balanced_shots(docs: tuple[CulturalDoc, ...], n: int) -> tuple[CulturalDoc, ...]:
    """The first `n` documents by gold-index round-robin in source order.

    A label whose documents run out is skipped, so a task with fewer than `n` distinct
    labels still yields `n` shots.
    """
    by_gold: dict[int, deque[CulturalDoc]] = {}
    for d in docs:
        by_gold.setdefault(d.gold, deque()).append(d)
    queues = [by_gold[g] for g in sorted(by_gold)]
    chosen: list[CulturalDoc] = []
    while len(chosen) < n and queues:
        for q in list(queues):
            if len(chosen) == n:
                break
            if q:
                chosen.append(q.popleft())
            else:
                queues.remove(q)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} documents available for {n} shots")
    return tuple(chosen)


def strip_blend_instruction_tail(question: str) -> str:
    """Remove a trailing "(Provide ...)" format instruction; leave "(e.g., ...)" hints alone."""
    return _BLEND_PROVIDE_TAIL.sub("", question).strip()


def blend_tasks(mc_rows: list[dict]) -> list[CulturalTask]:
    """One task per BLEnD country from the raw MC-file rows."""
    by_country = dedup_mc_rows(mc_rows)
    tasks: list[CulturalTask] = []
    for country_key, rows in by_country.items():
        if not rows:  # dedup_mc_rows pre-creates every BLEND_COUNTRIES key
            continue
        docs: list[CulturalDoc] = []
        for row in rows:
            question, choices, gold = parse_mc_row(row)
            if choices[gold].strip().lower() == BLEND_NOT_APPLICABLE:
                continue
            docs.append(
                CulturalDoc(
                    native_id=row["MCQID"],
                    country=BLEND_COUNTRIES[country_key],
                    question=strip_blend_instruction_tail(question),
                    choices=tuple(choices),
                    gold=gold,
                    category="",
                )
            )
        tasks.append(CulturalTask(SPEC_BY_NAME[blend_task_name(country_key)], tuple(docs)))
    return tasks


def culturalbench_task(rows: list[dict]) -> CulturalTask:
    """CulturalBench-Easy CSV rows (`prompt_question`, `prompt_option_a..d`, `answer`, `country`)."""
    docs: list[CulturalDoc] = []
    for row in rows:
        question = row["prompt_question"].strip()
        choices = tuple(row[f"prompt_option_{c}"].strip() for c in "abcd")
        docs.append(
            CulturalDoc(
                native_id=str(row["data_idx"]),
                country=row["country"].strip(),
                question=question,
                choices=choices,
                gold=LETTERS.index(row["answer"].strip().upper()),
                category="composite" if CULTURALBENCH_COMPOSITE_MARKER in question else "single",
            )
        )
    return CulturalTask(CULTURALBENCH_TASK, tuple(docs))


def fork_task(rows: list[dict]) -> CulturalTask:
    """FORK jsonl rows; Underspecified items are dropped."""
    docs: list[CulturalDoc] = []
    for row in rows:
        if row["Type"] == FORK_UNDERSPECIFIED:
            continue
        docs.append(
            CulturalDoc(
                native_id=row["ID"],
                country=row["Country"].strip(),
                question=row["Question"].strip(),
                choices=(row["Option A"].strip(), row["Option B"].strip()),
                gold="AB".index(row["Answer Key"].strip().upper()),
                category=row["Dimension"].strip(),
            )
        )
    return CulturalTask(FORK_TASK, tuple(docs))


def normad_task(rows: list[dict]) -> CulturalTask:
    """NormAd CSV rows; only yes / no gold labels survive."""
    docs: list[CulturalDoc] = []
    for row in rows:
        label = row["Gold Label"].strip().lower()
        if label not in NORMAD_CHOICES:
            continue
        docs.append(
            CulturalDoc(
                native_id=str(row["ID"]),
                country=row["Country"].strip().title(),
                question=row["Story"].strip(),
                choices=NORMAD_CHOICES,
                gold=NORMAD_CHOICES.index(label),
                category=row["Subaxis"].strip(),
            )
        )
    return CulturalTask(NORMAD_TASK, tuple(docs))
