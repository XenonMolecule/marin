# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cascade economics for two (or more) useful-classifiers scored on the SAME docs.

Answers one question: does a second, cheaper classifier earn a slot *after* fastText in a
``fastText -> X -> ModernBERT`` cascade? Peak F1 alone cannot answer it — a stage-1 filter is
operated at high recall, and what matters there is (a) how much junk it removes at that recall and
(b) whether its mistakes are DECORRELATED from the stage before it. Two models that are individually
excellent but agree on which junk looks useful stack to nothing.

Inputs are per-doc score JSONs written by ``score_fasttext_on_bert_test.py`` (``{ft_probs, labels}``)
and ``score_pooled_frozen_eval.py`` (``{probs, labels, ...}``), aligned index-for-index. The
``labels`` arrays are the alignment proof and are compared element-for-element.

Definitions used throughout:

- **junk exclusion** at a threshold = fraction of true-negative (junk) docs dropped = specificity.
  This is the quantity a stage-1 filter exists to maximize.
- **recall** = fraction of useful docs kept.
- **independence baseline** for a cascade = ``1 - (1 - e1) * (1 - e2)``, the exclusion the pair would
  reach if each model's junk misses were independent given their marginal exclusion rates ``e1``,
  ``e2``. Measured-minus-baseline is the decorrelation gap; a big negative gap means the models fail
  on the same junk and the second stage is nearly free of value.

Pure CPU, run locally::

    export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")
    uv run python -m experiments.baseline_collection.cascade_analysis \\
      --stage1 fastText=gs://.../ft_w640_on_frozen7k.json \\
      --stage2 pooled=gs://.../pooled_1M_on_frozen7k.json
    uv run python -m experiments.baseline_collection.cascade_analysis --selftest
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import fsspec
import numpy as np

RECALL_TARGETS = (0.99, 0.975, 0.95)
COMBINED_RECALL_TARGETS = (0.99, 0.98, 0.95)
VERDICT_RECALL = 0.98
# Extra junk exclusion (percentage points) a second stage must buy at VERDICT_RECALL to justify its
# own inference pass, tuning, and deployment surface in the cascade.
WORTH_IT_GAIN_PP = 5.0
PROB_KEYS = ("probs", "ft_probs", "scores")


@dataclass(frozen=True)
class ScoreSet:
    """Per-doc scores for one model over a fixed, ordered document set."""

    name: str
    probs: np.ndarray
    labels: np.ndarray

    @property
    def positive(self) -> np.ndarray:
        return self.labels == 1

    @property
    def negative(self) -> np.ndarray:
        return self.labels == 0


def load_scores(name: str, path: str) -> ScoreSet:
    """Read a ``{probs|ft_probs, labels}`` JSON into a ScoreSet."""
    with fsspec.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    key = next((k for k in PROB_KEYS if k in payload), None)
    if key is None:
        raise ValueError(f"{path}: no score column (looked for {PROB_KEYS}); keys={list(payload)}")
    return ScoreSet(name, np.asarray(payload[key], dtype=np.float64), np.asarray(payload["labels"], dtype=np.int32))


def check_alignment(sets: list[ScoreSet]) -> None:
    """Fail loudly unless every score file carries the identical label sequence (same doc order)."""
    reference = sets[0]
    for other in sets[1:]:
        if other.labels.shape != reference.labels.shape:
            raise ValueError(f"{other.name} has {other.labels.size} docs, {reference.name} has {reference.labels.size}")
        mismatch = int(np.sum(other.labels != reference.labels))
        if mismatch:
            first = int(np.argmax(other.labels != reference.labels))
            raise ValueError(
                f"{other.name} labels differ from {reference.name} at {mismatch} positions (first: index {first}) — "
                "the two score files are NOT index-aligned; re-read with the same order."
            )


# --- Single-model operating points -----------------------------------------


def best_f1(scores: ScoreSet) -> tuple[float, float, float, float]:
    """(best F1, threshold, precision, recall) over all thresholds. Keep = ``prob >= threshold``."""
    order = np.argsort(-scores.probs, kind="stable")
    labels = scores.labels[order]
    probs = scores.probs[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    total_pos = int((scores.labels == 1).sum())
    # Only consider cut points at score-value boundaries (a tie block must be kept or dropped whole).
    boundary = np.empty(probs.size, dtype=bool)
    boundary[:-1] = probs[:-1] != probs[1:]
    boundary[-1] = True
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(total_pos, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    f1 = np.where(boundary, f1, -1.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(probs[i]), float(precision[i]), float(recall[i])


def threshold_for_recall(probs: np.ndarray, positive: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose kept set still contains >= ``target_recall`` of the positives."""
    pos_scores = np.sort(probs[positive])[::-1]
    if pos_scores.size == 0:
        return -np.inf
    k = min(pos_scores.size, max(1, math.ceil(target_recall * pos_scores.size)))
    return float(pos_scores[k - 1])


def operating_point(scores: ScoreSet, threshold: float) -> tuple[float, float]:
    """(recall, junk exclusion) for ``keep = prob >= threshold``."""
    keep = scores.probs >= threshold
    recall = float(keep[scores.positive].mean())
    exclusion = float(1.0 - keep[scores.negative].mean())
    return recall, exclusion


# --- Two-stage cascade ------------------------------------------------------


@dataclass(frozen=True)
class CascadeResult:
    """One cascade operating point: stage-1 keeps ``p1 >= t1``, survivors keep ``p2 >= t2``."""

    t1: float
    t2: float
    recall: float
    exclusion: float
    stage2_load: float  # fraction of ALL docs stage 2 must actually score
    stage1_recall: float
    stage1_exclusion: float
    stage2_exclusion_standalone: float

    @property
    def independent_exclusion(self) -> float:
        """Exclusion if the two models' junk misses were independent."""
        return 1.0 - (1.0 - self.stage1_exclusion) * (1.0 - self.stage2_exclusion_standalone)

    @property
    def decorrelation_gap(self) -> float:
        """measured - independent (<= 0 means correlated errors; 0 means fully independent)."""
        return self.exclusion - self.independent_exclusion


def evaluate_cascade(stage1: ScoreSet, stage2: ScoreSet, t1: float, t2: float) -> CascadeResult:
    """Measure a fixed (t1, t2) cascade. A doc survives only if it clears BOTH stages."""
    pass1 = stage1.probs >= t1
    keep = pass1 & (stage2.probs >= t2)
    positive, negative = stage1.positive, stage1.negative
    s1_recall, s1_exclusion = operating_point(stage1, t1)
    _, s2_exclusion = operating_point(stage2, t2)
    return CascadeResult(
        t1=t1,
        t2=t2,
        recall=float(keep[positive].mean()),
        exclusion=float(1.0 - keep[negative].mean()),
        stage2_load=float(pass1.mean()),
        stage1_recall=s1_recall,
        stage1_exclusion=s1_exclusion,
        stage2_exclusion_standalone=s2_exclusion,
    )


def best_cascade_at_recall(stage1: ScoreSet, stage2: ScoreSet, combined_recall: float, grid: int = 200) -> CascadeResult:
    """Best (max junk exclusion) cascade whose COMBINED recall is >= ``combined_recall``.

    Sweeps how the recall budget is split: stage 1 runs at recall ``r1`` in ``[combined_recall, 1]``,
    then stage 2's threshold is set as high as the remaining budget allows.
    """
    positive = stage1.positive
    total_pos = int(positive.sum())
    need = math.ceil(combined_recall * total_pos)
    best: CascadeResult | None = None
    for r1 in np.linspace(combined_recall, 1.0, grid):
        t1 = threshold_for_recall(stage1.probs, positive, float(r1))
        survivors = positive & (stage1.probs >= t1)
        if int(survivors.sum()) < need:
            continue
        surviving_scores = np.sort(stage2.probs[survivors])[::-1]
        t2 = float(surviving_scores[need - 1])
        result = evaluate_cascade(stage1, stage2, t1, t2)
        if result.recall < combined_recall - 1e-12:
            continue
        if best is None or result.exclusion > best.exclusion:
            best = result
    if best is None:
        raise ValueError(f"no cascade reaches combined recall {combined_recall}")
    return best


# --- Correlations -----------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), the basis of Spearman's rho."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_ranks(a), _ranks(b))


# --- Reporting --------------------------------------------------------------


def report(stage1: ScoreSet, stage2: ScoreSet) -> None:
    n = stage1.labels.size
    n_pos = int(stage1.positive.sum())
    print(f"docs={n}  useful={n_pos} ({n_pos / n:.1%})  junk={n - n_pos}")
    print(f"alignment: label sequences identical for {stage1.name} and {stage2.name} (checked element-for-element)")

    print("\n== standalone ==")
    print(
        f"{'model':<12}{'bestF1':>9}{'thr':>8}{'prec':>8}{'rec':>8}   "
        + "".join(f"excl@R{r:<8.3f}" for r in RECALL_TARGETS)
    )
    for scores in (stage1, stage2):
        f1, thr, precision, recall = best_f1(scores)
        cells = []
        for target in RECALL_TARGETS:
            t = threshold_for_recall(scores.probs, scores.positive, target)
            actual_recall, exclusion = operating_point(scores, t)
            cells.append(f"{exclusion:.4f}(R={actual_recall:.3f}) ")
        print(f"{scores.name:<12}{f1:>9.4f}{thr:>8.3f}{precision:>8.4f}{recall:>8.4f}   " + "".join(cells))

    print(f"\n== cascade {stage1.name} -> {stage2.name}, both stages at the SAME recall target ==")
    print(
        f"{'R/stage':>9}{'comb.rec':>10}{'comb.excl':>11}{'indep':>9}{'gap':>9}"
        f"{'s1 excl':>9}{'s2 excl':>9}{'s2 load':>9}"
    )
    for target in RECALL_TARGETS:
        t1 = threshold_for_recall(stage1.probs, stage1.positive, target)
        t2 = threshold_for_recall(stage2.probs, stage2.positive, target)
        r = evaluate_cascade(stage1, stage2, t1, t2)
        print(
            f"{target:>9.3f}{r.recall:>10.4f}{r.exclusion:>11.4f}{r.independent_exclusion:>9.4f}"
            f"{r.decorrelation_gap:>9.4f}{r.stage1_exclusion:>9.4f}{r.stage2_exclusion_standalone:>9.4f}"
            f"{r.stage2_load:>9.4f}"
        )

    print("\n== cascade at a fixed COMBINED recall budget (best split of the budget) ==")
    print(
        f"{'comb.R':>9}{'comb.excl':>11}{'s1-only':>9}{'gain pp':>9}{'indep':>9}{'gap':>9}"
        f"{'r1':>8}{'t1':>8}{'t2':>8}{'s2 load':>9}"
    )
    for target in COMBINED_RECALL_TARGETS:
        r = best_cascade_at_recall(stage1, stage2, target)
        t1_solo = threshold_for_recall(stage1.probs, stage1.positive, target)
        _, solo_exclusion = operating_point(stage1, t1_solo)
        print(
            f"{target:>9.3f}{r.exclusion:>11.4f}{solo_exclusion:>9.4f}{100 * (r.exclusion - solo_exclusion):>9.2f}"
            f"{r.independent_exclusion:>9.4f}{r.decorrelation_gap:>9.4f}{r.stage1_recall:>8.3f}"
            f"{r.t1:>8.3f}{r.t2:>8.3f}{r.stage2_load:>9.4f}"
        )

    print("\n== score correlation ==")
    junk = stage1.negative
    print(
        f"all docs : pearson={pearson(stage1.probs, stage2.probs):.4f} "
        f"spearman={spearman(stage1.probs, stage2.probs):.4f}"
    )
    print(
        f"junk only: pearson={pearson(stage1.probs[junk], stage2.probs[junk]):.4f} "
        f"spearman={spearman(stage1.probs[junk], stage2.probs[junk]):.4f}"
    )

    best = best_cascade_at_recall(stage1, stage2, VERDICT_RECALL)
    t1_solo = threshold_for_recall(stage1.probs, stage1.positive, VERDICT_RECALL)
    _, solo_exclusion = operating_point(stage1, t1_solo)
    gain_pp = 100 * (best.exclusion - solo_exclusion)
    worth = "YES" if gain_pp >= WORTH_IT_GAIN_PP else "NO"
    print(
        f"\nVERDICT: at combined recall {VERDICT_RECALL:.2f}, adding {stage2.name} after {stage1.name} moves junk "
        f"exclusion {solo_exclusion:.4f} -> {best.exclusion:.4f} (+{gain_pp:.2f} pp; independence would give "
        f"{best.independent_exclusion:.4f}, gap {best.decorrelation_gap:+.4f}), while scoring "
        f"{best.stage2_load:.1%} of docs. Worth a cascade slot (>= {WORTH_IT_GAIN_PP:.0f} pp)? {worth}"
    )


# --- Self-test --------------------------------------------------------------


def _synthetic(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray) -> tuple[ScoreSet, ScoreSet]:
    return ScoreSet("A", probs_a, labels), ScoreSet("B", probs_b, labels)


def selftest() -> None:
    """Prove the cascade math on constructed cases where the answer is known by hand."""
    rng = np.random.default_rng(0)

    # 1. Identical models: stage 2 can add nothing; measured exclusion == stage-1 exclusion and the
    #    decorrelation gap is maximally negative.
    labels = np.array([1] * 100 + [0] * 100)
    probs = np.concatenate([rng.uniform(0.5, 1.0, 100), rng.uniform(0.0, 0.6, 100)])
    a, b = _synthetic(probs, probs.copy(), labels)
    identical = best_cascade_at_recall(a, b, 0.98)
    solo_t = threshold_for_recall(a.probs, a.positive, 0.98)
    _, solo_excl = operating_point(a, solo_t)
    assert identical.recall >= 0.98 - 1e-12, identical.recall
    assert abs(identical.exclusion - solo_excl) < 1e-9, (identical.exclusion, solo_excl)
    assert identical.decorrelation_gap < -1e-6, identical.decorrelation_gap

    # 2. Perfectly complementary models: A misses the first half of the junk, B misses the second
    #    half, and each keeps every positive. The cascade must exclude ALL junk (exclusion 1.0),
    #    beating either stage alone, with the gap at/above the independence baseline.
    labels = np.array([1] * 100 + [0] * 100)
    hi = np.ones(100)
    junk_a = np.concatenate([np.ones(50), np.zeros(50)])  # A is fooled by junk 0..49
    junk_b = np.concatenate([np.zeros(50), np.ones(50)])  # B is fooled by junk 50..99
    a, b = _synthetic(np.concatenate([hi, junk_a]), np.concatenate([hi, junk_b]), labels)
    comp = evaluate_cascade(a, b, 1.0, 1.0)
    assert comp.recall == 1.0, comp.recall
    assert comp.exclusion == 1.0, comp.exclusion
    assert comp.stage1_exclusion == 0.5 and comp.stage2_exclusion_standalone == 0.5
    assert abs(comp.independent_exclusion - 0.75) < 1e-12, comp.independent_exclusion
    assert comp.decorrelation_gap > 0.2, comp.decorrelation_gap
    assert comp.stage2_load == 0.75, comp.stage2_load  # 100 positives + 50 junk out of 200

    # 3. Independence baseline is exactly reproduced when the two models' junk misses are drawn
    #    independently: 40% and 30% miss rates over 20k junk docs -> exclusion ~= 1 - 0.4*0.3.
    n_junk = 20_000
    labels = np.concatenate([np.ones(1000, dtype=int), np.zeros(n_junk, dtype=int)])
    miss_a = rng.random(n_junk) < 0.4
    miss_b = rng.random(n_junk) < 0.3
    a, b = _synthetic(
        np.concatenate([np.ones(1000), miss_a.astype(float)]),
        np.concatenate([np.ones(1000), miss_b.astype(float)]),
        labels,
    )
    ind = evaluate_cascade(a, b, 1.0, 1.0)
    assert abs(ind.decorrelation_gap) < 0.01, ind.decorrelation_gap
    assert abs(ind.exclusion - ind.independent_exclusion) < 0.01

    # 4. best_f1 on a separable problem is 1.0; threshold_for_recall never undershoots the target.
    labels = np.array([1] * 50 + [0] * 50)
    sep = ScoreSet("sep", np.concatenate([np.full(50, 0.9), np.full(50, 0.1)]), labels)
    f1, thr, precision, recall = best_f1(sep)
    assert f1 == 1.0 and precision == 1.0 and recall == 1.0, (f1, thr, precision, recall)
    noisy = ScoreSet("noisy", rng.random(100), labels)
    for target in (0.99, 0.95, 0.9):
        t = threshold_for_recall(noisy.probs, noisy.positive, target)
        got, _ = operating_point(noisy, t)
        assert got >= target - 1e-12, (target, got)

    print(
        "selftest: OK (identical models gain nothing; complementary models gain everything; "
        "independent errors reproduce the baseline; F1/recall helpers exact)"
    )


def _parse_named(spec: str) -> ScoreSet:
    if "=" not in spec:
        raise ValueError(f"expected name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return load_scores(name, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", help="name=path JSON for the first (cheapest) filter.")
    ap.add_argument("--stage2", help="name=path JSON for the candidate second filter.")
    ap.add_argument("--selftest", action="store_true", help="Run the synthetic cascade-math checks and exit.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.stage1 or not args.stage2:
        ap.error("--stage1 and --stage2 are required unless --selftest")

    stage1 = _parse_named(args.stage1)
    stage2 = _parse_named(args.stage2)
    check_alignment([stage1, stage2])
    report(stage1, stage2)


if __name__ == "__main__":
    main()
