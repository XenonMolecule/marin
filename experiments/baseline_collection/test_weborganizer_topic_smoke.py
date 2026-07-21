# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WebOrganizer topic-smoke helpers.

The reservoir is the piece worth testing: it is the only non-obvious logic here, and it is the piece
that carries unchanged from the 500-doc smoke to a 20M-doc production pass. If its draw is biased or
its memory unbounded, both the viewer's per-category samples and the production run are wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.baseline_collection.weborganizer_topic_smoke import Doc, LabelReservoir, _softmax

LABELS = ["A", "B", "C"]
CONFIDENT_A = np.array([[0.9, 0.05, 0.05]])  # argmax=A, p=0.9
UNSURE_A = np.array([[0.4, 0.35, 0.25]])  # argmax=A, p=0.4


def test_softmax_normalizes_and_preserves_argmax():
    probs = _softmax(np.array([[1.0, 2.0, 3.0], [5.0, 0.0, -1.0]]))
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-6)
    assert probs.argmax(axis=-1).tolist() == [2, 0]


def test_min_prob_floor_excludes_low_confidence_docs():
    reservoir = LabelReservoir(LABELS, per_label=5, min_prob=0.5)
    reservoir.update([Doc("below", "t")], UNSURE_A)
    reservoir.update([Doc("above", "t")], CONFIDENT_A)

    assert reservoir.seen[0] == 2, "seen counts every doc, floor or not"
    assert reservoir.eligible[0] == 1, "only the confident doc clears the floor"
    assert [row["url"] for row in reservoir.kept[0]] == ["above"]


def test_label_starved_by_floor_is_visible_not_silent():
    """A label predicted but never confidently: seen > 0, eligible == 0, no examples. Must be legible."""
    reservoir = LabelReservoir(LABELS, per_label=5, min_prob=0.5)
    for _ in range(10):
        reservoir.update([Doc("u", "t")], UNSURE_A)

    assert reservoir.seen[0] == 10
    assert reservoir.eligible[0] == 0
    assert reservoir.kept[0] == []
    assert reservoir.rows("ds") == []


def test_reservoir_is_bounded_regardless_of_stream_length():
    reservoir = LabelReservoir(LABELS, per_label=10, min_prob=0.5, seed=0)
    for _ in range(50_000):
        reservoir.update([Doc("u", "t")], CONFIDENT_A)

    assert len(reservoir.kept[0]) == 10, "memory must not grow with the stream"
    assert reservoir.seen[0] == 50_000
    assert reservoir.eligible[0] == 50_000


def test_reservoir_draw_is_uniform_over_the_eligible_stream():
    """Algorithm R: every position must be kept with probability k/n — no recency or head bias.

    This is the property that makes the per-category samples trustworthy. A naive "keep the first k"
    would pass every other test in this file and fail this one.
    """
    n, k, trials = 200, 10, 4000
    hits = np.zeros(n)
    for trial in range(trials):
        reservoir = LabelReservoir(LABELS, per_label=k, min_prob=0.5, seed=trial)
        for i in range(n):
            reservoir.update([Doc(str(i), "t")], CONFIDENT_A)
        for row in reservoir.kept[0]:
            hits[int(row["url"])] += 1

    freq = hits / trials
    expected = k / n
    assert abs(freq.mean() - expected) < 0.005
    # Binomial sd per position is ~sqrt(p(1-p)/trials) ~= 0.0034; +-25% of p is >6 sd, so a real
    # positional bias (e.g. keep-first-k => freq 1.0 then 0.0) fails loudly while noise passes.
    assert freq.min() > expected * 0.75, f"position starved: {freq.min():.4f} vs {expected:.4f}"
    assert freq.max() < expected * 1.25, f"position favoured: {freq.max():.4f} vs {expected:.4f}"


def test_rows_carry_distribution_context_for_the_viewer():
    reservoir = LabelReservoir(LABELS, per_label=2, min_prob=0.5, seed=0)
    for _ in range(7):
        reservoir.update([Doc("u", "body")], CONFIDENT_A)

    rows = reservoir.rows("dclm_10k")
    assert len(rows) == 2
    assert all(row["dataset"] == "dclm_10k" and row["label"] == "A" for row in rows)
    # n_seen/n_eligible are per-label totals, not per-row — the viewer needs them to show how big the
    # category was relative to the handful of docs it is displaying.
    assert all(row["n_seen"] == 7 and row["n_eligible"] == 7 for row in rows)


@pytest.mark.parametrize("per_label", [1, 3, 25])
def test_reservoir_never_exceeds_capacity_or_invents_docs(per_label):
    n = 10
    reservoir = LabelReservoir(LABELS, per_label=per_label, min_prob=0.5, seed=1)
    for i in range(n):
        reservoir.update([Doc(str(i), "t")], CONFIDENT_A)

    kept = reservoir.kept[0]
    assert len(kept) == min(per_label, n)
    urls = [row["url"] for row in kept]
    assert len(set(urls)) == len(urls), "no duplicates"
    assert set(urls) <= {str(i) for i in range(n)}, "no invented docs"
