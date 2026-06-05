# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure logic in fasttext_useful_classifier: HTML→fastText
preprocessing, WARC-disjoint split assignment, and the threshold sweep."""

from __future__ import annotations

from experiments.baseline_collection.fasttext_useful_classifier import (
    LABEL_NO_USEFUL,
    LABEL_USEFUL,
    _precision_recall_sweep,
    _select_train_indices,
    body_strip,
    collect_leaderboard_rows,
    evaluate_thresholds,
    fasttext_line,
    held_out_indices,
    to_fasttext_text,
)


def test_precision_recall_sweep_recall_is_ratio_invariant_precision_is_not():
    # Same 2 useful (p=0.9, 0.8); compare a balanced vs negative-heavy test.
    pos = [(0.9, True), (0.8, True)]
    balanced = pos + [(0.7, False), (0.1, False)]  # 1 FP above 0.5
    heavy = pos + [(0.7, False)] * 10 + [(0.1, False)] * 10  # 10 FPs above 0.5
    rb = _precision_recall_sweep(balanced, [0.5])[0]
    rh = _precision_recall_sweep(heavy, [0.5])[0]
    assert rb["recall"] == rh["recall"] == 1.0  # recall invariant to negative count
    assert rb["precision"] == round(2 / 3, 4)  # 2 TP / (2 TP + 1 FP)
    assert rh["precision"] == round(2 / 12, 4)  # 2 TP / (2 TP + 10 FP) -> precision collapses
    assert rh["precision"] < rb["precision"]


def test_body_strip_keeps_body_drops_script_and_head():
    html = "<html><head><title>T</title></head><body><p>hello</p><script>var x=1;</script></body></html>"
    out = body_strip(html)
    assert "hello" in out
    assert "var x=1" not in out  # script removed
    assert "<title>" not in out  # head dropped (outside body)


def test_body_strip_falls_back_when_no_body():
    html = "<div>no body tag here</div><script>junk()</script>"
    out = body_strip(html)
    assert "no body tag here" in out
    assert "junk()" not in out


def test_body_strip_joins_multiple_bodies():
    html = "<body>one</body><body>two</body>"
    assert body_strip(html) == "onetwo"


def test_to_fasttext_text_collapses_whitespace_and_lowercases():
    html = "<body>\n\tHello   WORLD\n\n</body>"
    out = to_fasttext_text(html, "body_strip")
    assert "\n" not in out and "\t" not in out
    assert "  " not in out  # no double spaces
    assert out == "hello world"


def test_to_fasttext_text_raw_full_keeps_markup_lowercased_single_line():
    html = "<HTML>\n<BODY>Hi</BODY>\n</HTML>"
    out = to_fasttext_text(html, "raw_full")
    assert "\n" not in out
    assert out == "<html> <body>hi</body> </html>"


def test_fasttext_line_has_label_prefix_and_no_newline():
    line = fasttext_line(LABEL_USEFUL, "<body>Some Text</body>", "body_strip")
    assert line.startswith(LABEL_USEFUL + " ")
    assert "\n" not in line
    assert line == f"{LABEL_USEFUL} some text"


def test_resiliparse_representation_routes_and_collapses(monkeypatch):
    # resiliparse_text needs the lib; stub it to test routing + whitespace handling.
    monkeypatch.setattr(
        "experiments.baseline_collection.fasttext_useful_classifier.resiliparse_text",
        lambda html: "Main\n  Content  Here",
    )
    out = to_fasttext_text("<html>...</html>", "resiliparse")
    assert out == "main content here"


def test_held_out_indices_snapshot_stratified_and_disjoint():
    # 3 snapshots: A=idx{0,1,2}, B=idx{3,4}, C=idx{5,6,7,8}. k=1 -> val first, test second.
    snapshots = ["A", "A", "A", "B", "B", "C", "C", "C", "C"]
    val, test = held_out_indices(snapshots, k_per_snapshot=1)
    # val = lowest index per snapshot; test = second-lowest per snapshot.
    assert val == {0, 3, 5}
    assert test == {1, 4, 6}
    assert val.isdisjoint(test)
    # Every snapshot is represented in BOTH val and test.
    assert {snapshots[i] for i in val} == {"A", "B", "C"}
    assert {snapshots[i] for i in test} == {"A", "B", "C"}


def test_held_out_indices_skips_snapshot_without_enough_shards():
    # Snapshot "B" has only 1 shard -> it can fill val but not test at k=1.
    snapshots = ["A", "A", "B"]
    val, test = held_out_indices(snapshots, k_per_snapshot=1)
    assert val == {0, 2}  # A:0, B:2
    assert test == {1}  # A:1 ; B has no second shard
    assert val.isdisjoint(test)


# Snapshots clustered by index block (A=0-9, B=10-19, C=20-29, D=30-39) — the real
# corpus has this index↔snapshot correlation, which is what biases front-first sampling.
_BLOCKED = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10


def test_select_train_front_is_snapshot_biased():
    chosen = _select_train_indices(list(range(40)), _BLOCKED, "front", 10, seed=0)
    assert chosen == list(range(10))  # the index prefix
    assert {_BLOCKED[i] for i in chosen} == {"A"}  # only the earliest snapshot — biased


def test_select_train_stratified_covers_all_snapshots():
    chosen = _select_train_indices(list(range(40)), _BLOCKED, "stratified", 8, seed=0)
    assert len(chosen) == 8
    assert {_BLOCKED[i] for i in chosen} == {"A", "B", "C", "D"}  # 2 per snapshot
    assert chosen == sorted(chosen)


def test_select_train_random_spans_snapshots_and_is_seed_deterministic():
    a = _select_train_indices(list(range(40)), _BLOCKED, "random", 12, seed=7)
    b = _select_train_indices(list(range(40)), _BLOCKED, "random", 12, seed=7)
    assert a == b  # deterministic for a fixed seed
    assert len(a) == 12 and a == sorted(a)
    assert len({_BLOCKED[i] for i in a}) >= 3  # spread across most snapshots, unlike front


def test_select_train_respects_held_out_candidates_only():
    candidates = [i for i in range(40) if i not in {0, 1, 2}]  # 0-2 held out
    chosen = _select_train_indices(candidates, _BLOCKED, "random", 5, seed=1)
    assert set(chosen).isdisjoint({0, 1, 2})


def test_leaderboard_merges_pilot_and_natural_and_flags_degenerate():
    pilot = {
        "m_good": {
            "config": {"representation": "body_strip", "neg_per_pos": 12.0, "fixed_config": True},
            "held_out": {"n_train_warcs": 80, "n_train_snapshots": 35, "front_baseline_n_train_snapshots": 24},
            "split_counts": {"train": {"useful": 200, "no_useful": 2400}, "test": {"useful": 50, "no_useful": 600}},
            "best_f1_operating_point": {"f1": 0.88, "threshold": 0.2, "precision": 0.9, "recall": 0.86},
        }
    }
    natural = {
        "m_good": {
            "neg_to_pos_ratio": 11.8,
            "best_f1_operating_point": {"f1": 0.55, "threshold": 0.6, "precision": 0.5, "recall": 0.61},
            "threshold_sweep": [
                {"threshold": 0.6, "precision": 0.5, "recall": 0.61, "f1": 0.55},
                {"threshold": 0.1, "precision": 0.3, "recall": 0.92, "f1": 0.45},
            ],
        },
        "m_broken": {  # degenerate: best F1 at threshold 0 with recall 1.0
            "neg_to_pos_ratio": 11.8,
            "best_f1_operating_point": {"f1": 0.14, "threshold": 0.0, "precision": 0.08, "recall": 1.0},
            "threshold_sweep": [{"threshold": 0.0, "precision": 0.08, "recall": 1.0, "f1": 0.14}],
        },
    }
    rows = {r["model"]: r for r in collect_leaderboard_rows(pilot, natural)}
    good = rows["m_good"]
    assert good["sampling"] == "front"  # default when config lacks train_sample
    assert good["recipe"] == "fixed"
    assert good["natural_best_f1"] == 0.55
    assert good["natural_precision_at_recall_0_90"] == 0.3  # the only threshold reaching R>=0.90
    assert good["natural_suspect"] is False
    assert rows["m_broken"]["natural_suspect"] is True  # flagged, not trusted


class _FakeInner:
    """Mimics fasttext model.f.predict(text, k, threshold, mode) -> [(prob, label), ...]."""

    def predict(self, text, k, threshold, mode):
        p_useful = float(text)
        return [(p_useful, LABEL_USEFUL), (1 - p_useful, LABEL_NO_USEFUL)]


class _FakeModel:
    """Returns a fixed P(useful) per text (text == the prob as a string)."""

    def __init__(self):
        self.f = _FakeInner()


def test_evaluate_thresholds_precision_recall(tmp_path):
    # 2 useful (p=0.9, 0.4), 2 no_useful (p=0.6, 0.1). Encode P(useful) as the text.
    lines = [
        f"{LABEL_USEFUL} 0.9",
        f"{LABEL_USEFUL} 0.4",
        f"{LABEL_NO_USEFUL} 0.6",
        f"{LABEL_NO_USEFUL} 0.1",
    ]
    p = tmp_path / "test.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = evaluate_thresholds(_FakeModel(), str(p), [0.5])
    r = rows[0]
    # At t=0.5: predicted useful = {0.9 (useful TP), 0.6 (no_useful FP)}. 0.4 useful -> FN.
    # precision = 1/2, recall = 1/2, f1 = 0.5
    assert r["precision"] == 0.5
    assert r["recall"] == 0.5
    assert r["f1"] == 0.5


def test_evaluate_thresholds_low_threshold_maximizes_recall(tmp_path):
    lines = [f"{LABEL_USEFUL} 0.3", f"{LABEL_USEFUL} 0.8", f"{LABEL_NO_USEFUL} 0.05"]
    p = tmp_path / "test.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = evaluate_thresholds(_FakeModel(), str(p), [0.1])
    # t=0.1: both useful (0.3,0.8) predicted useful -> recall 1.0; no_useful 0.05 below -> no FP.
    assert rows[0]["recall"] == 1.0
    assert rows[0]["precision"] == 1.0
