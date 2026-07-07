# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MoralChoice loglik metric + steerability math (no torch needed)."""

from experiments.evals.moralchoice_loglik import (
    RULE_COLUMNS,
    build_requests,
    compute_metrics,
    steer_preamble,
)


def _row(sid, a1_violates=(), a2_violates=(), action1="do the good thing", action2="do the bad thing"):
    """Build a scenario row; *_violates are rule names whose a{1,2}_<rule> column is 'Yes'."""
    row = {
        "scenario_id": sid,
        "context": "You face a choice.",
        "action1": action1,
        "action2": action2,
        "generation_rule": "Do not deceive",
    }
    for r in RULE_COLUMNS:
        row[f"a1_{r}"] = "Yes" if r in a1_violates else "No"
        row[f"a2_{r}"] = "Yes" if r in a2_violates else "No"
    return row


def _pref(orig_a1: bool, swap_a1: bool):
    """A preference entry: whether action1 is preferred in each order (sum==norm here)."""
    return {"sum": {"orig": orig_a1, "swap": swap_a1}, "norm": {"orig": orig_a1, "swap": swap_a1}}


def test_build_requests_ab_and_swap():
    row = _row("s1")
    reqs = build_requests(row, "ab", system_prompt="")
    assert len(reqs) == 4  # 2 orders x 2 choices
    orig = {r.choice_action: r for r in reqs if r.order == "orig"}
    # In orig order, action1 is option A -> continuation " A"; action2 -> " B".
    assert orig[1].continuation == " A"
    assert orig[2].continuation == " B"
    swap = {r.choice_action: r for r in reqs if r.order == "swap"}
    # In swap order, action1 is now option B.
    assert swap[1].continuation == " B"
    assert swap[2].continuation == " A"
    # action1's text appears before action2's in orig, reversed in swap.
    assert orig[1].prompt.index("good") < orig[1].prompt.index("bad")
    assert swap[1].prompt.index("bad") < swap[1].prompt.index("good")


def test_build_requests_repeat_uses_action_text():
    row = _row("s1")
    reqs = build_requests(row, "repeat", system_prompt="")
    orig = {r.choice_action: r for r in reqs if r.order == "orig"}
    assert orig[1].continuation == " do the good thing"
    assert orig[2].continuation == " do the bad thing"


def test_steer_preamble_prepended():
    row = _row("s1")
    pre = steer_preamble("deceive")
    assert "truthfulness" in pre
    reqs = build_requests(row, "ab", system_prompt=pre)
    assert all(r.prompt.startswith(pre) for r in reqs)


def test_accuracy_perfect_and_worst():
    rows = [_row("s1"), _row("s2")]
    # Always prefer action1 (gold) in both orders -> acc 1.0, consistent, no position flip.
    prefs = {"s1": _pref(True, True), "s2": _pref(True, True)}
    m = compute_metrics(rows, prefs)
    assert m["acc"] == 1.0 and m["acc_norm"] == 1.0
    assert m["consistency"] == 1.0
    # Always prefer action2 -> acc 0.0.
    prefs0 = {"s1": _pref(False, False), "s2": _pref(False, False)}
    assert compute_metrics(rows, prefs0)["acc"] == 0.0


def test_position_bias_always_first():
    # Model always picks the FIRST-listed option: orig->action1 (True), swap->action2 (False).
    rows = [_row("s1")]
    prefs = {"s1": _pref(True, False)}
    m = compute_metrics(rows, prefs)
    assert m["first_position_bias"] == 1.0
    assert m["consistency"] == 0.0  # flips with order
    assert m["acc"] == 0.5  # averaged over orders


def test_compliance_only_counts_discriminating_scenarios():
    # s1: action2 deceives (action1 compliant). s2: neither deceives -> excluded for deceive.
    rows = [_row("s1", a2_violates=("deceive",)), _row("s2")]
    prefs = {"s1": _pref(True, True), "s2": _pref(True, True)}
    m = compute_metrics(rows, prefs)
    dec = m["compliance"]["deceive"]
    assert dec["n"] == 1  # only s1 discriminates on deceive
    assert dec["prefer_compliant"] == 1.0  # model prefers action1 = the non-deceptive action


def test_compliance_when_action1_is_the_violator():
    # action1 deceives -> compliant action is action2. Model prefers action1 -> low compliance.
    rows = [_row("s1", a1_violates=("deceive",))]
    prefs = {"s1": _pref(True, True)}  # prefers action1 (the violator)
    dec = compute_metrics(rows, prefs)["compliance"]["deceive"]
    assert dec["n"] == 1
    assert dec["prefer_compliant"] == 0.0
