# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the high_quality distillation chat builder.

Covers the format-fidelity bits that are easy to get subtly wrong (empty-think
prefix, DSPy output markers, body-strip, brace-safety of ``str.format``) and the
1:1 reservoir balance + parquet read filter.
"""

import random

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.baseline_collection.build_hq_distill_chat import (
    EMPTY_THINK,
    SYSTEM_MESSAGE,
    assistant_content,
    chat_row,
    read_pairs,
    reservoir_sample,
    stratified_caps,
)

_HTML = "<html><head><title>PageTitle</title></head><body><p>Hi there</p><script>junk()</script></body></html>"


def test_assistant_content_exact_bytes():
    # enable_thinking=False must reproduce exactly: empty think block, then DSPy
    # output fields. Whitespace is load-bearing.
    assert (
        assistant_content("EXTRACTED") == "<think>\n\n</think>\n\n[[ ## text ## ]]\nEXTRACTED\n\n[[ ## completed ## ]]"
    )


def test_chat_row_roles_and_system():
    row = chat_row(_HTML, "Hi there")
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][0]["content"] == SYSTEM_MESSAGE


def test_chat_row_user_is_body_stripped_dspy():
    user = chat_row(_HTML, "Hi there")["messages"][1]["content"]
    # DSPy scaffold present
    assert "[[ ## html ## ]]" in user
    assert "[[ ## extraction_spec ## ]]" in user
    assert "Respond with the corresponding output fields" in user
    # body kept, script + head dropped
    assert "<p>Hi there</p>" in user
    assert "junk()" not in user
    assert "<title>" not in user
    assert "PageTitle" not in user


def test_chat_row_assistant_useful():
    asst = chat_row(_HTML, "Hi there")["messages"][2]["content"]
    assert asst.startswith(EMPTY_THINK + "[[ ## text ## ]]\n")
    assert asst.endswith("[[ ## completed ## ]]")
    assert "Hi there" in asst


def test_chat_row_assistant_abstention():
    asst = chat_row(_HTML, "[NO_USEFUL_CONTENT]")["messages"][2]["content"]
    assert "[[ ## text ## ]]\n[NO_USEFUL_CONTENT]\n\n[[ ## completed ## ]]" in asst


def test_brace_safety():
    # Braces in the HTML value must be inserted literally (str.format only parses
    # the template, not the substituted value).
    html = "<body>literal {curly} and {{double}} braces</body>"
    user = chat_row(html, "x")["messages"][1]["content"]
    assert "literal {curly} and {{double}} braces" in user
    # The spec's own spinner example survives the template's double-brace escaping.
    assert "{word1|word2|word3}" in user


def test_reservoir_sample_balance_and_determinism():
    stream = [(f"h{i}", f"o{i}") for i in range(100)]
    a = reservoir_sample(iter(stream), 10, random.Random(0))
    b = reservoir_sample(iter(stream), 10, random.Random(0))
    assert len(a) == 10
    assert a == b  # same seed -> identical sample
    assert all(x in stream for x in a)
    # fewer items than k -> return all
    short = reservoir_sample(iter(stream[:5]), 10, random.Random(0))
    assert len(short) == 5


def test_stratified_caps_balances_across_snapshots():
    # Snapshot A has 1 WARC, snapshot B has 4. A naive per-WARC cap would give B
    # 4x the data; stratified_caps must equalize the two snapshots' totals.
    snapshots = ["A", "B", "B", "B", "B"]
    train_indices = [0, 1, 2, 3, 4]
    caps = stratified_caps(train_indices, snapshots, target_total=400)
    per_snapshot = 400 // 2  # two snapshots -> 200 each
    assert caps[0] == per_snapshot  # A's single WARC carries the whole quota
    assert all(caps[i] == per_snapshot // 4 for i in (1, 2, 3, 4))
    assert caps[0] == sum(caps[i] for i in (1, 2, 3, 4))  # A total == B total


def test_stratified_caps_floor_is_one():
    # Tiny budget must never produce a 0 cap (which would drop a snapshot entirely).
    caps = stratified_caps([0, 1, 2], ["A", "B", "C"], target_total=1)
    assert all(v >= 1 for v in caps.values())


def test_read_pairs_drops_empty(tmp_path):
    path = str(tmp_path / "shard.parquet")
    table = pa.table(
        {
            "raw_html": ["<html>a</html>", "", "<html>b</html>", "<html>c</html>"],
            "final_output": ["out1", "out2", "", "[NO_USEFUL_CONTENT]"],
        }
    )
    pq.write_table(table, path)
    pairs = list(read_pairs(path))
    assert pairs == [("<html>a</html>", "out1"), ("<html>c</html>", "[NO_USEFUL_CONTENT]")]
