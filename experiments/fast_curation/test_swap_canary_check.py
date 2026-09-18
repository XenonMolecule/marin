# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.baseline_collection.decode_warcs_clean import _warc_path_hash
from experiments.fast_curation.swap_canary_check import arm_c_shard_index, compare


def test_compare_classifies_equal_decoder_only_and_other_differences():
    b = [
        {"url": "u1", "text": "same"},
        {"url": "u2", "text": "b-version"},
        {"url": "u3", "text": "only in b"},
        {"url": "u4", "text": "dup second"},
        {"url": "u5", "text": "caf\u00e9 au lait"},
        {"url": "u6", "text": "caf\u00e9"},
    ]
    c = [
        {"url": "u1", "text": "same"},
        {"url": "u2", "text": "c-version"},
        {"url": "u4", "text": "dup first"},
        {"url": "u4", "text": "dup second"},
        {"url": "u5", "text": "caf\ufffd au lait"},  # decoder-only: same length, U+FFFD in C
        {"url": "u6", "text": "caf\ufffd\ufffd"},  # length differs: not explained
    ]
    report = compare(b, c)
    assert (report.n_b, report.n_url_matched, report.n_url_unmatched) == (6, 5, 1)
    assert (report.n_text_equal, report.n_differs_decoder, report.n_differs_other) == (2, 1, 2)
    assert [e["url"] for e in report.examples] == ["u2", "u6"]


def test_arm_c_shard_index_is_rank_among_sorted_hashes():
    paths = ["s3://commoncrawl/a.warc.gz", "s3://commoncrawl/b.warc.gz", "s3://commoncrawl/c.warc.gz"]
    ranks = {p: arm_c_shard_index(p, paths) for p in paths}
    assert sorted(ranks.values()) == [0, 1, 2]
    assert [p for p, _ in sorted(ranks.items(), key=lambda kv: kv[1])] == sorted(paths, key=_warc_path_hash)
