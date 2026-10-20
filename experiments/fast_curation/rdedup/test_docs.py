# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random

import dupekit
import numpy as np
from marin.datakit.normalize import generate_id

from experiments.fast_curation.rdedup import docs


def test_canonical_order_keys_sort_like_fuzzy_dups_id_norm_strings():
    rng = random.Random(0)
    ids = [rng.getrandbits(128) for _ in range(3000)]
    hi = np.array([i >> 64 for i in ids], dtype=np.uint64)
    lo = np.array([i & (2**64 - 1) for i in ids], dtype=np.uint64)
    k1, k2, k3 = docs.canonical_order_keys(hi, lo)
    by_keys = np.lexsort((k3, k2, k1)).tolist()
    # connected_components compares str(xxh3_128("source_000|<hex id>")) as strings.
    by_strings = sorted(
        range(len(ids)), key=lambda i: str(dupekit.hash_xxh3_128(f"source_000|{generate_id_from_int(ids[i])}".encode()))
    )
    assert by_keys == by_strings


def generate_id_from_int(value: int) -> str:
    return format(value, "032x")


def test_order_words_put_a_prefix_before_its_extensions():
    strings = ["12", "123", "13", "9", "1", "129", "0"]
    k1, k2, k3 = docs.order_words(strings)
    assert [strings[i] for i in np.lexsort((k3, k2, k1))] == sorted(strings)


def test_normalization_matches_datakit_rules():
    run = " " * 300
    texts = [None, "", "  \n\t ", "keep me", f"a{run}b", "keep me"]
    rows, normalized, compacted = docs.normalize_file(texts)
    assert rows.tolist() == [3, 4, 5]
    assert normalized[1] == "a" + " " * 128 + "b"
    assert compacted == 1
    assert format(docs.content_id(normalized[1]), "032x") == generate_id(normalized[1])
