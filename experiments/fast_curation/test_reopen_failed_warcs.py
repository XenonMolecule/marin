# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.fast_curation.reopen_failed_warcs import FailedWarc, failed_rows


def test_failed_rows_selects_only_warcs_without_input():
    rows = [
        {"warc_hash": "aa", "a_n_in": 120, "n_kept": 40},
        {"warc_hash": "bb", "a_n_in": 0, "n_kept": 0},
        {"warc_hash": "cc", "a_n_in": 3, "n_kept": 0},
    ]
    assert failed_rows(7, rows) == [FailedWarc(7, "bb")]


def test_failed_rows_rejects_kept_docs_without_input():
    with pytest.raises(ValueError, match="n_kept == 5"):
        failed_rows(7, [{"warc_hash": "dd", "a_n_in": 0, "n_kept": 5}])
