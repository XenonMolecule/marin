# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import math

import msgspec

from experiments.fast_curation.dedup import _carry_provenance


def _record(prob):
    return {
        "id": "h",
        "source_id": "d1",
        "text": "t",
        "doc_id": "d1",
        "url": "u",
        "pooled_prob": 0.9,
        "modernbert_prob": prob,
    }


def test_carry_provenance_agrees_across_jsonl_roundtrip_for_nan_score():
    # TEXT-line hi-accepts store NaN; JSONL has no NaN, so reshape and apply must both emit None.
    reshaped = _carry_provenance(_record(float("nan")))
    assert reshaped["modernbert_prob"] is None
    read_back = msgspec.json.decode(msgspec.json.Encoder().encode(reshaped))
    assert _carry_provenance({**read_back, "id": "h"}) == reshaped


def test_carry_provenance_keeps_finite_scores_and_drops_datakit_ids():
    out = _carry_provenance(_record(0.25))
    assert out["modernbert_prob"] == 0.25 and not math.isnan(out["modernbert_prob"])
    assert out["pooled_prob"] == 0.9
    assert "id" not in out and "source_id" not in out
