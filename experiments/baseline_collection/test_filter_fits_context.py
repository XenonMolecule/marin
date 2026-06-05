# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the keep-iff-full-fit context filter."""

import pytest
from transformers import AutoTokenizer

from experiments.baseline_collection.build_hq_distill_chat import chat_row
from experiments.baseline_collection.filter_fits_context import fits_context
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_short_example_fits(tok):
    msgs = chat_row("<body><p>hi there</p></body>", "hi there")["messages"]
    assert fits_context(msgs, tok, 4096, QWEN_3_CHAT_TEMPLATE)


def test_abstention_fits(tok):
    # The 22-token abstention target must survive (the whole point).
    msgs = chat_row("<body><p>junk</p></body>", "[NO_USEFUL_CONTENT]")["messages"]
    assert fits_context(msgs, tok, 4096, QWEN_3_CHAT_TEMPLATE)


def test_over_length_rejected(tok):
    # Scaffold alone exceeds a tiny window -> dropped, not truncated.
    msgs = chat_row("<body><p>hi</p></body>", "hi")["messages"]
    assert not fits_context(msgs, tok, 30, QWEN_3_CHAT_TEMPLATE)
