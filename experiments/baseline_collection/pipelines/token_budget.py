# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token counting and view-capping for the extraction pipelines.

The pipelines use ONE tokenizer everywhere (the served model's own Qwen3
tokenizer) for both chunk sizing and the filter-view cap — never a second
tokenizer. `QwenTokenCodec` wraps an already-loaded HF tokenizer (the one the
vLLM engine loaded) so tokenization is identical to the model's and no second
tokenizer is introduced.
"""

from __future__ import annotations

from typing import Any

# The certified filter-view middle-omission marker. The exact text matters only
# for reproducibility of where a view was capped.
VIEW_OMISSION_MARKER = "\n\n[... middle of page omitted ...]\n\n"


class QwenTokenCodec:
    """Token count/encode/decode over an already-loaded HF tokenizer.

    Wrapping the engine's tokenizer (rather than loading `Qwen/Qwen3-8B` again)
    keeps a single tokenizer per process and guarantees the chunk/view token
    boundaries match what the model actually sees.
    """

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))


def load_qwen_tokenizer(model_name: str = "Qwen/Qwen3-8B") -> Any:
    """Load the Qwen3 fast tokenizer standalone (tests / non-vLLM callers).

    `model_max_length` is set huge to silence length warnings; the codec never
    truncates via the tokenizer, only via explicit token budgets.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 10**9
    return tok


def cap_view(text: str, codec: QwenTokenCodec, max_tokens: int) -> str:
    """Cap a whole-document text view at `max_tokens`, keeping head + tail.

    Mirrors the reference filter-view cap: keep the first `max_tokens // 2`
    tokens, then `VIEW_OMISSION_MARKER`, then the last `max_tokens // 2` tokens.
    A view already within budget is returned unchanged.
    """
    ids = codec.encode(text)
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return codec.decode(ids[:half]) + VIEW_OMISSION_MARKER + codec.decode(ids[-half:])
