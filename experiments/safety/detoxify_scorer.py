# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detoxify wrapper with batched scoring and on-disk caching.

Used as the local replacement for Perspective API in our HELM-spec RTP eval.
The `original` checkpoint replicates Perspective's attribute schema (toxicity,
severe_toxicity, obscene, threat, insult, identity_attack) since both were
trained on Jigsaw Toxic Comment Challenge data.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DetoxifyScorer:
    """Score texts with Detoxify and cache results by SHA-256 hash."""

    def __init__(self, checkpoint: str = "original", cache_dir: str | Path | None = None) -> None:
        # Local import: detoxify pulls in torch + transformers eagerly.
        from detoxify import Detoxify

        self.checkpoint = checkpoint
        self._model = Detoxify(checkpoint)
        if cache_dir is not None:
            self.cache_dir: Path | None = Path(cache_dir) / f"detoxify-{checkpoint}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.checkpoint}\0{text}".encode()).hexdigest()

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        h = self._hash(text)
        # Shard by first two chars so we never end up with one giant directory.
        return self.cache_dir / h[:2] / f"{h}.json"

    def score_batch(self, texts: list[str], batch_size: int = 64) -> list[dict[str, float]]:
        """Return a list of attribute-score dicts, one per input text.

        Empty strings are scored as all-zeros without invoking the model.
        """
        results: list[dict[str, float] | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if not text:
                results[i] = {}
                continue
            cp = self._cache_path(text)
            if cp is not None and cp.exists():
                with cp.open() as f:
                    results[i] = json.load(f)
            else:
                to_compute.append((i, text))

        if to_compute:
            logger.info(
                "Detoxify: %d/%d uncached, scoring in batches of %d",
                len(to_compute),
                len(texts),
                batch_size,
            )
            for batch_start in range(0, len(to_compute), batch_size):
                batch = to_compute[batch_start : batch_start + batch_size]
                batch_texts = [t for _, t in batch]
                batch_out: dict[str, Any] = self._model.predict(batch_texts)
                for offset, (orig_idx, text) in enumerate(batch):
                    row = {k: float(v[offset]) for k, v in batch_out.items()}
                    results[orig_idx] = row
                    cp = self._cache_path(text)
                    if cp is not None:
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        with cp.open("w") as f:
                            json.dump(row, f)

        return [r if r is not None else {} for r in results]
