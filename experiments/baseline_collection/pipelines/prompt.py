# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt template loading and formatting for litellm inference."""

from pathlib import Path


class PromptFormatter:
    """Loads templates from disk and formats messages for litellm.completion()."""

    def __init__(self, template_dir: str = "static/training_data/prompt"):
        template_path = Path(template_dir)
        self._system = (template_path / "system.txt").read_text(encoding="utf-8")
        self._user = (template_path / "user.txt").read_text(encoding="utf-8")

    def format(self, html: str, spec: str) -> list[dict]:
        """Build messages list for litellm.completion()."""
        user = self._user.replace("{{html}}", html).replace("{{extraction_spec}}", spec)
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user},
        ]

    def format_user_content(self, html: str, spec: str) -> str:
        """Return just the user content string (for storing in parquet)."""
        return self._user.replace("{{html}}", html).replace("{{extraction_spec}}", spec)
