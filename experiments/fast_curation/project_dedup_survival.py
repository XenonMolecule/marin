# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project post-dedup corpus size to a larger WARC pool from ``survival.json`` points.

Each point is one dedup tree measured by ``reapply_verified_dedup.py``: the documents and
characters that ``cc_canonical`` and ``c075`` remove from the same normalized documents. Unique
content grows sublinearly in the pool, so each rule gets a power law ``unique = a * warcs^b``
fitted in log space, which is Heaps' law for documents.

A power law is an assumption, and the largest measured pool is far below the target. The report
therefore also prints the exponent between each pair of neighbouring points. If those exponents
fall as the pool grows, the curve is bending and the fit over-projects; the projection from the
last two points alone is then the better guide, and it is printed beside the full fit so the gap
between them is the visible uncertainty.

    uv run python -m experiments.fast_curation.project_dedup_survival \\
        --point 10364=gs://.../10364warcs/stats_c075/survival.json \\
        --point 154117=gs://.../154117warcs/stats_c075/survival.json \\
        --point 500000=gs://.../500000warcs/stats_c075/survival.json \\
        --target 2000000 --target 8000000
"""

import argparse
import dataclasses
import json
import math
import sys

import fsspec
import numpy as np

# llama3 characters per token measured on the 154k and 500k deduped trees.
LLAMA3_CHARS_PER_TOKEN = 4.302
RULES = ("cc_canonical", "c075")


@dataclasses.dataclass(frozen=True)
class SurvivalPoint:
    """Unique content one dedup tree keeps under each rule."""

    warcs: int
    docs: int
    chars: int
    unique_docs: dict[str, int]
    unique_chars: dict[str, int]


@dataclasses.dataclass(frozen=True)
class PowerLaw:
    """``unique = scale * warcs ** exponent``."""

    scale: float
    exponent: float

    def at(self, warcs: float) -> float:
        return self.scale * warcs**self.exponent


def load_point(warcs: int, survival_path: str) -> SurvivalPoint:
    with fsspec.open(survival_path, "r") as handle:
        survival = json.load(handle)
    return SurvivalPoint(
        warcs=warcs,
        docs=survival["docs"],
        chars=survival["chars"],
        unique_docs={rule: survival["docs"] - survival[f"{rule}_removed_docs"] for rule in RULES},
        unique_chars={rule: survival["chars"] - survival[f"{rule}_removed_chars"] for rule in RULES},
    )


def fit_power_law(warcs: list[int], unique: list[int]) -> PowerLaw:
    """Least-squares line through ``(log warcs, log unique)``."""
    if len(warcs) < 2:
        raise ValueError("a power law needs at least two points")
    exponent, intercept = np.polyfit(np.log(warcs), np.log(unique), 1)
    return PowerLaw(scale=float(math.exp(intercept)), exponent=float(exponent))


def local_exponents(warcs: list[int], unique: list[int]) -> list[float]:
    """The power-law exponent between each pair of neighbouring points."""
    return [math.log(unique[i + 1] / unique[i]) / math.log(warcs[i + 1] / warcs[i]) for i in range(len(warcs) - 1)]


def _tokens(chars: float) -> str:
    return f"{chars / LLAMA3_CHARS_PER_TOKEN / 1e12:.2f}T"


def report(points: list[SurvivalPoint], targets: list[int]) -> str:
    points = sorted(points, key=lambda point: point.warcs)
    warcs = [point.warcs for point in points]
    lines = ["Measured (after exact dedup; tokens estimated at 4.302 llama3 chars/token)"]
    for point in points:
        lines.append(f"  {point.warcs:>9,} WARCs  {point.docs / 1e6:>8.1f}M docs  {_tokens(point.chars)} tokens")
        for rule in RULES:
            lines.append(
                f"    {rule:<13} keeps {point.unique_docs[rule] / point.docs:6.1%} of docs, "
                f"{point.unique_chars[rule] / point.chars:6.1%} of chars -> {_tokens(point.unique_chars[rule])}"
            )
    for rule in RULES:
        unique = [point.unique_chars[rule] for point in points]
        fit = fit_power_law(warcs, unique)
        tail = fit_power_law(warcs[-2:], unique[-2:])
        exponents = ", ".join(f"{value:.3f}" for value in local_exponents(warcs, unique))
        lines.append(f"\n{rule}: unique chars ~ warcs^{fit.exponent:.3f}; exponent between neighbours: {exponents}")
        for target in targets:
            lines.append(
                f"  {target:>9,} WARCs  all-point fit {_tokens(fit.at(target))}"
                f"   last-two-point fit {_tokens(tail.at(target))}"
            )
    return "\n".join(lines)


def _parse_point(text: str) -> tuple[int, str]:
    warcs, _, path = text.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected WARCS=PATH, got {text!r}")
    return int(warcs), path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", required=True, type=_parse_point, metavar="WARCS=PATH")
    parser.add_argument("--target", action="append", required=True, type=int, help="WARC pool size to project to.")
    args = parser.parse_args()
    print(report([load_point(warcs, path) for warcs, path in args.point], args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
