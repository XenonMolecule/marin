# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-knowledge null gap for the cultural suite, per task, per benchmark and overall.

The cultural benchmarks are scored as gold-vs-distractor likelihoods, and on this sweep
the raw soft metrics are dominated by general language-modeling quality (BLEnD gold-bpb
is ~90% one quality axis; MMLU's choice_logprob curve is 92% LM). The honest knowledge
signal is `observed - null`, where the null keeps the model's own per-question choice
distribution but draws gold from the gold marginal — the `analyze_mmlu_signal` method,
reused verbatim (`extract_samples`, `signal`). Chance differs by benchmark (MC-4 vs
2-way), which the null absorbs by construction, so gaps are comparable across benchmarks.

Groups emitted per run: every task (`cultural_sl_verb_<task>`) and every benchmark
(`blend`, `culturalbench`, `fork`, `normad`; blend pools its 16 country tasks). There is
no `all` row: `signal` stacks a group's per-question arrays, so 2-way and 4-way tasks
cannot share a group. The suite-level view is the mean of the four benchmark gaps.

Usage (in-cloud; results.json log samples for every document):
    iris --cluster marin job run --region us-central1 --cpu 8 --memory 32GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.cultural.analyze_cultural_signal \\
               --out gs://marin-us-central1/metadata/cultural_sl_verb_5shot_signal.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.cultural.consolidate_cultural_results import RESULTS_PREFIX
from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import SPEC_BY_NAME
from experiments.scaling_law_sweeps.mmlu.analyze_mmlu_signal import (
    BUCKETS,
    FIELDS,
    RUN_RE,
    extract_samples,
    signal,
)

logger = logging.getLogger(__name__)

_TASK_KEY = re.compile(r"^(?P<task>cultural_sl_verb_.+)_\d+shot$")


def task_name_of(task_key: str) -> str | None:
    """`cultural_sl_verb_<task>_<N>shot` -> task name; None for the group row or foreign keys."""
    m = _TASK_KEY.match(task_key)
    if m is None or m["task"] not in SPEC_BY_NAME:
        return None
    return m["task"]


def benchmark_of(task_key: str) -> str | None:
    name = task_name_of(task_key)
    return SPEC_BY_NAME[name].benchmark if name else None


def _one(path: str, shots: int, seed: int) -> list[dict]:
    stem = path.split(f"/{shots}shot/")[1].split("/")[0]
    m = RUN_RE.match(stem)
    if m is None:
        logger.warning("unparsed: %s", stem)
        return []
    fs = marin_filesystem("gcs")
    uri = path if path.startswith("gs://") else f"gs://{path}"
    with fs.open(uri, "r") as fh:
        doc = json.load(fh)
    by_task = extract_samples(doc, task_name_of, all_group=None)
    by_benchmark = extract_samples(doc, benchmark_of, all_group=None)
    del doc
    groups = {**by_task, **by_benchmark}
    if not groups:
        logger.warning("no usable samples: %s", stem)
        return []
    rows = []
    for group, (logps, cpns, golds) in groups.items():
        row = dict(
            run_stem=stem,
            method=m.group("method"),
            hidden_dim=int(m.group("hidden")),
            budget=m.group("budget"),
            group=group,
        )
        row.update(signal(logps, cpns, golds, seed))
        rows.append(row)
    logger.info("%s: %d groups", stem, len(rows))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", required=True, help="Small CSV to write.")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent results.json reads.")
    args = ap.parse_args()

    fs = marin_filesystem("gcs")
    paths = []
    for b in BUCKETS:
        paths += list(fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"))
    logger.info("found %d results.json", len(paths))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, p, args.shots, i): p for i, p in enumerate(paths)}
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception as e:
                logger.error("FAILED %s: %s", futs[f], e)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(
        sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], float(r["budget"].replace("e+", "e")), r["group"]))
    )
    with fs.open(args.out, "w") as fh:
        fh.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
