# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate real MMLU knowledge from language-modeling improvement, per run.

THE QUESTION THIS ANSWERS. ``choice_logprob`` rises steeply with compute (-4.9 -> -1.5
across the 10k sweep) while ``acc`` stays near 0.25. Is that rise knowledge, or just the
model getting better at language modeling / less erratic over the 4 candidate strings?

THE METHOD. For each run we compute every metric twice:

  * OBSERVED — the real value, using the real gold answers.
  * NULL — the value the SAME model, with its OWN per-question probability distribution
    left completely untouched, would score if it had ZERO knowledge. Constructed by
    breaking only the link between a question and which of its choices is correct:
    gold is drawn from the sweep's gold marginal, independent of the question.

Because the null reuses the model's own distribution, everything that is not
question-specific knowledge — general fluency, calibration, how peaked the choice
distribution is, answer-position bias, length bias — is present IDENTICALLY in both
observed and null, and therefore cancels in the gap. What survives is only the part that
tracks WHICH answer is right for THIS question. That is the knowledge.

  gap = observed - null      (in metric units)
  sigma = gap / std(null)    (how many permutation-null SDs the gap is)

The nulls are exact expectations, not just samples (cheap and zero-variance):
  * null_acc  = sum_k P(pick=k) * P(gold=k)            — position bias x gold skew
  * null_clp  = mean_q sum_k P(gold=k) * log_softmax(lls_q)[k]
A permutation loop supplies only the SD for the sigma denominator.

Reading the output: if ``null_clp`` rises with compute just as steeply as ``obs_clp``,
then the headline choice_logprob curve is a LANGUAGE-MODELING curve — the null has no
knowledge in it by construction, so any rise it shows is not knowledge. The knowledge is
exactly ``clp_gap``, and only that.

`--by-category` adds one row per official MMLU category (the subject rows are grouped
via `mmlu_olmes.category_of`) next to the whole-suite `all` row; `group` is a CSV column.
`extract_samples` / `signal` are choice-count agnostic and are reused by
`cultural/analyze_cultural_signal.py`.

Runs IN-CLOUD: each results.json is ~226MB (lm-eval logs per-sample data for all 14042
questions), so 181 runs is ~41GB — read it in-region, emit a small CSV.

Usage::
    iris --cluster marin job run --region us-east5 --cpu 8 --memory 64GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.mmlu.analyze_mmlu_signal \\
               --out gs://marin-us-east5/metadata/mmlu_sl_verb_5shot_signal.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import category_of

logger = logging.getLogger(__name__)

NAT_TO_BIT = 1.4426950408889634  # 1/ln(2); matches lm_eval.api.task
LETTERS = "ABCD"

BUCKETS = ["marin-us-east5", "marin-eu-west4", "marin-us-central1", "marin-us-central2", "marin-us-east1"]
RESULTS_PREFIX = "metadata/mmlu_sl_verb_results"
N_PERM = 200
RUN_RE = re.compile(
    r"^curation-(?P<method>.+?)(?:_10k)?-expFM_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<hidden>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)$"
)

FIELDS = [
    "run_stem",
    "method",
    "hidden_dim",
    "budget",
    "group",
    "n",
    "obs_acc",
    "null_acc",
    "acc_gap",
    "acc_sigma",
    "obs_clp",
    "null_clp",
    "clp_gap",
    "clp_sigma",
    "obs_cpn",
    "null_cpn",
    "cpn_gap",
    "cpn_sigma",
    "obs_clpn",
    "null_clpn",
    "clpn_gap",
    "pick_A",
    "pick_B",
    "pick_C",
    "pick_D",
]


ALL_GROUP = "all"
MAX_CHOICES = 4  # pick_A..pick_D columns; 2-way tasks leave C/D at 0


def extract_samples(
    doc: dict,
    group_of: Callable[[str], str | None] | None = None,
    all_group: str | None = ALL_GROUP,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-group (log_softmax lls, choice_prob_norm dist, gold index) arrays for one run.

    `group_of` maps a samples task key to a group label (None = skip the task); every kept
    question also lands in `all_group` unless that is None. Choice count is inferred per
    task, so 2-way and 4-way tasks both work, but a group's arrays are stacked, so tasks of
    mixed choice counts must not share a group (pass `all_group=None` for such suites).
    """
    samples = doc.get("samples")
    if not samples:
        return {}
    buckets: dict[str, tuple[list, list, list]] = {}
    for task_key, recs in samples.items():
        labels = [all_group] if all_group is not None else []
        if group_of is not None:
            label = group_of(task_key)
            if label is None:
                continue
            labels.append(label)
        for r in recs:
            resp = r.get("filtered_resps")
            choices = r["doc"]["choices"]
            if not resp or len(resp) != len(choices):
                continue
            lls = np.array([x[0] for x in resp], dtype=np.float64)
            logp = lls - np.logaddexp.reduce(lls)
            # choice_prob_norm's distribution, rebuilt exactly as lm_eval does: softmax
            # over NEGATIVE per-byte bits, with byte lengths of the FULL "A. <text>" string.
            lettered = [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
            blen = np.array([max(1, len(c.encode("utf-8"))) for c in lettered], dtype=np.float64)
            w = np.exp(-((-lls / blen) * NAT_TO_BIT))
            cpn = w / max(w.sum(), 1e-8)
            for label in labels:
                logps, cpns, golds = buckets.setdefault(label, ([], [], []))
                logps.append(logp)
                cpns.append(cpn)
                golds.append(int(r["target"]))
    out = {}
    for label, (logps, cpns, golds) in buckets.items():
        if golds:
            out[label] = (np.array(logps), np.array(cpns), np.array(golds))
    return out


def signal(logps: np.ndarray, cpns: np.ndarray, golds: np.ndarray, seed: int) -> dict:
    """Observed vs zero-knowledge null for acc and choice_logprob; k = logps.shape[1]."""
    rng = np.random.default_rng(seed)
    n, k = logps.shape
    idx = np.arange(n)
    picks = logps.argmax(axis=1)

    pick_marg = np.bincount(picks, minlength=k) / n
    gold_marg = np.bincount(golds, minlength=k) / n
    pick_padded = np.zeros(MAX_CHOICES)
    pick_padded[:k] = pick_marg

    obs_acc = float((picks == golds).mean())
    null_acc = float((pick_marg * gold_marg).sum())  # exact E[acc | pick ⟂ gold]

    obs_clp = float(logps[idx, golds].mean())
    null_clp = float((logps * gold_marg).sum(axis=1).mean())  # exact E[clp | gold ⟂ question]

    # The length-normalized (per-byte) family. Its null is ~0.25 for ANY model because it
    # is a mean-of-PROB over a normalized distribution -- that is what "controls for
    # fluency" actually means, and it is the property choice_logprob lacks.
    obs_cpn = float(cpns[idx, golds].mean())
    null_cpn = float((cpns * gold_marg).sum(axis=1).mean())
    obs_clpn = float(np.log(cpns[idx, golds] + 1e-30).mean())
    null_clpn = float((np.log(cpns + 1e-30) * gold_marg).sum(axis=1).mean())

    # Permutation SDs only (the point estimates above are exact).
    acc_null_s, clp_null_s, cpn_null_s = [], [], []
    for _ in range(N_PERM):
        g = rng.permutation(golds)
        acc_null_s.append((picks == g).mean())
        clp_null_s.append(logps[idx, g].mean())
        cpn_null_s.append(cpns[idx, g].mean())
    acc_sd = float(np.std(acc_null_s)) or 1e-9
    clp_sd = float(np.std(clp_null_s)) or 1e-9
    cpn_sd = float(np.std(cpn_null_s)) or 1e-9

    return dict(
        n=n,
        obs_acc=obs_acc,
        null_acc=null_acc,
        acc_gap=obs_acc - null_acc,
        acc_sigma=(obs_acc - null_acc) / acc_sd,
        obs_clp=obs_clp,
        null_clp=null_clp,
        clp_gap=obs_clp - null_clp,
        clp_sigma=(obs_clp - null_clp) / clp_sd,
        obs_cpn=obs_cpn,
        null_cpn=null_cpn,
        cpn_gap=obs_cpn - null_cpn,
        cpn_sigma=(obs_cpn - null_cpn) / cpn_sd,
        obs_clpn=obs_clpn,
        null_clpn=null_clpn,
        clpn_gap=obs_clpn - null_clpn,
        pick_A=pick_padded[0],
        pick_B=pick_padded[1],
        pick_C=pick_padded[2],
        pick_D=pick_padded[3],
    )


def subject_category(task_key: str) -> str | None:
    """`mmlu_sl_verb_<subject>_<N>shot` -> its MMLU category; None for the group row."""
    m = re.match(r"^mmlu_sl_verb_(?P<subject>.+)_\d+shot$", task_key)
    if m is None:
        return None
    try:
        return category_of(m["subject"])
    except KeyError:
        return None


def _one(path: str, shots: int, seed: int, by_category: bool) -> list[dict]:
    stem = path.split(f"/{shots}shot/")[1].split("/")[0]
    m = RUN_RE.match(stem)
    if m is None:
        logger.warning("unparsed: %s", stem)
        return []
    fs = marin_filesystem("gcs")
    uri = path if path.startswith("gs://") else f"gs://{path}"
    with fs.open(uri, "r") as fh:
        doc = json.load(fh)
    groups = extract_samples(doc, subject_category if by_category else None)
    del doc
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
        logger.info(
            "%s [%s]: acc %.4f vs null %.4f (%+.4f, %+.1f sd) | clp %.4f vs null %.4f (%+.4f, %+.1f sd)",
            stem,
            group,
            row["obs_acc"],
            row["null_acc"],
            row["acc_gap"],
            row["acc_sigma"],
            row["obs_clp"],
            row["null_clp"],
            row["clp_gap"],
            row["clp_sigma"],
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", required=True, help="Small CSV to write.")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent results.json (each ~226MB).")
    ap.add_argument(
        "--by-category",
        action="store_true",
        help="Also emit one row per MMLU category (stem/humanities/social_sciences/other) next to `all`.",
    )
    args = ap.parse_args()

    fs = marin_filesystem("gcs")
    paths = []
    for b in BUCKETS:
        paths += list(fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"))
    logger.info("found %d results.json", len(paths))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, p, args.shots, i, args.by_category): p for i, p in enumerate(paths)}
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
