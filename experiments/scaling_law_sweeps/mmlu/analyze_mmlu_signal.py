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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rigging.filesystem import filesystem as marin_filesystem

logger = logging.getLogger(__name__)

NAT_TO_BIT = 1.4426950408889634  # 1/ln(2); matches lm_eval.api.task

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


def _extract(doc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-question (log_softmax lls, choice_prob_norm dist, gold index) for one run."""
    samples = doc.get("samples")
    if not samples:
        return None
    logps: list[np.ndarray] = []
    cpns: list[np.ndarray] = []
    golds: list[int] = []
    letters = "ABCD"
    for _task, recs in samples.items():
        for r in recs:
            resp = r.get("filtered_resps")
            if not resp or len(resp) != 4:
                continue
            lls = np.array([x[0] for x in resp], dtype=np.float64)
            logps.append(lls - np.logaddexp.reduce(lls))
            # choice_prob_norm's distribution, rebuilt exactly as lm_eval does: softmax
            # over NEGATIVE per-byte bits, with byte lengths of the FULL "A. <text>" string.
            choices = [f"{letters[i]}. {c}" for i, c in enumerate(r["doc"]["choices"])]
            blen = np.array([max(1, len(c.encode("utf-8"))) for c in choices], dtype=np.float64)
            w = np.exp(-((-lls / blen) * NAT_TO_BIT))
            cpns.append(w / max(w.sum(), 1e-8))
            golds.append(int(r["target"]))
    if not golds:
        return None
    return np.array(logps), np.array(cpns), np.array(golds)


def _signal(logps: np.ndarray, cpns: np.ndarray, golds: np.ndarray, seed: int) -> dict:
    """Observed vs zero-knowledge null for acc and choice_logprob."""
    rng = np.random.default_rng(seed)
    n = len(golds)
    idx = np.arange(n)
    picks = logps.argmax(axis=1)

    pick_marg = np.bincount(picks, minlength=4) / n
    gold_marg = np.bincount(golds, minlength=4) / n

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
        pick_A=pick_marg[0],
        pick_B=pick_marg[1],
        pick_C=pick_marg[2],
        pick_D=pick_marg[3],
    )


def _one(path: str, shots: int, seed: int) -> dict | None:
    stem = path.split(f"/{shots}shot/")[1].split("/")[0]
    m = RUN_RE.match(stem)
    if m is None:
        logger.warning("unparsed: %s", stem)
        return None
    fs = marin_filesystem("gcs")
    uri = path if path.startswith("gs://") else f"gs://{path}"
    with fs.open(uri, "r") as fh:
        doc = json.load(fh)
    got = _extract(doc)
    del doc
    if got is None:
        logger.warning("no usable samples: %s", stem)
        return None
    logps, cpns, golds = got
    row = dict(run_stem=stem, method=m.group("method"), hidden_dim=int(m.group("hidden")), budget=m.group("budget"))
    row.update(_signal(logps, cpns, golds, seed))
    logger.info(
        "%s: acc %.4f vs null %.4f (%+.4f, %+.1f sd) | clp %.4f vs null %.4f (%+.4f, %+.1f sd)",
        stem,
        row["obs_acc"],
        row["null_acc"],
        row["acc_gap"],
        row["acc_sigma"],
        row["obs_clp"],
        row["null_clp"],
        row["clp_gap"],
        row["clp_sigma"],
    )
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", required=True, help="Small CSV to write.")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent results.json (each ~226MB).")
    args = ap.parse_args()

    fs = marin_filesystem("gcs")
    paths = []
    for b in BUCKETS:
        paths += list(fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"))
    logger.info("found %d results.json", len(paths))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, p, args.shots, i): p for i, p in enumerate(paths)}
        for f in as_completed(futs):
            try:
                if (r := f.result()) is not None:
                    rows.append(r)
            except Exception as e:
                logger.error("FAILED %s: %s", futs[f], e)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], float(r["budget"].replace("e+", "e")))))
    with fs.open(args.out, "w") as fh:
        fh.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
