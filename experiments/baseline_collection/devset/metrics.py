# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metrics for devset extraction runs.

Signal sources (all keyed on record_id):
  1. keep/drop agreement F1 vs human/judge labels — overall + per register / web domain,
     with a label_mode knob to drop the weak labels (weak_keep/weak_drop).
  2. top-quality keep rate — fraction of top_quality docs the extractor kept.
  3. gold similarity — Levenshtein similarity (plus shingle-F1 / ROUGE-L) of the
     extracted text against the 393 gold extractions.

A doc only enters keep/drop F1 if BOTH sides made a decision: labeled keep/drop
(not unsure/None) and the extractor produced keep or drop (not error / context
overflow). The excluded counts are reported so nothing disappears silently.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from experiments.baseline_collection.devset.dataset import DevsetDoc
from experiments.baseline_collection.devset.runs import Decision, ExtractionRecord, Run
from experiments.baseline_collection.devset.token_f1 import doc_prf, doc_rouge_l, levenshtein_sim

LABEL_MODES = ("all", "strong", "human")


def _label_ok(doc: DevsetDoc, label_mode: str) -> bool:
    if doc.binary_label is None:
        return False
    if label_mode == "all":
        return True
    if label_mode == "strong":
        return not doc.is_weak_label
    if label_mode == "human":
        return doc.label_source == "human"
    raise ValueError(f"unknown label_mode {label_mode!r}; choose from {LABEL_MODES}")


def _group_key(doc: DevsetDoc, group_by: str) -> str:
    if group_by == "register":
        return doc.register
    if group_by == "web_domain":
        return doc.web_domain
    raise ValueError(f"unknown group_by {group_by!r}")


def classification_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    """pairs = (label_keep, pred_keep). Reports F1 for both classes + accuracy."""
    tp = sum(1 for l, p in pairs if l and p)  # kept a keep
    fp = sum(1 for l, p in pairs if not l and p)  # kept a drop (false keep)
    fn = sum(1 for l, p in pairs if l and not p)  # dropped a keep (false drop)
    tn = sum(1 for l, p in pairs if not l and not p)

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    keep_p, keep_r, keep_f1 = prf(tp, fp, fn)
    drop_p, drop_r, drop_f1 = prf(tn, fn, fp)
    n = len(pairs)
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "keep_precision": keep_p,
        "keep_recall": keep_r,
        "keep_f1": keep_f1,
        "drop_precision": drop_p,
        "drop_recall": drop_r,
        "drop_f1": drop_f1,
        # the labels are keep-heavy (~3:1), so keep_f1 alone rewards
        # over-extraction; macro_f1 is the anti-gaming headline metric
        "macro_f1": (keep_f1 + drop_f1) / 2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def keep_drop_report(
    run: Run,
    docs: list[DevsetDoc],
    label_mode: str = "all",
    group_by: str = "register",
) -> dict:
    """Keep/drop agreement F1 overall and per group."""
    pairs_by_group: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    all_pairs: list[tuple[bool, bool]] = []
    excluded = {"unlabeled": 0, "not_run": 0, "error": 0, "context": 0}

    records = run.records()
    for doc in docs:
        rec = records.get(doc.record_id)
        if rec is None:
            excluded["not_run"] += 1
            continue
        if not _label_ok(doc, label_mode):
            excluded["unlabeled"] += 1
            continue
        pred = rec.pred_keep
        if pred is None:
            excluded["error" if rec.decision == Decision.ERROR else "context"] += 1
            continue
        pair = (doc.binary_label, pred)
        all_pairs.append(pair)
        pairs_by_group[_group_key(doc, group_by)].append(pair)

    return {
        "label_mode": label_mode,
        "group_by": group_by,
        "overall": classification_metrics(all_pairs),
        "groups": {
            g: classification_metrics(ps) for g, ps in sorted(pairs_by_group.items(), key=lambda kv: -len(kv[1]))
        },
        "excluded": excluded,
    }


def top_quality_keep_rate(run: Run, docs: list[DevsetDoc]) -> dict:
    """Accuracy at keeping the top-quality docs (all are labeled keep-ish)."""
    records = run.records()
    kept = missed = undecided = 0
    missed_ids: list[str] = []
    for doc in docs:
        if not doc.top_quality:
            continue
        rec = records.get(doc.record_id)
        pred = rec.pred_keep if rec else None
        if pred is True:
            kept += 1
        elif pred is False:
            missed += 1
            missed_ids.append(doc.record_id)
        else:
            undecided += 1
    n = kept + missed
    return {
        "n_top_quality": kept + missed + undecided,
        "kept": kept,
        "dropped": missed,
        "undecided": undecided,
        "keep_rate": kept / n if n else 0.0,
        "dropped_ids": missed_ids,
    }


def gold_similarity(
    run: Run,
    docs: list[DevsetDoc],
    ngram_n: int = 4,
) -> dict:
    """Similarity of extracted text vs gold extractions, on gold docs.

    Per doc: char-level Levenshtein similarity, shingle token-F1, ROUGE-L F1.
    Aggregates over kept docs only (`kept_*`) and over all decided gold docs with
    dropped docs scored 0 (`overall_*`) — the latter penalizes wrongly dropping a
    gold doc. Error/context docs are excluded and counted.
    """
    records = run.records()
    rows: list[dict] = []
    excluded = {"not_run": 0, "error_or_context": 0, "gold_missing": 0}
    for doc in docs:
        if not doc.gold:
            continue
        gold_text = doc.gold_text()
        if gold_text is None:
            excluded["gold_missing"] += 1
            continue
        rec = records.get(doc.record_id)
        if rec is None:
            excluded["not_run"] += 1
            continue
        pred = rec.pred_keep
        if pred is None:
            excluded["error_or_context"] += 1
            continue
        if pred:
            lev = levenshtein_sim(gold_text, rec.text)
            shingle = doc_prf(gold_text, rec.text, ngram_n=ngram_n)
            rouge = doc_rouge_l(gold_text, rec.text)
        else:
            lev, shingle, rouge = 0.0, {"f1": 0.0}, {"f1": 0.0}
        rows.append(
            {
                "record_id": doc.record_id,
                "register": doc.register,
                "label": doc.label,
                "kept": pred,
                "lev_sim": lev,
                "token_f1": shingle["f1"],
                "rouge_l_f1": rouge["f1"],
            }
        )

    kept_rows = [r for r in rows if r["kept"]]

    def agg(rs: list[dict], key: str) -> dict[str, float]:
        vals = [r[key] for r in rs]
        if not vals:
            return {"mean": 0.0, "median": 0.0, "n": 0}
        return {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}

    return {
        "n_gold": len(rows),
        "n_kept": len(kept_rows),
        "n_dropped": len(rows) - len(kept_rows),
        "kept_lev_sim": agg(kept_rows, "lev_sim"),
        "kept_token_f1": agg(kept_rows, "token_f1"),
        "kept_rouge_l": agg(kept_rows, "rouge_l_f1"),
        "overall_lev_sim": agg(rows, "lev_sim"),
        "overall_token_f1": agg(rows, "token_f1"),
        "overall_rouge_l": agg(rows, "rouge_l_f1"),
        "excluded": excluded,
        "per_doc": rows,
    }


def disagreements(
    run: Run,
    docs: list[DevsetDoc],
    label_mode: str = "all",
) -> dict[str, list[tuple[DevsetDoc, ExtractionRecord]]]:
    """Docs where the extractor disagrees with the label.

    false_keep = label says drop, extractor kept it.
    false_drop = label says keep, extractor dropped it.
    """
    records = run.records()
    out: dict[str, list[tuple[DevsetDoc, ExtractionRecord]]] = {"false_keep": [], "false_drop": []}
    for doc in docs:
        rec = records.get(doc.record_id)
        if rec is None or not _label_ok(doc, label_mode):
            continue
        pred = rec.pred_keep
        if pred is None:
            continue
        if pred and not doc.binary_label:
            out["false_keep"].append((doc, rec))
        elif not pred and doc.binary_label:
            out["false_drop"].append((doc, rec))
    return out


def spec_stats(manifest: dict) -> dict:
    """Spec length (a compression target once a spec works well). Counts the main
    spec; the continuation spec (chunked runs) is reported separately."""
    out = {}
    for key, name in (("spec_text", "spec"), ("cont_spec_text", "cont_spec")):
        text = manifest.get(key)
        if not text:
            continue
        stats = {"chars": len(text), "lines": text.count("\n") + 1, "words": len(text.split())}
        try:
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")
            stats["tokens"] = len(enc.encode(text, disallowed_special=()))
        except Exception:
            stats["tokens"] = None
        out[name] = stats
    return out


def run_summary(run: Run, docs: list[DevsetDoc]) -> dict:
    """One-stop summary used by the analyze script."""
    decisions = defaultdict(int)
    for rec in run.records().values():
        decisions[rec.decision.value] += 1
    return {
        "run": run.name,
        "manifest": {k: v for k, v in run.manifest.items() if k not in ("spec_text", "cont_spec_text")},
        "spec_stats": spec_stats(run.manifest),
        "n_records": len(run),
        "decisions": dict(decisions),
        "keep_drop_all": keep_drop_report(run, docs, label_mode="all"),
        "keep_drop_strong": keep_drop_report(run, docs, label_mode="strong"),
        "top_quality": top_quality_keep_rate(run, docs),
        "gold": gold_similarity(run, docs),
    }
