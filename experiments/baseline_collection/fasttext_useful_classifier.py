# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train + autotune a fastText classifier: useful page vs ``[NO_USEFUL_CONTENT]``.

Labels come from the two halves of the high_quality_3000_distill dataset:
``data/`` (the teacher kept the page) → ``__label__useful``; ``data_no_useful/``
(the teacher abstained) → ``__label__no_useful``. The classifier input is the
page's ``raw_html``, preprocessed three ways (a bake-off):

  * ``body_strip`` — strip ``<script>`` and keep ``<body>`` contents (the user's
    ``small-rephraser`` ``preprocess_html_for_extraction``); then whitespace-
    collapse + lowercase. No truncation.
  * ``raw_full`` — the whole HTML, whitespace-collapsed + lowercased. No
    truncation.
  * ``resiliparse`` — main-content plain text via resiliparse (the DCLM-style
    extraction); whitespace-collapsed + lowercased.

A single ``pilot`` job (run in us-central2 where the parquet lives) does prep →
autotune-train → threshold-sweep eval and writes the model + metrics to GCS.
Splits are **WARC-disjoint** (a WARC's rows go entirely to one of train/val/test)
so near-duplicate pages can't leak across splits and inflate the score.

fastText is single-machine; this pilot samples ``--max-per-class`` rows so it
runs in minutes. Scaling is just larger ``--n-warcs`` / ``--max-per-class``.

Usage (smoke)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 16 --memory 64GB --disk 100GB --priority interactive \\
        --extra cpu --extra dclm --enable-extra-resources \\
        --region us-central2 --job-name ft-useful-smoke-bodystrip \\
        -- python experiments/baseline_collection/fasttext_useful_classifier.py pilot \\
           --representation body_strip --n-warcs 20 --max-per-class 10000 \\
           --autotune-duration 180 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

DATASET_ROOT = "gs://marin-us-central2/datasets/high_quality_3000_distill"
USEFUL_DIR = f"{DATASET_ROOT}/data"
NO_USEFUL_DIR = f"{DATASET_ROOT}/data_no_useful"
RESULTS_ROOT = "gs://marin-us-central2/classifiers/useful_fasttext"

LABEL_USEFUL = "__label__useful"
LABEL_NO_USEFUL = "__label__no_useful"

# --- HTML → fastText text (pure, unit-tested) ------------------------------

# From small-rephraser scripts/run_spec_bench.py: strip <script>, keep <body>.
_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")

REPRESENTATIONS = ("body_strip", "raw_full", "resiliparse")


def body_strip(html: str) -> str:
    """Strip ``<script>`` blocks and keep ``<body>`` contents (all bodies joined).

    Falls back to the script-stripped HTML when there is no ``<body>`` tag.
    Mirrors small-rephraser's ``preprocess_html_for_extraction``.
    """
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def resiliparse_text(html: str) -> str:
    """Extract main-content plain text via resiliparse (mirrors marin's
    ``extract_text_from_html`` / the resiliparse baseline). Returns ``""`` on
    parse failure."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    try:
        return extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, noscript=False)
    except Exception:
        return ""


def to_fasttext_text(html: str, representation: str) -> str:
    """Turn raw HTML into a single fastText input line body (no label).

    fastText is line-oriented, so all whitespace (newlines included) is collapsed
    to single spaces; text is lowercased. No truncation.
    """
    if representation == "body_strip":
        text = body_strip(html)
    elif representation == "resiliparse":
        text = resiliparse_text(html)
    else:  # raw_full
        text = html
    return _WS_RE.sub(" ", text).strip().lower()


def fasttext_line(label: str, html: str, representation: str) -> str:
    """A full fastText training line: ``__label__x <single-line text>``."""
    return f"{label} {to_fasttext_text(html, representation)}"


# --- Sampling + split ------------------------------------------------------


@dataclass(frozen=True)
class PilotConfig:
    representation: str
    train_warcs: int  # number of front (non-held-out) WARC shards used for training; grows with scale
    max_per_class: int  # cap on TRAIN examples per class
    autotune_duration: int
    autotune_metric: str
    tag: str
    k_holdout: int = 1  # WARCs per snapshot held out for EACH of val and test (frozen, snapshot-stratified)
    eval_per_class: int = 5000  # cap per class for val and test each
    neg_per_pos: float = 1.0  # negatives per positive within each WARC (1.0 = balanced; ~12 = true deployment ratio)
    # How to pick the train WARCs from the non-held-out pool. "front" (sorted index prefix) is biased because
    # index correlates with snapshot; "random"/"stratified" match the snapshot-uniform held-out test distribution.
    train_sample: str = "front"
    sample_seed: int = 0
    reuse_config: str = ""  # GCS metrics.json to copy tuned HPs from (skip autotune; for clean cross-scale compare)
    # Fixed-config path (skip autotune). Defaults are the DCLM recipe: epoch=5, lr=0.1, dim=100, bigrams, softmax.
    fixed_config: bool = False
    epoch: int = 5
    lr: float = 0.1
    dim: int = 100
    word_ngrams: int = 2
    # minCount=1 (the DCLM recipe) keeps every singleton markup token (unique URLs/hashes), which
    # bloats the model to many GB and risks overfitting; raising it prunes that noise -> tiny, deployable.
    min_count: int = 1
    minn: int = 0
    maxn: int = 0
    loss: str = "softmax"


def _read_first_value(path: str, column: str):
    """Read the first value of ``column`` from a parquet shard (footer + 1 row)."""
    for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=1):
        for v in batch.column(column).to_pylist():
            return v
    return None


def _shard_snapshots(shards: list[str]) -> list[str]:
    """The CC-MAIN snapshot of each shard (all rows in a shard share one WARC →
    one snapshot). Read in parallel — one tiny read per shard."""
    snaps: list[str] = [""] * len(shards)
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(_read_first_value, p, "snapshot"): i for i, p in enumerate(shards)}
        for fut in as_completed(futs):
            snaps[futs[fut]] = fut.result() or ""
    return snaps


def held_out_indices(snapshots: list[str], k_per_snapshot: int) -> tuple[set[int], set[int]]:
    """Snapshot-stratified, frozen held-out shard indices for val and test.

    Within each snapshot the shards are ordered by index (deterministic); the
    first ``k`` go to val, the next ``k`` to test. So both val and test cover
    every time period, and the choice never changes across runs.
    """
    by_snap: dict[str, list[int]] = {}
    for i, s in enumerate(snapshots):
        by_snap.setdefault(s, []).append(i)
    val: set[int] = set()
    test: set[int] = set()
    for idxs in by_snap.values():
        idxs = sorted(idxs)
        val.update(idxs[:k_per_snapshot])
        test.update(idxs[k_per_snapshot : 2 * k_per_snapshot])
    return val, test


def _select_train_indices(candidates: list[int], snapshots: list[str], mode: str, n: int, seed: int) -> list[int]:
    """Pick ``n`` train WARC indices from the non-held-out ``candidates``.

    - ``front``: the first ``n`` by index. Biased: index correlates with snapshot,
      so the prefix over-represents the earliest crawls vs the snapshot-uniform test.
    - ``random``: a seeded random subset across all snapshots (representative by volume).
    - ``stratified``: round-robin across snapshots so each snapshot contributes
      near-equally, matching the uniform (1-per-snapshot) held-out test distribution.
    """
    if mode == "front":
        return sorted(candidates[:n])
    if mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(candidates, min(n, len(candidates))))
    if mode == "stratified":
        rng = random.Random(seed)
        by_snap: dict[str, list[int]] = {}
        for i in candidates:
            by_snap.setdefault(snapshots[i], []).append(i)
        for idxs in by_snap.values():
            rng.shuffle(idxs)
        chosen: list[int] = []
        snaps = sorted(by_snap)  # deterministic snapshot order; per-snapshot picks are shuffled
        while len(chosen) < n and any(by_snap[s] for s in snaps):
            for s in snaps:
                if by_snap[s]:
                    chosen.append(by_snap[s].pop())
                    if len(chosen) >= n:
                        break
        return sorted(chosen)
    raise ValueError(f"unknown train_sample mode: {mode!r}")


def _read_html_rows(path: str, max_rows: int) -> list[str]:
    """Read up to ``max_rows`` ``raw_html`` values from a parquet shard, reading
    only as many row groups as needed."""
    out: list[str] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["raw_html"], batch_size=2048):
        for h in batch.column("raw_html").to_pylist():
            if h:
                out.append(h)
                if len(out) >= max_rows:
                    return out
    return out


def write_split_files(cfg: PilotConfig, workdir: str) -> tuple[dict, dict]:
    """Build balanced, WARC-disjoint train/val/test fastText files.

    Val/test are a frozen, snapshot-stratified held-out set (``k_holdout`` WARCs
    per snapshot each); train is drawn from the remaining WARCs (front-first, up
    to ``train_warcs``). Negatives are balanced 1:1 within each WARC.
    Returns (per-split counts, held-out metadata).
    """
    useful_shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    no_useful_shards = sorted(fsspec_glob(f"{NO_USEFUL_DIR}/*.parquet"))
    n_total = min(len(useful_shards), len(no_useful_shards))

    snapshots = _shard_snapshots(useful_shards[:n_total])
    val_idx, test_idx = held_out_indices(snapshots, cfg.k_holdout)
    held = val_idx | test_idx
    candidates = [i for i in range(n_total) if i not in held]
    train_idx = _select_train_indices(candidates, snapshots, cfg.train_sample, cfg.train_warcs, cfg.sample_seed)
    splits = {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}
    per_warc = {
        "train": max(1, cfg.max_per_class // max(len(train_idx), 1)),
        "val": max(1, cfg.eval_per_class // max(len(val_idx), 1)),
        "test": max(1, cfg.eval_per_class // max(len(test_idx), 1)),
    }

    handles = {s: open(os.path.join(workdir, f"{s}.txt"), "w", encoding="utf-8") for s in ("train", "val", "test")}
    counts = {s: {"useful": 0, "no_useful": 0} for s in ("train", "val", "test")}
    try:
        for split, idxs in splits.items():
            cap = per_warc[split]
            for i in idxs:
                pos = _read_html_rows(useful_shards[i], cap)
                neg = _read_html_rows(no_useful_shards[i], round(cfg.neg_per_pos * len(pos)))  # neg:pos within WARC
                for html in pos:
                    handles[split].write(fasttext_line(LABEL_USEFUL, html, cfg.representation) + "\n")
                for html in neg:
                    handles[split].write(fasttext_line(LABEL_NO_USEFUL, html, cfg.representation) + "\n")
                counts[split]["useful"] += len(pos)
                counts[split]["no_useful"] += len(neg)
    finally:
        for h in handles.values():
            h.close()

    holdout_meta = {
        "n_total_warcs": n_total,
        "distinct_snapshots": len(set(snapshots)),
        "val_warc_shards": splits["val"],
        "test_warc_shards": splits["test"],
        "val_snapshots": sorted({snapshots[i] for i in val_idx}),
        "test_snapshots": sorted({snapshots[i] for i in test_idx}),
        "n_train_warcs": len(train_idx),
        "train_sample": cfg.train_sample,
        "sample_seed": cfg.sample_seed,
        "n_train_snapshots": len({snapshots[i] for i in train_idx}),
        "front_baseline_n_train_snapshots": len({snapshots[i] for i in candidates[: cfg.train_warcs]}),
    }
    logger.info("split counts: %s | held-out val=%d test=%d warcs", counts, len(val_idx), len(test_idx))
    for s in ("train", "val", "test"):
        if counts[s]["useful"] == 0 or counts[s]["no_useful"] == 0:
            raise RuntimeError(f"split {s} is missing a class: {counts[s]}")
    return counts, holdout_meta


# --- Eval: threshold sweep on P(useful) ------------------------------------


def _true_label(line: str) -> str:
    return line.split(" ", 1)[0]


def predict_label_prob(model, text: str, label: str) -> float:
    """P(``label``) for one line.

    Uses the low-level ``model.f.predict`` (mirrors marin.transform.dclm_filter):
    the high-level ``model.predict`` builds a no-copy NumPy array that raises
    "Unable to avoid copy" under NumPy 2.x with fasttext-wheel.
    """
    for prob, lab in model.f.predict(text, -1, 0.0, "strict"):  # all labels, threshold 0
        if lab == label:
            return float(prob)
    return 0.0


def predict_useful_prob(model, text: str) -> float:
    """P(__label__useful) — convenience wrapper around predict_label_prob."""
    return predict_label_prob(model, text, LABEL_USEFUL)


def evaluate_thresholds(model, test_path: str, thresholds: list[float]) -> list[dict]:
    """Sweep P(useful) thresholds; report precision/recall/F1 for the useful class.

    A row is predicted ``useful`` iff P(useful) >= threshold (so a high threshold
    means "only call it useful when confident" → high precision / lower recall;
    a low threshold favors recall of useful).
    """
    probs: list[tuple[float, bool]] = []  # (P(useful), is_truly_useful)
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            truth_useful = _true_label(line) == LABEL_USEFUL
            text = line.split(" ", 1)[1] if " " in line else ""
            probs.append((predict_useful_prob(model, text), truth_useful))
    return _precision_recall_sweep(probs, thresholds)


def _precision_recall_sweep(probs: list[tuple[float, bool]], thresholds: list[float]) -> list[dict]:
    """Given (P(useful), is_truly_useful) pairs, report precision/recall/F1 for the
    useful class at each threshold. Recall is independent of the negative:positive
    ratio; precision is not (so it reflects whatever ratio ``probs`` was drawn from)."""
    total_useful = sum(1 for _, t in probs if t)
    rows = []
    for t in thresholds:
        tp = sum(1 for p, truth in probs if truth and p >= t)
        fp = sum(1 for p, truth in probs if not truth and p >= t)
        fn = total_useful - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {"threshold": round(t, 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        )
    return rows


# --- Pilot driver ----------------------------------------------------------

# Hyperparameters fastText autotune searches over; captured so we can see whether
# the tuned config shifts as training data grows.
_HP_KEYS = ("lr", "dim", "ws", "epoch", "minCount", "minCountLabel", "neg", "wordNgrams", "bucket", "minn", "maxn", "t")


def tuned_hyperparameters(model) -> dict:
    """Read the autotuned args off a trained fastText model (``model.f.getArgs()``)."""
    args = model.f.getArgs()
    out: dict = {}
    for k in _HP_KEYS:
        try:
            out[k] = getattr(args, k)
        except AttributeError:
            pass
    try:
        out["loss"] = str(args.loss)
    except AttributeError:
        pass
    return out


def _load_hparams(metrics_path: str) -> dict:
    """Load tuned hyperparameters from a prior run's metrics.json as
    ``train_supervised`` kwargs (so a larger run can reuse a smaller run's
    autotuned config and isolate the effect of data size)."""
    with fsspec.open(metrics_path) as f:
        hp = dict(json.load(f).get("tuned_hyperparameters") or {})
    kwargs = {
        k: hp[k]
        for k in ("lr", "dim", "ws", "epoch", "minCount", "neg", "wordNgrams", "bucket", "minn", "maxn", "t")
        if hp.get(k) is not None
    }
    loss = str(hp.get("loss", "")).lower()
    for name in ("softmax", "ova", "hs", "ns"):
        if name in loss:
            kwargs["loss"] = name
            break
    return kwargs


def run_pilot(cfg: PilotConfig) -> None:
    import fasttext  # optional dep (marin[dclm]); only available inside the job

    if cfg.representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {cfg.representation!r}; expected {REPRESENTATIONS}")

    out_dir = f"{RESULTS_ROOT}/{cfg.representation}{('_' + cfg.tag) if cfg.tag else ''}"
    with tempfile.TemporaryDirectory() as workdir:
        counts, holdout_meta = write_split_files(cfg, workdir)
        train_p = os.path.join(workdir, "train.txt")
        val_p = os.path.join(workdir, "val.txt")
        test_p = os.path.join(workdir, "test.txt")

        if cfg.fixed_config:
            logger.info(
                "training fixed config (no autotune): epoch=%d lr=%g dim=%d wordNgrams=%d minn=%d maxn=%d "
                "minCount=%d loss=%s",
                cfg.epoch,
                cfg.lr,
                cfg.dim,
                cfg.word_ngrams,
                cfg.minn,
                cfg.maxn,
                cfg.min_count,
                cfg.loss,
            )
            model = fasttext.train_supervised(
                input=train_p,
                epoch=cfg.epoch,
                lr=cfg.lr,
                dim=cfg.dim,
                wordNgrams=cfg.word_ngrams,
                minn=cfg.minn,
                maxn=cfg.maxn,
                minCount=cfg.min_count,
                loss=cfg.loss,
            )
        elif cfg.reuse_config:
            hp = _load_hparams(cfg.reuse_config)
            logger.info("training with reused config from %s: %s", cfg.reuse_config, hp)
            model = fasttext.train_supervised(input=train_p, **hp)
        else:
            logger.info("autotuning fastText (duration=%ds, metric=%s) ...", cfg.autotune_duration, cfg.autotune_metric)
            model = fasttext.train_supervised(
                input=train_p,
                autotuneValidationFile=val_p,
                autotuneDuration=cfg.autotune_duration,
                autotuneMetric=cfg.autotune_metric,
            )

        n_test, p_at_1, r_at_1 = model.test(test_p)
        sweep = evaluate_thresholds(model, test_p, [i / 50 for i in range(51)])
        # Best-F1 and the highest-recall point at >=0.90 precision (recall-of-useful focus).
        best_f1 = max(sweep, key=lambda r: r["f1"])
        hi_recall = max((r for r in sweep if r["precision"] >= 0.90), key=lambda r: r["recall"], default=None)

        hyperparams = tuned_hyperparameters(model)
        metrics = {
            "config": asdict(cfg),
            "tuned_hyperparameters": hyperparams,
            "held_out": holdout_meta,
            "split_counts": counts,
            "test_size": n_test,
            "test_precision_at_1": p_at_1,
            "test_recall_at_1": r_at_1,
            "best_f1_operating_point": best_f1,
            "max_recall_at_precision_0.90": hi_recall,
            "threshold_sweep": sweep,
        }
        logger.info(
            "RESULT %s: tuned=%s best_f1=%s hi_recall@p0.90=%s",
            cfg.representation,
            hyperparams,
            best_f1,
            hi_recall,
        )

        # Upload model.bin FIRST so it is the leading artifact: a death mid-upload must never
        # leave metrics.json (the leaderboard key) present without the model the eval needs.
        # Stream every upload (shutil.copyfileobj) — train.txt is hundreds of GB at 1M-scale, so
        # the old src.read()-into-memory OOM-killed the box regardless of how much RAM we gave it.
        local_model = os.path.join(workdir, "model.bin")
        model.save_model(local_model)
        with open(local_model, "rb") as src, fsspec.open(f"{out_dir}/model.bin", "wb") as dst:
            shutil.copyfileobj(src, dst)
        with fsspec.open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        for split in ("train", "val", "test"):
            with (
                open(os.path.join(workdir, f"{split}.txt"), "rb") as src,
                fsspec.open(f"{out_dir}/{split}.txt.gz", "wb", compression="gzip") as dst,
            ):
                shutil.copyfileobj(src, dst)
    logger.info("pilot done -> %s (metrics.json, model.bin, {train,val,test}.txt.gz)", out_dir)


# --- Full prep (Zephyr, ALL data, no class balancing) ----------------------
#
# Materializes the entire fastText corpus once, in parallel: every useful doc +
# EVERY abstention (no 1:1 balancing — natural ~11:1 ratio), one durable gzipped
# text shard per WARC, split into the frozen snapshot-stratified train/val/test.
# A later single-box step concatenates + trains. Per-WARC + skip-existing → fully
# resumable under preemption.


def _iter_html(path: str):
    """Stream ``raw_html`` values from a parquet shard (bounded memory)."""
    for batch in pq.ParquetFile(path).iter_batches(columns=["raw_html"], batch_size=2048):
        for h in batch.column("raw_html").to_pylist():
            if h:
                yield h


def _prep_one_warc(spec: dict) -> dict:
    """Write one WARC's fastText text shard (all useful + all no_useful). Streaming,
    atomic (tmp→rename), skip-existing for resumability."""
    rep = spec["representation"]
    out = f"{spec['prep_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz"
    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath):
        return {"index": spec["index"], "split": spec["split"], "skipped": True}
    tmp = f"{out}.tmp"
    n_pos = n_neg = 0
    with fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as f:
        for html in _iter_html(spec["useful"]):
            f.write(fasttext_line(LABEL_USEFUL, html, rep) + "\n")
            n_pos += 1
        for html in _iter_html(spec["no_useful"]):
            f.write(fasttext_line(LABEL_NO_USEFUL, html, rep) + "\n")
            n_neg += 1
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    return {"index": spec["index"], "split": spec["split"], "n_useful": n_pos, "n_no_useful": n_neg, "skipped": False}


def run_prep(representation: str, k_holdout: int, tag: str, limit_warcs: int | None, only_split: str = "all") -> None:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; expected {REPRESENTATIONS}")
    prep_root = f"{RESULTS_ROOT}/full_prep_{representation}{('_' + tag) if tag else ''}"

    useful_shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    no_useful_shards = sorted(fsspec_glob(f"{NO_USEFUL_DIR}/*.parquet"))
    n_total = min(len(useful_shards), len(no_useful_shards))
    snapshots = _shard_snapshots(useful_shards[:n_total])
    val_idx, test_idx = held_out_indices(snapshots, k_holdout)

    def split_of(i: int) -> str:
        return "val" if i in val_idx else "test" if i in test_idx else "train"

    specs = [
        {
            "index": i,
            "useful": useful_shards[i],
            "no_useful": no_useful_shards[i],
            "split": split_of(i),
            "representation": representation,
            "prep_root": prep_root,
        }
        for i in range(n_total)
    ]
    if only_split != "all":
        specs = [s for s in specs if s["split"] == only_split]
    if limit_warcs is not None:
        specs = specs[:limit_warcs]
    logger.info(
        "prep %d WARCs (train=%d val=%d test=%d) -> %s",
        len(specs),
        sum(1 for s in specs if s["split"] == "train"),
        sum(1 for s in specs if s["split"] == "val"),
        sum(1 for s in specs if s["split"] == "test"),
        prep_root,
    )
    manifest = f"{prep_root}/_prep_manifest-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_iterable(specs).map(_prep_one_warc).write_jsonl(manifest, skip_existing=False)
    ctx = ZephyrContext(name="ft-useful-prep", max_workers=256, resources=ResourceConfig(cpu=2, ram="8g"))
    ctx.execute(pipeline)
    logger.info("prep done -> %s/{train,val,test}/", prep_root)


# --- Eval a saved model on an external (natural-ratio) test set ------------


def _max_recall_at_precision(sweep: list[dict], floor: float) -> dict | None:
    cand = [r for r in sweep if r["precision"] >= floor]
    return max(cand, key=lambda r: r["recall"]) if cand else None


def run_eval(
    model_path: str, test_glob: str, out_path: str, quality_label: str = LABEL_USEFUL, negative_label: str = ""
) -> None:
    """Score a saved model.bin against an external fastText-formatted test set
    (e.g., the full-prep natural-ratio test) — recall is comparable to the
    balanced eval; precision now reflects the real negative:positive ratio.

    ``quality_label`` is the model's label whose probability means "useful/high
    quality" (``__label__useful`` for ours; ``__label__eli5`` for the DCLM model).
    Ground-truth is always the test line's ``__label__useful`` vs ``no_useful``.
    """
    import fasttext

    shards = sorted(fsspec_glob(test_glob))
    if not shards:
        raise RuntimeError(f"no test shards match {test_glob}")
    with tempfile.TemporaryDirectory() as wd:
        local = os.path.join(wd, "model.bin")
        with fsspec.open(model_path, "rb") as src, open(local, "wb") as dst:
            dst.write(src.read())
        model = fasttext.load_model(local)
        logger.info("model labels: %s", model.get_labels())
        if negative_label:
            logger.info("scoring quality = 1 - P(%s)", negative_label)

        def quality_score(text: str) -> float:
            # ``negative_label`` is robust to the positive label's name (binary model):
            # P(positive) = 1 - P(negative). Mirrors marin.transform.dclm_filter.
            if negative_label:
                return 1.0 - predict_label_prob(model, text, negative_label)
            return predict_label_prob(model, text, quality_label)

        probs: list[tuple[float, bool]] = []
        n_useful = n_no_useful = 0
        for shard in shards:
            with fsspec.open(shard, "rt", compression="gzip", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    truth_useful = _true_label(line) == LABEL_USEFUL
                    text = line.split(" ", 1)[1] if " " in line else ""
                    probs.append((quality_score(text), truth_useful))
                    n_useful += truth_useful
                    n_no_useful += not truth_useful

        # Dense in the low region: some classifiers (e.g. DCLM oh-eli5, keep
        # threshold 0.018) put almost all probability mass below 0.02, so a plain
        # 0.02-step grid would miss their entire operating range.
        fine = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.018, 0.025, 0.03, 0.04, 0.06, 0.08]
        thresholds = sorted(set(fine + [i / 50 for i in range(51)]))
        sweep = _precision_recall_sweep(probs, thresholds)
        metrics = {
            "model": model_path,
            "quality_label": quality_label,
            "negative_label": negative_label,
            "test_glob": test_glob,
            "n_useful": n_useful,
            "n_no_useful": n_no_useful,
            "neg_to_pos_ratio": round(n_no_useful / max(n_useful, 1), 2),
            "test_size": n_useful + n_no_useful,
            "best_f1_operating_point": max(sweep, key=lambda r: r["f1"]),
            "max_recall_at_precision_0.90": _max_recall_at_precision(sweep, 0.90),
            "max_recall_at_precision_0.95": _max_recall_at_precision(sweep, 0.95),
            "threshold_sweep": sweep,
        }
        with fsspec.open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Persist the raw (score, was-it-useful) pairs so any threshold sweep can be
        # recomputed offline in milliseconds — never re-run the slow eval again.
        preds_path = (out_path[:-5] if out_path.endswith(".json") else out_path) + ".preds.parquet"
        table = pa.table(
            {
                "score": pa.array([p for p, _ in probs], type=pa.float32()),
                "useful": pa.array([t for _, t in probs], type=pa.bool_()),
            }
        )
        local_preds = os.path.join(wd, "preds.parquet")
        pq.write_table(table, local_preds)
        with open(local_preds, "rb") as src, fsspec.open(preds_path, "wb") as dst:
            dst.write(src.read())
        logger.info("wrote per-example scores -> %s", preds_path)
    logger.info(
        "eval done -> %s | n=%d (ratio %.1f:1) best_f1=%s R@P0.90=%s",
        out_path,
        n_useful + n_no_useful,
        n_no_useful / max(n_useful, 1),
        metrics["best_f1_operating_point"],
        metrics["max_recall_at_precision_0.90"],
    )


LEADERBOARD_MD = f"{RESULTS_ROOT}/LEADERBOARD.md"
LEADERBOARD_JSON = f"{RESULTS_ROOT}/leaderboard.json"


def _load_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _operating_at_recall(sweep: list[dict], target_recall: float) -> dict | None:
    """Among thresholds reaching ``target_recall``, the one with the best precision."""
    candidates = [r for r in sweep if r["recall"] >= target_recall]
    return max(candidates, key=lambda r: r["precision"]) if candidates else None


def collect_leaderboard_rows(pilot_by_dir: dict[str, dict], natural_by_dir: dict[str, dict]) -> list[dict]:
    """One row per model dir, merging its pilot self-test metrics with its external
    natural-ratio eval (when present). Pure function over already-loaded JSON."""
    rows = []
    for name in sorted(set(pilot_by_dir) | set(natural_by_dir)):
        pm = pilot_by_dir.get(name, {})
        cfg = pm.get("config", {})
        held = pm.get("held_out", {})
        train_counts = pm.get("split_counts", {}).get("train", {})
        test_counts = pm.get("split_counts", {}).get("test", {})
        if cfg.get("fixed_config"):
            recipe = "fixed"
        elif cfg.get("reuse_config"):
            recipe = "reuse"
        else:
            recipe = "autotune"
        pilot_test_ratio = round(test_counts.get("no_useful", 0) / max(test_counts.get("useful", 0), 1), 1)
        row = {
            "model": name,
            "representation": cfg.get("representation"),
            "sampling": held.get("train_sample", cfg.get("train_sample", "front")),
            "neg_per_pos": cfg.get("neg_per_pos"),
            "recipe": recipe,
            "train_pos": train_counts.get("useful"),
            "train_neg": train_counts.get("no_useful"),
            "n_train_warcs": held.get("n_train_warcs"),
            "n_train_snapshots": held.get("n_train_snapshots"),
            "front_snapshots": held.get("front_baseline_n_train_snapshots"),
            "pilot_test_ratio": pilot_test_ratio,
            "pilot_test_f1": round(pm.get("best_f1_operating_point", {}).get("f1", 0), 4) if pm else None,
        }
        nm = natural_by_dir.get(name)
        if nm:
            best = nm["best_f1_operating_point"]
            r90 = _operating_at_recall(nm["threshold_sweep"], 0.90)
            r95 = _operating_at_recall(nm["threshold_sweep"], 0.95)
            # Degenerate: best F1 is the threshold-0 "predict everything useful" point — the
            # model's label/score wiring is broken (e.g. wrong quality_label). Flag, don't rank.
            row["natural_suspect"] = best["threshold"] == 0.0 and best["recall"] >= 0.999
            row.update(
                natural_ratio=nm.get("neg_to_pos_ratio"),
                natural_best_f1=round(best["f1"], 4),
                natural_f1_threshold=best["threshold"],
                natural_f1_precision=round(best["precision"], 4),
                natural_f1_recall=round(best["recall"], 4),
                natural_precision_at_recall_0_90=round(r90["precision"], 4) if r90 else None,
                natural_precision_at_recall_0_95=round(r95["precision"], 4) if r95 else None,
            )
        rows.append(row)
    return rows


def _leaderboard_markdown(rows: list[dict]) -> str:
    """Render the merged rows as a Markdown leaderboard, sorted by natural-ratio F1."""

    def _rank_key(r: dict):
        # Pending and degenerate evals sink below legitimately-scored runs.
        unranked = r.get("natural_best_f1") is None or r.get("natural_suspect")
        return (unranked, -(r.get("natural_best_f1") or 0))

    ranked = sorted(rows, key=_rank_key)
    lines = [
        "# useful-vs-NO_USEFUL classifier leaderboard",
        "",
        "Deployment number = **natural-ratio best F1** (frozen ~12:1 test). `pilot F1` is each",
        "run's own held-out test (ratio varies with `neg/pos`). `pending` = natural eval not yet run.",
        "",
        "| model | rep | sampling | neg/pos | recipe | train pos×neg | train snaps (front) | pilot F1 (ratio) | **natural F1** | P@F1 | R@F1 | P@R≥.90 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        tp, tn = r.get("train_pos"), r.get("train_neg")
        train = f"{tp:,}×{tn:,}" if tp and tn else "—"
        snaps = f"{r.get('n_train_snapshots')}({r.get('front_snapshots')})"
        pilot = f"{r['pilot_test_f1']} @{r['pilot_test_ratio']}:1" if r.get("pilot_test_f1") is not None else "—"
        nat = r.get("natural_best_f1")
        if nat is None:
            nat_s = "pending"
        elif r.get("natural_suspect"):
            nat_s = f"⚠ {nat} (degenerate)"
        else:
            nat_s = f"**{nat}**"
        pf1 = r.get("natural_f1_precision", "—")
        rf1 = r.get("natural_f1_recall", "—")
        p90 = r.get("natural_precision_at_recall_0_90")
        p90_s = p90 if p90 is not None else "—"
        lines.append(
            f"| {r['model']} | {r.get('representation') or '—'} | {r['sampling']} | "
            f"{r.get('neg_per_pos') or '—'} | {r['recipe']} | {train} | {snaps} | {pilot} | "
            f"{nat_s} | {pf1} | {rf1} | {p90_s} |"
        )
    return "\n".join(lines) + "\n"


def run_leaderboard(out_md: str, out_json: str) -> None:
    """Scan every result JSON under RESULTS_ROOT and write a consolidated leaderboard
    (Markdown + JSON) to the shared GCS location."""
    pilot_by_dir, natural_by_dir = {}, {}
    for path in fsspec_glob(f"{RESULTS_ROOT}/*/metrics.json"):
        pilot_by_dir[path.rsplit("/", 2)[-2]] = _load_json(path)
    for path in fsspec_glob(f"{RESULTS_ROOT}/*/eval*.json"):
        # Prefer eval_natural.json when a dir has several eval files.
        name = path.rsplit("/", 2)[-2]
        if name not in natural_by_dir or path.endswith("eval_natural.json"):
            natural_by_dir[name] = _load_json(path)
    rows = collect_leaderboard_rows(pilot_by_dir, natural_by_dir)
    with fsspec.open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    md = _leaderboard_markdown(rows)
    with fsspec.open(out_md, "w") as f:
        f.write(md)
    logger.info("leaderboard: %d models -> %s , %s", len(rows), out_md, out_json)
    print(md)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot", help="Prep + autotune-train + threshold-sweep eval in one job.")
    p.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    p.add_argument("--train-warcs", type=int, default=60, help="Non-held-out front WARC shards for training (scales).")
    p.add_argument("--max-per-class", type=int, default=200000, help="Cap on TRAIN examples per class.")
    p.add_argument(
        "--k-holdout", type=int, default=1, help="WARCs per snapshot held out for EACH of val and test (frozen)."
    )
    p.add_argument("--eval-per-class", type=int, default=5000, help="Cap per class for val and test each.")
    p.add_argument(
        "--neg-per-pos",
        type=float,
        default=1.0,
        help="Negatives per positive within each WARC (1.0 = balanced; ~12 = true deployment ratio).",
    )
    p.add_argument("--autotune-duration", type=int, default=900, help="fastText autotune seconds.")
    p.add_argument(
        "--autotune-metric",
        default="f1:__label__useful",
        help="fastText autotune metric, e.g. f1:__label__useful or recallAtPrecision:90:__label__useful.",
    )
    p.add_argument(
        "--reuse-config",
        default="",
        help="GCS metrics.json from a prior run; reuse its tuned HPs and skip autotune (clean cross-scale compare).",
    )
    p.add_argument("--fixed-config", action="store_true", help="Skip autotune; train a fixed config (DCLM recipe).")
    p.add_argument("--epoch", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--word-ngrams", type=int, default=2)
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Prune tokens occurring < this many times. 1=DCLM recipe (huge model); 5-10 shrinks 10-50x.",
    )
    p.add_argument("--minn", type=int, default=0)
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--loss", default="softmax")
    p.add_argument(
        "--train-sample",
        default="front",
        choices=["front", "random", "stratified"],
        help="How to pick train WARCs: front (biased prefix), random, or snapshot-stratified (matches test).",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="Seed for random/stratified train-WARC sampling.")
    p.add_argument("--tag", default="", help="Suffix for the output dir (e.g. 'smoke').")

    pe = sub.add_parser("eval", help="Score a saved model.bin against an external (natural-ratio) test set.")
    pe.add_argument("--model", required=True, help="GCS path to model.bin.")
    pe.add_argument(
        "--test-glob",
        default=f"{RESULTS_ROOT}/full_prep_body_strip/test/*.txt.gz",
        help="Glob of fastText-formatted test shards (default: full-prep natural-ratio test).",
    )
    pe.add_argument("--out", required=True, help="GCS path for the eval metrics.json.")
    pe.add_argument(
        "--quality-label",
        default=LABEL_USEFUL,
        help="Model label whose prob = 'useful/high-quality' (ours: __label__useful).",
    )
    pe.add_argument(
        "--negative-label",
        default="",
        help="If set, quality = 1 - P(this label) — robust to the positive label's name (DCLM: __label__cc).",
    )

    pp = sub.add_parser("prep", help="Parallel (Zephyr) full-corpus prep: ALL data, no balancing, split-tagged.")
    pp.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    pp.add_argument("--k-holdout", type=int, default=1, help="WARCs per snapshot held out for val and test each.")
    pp.add_argument("--limit-warcs", type=int, default=None, help="Smoke: only prep the first N WARC shards.")
    pp.add_argument("--only-split", default="all", choices=["all", "train", "val", "test"], help="Prep only this split.")
    pp.add_argument("--tag", default="", help="Suffix for the prep output dir.")

    pl = sub.add_parser("leaderboard", help="Scan all result JSONs and write the consolidated leaderboard.")
    pl.add_argument("--out-md", default=LEADERBOARD_MD, help="GCS/local path for the Markdown leaderboard.")
    pl.add_argument("--out-json", default=LEADERBOARD_JSON, help="GCS/local path for the leaderboard rows JSON.")
    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args.model, args.test_glob, args.out, args.quality_label, args.negative_label)
    elif args.command == "leaderboard":
        run_leaderboard(args.out_md, args.out_json)
    elif args.command == "prep":
        run_prep(args.representation, args.k_holdout, args.tag, args.limit_warcs, args.only_split)
    elif args.command == "pilot":
        run_pilot(
            PilotConfig(
                representation=args.representation,
                train_warcs=args.train_warcs,
                max_per_class=args.max_per_class,
                k_holdout=args.k_holdout,
                eval_per_class=args.eval_per_class,
                neg_per_pos=args.neg_per_pos,
                autotune_duration=args.autotune_duration,
                autotune_metric=args.autotune_metric,
                reuse_config=args.reuse_config,
                fixed_config=args.fixed_config,
                epoch=args.epoch,
                lr=args.lr,
                dim=args.dim,
                word_ngrams=args.word_ngrams,
                min_count=args.min_count,
                minn=args.minn,
                maxn=args.maxn,
                loss=args.loss,
                train_sample=args.train_sample,
                sample_seed=args.sample_seed,
                tag=args.tag,
            )
        )


if __name__ == "__main__":
    main()
