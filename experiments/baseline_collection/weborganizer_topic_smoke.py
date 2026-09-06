# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the WebOrganizer/TopicClassifier (24-way web topic labels) on our curated corpora.

This is PHASE 0 of the topic-labelling pipeline: it runs the reference HF implementation on CPU over
a small real sample and answers the three questions that gate the TPU build-out:

1. **Correctness** — does the reference stack load and produce sane labels on OUR documents?
2. **Golden logits** — emits ``logits`` at ``max_length=8192`` per doc, which becomes the parity
   oracle for the Levanter/JAX port (mirrors how the ModernBERT migration was validated).
3. **Truncation curve** — the dominant throughput lever. The model is a 140M gte-base-en-v1.5 with
   FULL global attention at every layer, so cost is quadratic-ish in context. WebOrganizer trains and
   infers at ``max_length=8192``, but if short truncation preserves the label we buy a large constant
   factor. This scores each doc at several ``max_length`` values and reports agreement vs the 8192
   reference, restricted to the docs that the cutoff actually truncates (the only ones that can move).

The model input template is ``"{url}\\n\\n{text}"`` (WebOrganizer ``annotate_data/domains.py``). The
``-NoURL`` variant takes bare ``{text}`` and exists for the deduped corpora where URL was dropped.

The reference config sets ``use_memory_efficient_attention`` / ``unpad_inputs``, which are xformers
CUDA-only paths; both are forced off here so the model runs on CPU (and, later, XLA).

Run in-region as a CPU Iris job (never read these corpora cross-region)::

    uv run iris --cluster marin job run --region us-central2 --memory 32GB \\
      --priority interactive --no-wait --job-name wo-topic-smoke-dclm \\
      -e HF_TOKEN hf_... -- \\
      python -m experiments.baseline_collection.weborganizer_topic_smoke probe --dataset dclm_10k --limit 300

``--dataset high_quality_10k`` lives in us-central1, so pin ``--region us-central1`` for that arm; every
other corpus is us-central2. Never run this cross-region — the corpora are large and egress is billed.
"""

from __future__ import annotations

import argparse
import enum
import gzip
import io
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

URL_MODEL = "WebOrganizer/TopicClassifier"
NOURL_MODEL = "WebOrganizer/TopicClassifier-NoURL"
INPUT_TEMPLATE = "{url}\n\n{text}"  # WebOrganizer annotate_data/domains.py EmbedOptions.input_template
REFERENCE_MAX_LENGTH = 8192  # what WebOrganizer trains + annotates at; the truncation baseline
TRUNCATION_GRID = (256, 512, 1024, 2048, 4096, 8192)
MAX_TEXT_CHARS = 1_000_000  # cap pathological multi-MB docs before tokenizing (matches score_modernbert_useful)
OUT_ROOT = "documents/weborganizer_topic/smoke"
EXAMPLES_PER_LABEL = 8  # per-label random sample kept for the drill-down viewer
EXAMPLE_MIN_PROB = 0.5  # sample only above this predicted probability (stay out of the coin-flip zone)

# The consolidated (all-regions-in-one-bucket) LLM-extraction archive that `dedup_extracted.py` reads,
# and the manifest index beside it. Everything is in us-central1 by construction.
LLM_ARCHIVE = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region"
LLM_RESOLVED = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved"
# Group manifest of the llm_pipeline_v1_1 10k run's us-east5 batches at 10,363 completed WARCs (2026-09-03).
LPV11_TOPIC_MANIFESTS = "gs://marin-us-east5/metadata/lpv1_1_topic_sample"


class Format(enum.StrEnum):
    JSONL_GZ = "jsonl.gz"
    PARQUET = "parquet"


@dataclass(frozen=True)
class Corpus:
    """A curated corpus at its DOCUMENT layer (the tokenized caches carry no URL)."""

    path: str
    region: str
    format: Format
    text_field: str = "text"
    url_field: str | None = "url"  # None => URL was dropped; only the -NoURL model applies
    # Shard discovery. Most corpora are a flat directory we can glob. The LLM-extraction bands are
    # not: they are ~240k tiny per-WARC batch files across 5 region subtrees, where a glob is
    # hopeless. For those, `manifest` points at the consolidated `resolved_{spec}.jsonl.gz` index
    # (rows of {region, warc_hash, batch_idx, num_records}) and `spec` is the path segment — empty
    # for `low_quality`, which predates the registry and writes to the UNPREFIXED legacy path.
    manifest: str | None = None
    spec: str | None = None
    # An UN-consolidated extraction run: a group manifest (rows {warc, region, shards}) written by
    # `grid_projection_manifest.py --only-region`, whose fully-qualified shard URLs all live in
    # `region`. `path` is informational.
    group_manifest: str | None = None


# The 10,364-WARC (`dclm_400m_1x_10k`) pool at its DOCUMENT layer — all URL-bearing, verified 2026-07-17.
# The `_deduped`/`_decon` trees are deliberately absent: dedup_extracted.py:196 projects records to
# `{text}` only, dropping url/warc_record_id. dclm/nemotron/fineweb_edu/resiliparse below are the
# PRE-dedup document layer (URL native); high_quality is the POST-dedup+decon survivor set with URL
# already restored by build_high_quality_hf_export.py's content-hash join (99.99% matched).
CORPORA: dict[str, Corpus] = {
    "dclm_10k": Corpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977", "us-central2", Format.JSONL_GZ
    ),
    "nemotron_full_10k": Corpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9", "us-central2", Format.JSONL_GZ
    ),
    "fineweb_edu_10k": Corpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9", "us-central2", Format.JSONL_GZ
    ),
    "resiliparse_10k": Corpus(
        "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f", "us-central2", Format.JSONL_GZ
    ),
    "high_quality_10k": Corpus(
        "gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined",
        "us-central1",
        Format.PARQUET,
    ),
    # --- LLM-extraction quality bands, from the 3000-WARC pool -----------------------------------
    # These are Qwen3-8B PROMPT VARIANTS over the same raw HTML, not score thresholds on a corpus:
    # low_quality (the original prompt, aka llm_curated) -> med_low_quality (v6) -> med_quality (v8)
    # -> high_quality (extreme_quality_v5, ~7.5% keep). So the bands differ by how hard the extractor
    # was told to reject a page, which is exactly the axis the topic mix should illuminate.
    #
    # We read the PRE-dedup consolidated archive because dedup projects rows to {text} and destroys
    # url. It all lives in us-central1, so unlike the 10k corpora these need NO mirror — v5p is local.
    #
    # ⚠ Coverage differs per band and they are NOT interchangeable: med_quality has 3,000 WARCs but
    # med_low_quality has only 100. And this is the 3k WARC pool, a DIFFERENT draw from the 10k pool
    # the other corpora use — cross-pool comparisons are indicative, not controlled.
    "med_quality_3k": Corpus(
        LLM_ARCHIVE,
        "us-central1",
        Format.JSONL_GZ,
        manifest=f"{LLM_RESOLVED}/resolved_med_quality.jsonl.gz",
        spec="med_quality",
    ),
    "med_low_quality_3k": Corpus(
        LLM_ARCHIVE,
        "us-central1",
        Format.JSONL_GZ,
        manifest=f"{LLM_RESOLVED}/resolved_med_low_quality.jsonl.gz",
        spec="med_low_quality",
    ),
    # low_quality is the LEGACY spec: no `{spec}/` path segment, and its manifest is the unsuffixed
    # resolved.jsonl.gz. Getting either wrong silently yields zero shards.
    "low_quality_3k": Corpus(
        LLM_ARCHIVE,
        "us-central1",
        Format.JSONL_GZ,
        manifest=f"{LLM_RESOLVED}/resolved.jsonl.gz",
        spec=None,
    ),
    # --- llm_pipeline_v1_1 over the 10k pool (the Qwen3-8B twin pipeline), NOT consolidated --------
    # The run's raw output is `data-{warc_hash}/batch_NNNN.jsonl.gz` across five regional buckets
    # (steal mode splits a WARC's batches across regions), and the 2026-08-12 consolidated archive
    # covers only 6,943 of the 10,363 completed WARCs — and only partly. Which fleet claimed a WARC
    # is unrelated to its content, so the us-east5 share (152,963 batches of 2,844 completed WARCs,
    # ~7M docs, listed 2026-09-03) is a random subsample of the run and is read natively there.
    # Pre-dedup, URL native, kept docs only.
    "llm_pipeline_v1_1_10k": Corpus(
        "gs://marin-us-east5/documents/baseline_llm_extraction/llm_pipeline_v1_1",
        "us-east5",
        Format.JSONL_GZ,
        group_manifest=f"{LPV11_TOPIC_MANIFESTS}/manifest_us-east5.jsonl",
    ),
}


@dataclass(frozen=True)
class Doc:
    url: str
    text: str


def _iter_jsonl_gz(path: str, text_field: str, url_field: str | None) -> Iterator[Doc]:
    with fsspec.open(path, "rb") as fh:
        with gzip.open(io.BufferedReader(fh), "rt", encoding="utf-8") as gz:  # type: ignore[arg-type]
            for line in gz:
                record = json.loads(line)
                text = record.get(text_field)
                if not text:
                    continue
                yield Doc(url=record.get(url_field) or "" if url_field else "", text=text)


def _iter_parquet(path: str, text_field: str, url_field: str | None) -> Iterator[Doc]:

    columns = [text_field] + ([url_field] if url_field else [])
    with fsspec.open(path, "rb") as fh:
        table = pq.ParquetFile(fh).read(columns=columns)
    urls = table.column(url_field).to_pylist() if url_field else [""] * table.num_rows
    for text, url in zip(table.column(text_field).to_pylist(), urls, strict=True):
        if text:
            yield Doc(url=url or "", text=text)


def read_sample(corpus: Corpus, limit: int) -> list[Doc]:
    """Read the first `limit` documents from the corpus's first shard (single-shard read is enough for a smoke)."""
    pattern = f"{corpus.path}/*.{corpus.format.value}"
    shards = sorted(fsspec_glob(pattern))
    if not shards:
        raise ValueError(f"no shards matched {pattern!r}")
    reader = _iter_jsonl_gz if corpus.format is Format.JSONL_GZ else _iter_parquet
    docs: list[Doc] = []
    for doc in reader(shards[0], corpus.text_field, corpus.url_field):
        docs.append(Doc(url=doc.url, text=doc.text[:MAX_TEXT_CHARS]))
        if len(docs) >= limit:
            break
    logger.info("sample: %d docs from %s (%s)", len(docs), shards[0], corpus.region)
    return docs


def load_model(model_name: str):
    """Load the reference HF classifier with the xformers-only attention paths disabled (CPU/XLA safe)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_memory_efficient_attention = False
    config.unpad_inputs = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True)
    _materialize_rope_buffers(model.new.embeddings, config.max_position_embeddings)
    model.eval()
    return config, tokenizer, model


def _materialize_rope_buffers(embeddings, max_position_embeddings: int) -> None:
    """Rebuild the non-persistent buffers the remote code computes in ``__init__``.

    transformers >= 5 initializes remote-code modules on the meta device and only restores tensors
    that exist in the checkpoint, so ``position_ids``, ``inv_freq``, ``cos_cached`` and ``sin_cached``
    come back uninitialized (zeros or garbage). Zero rope silently yields finite-but-wrong logits;
    garbage yields NaN — the 2026-08-16 lpv11 topic run produced both. Recompute them exactly as
    ``__init__`` would, then refuse to continue if they still look degenerate.
    """
    embeddings.register_buffer("position_ids", torch.arange(max_position_embeddings), persistent=False)
    rotary = embeddings.rotary_emb
    inv_freq = 1.0 / (rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim))
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    # NTKScalingRotaryEmbedding re-derives inv_freq for seq_len > max_position_embeddings from
    # (base, scaling_factor, dim) alone, so this reproduces the NTK cache; the plain class reuses
    # the inv_freq set above.
    rotary._set_cos_sin_cache(int(rotary.max_seq_len_cached), rotary.inv_freq.device, torch.float32)
    for name in ("inv_freq", "cos_cached", "sin_cached"):
        buf = getattr(rotary, name)
        if not torch.isfinite(buf).all() or float(buf.abs().max()) == 0.0:
            raise RuntimeError(f"rope buffer {name} is degenerate after materialization")
    if int(embeddings.position_ids[-1]) != max_position_embeddings - 1:
        raise RuntimeError("position_ids buffer is not arange after materialization")


def _render(docs: list[Doc], use_url: bool) -> list[str]:
    if not use_url:
        return [d.text for d in docs]
    return [INPUT_TEMPLATE.format(url=d.url, text=d.text) for d in docs]


@torch.inference_mode()
def score(
    model,
    tokenizer,
    pages: list[str],
    max_length: int,
    batch_size: int,
    on_batch: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, float]:
    """Return (logits [N, num_labels] float32, elapsed_seconds). Batches are padded to the longest member.

    `on_batch(start, logits)` fires per batch as it completes, so callers can collect streaming
    side-outputs (e.g. the per-label reservoir) in the same pass rather than re-reading afterwards.
    """
    out: list[np.ndarray] = []
    started = time.monotonic()
    for start in range(0, len(pages), batch_size):
        batch = tokenizer(
            pages[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(**batch).logits.float().numpy()
        out.append(logits)
        if on_batch is not None:
            on_batch(start, logits)
    return np.concatenate(out, axis=0), time.monotonic() - started


def token_lengths(tokenizer, pages: list[str]) -> np.ndarray:
    """Untruncated token length per page — sizes the length-bucketing and bounds the truncation loss."""
    return np.array([len(tokenizer(p, truncation=False)["input_ids"]) for p in pages], dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class LabelReservoir:
    """Streaming per-label reservoir sample (Algorithm R), collected AS the scorer runs.

    Holds at most ``per_label`` docs per label regardless of corpus size, so the same object works
    for a 500-doc smoke and a 20M-doc production pass — no post-hoc second read of the corpus.

    The draw is uniform over each label's *eligible* stream (predicted ``p >= min_prob``), not
    confidence-ranked: a top-N-by-confidence list only ever shows the easy core of a class, which
    tells you nothing about what the label actually captures. The floor keeps the draw out of the
    model's coin-flip zone without collapsing it to the argmax.

    ``seen`` counts every doc per label (the true distribution) while ``eligible`` counts only those
    that cleared the floor, so a label whose floor starved its sample is visible rather than silently
    under-sampled.
    """

    def __init__(self, label_names: list[str], per_label: int, min_prob: float, seed: int = 0):
        self.label_names = label_names
        self.per_label = per_label
        self.min_prob = min_prob
        self.rng = np.random.default_rng(seed)
        self.seen = np.zeros(len(label_names), dtype=np.int64)
        self.eligible = np.zeros(len(label_names), dtype=np.int64)
        self.kept: dict[int, list[dict]] = {i: [] for i in range(len(label_names))}

    def update(self, docs: list[Doc], probs: np.ndarray) -> None:
        """Feed one batch: `docs` aligned with `probs` [batch, num_labels]."""
        for doc, row in zip(docs, probs, strict=True):
            label_id = int(row.argmax())
            self.seen[label_id] += 1
            prob = float(row[label_id])
            if prob < self.min_prob:
                continue
            self.eligible[label_id] += 1
            bucket = self.kept[label_id]
            if len(bucket) < self.per_label:
                bucket.append({"url": doc.url, "text": doc.text, "prob": prob})
                continue
            # Algorithm R: the n-th eligible item replaces a uniformly chosen slot with prob k/n.
            j = int(self.rng.integers(0, self.eligible[label_id]))
            if j < self.per_label:
                bucket[j] = {"url": doc.url, "text": doc.text, "prob": prob}

    def rows(self, dataset: str) -> list[dict]:
        return [
            {
                "dataset": dataset,
                "label": self.label_names[label_id],
                "prob": row["prob"],
                "n_seen": int(self.seen[label_id]),
                "n_eligible": int(self.eligible[label_id]),
                "url": row["url"],
                "text": row["text"],
            }
            for label_id, bucket in self.kept.items()
            for row in bucket
        ]


def run_probe(
    dataset: str,
    limit: int,
    batch_size: int,
    use_url: bool,
    grid: tuple[int, ...],
    examples_per_label: int,
    example_min_prob: float,
) -> None:

    corpus = CORPORA[dataset]
    if use_url and corpus.url_field is None:
        raise ValueError(f"corpus {dataset!r} has no URL field; rerun with --no-url")
    model_name = URL_MODEL if use_url else NOURL_MODEL

    docs = read_sample(corpus, limit)
    config, tokenizer, model = load_model(model_name)
    pages = _render(docs, use_url)
    logger.info("model=%s labels=%d docs=%d", model_name, config.num_labels, len(pages))

    lengths = token_lengths(tokenizer, pages)
    pct = np.percentile(lengths, [50, 75, 90, 95, 99]).astype(int)
    logger.info(
        "TOKEN LENGTHS: mean=%d p50=%d p75=%d p90=%d p95=%d p99=%d max=%d",
        int(lengths.mean()),
        *pct,
        int(lengths.max()),
    )
    for cutoff in grid:
        share = float((lengths > cutoff).mean())
        logger.info("  truncated at %5d: %5.1f%% of docs", cutoff, 100 * share)

    label_names = [config.id2label[i] for i in range(config.num_labels)]
    reservoir = LabelReservoir(label_names, examples_per_label, example_min_prob)

    def _collect(start: int, logits: np.ndarray) -> None:
        reservoir.update(docs[start : start + len(logits)], _softmax(logits))

    results: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for max_length in sorted(grid, reverse=True):  # reference (8192) first
        # Only the reference pass feeds the reservoir — the truncated passes re-score the same docs.
        on_batch = _collect if max_length == REFERENCE_MAX_LENGTH else None
        logits, secs = score(model, tokenizer, pages, max_length, batch_size, on_batch=on_batch)
        results[max_length] = logits
        timings[max_length] = len(pages) / secs if secs else 0.0
        logger.info(
            "SCORED max_length=%5d in %6.1fs = %6.2f docs/s (CPU, batch=%d)",
            max_length,
            secs,
            len(pages) / secs if secs else 0.0,
            batch_size,
        )

    reference = results[REFERENCE_MAX_LENGTH]
    ref_choice = reference.argmax(-1)
    logger.info("TRUNCATION AGREEMENT vs max_length=%d:", REFERENCE_MAX_LENGTH)
    logger.info("  %6s  %8s  %8s  %10s", "maxlen", "all", "affected", "n_affected")
    agreement: list[dict] = []
    for max_length in sorted(grid):
        choice = results[max_length].argmax(-1)
        agree_all = float((choice == ref_choice).mean())
        affected = lengths > max_length  # only docs this cutoff actually truncates can disagree
        agree_affected = float((choice[affected] == ref_choice[affected]).mean()) if affected.any() else 1.0
        agreement.append(
            {
                "max_length": max_length,
                "agree_all": agree_all,
                "agree_affected": agree_affected,
                "n_affected": int(affected.sum()),
                "docs_per_sec_cpu": timings[max_length],
            }
        )
        logger.info(
            "  %6d  %7.2f%%  %7.2f%%  %10d", max_length, 100 * agree_all, 100 * agree_affected, int(affected.sum())
        )

    counts = np.bincount(ref_choice, minlength=config.num_labels)
    logger.info("REFERENCE LABEL DISTRIBUTION (%s):", dataset)
    for i in np.argsort(-counts):
        if counts[i]:
            logger.info("  %-22s %5d  (%4.1f%%)", label_names[i], int(counts[i]), 100 * counts[i] / len(pages))

    out_dir = f"gs://marin-{corpus.region}/{OUT_ROOT}"
    stem = f"{dataset}_{'url' if use_url else 'nourl'}"

    # Durable summary — stdout is not a reliable channel here (finelog is frequently unavailable).
    summary = {
        "dataset": dataset,
        "model": model_name,
        "n_docs": len(docs),
        "source_shard_region": corpus.region,
        "token_lengths": {
            "mean": float(lengths.mean()),
            "p50": int(pct[0]),
            "p75": int(pct[1]),
            "p90": int(pct[2]),
            "p95": int(pct[3]),
            "p99": int(pct[4]),
            "max": int(lengths.max()),
        },
        "truncation": agreement,
        "label_distribution": {label_names[i]: int(counts[i]) for i in np.argsort(-counts) if counts[i]},
    }
    with fsspec.open(f"{out_dir}/{stem}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("wrote summary -> %s/%s_summary.json", out_dir, stem)

    # Per-label random sample (p >= floor), collected DURING the reference pass by the reservoir.
    # Feeds the distribution-comparison viewer's drill-down.
    example_rows = reservoir.rows(dataset)
    # Labels the model predicted, but never confidently enough to sample — worth seeing, not hiding.
    starved = [name for i, name in enumerate(label_names) if reservoir.seen[i] > 0 and reservoir.eligible[i] == 0]
    logger.info(
        "EXAMPLES: %d rows across %d labels (<=%d each, p>=%.2f)",
        len(example_rows),
        sum(1 for b in reservoir.kept.values() if b),
        examples_per_label,
        example_min_prob,
    )
    if starved:
        logger.info("  labels with NO doc above the p>=%.2f floor: %s", example_min_prob, ", ".join(starved))
    with fsspec.open(f"{out_dir}/{stem}_examples.parquet", "wb") as fh:
        pq.write_table(pa.Table.from_pylist(example_rows), fh)
    logger.info("wrote %d examples -> %s/%s_examples.parquet", len(example_rows), out_dir, stem)

    # Golden logits = the parity oracle for the Levanter/JAX port.
    table = pa.table(
        {
            "url": [d.url for d in docs],
            "text": [d.text for d in docs],
            "token_length": lengths.tolist(),
            "logits": [row.tolist() for row in reference],
            "choice": ref_choice.tolist(),
            "label": [label_names[i] for i in ref_choice],
        }
    )
    with fsspec.open(f"{out_dir}/{stem}_golden.parquet", "wb") as fh:
        pq.write_table(table, fh)
    logger.info("wrote %d golden rows -> %s/%s_golden.parquet", len(docs), out_dir, stem)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    probe = sub.add_parser("probe", help="Score a real sample at several context lengths; emit golden logits.")
    probe.add_argument("--dataset", required=True, choices=list(CORPORA))
    probe.add_argument("--limit", type=int, default=300, help="Docs to sample from the first shard.")
    probe.add_argument("--batch-size", type=int, default=8, help="CPU batch size.")
    probe.add_argument("--no-url", action="store_true", help="Use the -NoURL model on bare text.")
    probe.add_argument(
        "--grid",
        type=int,
        nargs="+",
        default=list(TRUNCATION_GRID),
        help=f"Context lengths to compare; must include {REFERENCE_MAX_LENGTH}.",
    )
    probe.add_argument(
        "--examples-per-label",
        type=int,
        default=EXAMPLES_PER_LABEL,
        help="Random docs to keep per predicted label (reservoir-sampled during the run).",
    )
    probe.add_argument(
        "--example-min-prob",
        type=float,
        default=EXAMPLE_MIN_PROB,
        help="Only sample docs whose predicted probability clears this floor.",
    )
    args = parser.parse_args()

    grid = tuple(sorted(set(args.grid)))
    if REFERENCE_MAX_LENGTH not in grid:
        raise ValueError(f"--grid must include the reference length {REFERENCE_MAX_LENGTH}")
    if not 0.0 <= args.example_min_prob <= 1.0:
        raise ValueError(f"--example-min-prob must be in [0,1], got {args.example_min_prob}")
    run_probe(
        args.dataset,
        args.limit,
        args.batch_size,
        not args.no_url,
        grid,
        args.examples_per_label,
        args.example_min_prob,
    )


if __name__ == "__main__":
    main()
