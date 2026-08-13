# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label a random sample of each curated 10k corpus with WebOrganizer topic domains (24-way), on TPU.

Produces the two artifacts the distribution viewer needs, per (dataset, chunk):
  * ``counts``   — per-label document counts → the distribution comparison across datasets
  * ``examples`` — a per-label uniform random sample of docs (p >= floor) → the drill-down

Runs the JAX port (`weborganizer_gte_jax`, parity-tested against HF) rather than the reference HF
PyTorch path, whose fast kernels are CUDA-only: on CPU an 8k-token doc scores at 0.03 docs/s, which
makes millions of docs impossible. bf16 on TPU matches the reference, which also runs bf16
(WebOrganizer `annotate_data/domains.py` does `model.to(torch.bfloat16)`).

**Sequence length.** `--max-length` defaults to 8192, the length WebOrganizer trains and annotates at,
so labels match the published model's behaviour. That costs nothing extra for typical docs because of:

**Length bucketing.** Padding a whole batch to 8192 would make every short doc cost as much as the
longest one, and gte-base has FULL global attention at every layer (no local window), so that cost is
quadratic. Instead docs are sorted by token length and each batch is padded only to the next bucket
(128…8192), with the batch size scaled so tokens-per-batch stays ~constant. Seven bucket shapes means
XLA compiles at most seven programs, not one per length.

**Sampling.** Shards are randomly permuted (fixed seed) and dealt round-robin to chunks, then rows are
taken with a stride within each shard — random at the WARC/shard level, which is the right
granularity (a shard is one crawl for nemotron, one WARC for resiliparse).

Per-chunk outputs are independent, each chunk writes a done-marker, so partial progress is usable and
reruns skip finished chunks. Fan out one Iris job per chunk, in the corpus's own region::

    uv run iris --cluster marin job run --region us-central2 --tpu v6e-4 \\
      --enable-extra-resources --extra tpu --memory 64GB --priority interactive --no-wait \\
      --job-name wo-label-dclm-000 -e HF_TOKEN hf_... -- \\
      python -m experiments.baseline_collection.weborganizer_topic_label chunk \\
        --dataset dclm_10k --target-docs 1000000 --num-chunks 8 --chunk-idx 0

Then merge (cheap, CPU) into the viewer's inputs::

    python -m experiments.baseline_collection.weborganizer_topic_label merge --dataset dclm_10k
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.baseline_collection.weborganizer_gte_jax import (
    NUM_HEADS,
    QUERY_BLOCK,
    extract_params,
    forward,
    to_device,
)
from experiments.baseline_collection.weborganizer_topic_smoke import (
    CORPORA,
    EXAMPLE_MIN_PROB,
    EXAMPLES_PER_LABEL,
    NOURL_MODEL,
    URL_MODEL,
    Doc,
    Format,
    LabelReservoir,
    _iter_jsonl_gz,
    _iter_parquet,
    _render,
    _softmax,
    load_model,
)
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

OUT_ROOT = "documents/weborganizer_topic/labels_v1"
# The 10k document corpora (the only place URLs exist) are NOT mirrored out of us-central2, whose only
# TPU family is the reserved/preemptible v4 — and v4 is frequently dry, which traps the four corpora
# that live there. The `mirror` stage samples IN us-central2 (cheap, in-region) and ships only the
# SAMPLED docs to a compute-rich region, so ~3 KB/doc x 1M docs ~= 3 GB/corpus crosses instead of the
# whole corpus. At US inter-region egress that is cents per corpus, and it unlocks v6e.
MIRROR_ROOT = "documents/weborganizer_topic/mirror_v1"
MIRROR_REGION = "us-east5"
DEFAULT_MAX_LENGTH = 8192  # what WebOrganizer trains + annotates at
BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)
TOKENS_PER_BATCH = 262_144  # per-batch token budget; batch size = budget // bucket
# Cap on live attention-score elements PER DEVICE (~1.6 GB in fp32). Attention is quadratic in
# sequence length, so this — not the token budget — is what binds at the long buckets.
ATTN_ELEMS_PER_DEVICE = 400_000_000
# Take every row of the slice we read from each shard. A stride >1 caps the reachable sample at
# corpus_size/stride, which silently truncated fineweb_edu (2.35M docs) to 336k against a 1M target.
# It also buys nothing now that `read_chunk` gets its uniformity by spanning EVERY assigned shard
# rather than by skipping rows — and gzip has no cheap seek, so skipping a row costs the same
# decompression as reading it. Shard-level coverage is what decorrelates the sample; the stride just
# spent I/O and capped N.
ROW_STRIDE = 1
SHARD_PERMUTATION_SEED = 0
PAD_TOKEN_ID = 0  # gte config: pad_token_id=0
# Manifest corpora only: a batch file holds ~130 docs, so an uncapped shard list means one doc per
# file and hundreds of thousands of GCS opens. 6000 batches x ~42 docs still covers ~every WARC.
MAX_SHARDS_PER_CHUNK = 6000
# Hard cap on stored text per doc. The model only ever sees the first `max_length` (8192) tokens, so
# keeping the smoke's 1,000,000-char cap just holds megabytes per doc that get truncated away anyway —
# ruinous for resiliparse, whose raw HTML->text pages are enormous. 64k chars is >=16k tokens even at
# a pessimistic 4 chars/token, so it cannot clip anything the model would have read.
LABEL_TEXT_CHARS = 64_000

# True size of each FULL corpus, from curation_plan.METHODS (the tokenized Levanter caches' own
# .stats.json). We label a ~1M-doc sample of each, so these are what let the viewer show corpora "to
# scale" against one another — resiliparse is ~145x fineweb_edu, which no normalised view can convey.
#
# Caveat worth carrying: these are the counts of the corpus AS TRAINED. For high_quality and
# resiliparse that is post dedup+decontam, while we label the PRE-dedup document layer (the only place
# URLs survive). So absolute mass for those two is "what a training run would see", which is the
# number that matters — but it is not literally the doc layer we sampled.
CORPUS_TOTAL_TOKENS: dict[str, int] = {
    "dclm_10k": 7_331_583_927,
    "nemotron_full_10k": 10_130_086_896,
    "fineweb_edu_10k": 2_346_934_380,
    "high_quality_10k": 21_296_896_949,
    "resiliparse_10k": 339_971_302_028,
    # 3k-pool LLM-extraction bands. NOTE the N these were registered at differs — med_quality has a
    # 3000-WARC entry, low_quality tops out at 2000, med_low_quality exists only at 100 — so their
    # absolute masses are NOT on a common footing with each other or with the 10k corpora.
    "med_quality_3k": 26_930_000_000,  # med_quality_3000warcs-73cd32
    "low_quality_3k": 28_850_000_000,  # low_quality @2000warcs (no 3000W entry registered)
    "med_low_quality_3k": 1_450_000_000,  # med_low_quality @100warcs ONLY
}


def manifest_shards(corpus) -> list[str]:
    """Shard paths for an LLM-extraction band, from its `resolved_{spec}.jsonl.gz` index.

    Globbing is not an option here: the archive is ~240k tiny per-WARC batch files under five region
    subtrees. The manifest is one file listing {region, warc_hash, batch_idx}, from which the archive
    path is derived. `low_quality` predates the spec registry and lives at the UNPREFIXED path, hence
    the conditional segment.

    Batches with zero records are dropped, mirroring dedup_extracted.py.
    """
    rows: list[tuple[str, str, int]] = []
    with fsspec.open(corpus.manifest, "rb", compression="gzip") as fh:
        for line in fh:  # type: ignore[union-attr]
            row = json.loads(line)
            if not int(row.get("num_records", 0)):
                continue
            rows.append((row["region"], row["warc_hash"], int(row["batch_idx"])))
    seg = f"{corpus.spec}/" if corpus.spec else ""
    shards = [f"{corpus.path}/{r}/{seg}data-{h}/batch_{b:04d}.jsonl.gz" for r, h, b in rows]
    warcs = len({h for _, h, _ in rows})
    logger.info("manifest %s: %d batches across %d WARCs", corpus.spec or "low_quality(legacy)", len(shards), warcs)
    return sorted(shards)


def chunk_shards(corpus, num_chunks: int, chunk_idx: int) -> list[str]:
    """Randomly permute the corpus's shards (fixed seed) and deal shard i to chunk i % num_chunks.

    For manifest corpora the shard list is capped: each batch file holds only ~130 docs, so keeping
    all ~240k would mean one doc per file and 240k GCS opens per chunk. A random subsample of the
    batches still lands across essentially every WARC (batches are spread over WARCs), so coverage
    survives while the open count stays sane.
    """
    if corpus.manifest:
        shards = manifest_shards(corpus)
    else:
        shards = sorted(fsspec_glob(f"{corpus.path}/*.{corpus.format.value}"))
    if not shards:
        raise ValueError(f"no shards for {corpus.path} (manifest={corpus.manifest}, spec={corpus.spec})")
    rng = np.random.default_rng(SHARD_PERMUTATION_SEED)
    order = rng.permutation(len(shards))
    mine = [shards[i] for i in order[chunk_idx::num_chunks]]
    if not mine:
        # More chunks than shards: this chunk owns nothing. Say so plainly — the old behaviour was to
        # hand an empty list to read_chunk and die in `quota // len(shards)` with a bare
        # ZeroDivisionError that says nothing about the real mistake.
        raise ValueError(
            f"chunk {chunk_idx}/{num_chunks} was dealt 0 of {len(shards)} shards — "
            f"--num-chunks must be <= the shard count ({len(shards)})"
        )
    if corpus.manifest and len(mine) > MAX_SHARDS_PER_CHUNK:
        keep = np.random.default_rng(SHARD_PERMUTATION_SEED + chunk_idx).choice(
            len(mine), size=MAX_SHARDS_PER_CHUNK, replace=False
        )
        mine = [mine[i] for i in sorted(keep.tolist())]
        logger.info("capped shard list to %d batches for this chunk", len(mine))
    return mine


def read_chunk(corpus, shards: list[str], quota: int, stride: int = ROW_STRIDE) -> list[Doc]:
    """Take an EQUAL slice of `quota` from EVERY assigned shard, striding rows within each.

    Draining the quota from the first few shards and stopping would be much cheaper to write and
    badly biased: a shard here is one crawl (nemotron) or one WARC (resiliparse), so a quota filled
    from the head of the shard list is a sample of a handful of WARCs — correlated by crawl date and
    domain — wearing a big-N costume. resiliparse was the worst case: 250k docs would come from ~53
    of 2,591 assigned WARCs (2%).

    Spreading `quota / len(shards)` across all of them costs **the same I/O** (same rows read, just a
    few from many shards instead of many from few) while spanning every shard the chunk owns.

    Residual bias, stated plainly: within a shard we read from the start, so this takes roughly the
    first `per_shard * stride` records of each. That is a mild intra-shard position bias, spread over
    thousands of shards — far weaker than whole-shard clustering. Falling short of quota is logged
    rather than hidden; a uniform 84% of the target beats a clustered 100%.
    """
    reader = _iter_jsonl_gz if corpus.format is Format.JSONL_GZ else _iter_parquet
    per_shard = max(1, -(-quota // len(shards)))  # ceil
    docs: list[Doc] = []
    logger.info("balanced read: %d shards x <=%d docs (stride %d) -> quota %d", len(shards), per_shard, stride, quota)
    for n_shards, shard in enumerate(shards, start=1):
        taken = 0
        for row, doc in enumerate(reader(shard, corpus.text_field, corpus.url_field)):
            if row % stride:
                continue
            docs.append(Doc(url=doc.url, text=doc.text[:LABEL_TEXT_CHARS]))
            taken += 1
            if taken >= per_shard or len(docs) >= quota:
                break
        if len(docs) >= quota:
            logger.info("quota met: %d docs spanning %d shards", len(docs), n_shards)
            return docs
        if n_shards % 200 == 0:
            logger.info("  %d docs after %d/%d shards", len(docs), n_shards, len(shards))
    logger.info(
        "read %d/%d docs spanning ALL %d shards (%.0f%% of quota; shards ran dry, sample still uniform)",
        len(docs),
        quota,
        len(shards),
        100 * len(docs) / quota,
    )
    return docs


def mirrored_corpus(dataset: str):
    """The mirror of `dataset` in MIRROR_REGION: already-sampled `{url,text}` parquet."""
    from experiments.baseline_collection.weborganizer_topic_smoke import Corpus

    return Corpus(f"gs://marin-{MIRROR_REGION}/{MIRROR_ROOT}/{dataset}", MIRROR_REGION, Format.PARQUET)


def run_mirror(dataset: str, target_docs: int, num_chunks: int, chunk_idx: int) -> None:
    """Sample `target_docs/num_chunks` docs IN the corpus's own region and ship them to MIRROR_REGION.

    This is the escape hatch for corpora trapped in a TPU-poor region: us-central2 holds the only
    URL-bearing copy of dclm/nemotron/fineweb_edu/resiliparse and its only TPU family is v4, which is
    often dry. Reading those shards from a us-east5 job would drag the WHOLE corpus across; sampling
    here first means only the ~3 KB/doc we actually score crosses the wire.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = CORPORA[dataset]
    out_path = f"gs://marin-{MIRROR_REGION}/{MIRROR_ROOT}/{dataset}/part-{chunk_idx:04d}-of-{num_chunks:04d}.parquet"
    fs, _ = fsspec.core.url_to_fs(out_path)
    if fs.exists(out_path):
        logger.info("mirror shard already present -> %s", out_path)
        return

    shards = chunk_shards(corpus, num_chunks, chunk_idx)
    docs = read_chunk(corpus, shards, max(1, target_docs // num_chunks))
    if not docs:
        raise RuntimeError(f"mirror chunk {chunk_idx} read zero docs")
    table = pa.table({"url": [d.url for d in docs], "text": [d.text for d in docs]})
    with fsspec.open(out_path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info("mirrored %d docs (%s -> %s) -> %s", len(docs), corpus.region, MIRROR_REGION, out_path)


def _bucket_for(length: int, max_length: int) -> int:
    for bucket in BUCKETS:
        if bucket >= length:
            return min(bucket, max_length)
    return max_length


TOKENIZE_BATCH = 2_000


def tokenize_all(tokenizer, pages: list[str], max_length: int) -> list[np.ndarray]:
    """Tokenize every page, returning int32 arrays.

    Two things here are load-bearing for memory, and getting either wrong OOM-kills the container:

    1. **int32 arrays, not Python lists.** A Python int costs ~28 bytes plus a list slot, so holding
       250k docs x 8192 tokens as lists is ~57 GB — which is exactly how the first dclm/fineweb/
       resiliparse runs died (exit 137). As int32 the same data is ~8 GB, and typical docs are far
       shorter than 8192 so it lands well under that.
    2. **Tokenize in slices.** One `tokenizer(pages)` call over 250k docs materialises every Python
       list at once before we can convert, so the peak is the same 57 GB even if the result is small.
       Slicing caps the transient at TOKENIZE_BATCH docs.
    """
    out: list[np.ndarray] = []
    for start in range(0, len(pages), TOKENIZE_BATCH):
        encoded = tokenizer(pages[start : start + TOKENIZE_BATCH], truncation=True, max_length=max_length)
        out.extend(np.asarray(ids, dtype=np.int32) for ids in encoded["input_ids"])
    return out


def batch_size_for(bucket: int, n_devices: int) -> int:
    """Docs per batch at this bucket: bounded by BOTH a token budget and attention memory.

    Attention cost is quadratic in sequence length, so a token-only budget silently explodes at long
    buckets (at T=8192 a 32-doc batch needs ~103 GB of scores and OOMs). Peak live attention memory
    per device is `per_device * heads * QUERY_BLOCK * T`, so bound that too. Always a multiple of
    n_devices, so the batch shards evenly across chips.
    """
    by_tokens = (TOKENS_PER_BATCH // bucket) // n_devices
    by_attention = ATTN_ELEMS_PER_DEVICE // (NUM_HEADS * min(QUERY_BLOCK, bucket) * bucket)
    per_device = max(1, min(by_tokens, by_attention))
    return per_device * n_devices


def score_bucketed(params, tokenizer, pages: list[str], max_length: int, mesh, on_batch):
    """Score every page, bucketing by token length. Returns (logits, token_lengths) in ORIGINAL order.

    Token lengths come back because they are already computed here for bucketing, and the downstream
    "to scale" comparison needs topic mass in TOKENS, not documents — a corpus's code pages are far
    longer than its average page, so doc-share x corpus-tokens would misstate it badly.

    Batches are sharded over the chips on a 1-D `data` mesh (params replicated -- the model is 137M,
    it fits anywhere). Every batch is padded to a fixed per-bucket size so XLA compiles at most one
    program per bucket rather than one per ragged tail.

    `on_batch(indices, logits)` fires per batch with the ORIGINAL indices of those pages, so callers
    can collect streaming side-outputs without caring about the bucketing reorder.
    """
    n_devices = mesh.devices.size
    batch_shard = NamedSharding(mesh, P("data", None))
    jitted = jax.jit(forward)

    encodings = tokenize_all(tokenizer, pages, max_length)
    lengths = np.array([len(e) for e in encodings])
    buckets = np.array([_bucket_for(n, max_length) for n in lengths])

    num_labels = params["classifier_b"].shape[0]
    out = np.zeros((len(pages), num_labels), dtype=np.float32)

    for bucket in sorted(set(buckets.tolist())):
        members = np.flatnonzero(buckets == bucket)
        per_batch = batch_size_for(bucket, n_devices)
        logger.info(
            "  bucket %5d: %7d docs, batch=%d (%d/chip)", bucket, len(members), per_batch, per_batch // n_devices
        )
        for start in range(0, len(members), per_batch):
            idx = members[start : start + per_batch]
            # Pad to the FULL per_batch (not just a device multiple) so the shape stays static.
            ids = np.full((per_batch, bucket), PAD_TOKEN_ID, dtype=np.int32)
            mask = np.zeros((per_batch, bucket), dtype=np.int32)
            for row, i in enumerate(idx):
                tokens = encodings[i][:bucket]
                ids[row, : len(tokens)] = tokens
                mask[row, : len(tokens)] = 1
            logits = np.asarray(
                jitted(params, jax.device_put(ids, batch_shard), jax.device_put(mask, batch_shard)),
                dtype=np.float32,
            )[: len(idx)]
            out[idx] = logits
            if on_batch is not None:
                on_batch(idx, logits)
    return out, lengths


def run_chunk(
    dataset: str,
    target_docs: int,
    num_chunks: int,
    chunk_idx: int,
    max_length: int,
    use_url: bool,
    examples_per_label: int,
    example_min_prob: float,
    source: str = "native",
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Reading the mirror means the docs are ALREADY sampled, so take every row (stride 1) and let the
    # outputs land in the mirror region alongside the compute.
    from_mirror = source == "mirror"
    corpus = mirrored_corpus(dataset) if from_mirror else CORPORA[dataset]
    stride = 1 if from_mirror else ROW_STRIDE
    out_dir = f"gs://marin-{corpus.region}/{OUT_ROOT}/{dataset}"
    stem = f"chunk-{chunk_idx:04d}-of-{num_chunks:04d}"
    done_marker = f"{out_dir}/done/{stem}"
    fs, _ = fsspec.core.url_to_fs(done_marker)
    if fs.exists(done_marker):
        logger.info("chunk %d already done -> %s", chunk_idx, done_marker)
        return

    n_devices = jax.device_count()
    logger.info("chunk %d/%d on %d devices: %s", chunk_idx, num_chunks, n_devices, jax.devices())

    quota = max(1, target_docs // num_chunks)
    shards = chunk_shards(corpus, num_chunks, chunk_idx)
    t_read = time.monotonic()
    docs = read_chunk(corpus, shards, quota, stride=stride)
    logger.info("read %d docs in %.1fs", len(docs), time.monotonic() - t_read)
    if not docs:
        raise RuntimeError(f"chunk {chunk_idx} read zero docs")

    model_name = URL_MODEL if use_url else NOURL_MODEL
    config, tokenizer, hf_model = load_model(model_name)
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    params = to_device(extract_params(hf_model), jnp.bfloat16)  # reference also runs bf16
    del hf_model

    # Replicate the (137M) params across chips once, so each sharded batch finds them locally
    # instead of JAX broadcasting them on every call.
    mesh = Mesh(np.array(jax.devices()), ("data",))
    params = jax.device_put(params, NamedSharding(mesh, P()))

    pages = _render(docs, use_url)
    reservoir = LabelReservoir(label_names, examples_per_label, example_min_prob, seed=chunk_idx)

    def _collect(indices: np.ndarray, logits: np.ndarray) -> None:
        reservoir.update([docs[i] for i in indices], _softmax(logits))

    t_score = time.monotonic()
    logits, token_lengths = score_bucketed(params, tokenizer, pages, max_length, mesh, _collect)
    secs = time.monotonic() - t_score
    docs_per_sec = len(pages) / secs if secs else 0.0
    logger.info(
        "SCORED %d docs in %.1fs = %.2f docs/s = %.2f docs/s/chip (max_length=%d)",
        len(pages),
        secs,
        docs_per_sec,
        docs_per_sec / n_devices,
        max_length,
    )

    choice = logits.argmax(-1)
    counts = np.bincount(choice, minlength=config.num_labels)
    # Topic mass in TOKENS, not documents. This is the unit that matters downstream: training consumes
    # tokens, and doc length varies a lot BY topic (a code page dwarfs a forum post), so doc-share is
    # not a stand-in. Truncated at max_length, same as the model sees.
    token_counts = np.bincount(choice, weights=token_lengths, minlength=config.num_labels)
    probs = _softmax(logits)

    summary = {
        "dataset": dataset,
        "chunk_idx": chunk_idx,
        "num_chunks": num_chunks,
        "model": model_name,
        "max_length": max_length,
        "n_docs": len(docs),
        "n_shards_assigned": len(shards),
        "docs_per_sec": docs_per_sec,
        "docs_per_sec_per_chip": docs_per_sec / n_devices,
        "score_seconds": secs,
        "n_tokens": int(token_lengths.sum()),
        "counts": {label_names[i]: int(counts[i]) for i in range(config.num_labels)},
        "token_counts": {label_names[i]: int(token_counts[i]) for i in range(config.num_labels)},
    }
    with fsspec.open(f"{out_dir}/counts/{stem}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with fsspec.open(f"{out_dir}/examples/{stem}.parquet", "wb") as fh:
        pq.write_table(pa.Table.from_pylist(reservoir.rows(dataset)), fh)
    # Per-doc labels, so the sample can be re-analysed or joined back without re-scoring.
    with fsspec.open(f"{out_dir}/docs/{stem}.parquet", "wb") as fh:
        pq.write_table(
            pa.table(
                {
                    "url": [d.url for d in docs],
                    "label": [label_names[i] for i in choice],
                    "prob": probs.max(-1).tolist(),
                    "token_length": token_lengths.tolist(),
                }
            ),
            fh,
        )
    with fsspec.open(done_marker, "w") as fh:
        fh.write(str(len(docs)))
    logger.info("chunk %d DONE -> %s", chunk_idx, out_dir)


def run_merge(dataset: str, examples_per_label: int, source: str = "native") -> None:
    """Merge finished chunks into the viewer's two inputs: a distribution JSON and an examples parquet.

    `source` must match what `chunk` ran with: a mirror-sourced run writes its outputs beside the
    mirror (us-east5), not in the corpus's native region, so merging with the wrong one looks in an
    empty directory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = mirrored_corpus(dataset) if source == "mirror" else CORPORA[dataset]
    out_dir = f"gs://marin-{corpus.region}/{OUT_ROOT}/{dataset}"

    count_files = sorted(fsspec_glob(f"{out_dir}/counts/*.json"))
    if not count_files:
        raise RuntimeError(f"no finished chunks under {out_dir}/counts/")
    totals: dict[str, int] = {}
    token_totals: dict[str, int] = {}
    n_docs = 0
    n_tokens = 0
    for path in count_files:
        with fsspec.open(path) as fh:
            payload = json.load(fh)
        n_docs += payload["n_docs"]
        n_tokens += payload.get("n_tokens", 0)
        for label, count in payload["counts"].items():
            totals[label] = totals.get(label, 0) + count
        for label, count in payload.get("token_counts", {}).items():
            token_totals[label] = token_totals.get(label, 0) + count

    with fsspec.open(f"{out_dir}/distribution.json", "w") as fh:
        json.dump(
            {
                "dataset": dataset,
                "n_docs": n_docs,
                "n_tokens": n_tokens,
                "n_chunks": len(count_files),
                "counts": totals,
                "shares": {label: count / n_docs for label, count in totals.items()},
                # Token share is the training-relevant mix; doc share weights a one-line page the same
                # as a 8k-token manual.
                "token_counts": token_totals,
                "token_shares": {label: count / n_tokens for label, count in token_totals.items()} if n_tokens else {},
                # Corpus totals let the viewer scale a 1M-doc SAMPLE up to the real corpus, so corpora
                # of wildly different size can be compared in absolute terms.
                "corpus_total_tokens": CORPUS_TOTAL_TOKENS.get(dataset),
            },
            fh,
            indent=2,
        )
    logger.info("merged %d chunks, %d docs -> %s/distribution.json", len(count_files), n_docs, out_dir)
    for label in sorted(totals, key=lambda x: -totals[x]):
        if totals[label]:
            logger.info("  %-22s %8d  (%5.2f%%)", label, totals[label], 100 * totals[label] / n_docs)

    # Each chunk's reservoir is a uniform draw over its own equal-quota slice, so a uniform draw over
    # the pooled rows is a uniform draw over the corpus.
    example_files = sorted(fsspec_glob(f"{out_dir}/examples/*.parquet"))
    pooled = pa.concat_tables([pq.read_table(f) for f in example_files]).to_pylist()
    by_label: dict[str, list[dict]] = {}
    for row in pooled:
        by_label.setdefault(row["label"], []).append(row)
    rng = np.random.default_rng(0)
    merged: list[dict] = []
    for rows in by_label.values():
        idx = rng.choice(len(rows), size=min(examples_per_label, len(rows)), replace=False)
        merged.extend(rows[i] for i in sorted(idx.tolist()))
    with fsspec.open(f"{out_dir}/examples.parquet", "wb") as fh:
        pq.write_table(pa.Table.from_pylist(merged), fh)
    logger.info("merged examples: %d rows across %d labels -> %s/examples.parquet", len(merged), len(by_label), out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "stage",
        choices=["chunk", "merge", "mirror"],
        help="Score one chunk, merge finished chunks, or mirror a sample to a compute-rich region.",
    )
    parser.add_argument("--dataset", required=True, choices=list(CORPORA))
    parser.add_argument("--target-docs", type=int, default=1_000_000, help="Docs to sample across ALL chunks.")
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--no-url", action="store_true", help="Use the -NoURL model on bare text.")
    parser.add_argument("--examples-per-label", type=int, default=EXAMPLES_PER_LABEL)
    parser.add_argument("--example-min-prob", type=float, default=EXAMPLE_MIN_PROB)
    parser.add_argument(
        "--source",
        choices=["native", "mirror"],
        default="native",
        help="Read the corpus in its own region, or the pre-sampled mirror in a compute-rich region.",
    )
    args = parser.parse_args()

    if args.stage == "merge":
        run_merge(args.dataset, args.examples_per_label, args.source)
        return
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError(f"--chunk-idx {args.chunk_idx} out of range for --num-chunks {args.num_chunks}")
    if args.stage == "mirror":
        run_mirror(args.dataset, args.target_docs, args.num_chunks, args.chunk_idx)
        return
    run_chunk(
        args.dataset,
        args.target_docs,
        args.num_chunks,
        args.chunk_idx,
        args.max_length,
        not args.no_url,
        args.examples_per_label,
        args.example_min_prob,
        args.source,
    )


if __name__ == "__main__":
    main()
