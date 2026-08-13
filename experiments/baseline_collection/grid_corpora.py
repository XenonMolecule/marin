# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus registry and shard plumbing for the quality x domain grid.

Deliberately free of JAX and torch imports so the mirror stage — and a CPU-only
quality catch-up run — can import this without pulling an accelerator stack in.

The grid labels every document of a corpus, so unlike
:mod:`experiments.baseline_collection.weborganizer_topic_smoke` (which samples a
quota per shard) shard discovery here is exhaustive and the mapping from input
shard to output file is **1:1 on basename**. That is not cosmetic: the datakit
store joins tokenize/decon/cluster/quality positionally and hard-fails on a
row-count or id mismatch, so a shard that gets split or reordered anywhere in
this pipeline silently breaks everything downstream of it.
"""

from __future__ import annotations

import enum
import gzip
import io
import json
import logging
import posixpath
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

# Where the mirrored source shards and the two attribute tables live. Everything
# is written under one region so the scorer never reads across regions.
COMPUTE_REGION = "us-central1"
# Every OUTPUT path is built from this, so a test (or a smoke run via
# --output-base) can redirect the whole result tree by patching one name. fsspec
# treats a local path the same as a gs:// one, so nothing else has to change.
OUTPUT_BASE = f"gs://marin-{COMPUTE_REGION}"
# Deliberately separate from OUTPUT_BASE. The mirror is an INPUT, and folding it
# into the output root means redirecting results also redirects where the scorer
# looks for source shards — which silently turns a smoke run into "no shards
# found" against an empty scratch path.
MIRROR_BASE = f"gs://marin-{COMPUTE_REGION}"
MIRROR_ROOT = "mirror/grid_v1"
QUALITY_ROOT = "datakit/quality"
TOPIC_ROOT = "datakit/cluster_assign"
METADATA_ROOT = "metadata/grid_v1"
GRID_SUFFIX = "gridv1"

# WebOrganizer's taxonomy size. Named `cluster_24` in the output to match the
# upstream `cluster_<K>` column convention, so an AssignmentAttrData built over
# it validates against datakit_store._validate_cluster_view without special-casing.
NUM_TOPICS = 24

# Canonical for the grid pipeline, and deliberately here rather than imported
# from weborganizer_topic_label: that module pulls torch at import time, and a
# quality-only run must not need it. Values match the proven topic runs.
DEFAULT_MAX_LENGTH = 8192  # what WebOrganizer trains and annotates at
# The model reads at most DEFAULT_MAX_LENGTH tokens, so a larger cap would only
# hold megabytes per doc that get truncated away — ruinous on resiliparse, whose
# HTML-to-text pages run to megabytes. 64k chars is >=16k tokens even at a
# pessimistic 4 chars/token, so it cannot clip anything the model would read.
LABEL_TEXT_CHARS = 64_000


class Format(enum.StrEnum):
    JSONL_GZ = "jsonl.gz"
    PARQUET = "parquet"


class Stage(enum.StrEnum):
    """The two independently runnable scoring stages."""

    TOPIC = "topic"
    QUALITY = "quality"


@dataclass(frozen=True)
class GridCorpus:
    """A corpus at the document layer we label, with its provenance columns.

    Attributes:
        path: Directory of shards, either native or (after mirroring) local.
        region: Region the ``path`` lives in.
        format: Shard encoding.
        native_id_field: Corpus-specific stable id carried through to the output
            alongside the content-hash ``id``. ``None`` where the corpus has none.
        post_decon: Whether ``path`` is already the post-dedup/decon survivor set.
            Recorded so a run can never quietly mislabel which layer it scored.
    """

    path: str
    region: str
    format: Format
    native_id_field: str | None = None
    post_decon: bool = False


# The document layer for each corpus: the one that carries BOTH url and text.
# Topic classification needs the url ("{url}\n\n{text}" is WebOrganizer's input
# template), and every `_deduped`/`_decon` tree except high_quality's is url-less
# because dedup_extracted.py projects records to {text}.
#
# high_quality is the exception and the reason it is the only post_decon=True
# entry: build_high_quality_hf_export.py re-joined the decon_deduped survivors
# against the raw extraction batches to restore url/warc_record_id, so this path
# IS the post-dedup+decon corpus, with 99.991% of 19,968,996 docs carrying full
# WARC metadata.
GRID_CORPORA: dict[str, GridCorpus] = {
    "dclm_10k": GridCorpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977",
        "us-central2",
        Format.JSONL_GZ,
        native_id_field="warc_record_id",
    ),
    "high_quality_10k": GridCorpus(
        "gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined",
        "us-central1",
        Format.PARQUET,
        native_id_field="warc_record_id",
        post_decon=True,
    ),
    "nemotron_full_10k": GridCorpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9",
        "us-central2",
        Format.JSONL_GZ,
        native_id_field="nemotron_id",
    ),
    "fineweb_edu_10k": GridCorpus(
        "gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9",
        "us-central2",
        Format.JSONL_GZ,
    ),
    # Landed 2026-07-29. Sharded by (crawl, part) rather than by WARC, so it has
    # 21,531 shards of which **13,541 (63%) are empty** — that combination simply
    # yielded no surviving documents. Empty shards are normal here, not a sign of
    # a partial write, and each still produces a zero-row output file to keep the
    # co-partitioning contract intact.
    "fineweb_cc_10k": GridCorpus(
        "gs://marin-us-central2/documents/baseline_fineweb_cc/10364warcs",
        "us-central2",
        Format.JSONL_GZ,
        native_id_field="id",
    ),
    # Reconstructed post-decon tree. The original was deleted after tokenization,
    # so it was rebuilt from the raw extraction by recovering the surviving
    # document set out of `tokenized/resiliparse_decon_10364warcs-beaaf5` — see
    # `recover_tokenized_text.py` and `.agents/projects/resiliparse_grid_url_recovery.md`.
    # Text is copied verbatim from raw (never from a decode); only the *identity*
    # of the survivors came from the cache. No native id: the raw extraction
    # carries `{text, url}` and nothing else.
    # Lives in us-east5, unlike the other five: it is the only corpus whose grid is
    # built outside COMPUTE_REGION, so its labelling runs need
    # `--region us-east5 --output-base gs://marin-us-east5 --source native`.
    # us-east5 is where the mixing swarm and the other five stores already sit, and
    # it is the only region with the v5p-8 shape the topic stage needs (us-central2,
    # where the tree was rebuilt, is v4-only).
    "resiliparse_10k": GridCorpus(
        "gs://marin-us-east5/documents/baseline_resiliparse_decon_deduped_urls/10364warcs/deduped",
        "us-east5",
        Format.JSONL_GZ,
        post_decon=True,
    ),
}


def mirrored(dataset: str) -> GridCorpus:
    """The in-compute-region mirror of ``dataset``, same format and columns."""
    corpus = GRID_CORPORA[dataset]
    return GridCorpus(
        f"{MIRROR_BASE}/{MIRROR_ROOT}/{dataset}",
        COMPUTE_REGION,
        corpus.format,
        native_id_field=corpus.native_id_field,
        post_decon=corpus.post_decon,
    )


def resolve(dataset: str, source: str) -> GridCorpus:
    """Pick the native or mirrored view of ``dataset``.

    Raises:
        ValueError: If ``source`` is not ``"native"`` or ``"mirror"``.
    """
    if source == "native":
        return GRID_CORPORA[dataset]
    if source == "mirror":
        return mirrored(dataset)
    raise ValueError(f"unknown source {source!r}; expected 'native' or 'mirror'")


def list_shards(corpus: GridCorpus) -> list[str]:
    """Every shard of ``corpus``, sorted so assignment is reproducible.

    Raises:
        ValueError: If the directory holds no shards, which almost always means
            a stale hashed path rather than a genuinely empty corpus.
    """
    shards = sorted(fsspec_glob(f"{corpus.path}/*.{corpus.format.value}"))
    if not shards:
        raise ValueError(f"no *.{corpus.format.value} shards under {corpus.path}")
    return shards


def assign_shards(shards: list[str], num_chunks: int, chunk_idx: int) -> list[str]:
    """Deal shards round-robin to chunks.

    Round-robin rather than contiguous blocks because shard size correlates with
    position for the crawl-ordered corpora (nemotron's shards are one crawl
    each), so contiguous blocks would hand one chunk all the big shards.

    Raises:
        ValueError: If this chunk is dealt nothing, i.e. ``num_chunks`` exceeds
            the shard count. The old code died in a bare ZeroDivisionError here,
            which named nothing.
    """
    mine = shards[chunk_idx::num_chunks]
    if not mine:
        raise ValueError(
            f"chunk {chunk_idx}/{num_chunks} was dealt 0 of {len(shards)} shards — "
            f"--num-chunks must be <= the shard count ({len(shards)})"
        )
    return mine


def iter_records(shard: str, shard_format: Format) -> Iterator[dict]:
    """Yield every record of ``shard`` in file order.

    File order is the co-partitioning contract: the output row at position i must
    correspond to the input row at position i.
    """
    if shard_format is Format.JSONL_GZ:
        with fsspec.open(shard, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as text_fh:
            for line in text_fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with fsspec.open(shard, "rb") as fh:
        table = pq.ParquetFile(io.BytesIO(fh.read())).read()
    yield from table.to_pylist()


@dataclass(frozen=True)
class ShardDocs:
    """One shard's documents, in file order.

    ``ids`` are datakit content hashes — the join key that lets a label set
    computed on one layer attach to any pre/post dedup/decon variant of the same
    corpus, because dedup and decon drop whole documents without mutating text.
    """

    ids: list[str]
    native_ids: list[str | None]
    urls: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.ids)


def read_shard(shard: str, corpus: GridCorpus, text_cap_chars: int) -> ShardDocs:
    """Read a whole shard into memory, computing the datakit id for each doc.

    ``text_cap_chars`` bounds resident memory on corpora with pathological pages
    (resiliparse's HTML-to-text output runs to megabytes). It is applied AFTER
    the id hash so ids stay consistent with the untruncated text that dedup and
    decon keyed on — truncating first would silently produce ids that join
    against nothing.
    """
    ids: list[str] = []
    native_ids: list[str | None] = []
    urls: list[str] = []
    texts: list[str] = []
    for record in iter_records(shard, corpus.format):
        text = record.get("text") or ""
        ids.append(generate_id(text))
        native_ids.append(record.get(corpus.native_id_field) if corpus.native_id_field else None)
        urls.append(record.get("url") or "")
        texts.append(text[:text_cap_chars])
    return ShardDocs(ids=ids, native_ids=native_ids, urls=urls, texts=texts)


def stage_output_dir(dataset: str, stage: Stage) -> str:
    """Root directory for ``stage``'s attribute table.

    The quality tree mirrors upstream's ``outputs/main`` + ``outputs/samples``
    layout so an upstream ``QualityScores`` artifact can point straight at it.
    """
    if stage is Stage.QUALITY:
        return f"{OUTPUT_BASE}/{QUALITY_ROOT}/{dataset}_{GRID_SUFFIX}"
    return f"{OUTPUT_BASE}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}"


def stage_shard_output(dataset: str, stage: Stage, shard: str) -> str:
    """Output parquet for one input shard — same basename, always ``.parquet``."""
    stem = output_stem(shard)
    if stage is Stage.QUALITY:
        return f"{stage_output_dir(dataset, stage)}/outputs/main/{stem}.parquet"
    return f"{stage_output_dir(dataset, stage)}/{stem}.parquet"


def quality_samples_output(dataset: str, shard: str) -> str:
    """Output parquet for the ~2% text sample that feeds the stage report."""
    return f"{stage_output_dir(dataset, Stage.QUALITY)}/outputs/samples/{output_stem(shard)}.parquet"


def counts_output(dataset: str, stage: Stage, shard: str) -> str:
    """Per-shard tally, merged later into the grid distribution."""
    return f"{stage_output_dir(dataset, stage)}/counts/{output_stem(shard)}.json"


def output_stem(shard: str) -> str:
    """Basename of ``shard`` with its format extension stripped.

    Both ``a.jsonl.gz`` and ``a.parquet`` become ``a``, so the topic and quality
    tables for a shard share a stem regardless of the source encoding.
    """
    name = posixpath.basename(shard)
    for suffix in (".jsonl.gz", ".parquet", ".json.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def done_marker(dataset: str, stage: Stage, shard: str) -> str:
    """Per-shard, PER-STAGE completion marker.

    Per-stage is what makes a fused run and two separate runs interchangeable: a
    quality-only catch-up pass months later skips exactly the shards it already
    scored and re-does none of the topic work.
    """
    return f"{stage_output_dir(dataset, stage)}/_done/{output_stem(shard)}"


def pending_shards(dataset: str, stage: Stage, shards: list[str]) -> list[str]:
    """Shards of ``stage`` still to do, by bulk-listing existing done markers.

    One list call rather than one exists() per shard: at resiliparse's 10,364
    shards (and nemotron's 24,390) the per-shard form costs more wall-clock in
    GCS round-trips than a small shard takes to score.
    """
    marker_dir = f"{stage_output_dir(dataset, stage)}/_done"
    done = {posixpath.basename(path) for path in fsspec_glob(f"{marker_dir}/*")}
    pending = [shard for shard in shards if output_stem(shard) not in done]
    logger.info(
        "%s/%s: %d shards, %d already done, %d pending",
        dataset,
        stage.value,
        len(shards),
        len(shards) - len(pending),
        len(pending),
    )
    return pending


def write_done(dataset: str, stage: Stage, shard: str, n_docs: int) -> None:
    """Mark ``shard`` complete for ``stage``, recording the row count."""
    with fsspec.open(done_marker(dataset, stage, shard), "w") as fh:
        fh.write(str(n_docs))
