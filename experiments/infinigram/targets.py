# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The infini-gram index registry: which document tier of each dataset to index.

We index the *final training-corpus* tier of each dataset (the deduped/deconned
``.jsonl.gz`` documents models actually train on), in two collections:

* ``FULL``  -- the 10,364-WARC extraction
* ``SMALL`` -- the random-300-WARC subset

Every dataset is described once by a :class:`DatasetSpec` giving its region and a
:class:`IndexSource` per collection. A source is *either* concrete gs:// globs or
a ``prefix`` with a single ``*`` confighash segment resolved at run time by
listing the bucket (executor step dirs carry a content hash we must not hardcode;
10k and 300 dirs have different hashes than the 3000-WARC dirs). Datasets still
processing are registered with the source they *will* land at -- resolution fails
only that dataset's job until the data exists, so the registry stays append-only.

Source paths verified against live GCS 2026-07-23. See ``resolve.py`` for how a
spec becomes concrete shard URLs.
"""

from dataclasses import dataclass
from enum import StrEnum


class Collection(StrEnum):
    """Which WARC draw of a dataset an index covers."""

    FULL = "full"  # 10,364-WARC extraction
    SMALL = "small"  # random-300-WARC subset


# Regional bucket per Iris region. A dataset's index is always read/written
# in-region to avoid cross-region egress (AGENTS.md hard rule).
REGION_BUCKET: dict[str, str] = {
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "us-west4": "gs://marin-us-west4",
    "eu-west4": "gs://marin-eu-west4",
}

# Canonical home for finished indices, one dir per (collection, dataset).
INDEX_ROOT_TEMPLATE = "{bucket}/infinigram_indices/{collection}/{dataset}"

# Random-300 WARC manifest (provenance only; the SMALL document subsets are drawn
# from this nested seed-0 sample). See experiments/distill/random_subsets/.
RANDOM_300_MANIFEST = "experiments/distill/random_subsets/random_warcs_300.txt"

# Provenance for the LLM-extraction datasets, whose deduped tiers are text-only.
# Every region's raw batches are mirrored into the us-central1 bucket, so reading
# them for the text-hash join is in-region. Each batch record carries
# {text, url, warc_record_id, warc_file, snapshot}. Keyed by dataset -> spec_id.
_LLM_PROVENANCE_BASE = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region"


def _llm_provenance_globs(spec_id: str) -> tuple[str, ...]:
    return (f"{_LLM_PROVENANCE_BASE}/*/{spec_id}/data-*/batch_*.jsonl.gz",)


@dataclass(frozen=True)
class IndexSource:
    """Where a dataset+collection's document shards live in GCS.

    Exactly one of ``globs`` (concrete patterns) or ``prefix`` (a pattern with a
    single ``*`` confighash segment, resolved by listing) is set. ``landed``
    records whether the tier already exists -- purely informational for launcher
    filtering; resolution always re-checks against live GCS.
    """

    globs: tuple[str, ...] = ()
    prefix: str | None = None
    landed: bool = True

    def __post_init__(self) -> None:
        if bool(self.globs) == bool(self.prefix):
            raise ValueError("IndexSource needs exactly one of globs / prefix")

    @classmethod
    def at(cls, *globs: str, landed: bool = True) -> "IndexSource":
        """Concrete shard globs, e.g. ``.../deduped/data-*.jsonl.gz``."""
        return cls(globs=tuple(globs), landed=landed)

    @classmethod
    def under(cls, prefix: str, *, landed: bool = True) -> "IndexSource":
        """A confighash prefix with one ``*`` to resolve, e.g. ``.../foo-*/data-*.jsonl.gz``."""
        return cls(prefix=prefix, landed=landed)


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset and where its FULL / SMALL document tiers live."""

    dataset: str
    region: str
    full: IndexSource | None = None
    small: IndexSource | None = None
    # gs:// globs to a provenance tier carrying {text, url, warc_record_id, ...}
    # for datasets whose document tier is text-only. Empty when ids are already
    # inline (baselines). Joined on exact text hash at stage time.
    provenance_globs: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.region not in REGION_BUCKET:
            raise ValueError(f"unknown region {self.region!r} for dataset {self.dataset!r}")
        if self.full is None and self.small is None:
            raise ValueError(f"dataset {self.dataset!r} has no sources")

    def source(self, collection: Collection) -> IndexSource | None:
        return self.full if collection is Collection.FULL else self.small


@dataclass(frozen=True)
class IndexTarget:
    """A single index to build: one dataset in one collection."""

    dataset: str
    collection: Collection
    region: str
    source: IndexSource
    provenance_globs: tuple[str, ...] = ()

    @property
    def index_dir(self) -> str:
        """Canonical GCS output dir for this index."""
        return INDEX_ROOT_TEMPLATE.format(
            bucket=REGION_BUCKET[self.region],
            collection=self.collection.value,
            dataset=self.dataset,
        )

    @property
    def name(self) -> str:
        return f"{self.dataset}-{self.collection.value}"


# ---------------------------------------------------------------------------
# Registry. Add a dataset = add one DatasetSpec. FULL/SMALL sources verified
# against live GCS where they exist; not-yet-landed tiers use landed=False with
# the path they will land at.
#
# Baseline join-filtered tiers (dclm/nemotron/fineweb*/resiliparse) live under
# .../10364warcs_core_v2/survivors/ (central2). LLM-extraction tiers
# (high/med/low_quality, llm_pipeline_v1) live under
# .../baseline_{spec}_deduped/{n}warcs/deduped/ (central1).
# ---------------------------------------------------------------------------

_C2 = "us-central2"
_C1 = "us-central1"
_E5 = "us-east5"

DATASETS: dict[str, DatasetSpec] = {
    spec.dataset: spec
    for spec in [
        DatasetSpec(
            dataset="dclm",
            region=_C2,
            full=IndexSource.at(
                "gs://marin-us-central2/documents/baseline_dclm_decon/10364warcs_core_v2/survivors/data-*.jsonl.gz"
            ),
            # SMALL = the exact 300-WARC subset the models trained on (non-deduped
            # filtered subset; carries url + warc_record_id inline, no join needed).
            small=IndexSource.under("gs://marin-us-central2/filtered_subsets/dclm_random_300warcs-*/data-*.jsonl.gz"),
            notes="SMALL = filtered_subsets/dclm_random_300warcs (the trained 300-WARC corpus).",
        ),
        DatasetSpec(
            dataset="nemotron_full",
            region=_C2,
            full=IndexSource.at(
                "gs://marin-us-central2/documents/baseline_nemotron_decon/10364warcs_core_v2/survivors/data-*.jsonl.gz"
            ),
            small=IndexSource.under(
                "gs://marin-us-central2/filtered_subsets/nemotron_full_random_300warcs-*/data-*.jsonl.gz"
            ),
            notes="SMALL = filtered_subsets/nemotron_full_random_300warcs (the trained 300-WARC corpus).",
        ),
        DatasetSpec(
            dataset="fineweb",
            region=_C2,
            full=IndexSource.under(
                "gs://marin-us-central2/documents/baseline_fineweb_cc/10364warcs*/**/data-*.jsonl.gz",
                landed=False,
            ),
            small=IndexSource.under(
                "gs://marin-us-central2/documents/baseline_fineweb_cc/300warcs*/**/data-*.jsonl.gz",
                landed=False,
            ),
            notes="Only smoke_CC-MAIN-2013-20 present so far; full/small tiers still landing.",
        ),
        DatasetSpec(
            dataset="fineweb_edu",
            region=_C2,
            full=IndexSource.at(
                "gs://marin-us-central2/documents/baseline_fineweb_edu_decon/"
                "10364warcs_core_v2/survivors/data-*.jsonl.gz",
                landed=False,
            ),
            small=IndexSource.under(
                "gs://marin-us-central2/documents/baseline_fineweb_edu_decon/300warcs*/survivors/data-*.jsonl.gz",
                landed=False,
            ),
            notes="10364warcs_core_v2 has only inspect/ so far; survivors leaf still landing.",
        ),
        DatasetSpec(
            dataset="resiliparse",
            region=_C2,
            # 10k resiliparse extraction (flat data-*-of-10364 under the hashed step dir).
            full=IndexSource.under("gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-*/data-*.jsonl.gz"),
            small=IndexSource.under(
                "gs://marin-us-central2/extracted/dclm_400m_1x_300_resiliparse-*/data-*.jsonl.gz",
                landed=False,
            ),
        ),
        DatasetSpec(
            dataset="high_quality",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_high_quality_decon_deduped/"
                "10364warcs/deduped/data-*.jsonl.gz"
            ),
            # SMALL = the trained 300-WARC subset (carries url inline -> no join).
            small=IndexSource.under(
                "gs://marin-us-central1/filtered_subsets/high_quality_random_300warcs-*/data-*.jsonl.gz"
            ),
            provenance_globs=_llm_provenance_globs("high_quality"),
            notes="SMALL = filtered_subsets/high_quality_random_300warcs (trained corpus, url inline).",
        ),
        DatasetSpec(
            dataset="high_quality_v2",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_high_quality_v2_decon_deduped/"
                "10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            # SMALL: v2 decon_deduped 300warcs tree carries url inline (no join).
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_high_quality_v2_decon_deduped/300warcs/deduped/data-*.jsonl.gz"
            ),
            notes="high_quality pipeline v2; dedup+decon trained tier, url inline.",
        ),
        DatasetSpec(
            dataset="llm_pipeline_v1",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_pipeline_v1_decon_deduped/"
                "10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            # SMALL uses the deduped (non-decon) tier to match llm_simple_v1 for a
            # clean pipeline-vs-pipeline comparison (decon drops ~0 docs at 300).
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_pipeline_v1_deduped/300warcs/deduped/data-*.jsonl.gz"
            ),
            provenance_globs=_llm_provenance_globs("llm_pipeline_v1"),
            notes="SMALL (deduped 300warcs) landed; FULL 10k still processing.",
        ),
        DatasetSpec(
            dataset="llm_simple_v1",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_simple_v1_decon_deduped/"
                "10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            # SMALL landed as the post-dedup (pre-decon) tier; final leaf is ``reshape/``.
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_simple_v1_deduped/300warcs/reshape/data-*.jsonl.gz"
            ),
            provenance_globs=_llm_provenance_globs("llm_simple_v1"),
            notes="One-call oc3 pipeline. SMALL uses the deduped (non-decon) 300warcs/reshape tier "
            "(the decon tier isn't built for 300); FULL 10k still processing.",
        ),
        DatasetSpec(
            dataset="fastpipe_v3",
            region="us-east5",
            # fastText+JustText->ModernBERT cascade. kept_text is one parquet per WARC
            # carrying url/doc_id/snapshot inline; the SMALL 300-WARC index is built by
            # selecting the manifest's per-WARC shards (build --source-warc-manifest).
            full=IndexSource.at(
                "gs://marin-us-east5/documents/fast_curation/fastpipe_v3-da3893385e/kept_text/data-*.parquet"
            ),
            small=IndexSource.at(
                "gs://marin-us-east5/documents/fast_curation/fastpipe_v3-da3893385e/kept_text/data-*.parquet"
            ),
            notes="kept_text per-WARC parquet (url inline). SMALL via --source-warc-manifest hash selection.",
        ),
        DatasetSpec(
            dataset="med_quality",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_med_quality_deduped/10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_med_quality_deduped/300warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            provenance_globs=_llm_provenance_globs("med_quality"),
            notes="Random-300 being regenerated; deduped/{n}warcs tree exists for other N.",
        ),
        DatasetSpec(
            dataset="low_quality",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_low_quality_deduped/10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_low_quality_deduped/300warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
            # low_quality is the LEGACY flat namespace; its raw batches carry the
            # same {text,url,warc_record_id,...} schema at the unprefixed path.
            provenance_globs=("gs://marin-us-central1/documents/baseline_llm_extraction/data-*/batch_*.jsonl.gz",),
            notes="Random-300 being regenerated. Legacy raw tier (pre-dedup) is the flat "
            "documents/baseline_llm_extraction/data-*/ namespace (extraction_specs.LEGACY_SPEC_ID).",
        ),
        # fastpipe_v3 keep-top-X% quality bands (ModernBERT-prob), us-east5. The
        # 300-WARC deduped docs (subset_to_warcs.py) are text-only (+ modernbert_prob);
        # no url is available without a separate DCLM-text join, so no provenance here.
        *[
            DatasetSpec(
                dataset=f"fastpipe_v3_{band}",
                region=_E5,
                full=IndexSource.at(
                    f"gs://marin-us-east5/documents/baseline_fastpipe_v3{suffix}_decon_deduped/"
                    "10364warcs/deduped/data-*.jsonl.gz",
                    landed=False,
                ),
                small=IndexSource.at(
                    f"gs://marin-us-east5/documents/baseline_fastpipe_v3{suffix}_decon_deduped/"
                    "300warcs/deduped/data-*.jsonl.gz"
                ),
                notes="ModernBERT keep-top-X% band; text-only (+modernbert_prob), no url.",
            )
            for band, suffix in (("100", ""), ("80", "_80"), ("60", "_60"), ("40", "_40"), ("20", "_20"))
        ],
    ]
}


def get_target(dataset: str, collection: Collection) -> IndexTarget:
    """Look up a build target. Raises with the known set on an unknown dataset,
    or if the dataset has no source for the requested collection."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}. Known: {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    source = spec.source(collection)
    if source is None:
        raise ValueError(f"dataset {dataset!r} has no {collection.value} source")
    return IndexTarget(
        dataset=dataset,
        collection=collection,
        region=spec.region,
        source=source,
        provenance_globs=spec.provenance_globs,
    )


def all_targets(*, only_landed: bool = False) -> list[IndexTarget]:
    """Every (dataset, collection) target in the registry.

    Args:
        only_landed: skip sources marked not-yet-landed. Resolution still
            re-checks GCS, so this is just a convenience filter for launchers.
    """
    out: list[IndexTarget] = []
    for spec in DATASETS.values():
        for collection in Collection:
            source = spec.source(collection)
            if source is None or (only_landed and not source.landed):
                continue
            out.append(
                IndexTarget(
                    dataset=spec.dataset,
                    collection=collection,
                    region=spec.region,
                    source=source,
                    provenance_globs=spec.provenance_globs,
                )
            )
    return out
