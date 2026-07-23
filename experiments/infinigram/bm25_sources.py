# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BM25 source registry: the REAL training-document tier for each dataset/scale.

The infini-gram registry (:mod:`experiments.infinigram.targets`) points at
dedup/decon document trees. But for the BM25 *backup*, the user wants the index
built on the exact documents each model was trained on -- which for several
datasets is a *different* tier (verified 2026-07-23 by tracing every training
cache in ``curation_plan.METHODS`` back through the tokenizer to its document
glob, then confirming each glob resolves and carries ``url`` inline in-cluster):

* dclm / nemotron_full: RAW ``filtered/`` (no marin dedup/decon)  -- url inline
* fineweb_edu: RAW ``filtered/``                                  -- url inline
* high_quality: dedup + decon survivors                           -- url inline
* llm_pipeline_v1 (300): dedup + decon                            -- url inline
* dclm / nemotron_full / high_quality (300): the ``filtered_subsets/
  *_random_300warcs-*`` seed-0 subsets                            -- url inline

Because every confirmed tier carries ``url`` inline, BM25 needs no provenance
join here. Tiers still being located (resiliparse deduped, fineweb_cc) are marked
``landed=False`` so the launcher polls without failing the whole run.

This module reuses :class:`~experiments.infinigram.targets.IndexSource` /
:class:`IndexTarget` (the resolver and the ``bm25_indices/`` output path both work
on any IndexTarget), so only the *sources* differ from the infini-gram registry.
"""

from dataclasses import dataclass

from experiments.infinigram.targets import REGION_BUCKET, Collection, IndexSource, IndexTarget, _llm_provenance_globs

_C1 = "us-central1"
_C2 = "us-central2"


@dataclass(frozen=True)
class Bm25DatasetSpec:
    """One dataset and where its FULL / SMALL *training* document tiers live.

    ``{full,small}_provenance`` gs:// globs are set only for text-only tiers whose
    ``url`` must be recovered by a text-hash join (the LLM dedup/decon trees);
    tiers that carry ``url`` inline leave them empty.
    """

    dataset: str
    region: str
    full: IndexSource | None = None
    small: IndexSource | None = None
    full_provenance: tuple[str, ...] = ()
    small_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.region not in REGION_BUCKET:
            raise ValueError(f"unknown region {self.region!r} for dataset {self.dataset!r}")
        if self.full is None and self.small is None:
            raise ValueError(f"dataset {self.dataset!r} has no sources")

    def source(self, collection: Collection) -> IndexSource | None:
        return self.full if collection is Collection.FULL else self.small

    def provenance(self, collection: Collection) -> tuple[str, ...]:
        return self.full_provenance if collection is Collection.FULL else self.small_provenance


BM25_SPECS: dict[str, Bm25DatasetSpec] = {
    spec.dataset: spec
    for spec in [
        # dclm / nemotron_full: RAW filtered extraction (what was actually trained).
        Bm25DatasetSpec(
            dataset="dclm",
            region=_C2,
            full=IndexSource.at(
                "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977/data-*.jsonl.gz"
            ),
            small=IndexSource.under("gs://marin-us-central2/filtered_subsets/dclm_random_300warcs-*/data-*.jsonl.gz"),
        ),
        Bm25DatasetSpec(
            dataset="nemotron_full",
            region=_C2,
            full=IndexSource.at("gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9/*.jsonl.gz"),
            small=IndexSource.under(
                "gs://marin-us-central2/filtered_subsets/nemotron_full_random_300warcs-*/data-*.jsonl.gz"
            ),
        ),
        # high_quality: dedup + decon. FULL tier is text-only (needs url join); the
        # 300 filtered_subsets draw carries url inline.
        Bm25DatasetSpec(
            dataset="high_quality",
            region=_C1,
            full=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_high_quality_decon_deduped/"
                "10364warcs/deduped/data-*.jsonl.gz"
            ),
            small=IndexSource.under(
                "gs://marin-us-central1/filtered_subsets/high_quality_random_300warcs-*/data-*.jsonl.gz"
            ),
            full_provenance=_llm_provenance_globs("high_quality"),
        ),
        # llm_pipeline_v1: only trained at 300 (dedup + decon, text-only -> url join).
        Bm25DatasetSpec(
            dataset="llm_pipeline_v1",
            region=_C1,
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_pipeline_v1_decon_deduped/"
                "300warcs/deduped/data-*.jsonl.gz"
            ),
            small_provenance=_llm_provenance_globs("llm_pipeline_v1"),
        ),
        # llm_simple_v1 (one-call twin pipeline): 300 dedup + decon, text-only.
        Bm25DatasetSpec(
            dataset="llm_simple_v1",
            region=_C1,
            small=IndexSource.at(
                "gs://marin-us-central1/documents/baseline_llm_simple_v1_decon_deduped/"
                "300warcs/deduped/data-*.jsonl.gz"
            ),
            small_provenance=_llm_provenance_globs("llm_simple_v1"),
        ),
        # fineweb_edu: RAW filtered at 10k (no 300 training run).
        Bm25DatasetSpec(
            dataset="fineweb_edu",
            region=_C2,
            full=IndexSource.under("gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-*/*.jsonl.gz"),
        ),
        # fineweb_cc: BLOCKED. Training documents do not exist in GCS -- only
        # baseline_fineweb_cc/smoke_CC-MAIN-2013-20 is present (fineweb was never
        # produced at 10k/300, per the user). landed=False so the launcher skips it.
        Bm25DatasetSpec(
            dataset="fineweb",
            region=_C2,
            full=IndexSource.at(
                "gs://marin-us-central2/documents/baseline_fineweb_cc_deduped/10364warcs/deduped/data-*.jsonl.gz",
                landed=False,
            ),
        ),
        # resiliparse = the COMPLETE, unfiltered lookup index (recover sources every
        # other pipeline dropped). Use the RAW 10k extraction (url inline, no dedup/
        # decon/quality filter -> maximal coverage). The deduped tier was cleaned up
        # and is NOT wanted here anyway. FULL confirmed (10364 shards, url inline);
        # SMALL 300 is being subset from the 10k by another thread -> poll for it.
        Bm25DatasetSpec(
            dataset="resiliparse",
            region=_C2,
            full=IndexSource.at("gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"),
            small=IndexSource.under(
                "gs://marin-us-central2/filtered_subsets/resiliparse_random_300warcs-*/data-*.jsonl.gz",
                landed=False,
            ),
        ),
    ]
}


def get_bm25_target(dataset: str, collection: Collection) -> IndexTarget:
    """Look up a BM25 build target on the real training tier."""
    if dataset not in BM25_SPECS:
        raise ValueError(f"unknown dataset {dataset!r}. Known: {sorted(BM25_SPECS)}")
    spec = BM25_SPECS[dataset]
    source = spec.source(collection)
    if source is None:
        raise ValueError(f"dataset {dataset!r} has no {collection.value} training tier")
    return IndexTarget(
        dataset=dataset,
        collection=collection,
        region=spec.region,
        source=source,
        provenance_globs=spec.provenance(collection),
    )


def all_bm25_targets(*, only_landed: bool = False) -> list[IndexTarget]:
    """Every (dataset, collection) BM25 target on the real training tiers."""
    out: list[IndexTarget] = []
    for spec in BM25_SPECS.values():
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
                    provenance_globs=spec.provenance(collection),
                )
            )
    return out
