# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure math helpers for the data curation IsoFLOP sweep.

Split out from `data_curation_isoflop.py` so the slicing/ceiling/projection
math is unit-testable without needing executor/levanter/fray imports.
"""

import json
from dataclasses import dataclass

import fsspec

from levanter.data.text import (
    DatasetComponent,
    LMMixtureDatasetConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)

TOTAL_WARCS_CC: int = 7_925_398
"""Total WARC files across all Common Crawl snapshots.

Confirmed by the user from the most recent crawl (CC-MAIN-2026-12). Used as
the denominator for the scale factor `s = total_warcs / sampled_warcs`.
"""


# Canonical uncheatable_eval cache hashes for the meta-llama/Meta-Llama-3.1-8B
# tokenizer — i.e. delphi's exact hashes (verified by resolving
# `uncheatable_eval_tokenized(tokenizer="meta-llama/Meta-Llama-3.1-8B")` via
# the executor's deterministic hashing). These hashes identify the cache
# directory; the absolute gs:// URL is composed per-region at read time.
#
# The data must be pre-copied into every training region's bucket at
# `gs://marin-{region}/tokenized/uncheatable_eval/{ds}-{hash}/`. One-time
# cross-region copy from the origin region.
#
# To verify these match delphi's latest, run:
#     uv run python -c "from experiments.evals.exp1600_uncheatable_evals import \
#         uncheatable_eval_tokenized; from experiments.llama import llama3_tokenizer; \
#         from marin.execution.executor import Executor; \
#         steps = uncheatable_eval_tokenized(tokenizer=llama3_tokenizer); \
#         ex = Executor(prefix='gs://marin-us-central2', \
#             executor_info_base_path='gs://marin-us-central2/experiments'); \
#         [ex.compute_version(s, is_pseudo_dep=False) for s in steps.values()]; \
#         [print(n, ex.output_paths[s]) for n, s in steps.items()]"
_UNCHEATABLE_EVAL_CACHE_HASHES: dict[str, str] = {
    "wikipedia_english": "6330df",
    "github_python": "baab41",
    "github_cpp": "a9de07",
    "bbc_news": "4df59f",
    "arxiv_physics": "f4ad8c",
    "arxiv_computer_science": "2b4f07",
    "ao3_english": "bb5666",
}


# Canonical paloma cache hashes for meta-llama/Meta-Llama-3.1-8B tokenizer —
# delphi's exact hashes (resolved via `paloma_tokenized(tokenizer=llama3_tokenizer)`
# through the executor's deterministic versioning, then verified each cache
# exists at `gs://marin-us-central2/tokenized/paloma/{ds}-{hash}/`).
#
# Adding paloma alongside uncheatable_eval makes our validation 1:1 with
# delphi's `default_validation_sets()`.
_PALOMA_CACHE_HASHES: dict[str, str] = {
    "4chan": "496ad5",
    "c4_100_domains": "2b6db7",
    "c4_en": "cf1f79",
    "dolma-v1_5": "d3bed7",
    "dolma_100_programing_languages": "369132",
    "dolma_100_subreddits": "f25f70",
    "falcon-refinedweb": "75d43b",
    "gab": "ccaced",
    "m2d2_s2orc_unsplit": "7dbcc1",
    "m2d2_wikipedia_unsplit": "b33d23",
    "manosphere_meta_sep": "a07891",
    "mc4": "ea36a2",
    "ptb": "628036",
    "redpajama": "9d4ddd",
    "twitterAAE_HELM_fixed": "2e17c1",
    "wikitext_103": "1f5636",
}


# LIMA ("Less Is More for Alignment", Zhou et al. 2023) -- 1,330 high-quality
# conversations, used as a high-signal validation set for the data-curation
# sweep (complements Paloma's distributional-fit signal with an alignment-
# flavored signal). The cache is produced by `experiments/lima.py` which is
# deterministic across regions (sorted inputs, canonical JSON, pinned HF
# revision + tokenizer).
#
# The path layout differs from paloma/uncheatable_eval (flat rather than
# nested under a `lima/` prefix) because LIMA is a single dataset — so we
# compose the val_path directly in the `with_lima` branch of
# `as_lm_mixture_config` rather than going through `_add_validation_components`.
_LIMA_CACHE_HASH: str = "41ca0d"


@dataclass(frozen=True)
class CurationMethod:
    """One data curation method in the benchmark.

    `tokenized_rel_path` is the bare relative path under the region bucket,
    e.g. `tokenized/baseline_dclm-23e9be/`. At read time it's composed with
    the local region into `gs://marin-{region}/{tokenized_rel_path}` —
    guaranteeing zero cross-region egress.

    The data must be pre-copied into every region bucket that a training
    run might land on (one-time manual cross-region copy). The standalone
    training child verifies the resolved gs:// path is local before handing
    the config to Levanter.

    `d_obs_tokens` is the canonical observed token count on `sampled_warcs`
    WARC files, read from `{source_bucket}/{tokenized_rel_path}/train/.stats.json`
    via `load_d_obs_from_stats` at method-registration time.
    """

    name: str
    tokenized_rel_path: str
    d_obs_tokens: int
    sampled_warcs: int = 3_000
    total_warcs: int = TOTAL_WARCS_CC
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    reproduce_per_region: bool = False
    """If True, treat tokenization as reproducible per-region (don't mirror).

    Informational — the materialization strategy is implemented by the
    one-time manual copy script. True for large but cheaply re-tokenized
    caches (e.g. Resiliparse, 571 GB) where re-running tokenization locally
    is cheaper than cross-region copy; False for caches we pre-copied.
    """
    pin_region: str | None = None
    """Hard region pin for training jobs that use this method.

    Set when the method's cache exists in only one region (e.g. BOS-fixed
    rebuilds at us-central1) or when ExpC re-uses fixed-model artifacts that
    were pinned to a specific region. None = float across regions via the
    SOFT region preference in the launcher (see launch_curation_sweep.py).
    """

    def __post_init__(self) -> None:
        if self.tokenized_rel_path.startswith(("gs://", "mirror://", "http://", "https://", "/")):
            raise ValueError(
                f"tokenized_rel_path must be a bare relative path like "
                f"'tokenized/<hash>/', not a URL or absolute path. "
                f"Got: {self.tokenized_rel_path!r}"
            )

    @property
    def s(self) -> float:
        """Scale factor: `total_warcs / sampled_warcs`."""
        return self.total_warcs / self.sampled_warcs

    @property
    def d_proj(self) -> float:
        """Linear projection of `d_obs` to the full Common Crawl.

        Invariant under re-sampling: if you double `sampled_warcs`,
        `d_obs_tokens` doubles and `s` halves, leaving `d_proj` unchanged.
        """
        return self.d_obs_tokens * self.s

    def local_cache_dir(self, region: str) -> str:
        """Compose the absolute gs:// URL in the given region's bucket.

        Uses `region_tracker.REGION_TO_BUCKET` for region→bucket mapping
        because some regions have non-obvious bucket names (e.g. the region
        is `europe-west4` but the bucket is `gs://marin-eu-west4`, short form).
        A naive `f"gs://marin-{region}"` would build `gs://marin-europe-west4/`
        which does not exist.

        Data must be pre-copied there. No cross-region egress at read time.
        """
        from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

        bucket = REGION_TO_BUCKET[region]  # e.g. "gs://marin-eu-west4"
        return f"{bucket}/{self.tokenized_rel_path.rstrip('/')}/"

    def as_lm_mixture_config(
        self,
        region: str,
        with_uncheatable_eval: bool = True,
        with_paloma: bool = True,
        with_lima: bool = True,
    ) -> LMMixtureDatasetConfig:
        # Capture the local region's bucket once — used for both the training
        # component and all validation components. See `local_cache_dir` for
        # why we go through REGION_TO_BUCKET (region != bucket-stem).
        from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

        region_bucket = REGION_TO_BUCKET[region]
        """Build a single-source `LMMixtureDatasetConfig` with LOCAL-only gs:// paths.

        Every component's `cache_dir` is `gs://marin-{region}/...` — no mirror://,
        no cross-region resolution, no abstraction. Tensorstore reads locally.
        Caller must have pre-copied training + validation caches into the region.

        Validation sets (default both on, matching delphi's `default_validation_sets`):
          - `with_uncheatable_eval=True`: adds 7 uncheatable_eval datasets
          - `with_paloma=True`: adds 16 paloma datasets

        Each validation component has `train_weights[name] = 0.0`, which Levanter's
        `_has_nonzero_weight()` skip in `datasets.py:650` excludes from the training
        stream. Marin convention `missing_weights_are_validation=True`.
        """
        cache_str = self.local_cache_dir(region)
        source = UrlDatasetSourceConfig(
            cache_dir=cache_str,
            train_urls=[],
            validation_urls=[],
            format=TextLmDatasetFormat(),
            tags=[self.name],
        )
        components: dict = {
            self.name: DatasetComponent(
                source=source,
                cache_dir=cache_str,
                format=TextLmDatasetFormat(),
                tags=[self.name],
            ),
        }
        train_weights: dict = {self.name: 1.0}

        def _add_validation_components(prefix: str, hashes: dict[str, str]) -> None:
            """Add each (dataset → hash) entry as a validation-only DatasetComponent."""
            for ds_name, cache_hash in hashes.items():
                key = f"{prefix}/{ds_name}"
                val_path = f"{region_bucket}/tokenized/{prefix}/{ds_name}-{cache_hash}/"
                val_source = UrlDatasetSourceConfig(
                    cache_dir=val_path,
                    train_urls=[],
                    validation_urls=[],
                    format=TextLmDatasetFormat(),
                    tags=[key],
                )
                components[key] = DatasetComponent(
                    source=val_source,
                    cache_dir=val_path,
                    format=TextLmDatasetFormat(),
                    tags=[key],
                )
                # Validation-only: weight 0 in training mixture (Marin convention,
                # Levanter excludes via `_has_nonzero_weight()` in datasets.py:650).
                train_weights[key] = 0.0

        if with_uncheatable_eval:
            _add_validation_components("uncheatable_eval", _UNCHEATABLE_EVAL_CACHE_HASHES)
        if with_paloma:
            _add_validation_components("paloma", _PALOMA_CACHE_HASHES)
        if with_lima:
            # LIMA sits directly under `tokenized/` (flat layout) -- see
            # `experiments/lima.py` for the pipeline that produces the cache.
            key = "lima"
            lima_path = f"{region_bucket}/tokenized/lima_text-{_LIMA_CACHE_HASH}/"
            lima_source = UrlDatasetSourceConfig(
                cache_dir=lima_path,
                train_urls=[],
                validation_urls=[],
                format=TextLmDatasetFormat(),
                tags=[key],
            )
            components[key] = DatasetComponent(
                source=lima_source,
                cache_dir=lima_path,
                format=TextLmDatasetFormat(),
                tags=[key],
            )
            train_weights[key] = 0.0

        return LMMixtureDatasetConfig(
            tokenizer=self.tokenizer,
            components=components,
            train_weights=train_weights,
            shuffle=True,
            permutation_type="feistel",
        )


def t_exp_ceiling(method: CurationMethod, t_target: float, *, allow_data_rich: bool = False) -> float:
    """Max `T_exp` that can faithfully simulate `t_target` for this method.

    Two regimes, switched by the target's epoch count `T_target / D_proj`:

    1. Slicing regime (target_epochs >= 1): the target run epochs the full
       projected pool, so the simulation must epoch a slice. Slice fits in
       D_obs constraints `slice = T_exp * D_proj / T_target <= D_obs`,
       giving `T_exp <= T_target / s`. Always returned when
       `allow_data_rich=False` (the default, used by ExpB).

    2. Data-rich regime (target_epochs < 1, only when `allow_data_rich=True`):
       the target run sees less than one full epoch of D_proj. Under
       uniformity, training T_exp i.i.d. tokens from the cache is
       statistically equivalent to drawing T_exp tokens from D_proj — no
       slicing needed. Ceiling is the physical cache size: `T_exp <= D_obs`.
       Used by ExpC for llm_curated and resiliparse, where 33T target tokens
       sit well below D_proj and the cache is much larger than T_target/s.

    Two ways to raise the slicing-regime ceiling:

    1. Increase `t_target`. Cheap in code (add to `T_TARGETS`), but changes
       the scientific question: the simulated regime becomes more aggressive
       (more target epochs), and data-constrained methods bend harder.
    2. Increase `method.sampled_warcs`. Keeps the target regime fixed but
       requires re-running the extraction pipeline for that method. NOT cheap
       for LLM-based extraction methods.
    """
    if allow_data_rich and t_target < method.d_proj:
        return float(method.d_obs_tokens)
    return t_target / method.s


def slice_tokens_for(method: CurationMethod, t_exp: float, t_target: float) -> float:
    """Experiment B slice size: `T_exp * D_proj / T_target`.

    By construction this makes `T_exp / slice == T_target / D_proj`, so the
    experiment's epoch count matches the target regime's epoch count.
    """
    return t_exp * method.d_proj / t_target


def implicit_target_exp_a(method: CurationMethod, t_exp: float) -> float:
    """Experiment A's implicit target: `T_exp * s`.

    Derivation: in Experiment A we don't slice, so the training sees each
    observed token `T_exp / D_obs` times on average. The faithful target
    regime matching that epoch count uses `T_target / D_proj` epochs.
    Setting them equal gives `T_target = T_exp * D_proj / D_obs = T_exp * s`.
    """
    return t_exp * method.s


def load_d_obs_from_stats(tokenized_path: str) -> int:
    """Read `total_tokens` from `{path}/train/.stats.json`.

    This is the canonical source of per-method D_obs. Stats are written by
    the tokenize step; `total_tokens` is the JaggedArrayStore byte count
    divided by dtype size.
    """
    stats_path = f"{tokenized_path.rstrip('/')}/train/.stats.json"
    with fsspec.open(stats_path, "r") as f:
        return int(json.load(f)["total_tokens"])
