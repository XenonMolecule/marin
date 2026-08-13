# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure enumeration helper for the data-curation IsoFLOP sweep.

Imported by both the coordinator (`launch_curation_sweep.py`) and the standalone
child (`run_curation_train_standalone.py`). Pure functions, no iris/fray/executor
dependencies — safe to test in isolation and runs fast even without GCS access.

The `PlannedRun` dataclass is the contract between coordinator and child:
the coordinator computes one per (method, experiment, candidate) and serializes
to CLI args; the child rehydrates, region-locks, and runs Levanter training.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from iris.cluster.types import get_tpu_topology
from marin.scaling_laws import CandidateConfig, pick_v4_type, pick_v5p_type

from experiments.scaling_law_sweeps.completed_adamh import (
    SEQ_LEN,
    _compute_tensor_parallel_size,
    completed_adamh_heuristic,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    CurationMethod,
    GridMixCurationMethod,
    ReweightedCurationMethod,
    implicit_target_exp_a,
    load_d_obs_from_stats,
    slice_tokens_for,
    t_exp_ceiling,
)

logger = logging.getLogger(__name__)


# --- v6e (Trillium) TPU picker -----------------------------------------------
# marin.scaling_laws doesn't export a v6e picker yet; keep this local until it's
# upstreamed. v6e-{slice_size} is named by CHIP count (unlike v4/v5p which name
# by core count). HBM per chip = 32 GiB.
#
# Available v6e slice sizes from iris TpuTopologyInfo:
#   v6e-4, v6e-8, v6e-16, v6e-32, v6e-64, v6e-128, v6e-256
# vm_count matters for device_variant_constraint compatibility:
#   v6e-4  → vm_count=1   (matches v4-8 / v5p-8 tier)
#   v6e-8  → vm_count=1   (matches v4-8 / v5p-8 tier, MORE memory)
#   v6e-16 → vm_count=4   (DIFFERENT from v4-16 / v5p-16 which are vm_count=2)
# So v6e can only be included as a constraint alternative at the vm_count=1
# tier — i.e. for plans that would otherwise use v4-8 / v5p-8.
_V6E_HBM_PER_CHIP_GIB = 32
_V6E_SINGLE_VM_SLICES = [4, 8]  # slice sizes with vm_count=1 (compatible alt for v4-8/v5p-8)


def pick_v6e_type_single_vm(estimated_memory_bytes: int) -> str | None:
    """Smallest v6e slice with vm_count=1 that fits the memory.

    Returns None if memory exceeds v6e-8 (i.e. the plan needs vm_count>1 variants,
    where v6e topology diverges from v4/v5p and can't be a compatible alternative).
    """
    chip_bytes = _V6E_HBM_PER_CHIP_GIB * 1024**3
    chips_req = math.ceil(estimated_memory_bytes / chip_bytes)
    for s in _V6E_SINGLE_VM_SLICES:
        if s >= chips_req:
            return f"v6e-{s}"
    return None  # too big — v6e-16+ has vm_count!=1, can't mix with v4/v5p


import math  # noqa: E402  # re-import for pick_v6e; python allows this

# --- Compute budgets ----------------------------------------------------------
# Matches delphi's `completed_adamh` defaults (7 log-spaced FLOP points).
BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)


# --- Experiment B target tokens ----------------------------------------------
# Grounded per internal review (20T ≈ 1T-param Chinchilla).
DEFAULT_T_TARGETS: tuple[float, ...] = (20e12,)


# --- Experiment C target tokens ----------------------------------------------
# 33T pushes deeper into the Chinchilla extrapolation while staying within
# llm_curated's D_obs (56B) and resiliparse's D_obs (143B), so the data-rich
# methods don't need re-extraction. See EXPERIMENT_C_MATH.md.
DEFAULT_T_TARGET_C: float = 33e12

# 10k WARC re-extraction (definitive count: 10,364 — the file lacks a trailing
# newline so `wc -l` returns 10,363, but `grep -c .` and the GCS object count
# both give 10,364).
EXPC_SAMPLED_WARCS: int = 10_364


def expc_uniform_t_exp_cap(t_target: float, sampled_warcs: int = EXPC_SAMPLED_WARCS) -> float:
    """Uniform T_exp cap for ExpC, applied to all methods regardless of D_obs.

    Cross-method comparability: even though llm_curated and resiliparse have
    D_obs much larger than the slicing-regime ceiling (= T_target * sampled_warcs
    / TOTAL_WARCS_CC), we cap all methods at the same value so candidate sets
    are aligned at every budget. = 43.14B at T=33T, sampled_warcs=10,364.

    From data_curation_math.TOTAL_WARCS_CC = 7,925,398.
    """
    from experiments.scaling_law_sweeps.data_curation_math import TOTAL_WARCS_CC

    return t_target * sampled_warcs / TOTAL_WARCS_CC


# --- Slice safety floor -------------------------------------------------------
MIN_SLICE_TOKENS_DEFAULT: float = 25e6


# --- D_obs constants for the methods registry --------------------------------
# Hardcoded to avoid a slow GCS read at every coordinator/child startup.
# Update from `{path}/train/.stats.json:total_tokens` after re-tokenization.
_D_OBS_DEFAULTS: dict[str, int] = {
    # --- Random 3000-WARC sample (uniform draw from the DCLM 400m-1x pool, seed 0;
    #     manifest experiments/distill/subsets/baseline_warcs_3000_random.txt).
    #     INDEPENDENT from the head-biased baseline_warcs_3000.txt runs below —
    #     distinct cache hashes so nothing collides. Built 2026-05-30 via pipeline.py
    #     (--products dclm/nemotron_full/resiliparse) + dedup_resiliparse_warc_scaling.py.
    #     token counts from train/.stats.json:total_tokens.
    "baseline_dclm-cf177e": 2_114_092_590,
    "baseline_nemotron_full-75f981": 2_943_995_743,
    "resiliparse_random_dedup_3000warcs-7de1e2": 109_186_106_815,
    # --- Random 100-WARC sample from the 10k pool (uniform draw seed 0, manifest
    #     experiments/distill/subsets/baseline_warcs_100_random.txt; 60 crawls spanning
    #     2013-2022). The COUNTERPART to the head-biased *_100 methods, whose manifest
    #     (subsets/baseline_warcs_100.txt = first 100 of the date-SORTED 3k manifest)
    #     covers only 2 crawls, BOTH 2013. Built 2026-07-15 by subset_random100_10k.py,
    #     which subsets the EXISTING 10,364-WARC extractions (no re-extraction) — all
    #     three methods use the IDENTICAL 100 WARCs.
    #     NOTE the biased head has MORE tokens/WARC (2013 WARCs are denser): dclm
    #     98M(biased) vs 68M(random); hq 414M vs 203M. So the old N=100 differed from a
    #     random sample in BOTH content mix and data volume. (hq's larger gap also has a
    #     confound: the 10k source is decon+dedup'd, the 3k-sourced biased hq is not.)
    #     Caches tokenized in the source region + mirrored to us-east5; pin_region
    #     us-east5 so children land where the mirror + free capacity are.
    #     Biased/random tokens-per-100-WARCs by method: dclm 98M/68M (1.44x),
    #     hq 414M/203M (2.0x), nemotron_full 317M/103M (3.1x) — the head-biased 2013
    #     WARCs yield MORE tokens for every method. nemotron's 3.1x is the largest,
    #     consistent with Nemotron-CC covering the 2013-era crawls far more densely
    #     than the 2013-2022 average.
    "dclm_random_100warcs-350025": 67_666_557,
    "high_quality_random_100warcs-88f468": 202_814_071,
    "nemotron_full_random_100warcs-ade768": 103_440_840,
    # Random 300/500-WARC samples (10k pool, seed 0; NESTED: 100 c 300 c 500). Same
    # subset_random100_10k.py path (WARC_N=300/500). Tokens scale ~linearly with N,
    # confirming clean nesting. The N=100->300->500 ladder over identical architectures
    # (256/512/768/1536) probes whether HQ flips to WORSE as data grows.
    "dclm_random_300warcs-cbd706": 214_838_835,
    "nemotron_full_random_300warcs-39307a": 318_293_446,
    "high_quality_random_300warcs-c6c5a5": 639_355_979,
    # high_quality_v2 (new extractor) over the same 300 random WARCs — 3x the tokens of v1 (keeps more data).
    "high_quality_v2_decon_300warcs-3d73da": 1_972_978_146,
    "llm_pipeline_v1_decon_300warcs-b834c5": 3_353_097_168,
    "llm_pipeline_v1_1_decon_300warcs-1563f5": 4_261_414_828,
    # N=3000 nested-ladder draw (random_warcs_3000_nested.txt, 100⊂300⊂…⊂3000).
    # 25.16M docs / 936 shards. NOTE the 3k baselines (dclm/nemotron/resiliparse_random_3000)
    # are on the INDEPENDENT draw and overlap this set by only 1423/3000.
    "llm_pipeline_v1_1_decon_3000warcs-aa7070": 36_648_839_862,
    "llm_simple_v1_decon_300warcs-b8a945": 4_259_336_845,
    # fastpipe_v3 bands @ N=300 (global 10k dedup+decon, subset to the 300 by text, then thresholded).
    "fastpipe_v3_decon_300warcs-095914": 1_425_374_475,
    "fastpipe_v3_80_decon_300warcs-a2ea2c": 1_196_359_639,
    "fastpipe_v3_60_decon_300warcs-888150": 902_223_137,
    "fastpipe_v3_40_decon_300warcs-e667fb": 569_103_297,
    "fastpipe_v3_20_decon_300warcs-ff2322": 274_632_363,
    "dclm_random_500warcs-b8780e": 356_033_903,
    "nemotron_full_random_500warcs-3f7ddf": 520_096_252,
    "high_quality_random_500warcs-5cc14b": 1_050_918_135,
    "dclm_random_1000warcs-5a69c1": 706_817_668,
    "nemotron_full_random_1000warcs-83bd90": 1_028_007_768,
    "high_quality_random_1000warcs-2470d1": 2_089_503_582,
    "dclm_random_2000warcs-7ba7b6": 1_424_810_618,
    "nemotron_full_random_2000warcs-7e5953": 2_035_820_453,
    "high_quality_random_2000warcs-06a890": 4_181_779_555,
    "baseline_dclm-23e9be": 2_663_454_015,
    "baseline_nemotron-c67de9": 1_919_401_016,
    "baseline_fineweb_edu-7a3bc5": 817_221_529,
    "baseline_resiliparse-7278c1": 142_652_598_588,
    # BOS-fixed rebuilds (tokenized on us-central1 only — not mirrored to other
    # regions). Replace the original nemotron_full/llm_curated caches, which
    # were rebuilt with BOS tokens.
    "baseline_nemotron_full_bos_fixed-4b1ce7": 2_700_501_906,
    "baseline_llm_curated_bos_fixed-d04ef8": 56_008_357_279,
    # Deduped llm_curated (fuzzy doc dedup on top of bos_fixed corpus, applied
    # at document level on the full 56B-token corpus → 43.64B tokens, 76.2M
    # docs). Tokenized cache mirrored to us-central1 / us-central2 / us-east5
    # on 2026-05-05. See .agents/projects/dedup_observations.md for params
    # (286 perms / 26 bands / 5-char n-gram / Jaccard ~0.75).
    "baseline_llm_curated_deduped-c444e2": 43_644_701_678,
    # ExpC 10k re-extractions — extraction in progress, hashes/d_obs not yet
    # known. Sentinel hashes ("TBD_*") are placeholders so the methods can be
    # registered now and resolved by the launcher; `_iter_valid_candidates`
    # raises if any of these are enumerated before real values are filled in.
    # When tokenization completes, replace the dict KEY with the real hash
    # AND update the d_obs to total_tokens from train/.stats.json. Then update
    # the corresponding `_method(...)` call below.
    "dclm_400m_1x_10k_dclm-3df0ba": 7331583927,
    "dclm_400m_1x_10k_nemotron_full-3dcb75": 10130086896,
    # CORE-v2-decontaminated (n=15/DF<=10, same treatment as high_quality/resiliparse)
    # variants of the dclm/nemotron 10k caches. Tokenized us-central2; input_ids +
    # ledger + stats lean-mirrored to us-east5 (pin_region enforces training there).
    # Token deltas vs non-decon are tiny (dclm -4.1M/0.056%, nemo -5.1M/0.05%) — the
    # decon flag rates were ~0.01-0.02% genuine, so this measures whether removing
    # eval leakage moves the curves at all. See decontam_dclm_nemo_fineweb_scope.md.
    "dclm_400m_1x_10k_dclm_decon-177776": 7327476298,
    "dclm_400m_1x_10k_nemotron_full_decon-e271a5": 10125029374,
    # FineWeb-Edu 10k — us-central2 only (not mirrored). pin_region enforces it.
    "dclm_400m_1x_10k_fineweb_edu-0a3143": 2346934380,
    # high_quality (LLM-extracted) 10k: dedup + CORE-v2 decontam (n=15/DF<=10),
    # tokenized us-central1 only (not mirrored). d_obs from train/.stats.json.
    "high_quality_decon_10364warcs-6451c8": 21296896949,
    # hq dilution-ablation variants (us-central1 only; built by build_hq_variants.py
    # from the hq HF export docs). dense = fact-bearing expository partition (upweight
    # component for hq_reweight); epoch_sub344 = uniform 34.4% subsample matching
    # DCLM's token count (7.332B) for the epoch-control ablation.
    "hq_dense-7024eb": 2588921892,
    "hq_epoch_sub344-f4dcd4": 7329749552,
    # FineWeb-CC 10k: full HF FineWeb for our WARCs (HF dedup trusted; no extra
    # dedup/decontam). Tokenized us-central2 only (not mirrored).
    "fineweb_cc_10364warcs-ddfeda": 28004233781,
    # resiliparse (raw HTML->text) 10k: dedup + CORE-v2 decontam (n=15/DF<=10),
    # same full treatment as high_quality. Tokenized us-central2 only (not mirrored).
    "resiliparse_decon_10364warcs-beaaf5": 339971302028,
    # System-prompt-conditioned DCLM 400m-1x: [S][D] sequences (Qwen3-30B-A3B
    # system prompts prepended to each DCLM doc), tokenized from the
    # conditioned_text field. us-east5 ONLY (cache not mirrored). total_tokens
    # from train/.stats.json (5,918,974 docs).
    "sysprompt_dclm_qfull1-0ba2ee": 7818175437,
    # --- 998M three-way system-prompt ablation on the DCLM-30B corpus (9e19) ---
    # A = [S][D] (conditioned_text), C = the SAME docs without [S] (text),
    # B = A's docs + extra docs, token-matched to A. From train/.stats.json.
    # Ratios confirm the design: B/A = 100.03%, C/A = 98.35% (the [S] overhead).
    "sysprompt30b_998m_9e19_A-94f799": 14810496434,
    "sysprompt30b_998m_9e19_B-64d3a5": 14815377151,
    "sysprompt30b_998m_9e19_C-c94a3d": 14565835827,
    # --- WARC-scaling sweep subsamples (N ∈ {100, 500, 1000, 2000}) ---
    # Read 2026-04-29 from {bucket}/tokenized/{key}/train/.stats.json.
    # dclm: source us-central2; mirrored to us-central1.
    "baseline_dclm_100warcs": 97_639_335,
    "baseline_dclm_500warcs": 441_260_426,
    "baseline_dclm_1000warcs": 953_517_521,
    "baseline_dclm_2000warcs": 1_895_152_428,
    # resiliparse: source us-central2; mirrored to us-central1.
    "baseline_resiliparse_100warcs": 4_422_854_170,
    "baseline_resiliparse_500warcs": 21_330_844_102,
    "baseline_resiliparse_1000warcs": 46_283_090_573,
    "baseline_resiliparse_2000warcs": 93_393_249_192,
    # resiliparse with per-N fuzzy dedup. N=100 smoke ran in eu-west4 then
    # mirrored to us-east5/us-central1/us-central2 (one-off cross-continent;
    # larger N's will use a different routing).
    "resiliparse_100warcs-7f8246": 3_952_774_658,
    # resiliparse full-corpus (3000-WARC) fuzzy dedup output, built as a side
    # experiment by tokenizing gs://marin-us-central2/documents/
    # baseline_resiliparse_deduped/data-XXXXX-of-03000.jsonl.gz (flat layout).
    # us-central2 ONLY — pin_region enforces it. Acts as the N=3000 anchor for
    # the WARC-scaling sweep's resiliparse_dedup curve.
    "baseline_resiliparse_deduped-471baf": 94_332_023_608,
    # resiliparse per-N fuzzy dedup, N=500/1000/2000. Built 2026-05-17 via
    # dedup_resiliparse_warc_scaling.py in us-east5 (raw extraction
    # pre-mirrored from us-central2). Tokenized in us-east5; pin_region on
    # the methods below enforces region locality.
    "resiliparse_dedup_300warcs-24d661": 11_361_792_146,
    "resiliparse_dedup_500warcs-a491f7": 18_484_848_433,
    "resiliparse_dedup_1000warcs-4384bb": 35_289_322_730,
    "resiliparse_dedup_2000warcs-7e2171": 66_568_599_303,
    # nemotron_full BOS-fixed: us-central1 only.
    "baseline_nemotron_full_bos_fixed_100warcs": 316_603_329,
    "baseline_nemotron_full_bos_fixed_500warcs": 799_434_481,
    "baseline_nemotron_full_bos_fixed_1000warcs": 1_292_527_311,
    "baseline_nemotron_full_bos_fixed_2000warcs": 1_968_904_506,
    # Nemotron-CC-HQ (quality=high only, both kind=actual + kind=synthetic).
    # 3k canonical built by tokenizing existing filtered/baseline_nemotron_qhigh-v1/
    # via run_quality_filter_region.py --skip-filter. Per-N subsamples built by
    # subset_baselines.py --methods nemotron_qhigh. Tokenized on us-central1 only;
    # pinned. d_obs values PENDING until tokenize lands.
    "baseline_nemotron_qhigh-v1": 801_420_584,
    "baseline_nemotron_qhigh_100warcs": 93_509_322,
    "baseline_nemotron_qhigh_500warcs": 235_506_216,
    "baseline_nemotron_qhigh_1000warcs": 386_117_427,
    "baseline_nemotron_qhigh_2000warcs": 585_125_784,
    # llm_curated BOS-fixed: us-central1 only.
    "baseline_llm_curated_bos_fixed_100warcs": 1_862_581_865,
    "baseline_llm_curated_bos_fixed_500warcs": 8_857_302_468,
    "baseline_llm_curated_bos_fixed_1000warcs": 19_007_039_689,
    "baseline_llm_curated_bos_fixed_2000warcs": 37_856_619_271,
    # llm_curated_dclm_filtered: full DCLM curation pipeline (filter +
    # bff dedup at old-both/13-gram/0.8) on the full 200-shard / 3000-WARC
    # llm_curated extraction. Built 2026-05-06; cache mirrored to us-central1 /
    # us-central2 / us-east5. 5.08M docs / 3.75B tokens after curation
    # (~6.7% of the 56B-token llm_curated_bos_fixed corpus survived filter+dedup).
    "llm_curated_dclm_filtered_v1": 3_746_596_900,
    # llm_curated_dclm_filtered WARC-scaling subsamples: same DCLM filter + bff
    # dedup pipeline applied to the per-N-WARC subset of the llm_curated
    # documents. Built 2026-05-09 via subset_one.py. Mirrored to us-central1 /
    # us-central2 / us-east5 (matching the 3000-WARC parent's mirror set).
    "baseline_llm_curated_dclm_filtered_100warcs": 139_571_843,
    "baseline_llm_curated_dclm_filtered_500warcs": 680_741_252,
    "baseline_llm_curated_dclm_filtered_1000warcs": 1_483_083_390,
    "baseline_llm_curated_dclm_filtered_2000warcs": 2_933_069_326,
    "low_quality_100warcs-98f9ef": 1_672_294_999,
    # low_quality_N WARC-scaling subsamples built 2026-05-17 via
    # dedup_extracted.py + tokenize_deduped_extracted.py in us-east5.
    # Mirrored to us-central1 / us-east1 / us-west4 (US-only — no eu-west4).
    "low_quality_500warcs-8e47af": 7_390_911_794,
    "low_quality_1000warcs-adbb79": 15_239_047_004,
    "low_quality_2000warcs-71e35e": 28_853_042_069,
    "high_quality_100warcs-e2ecfb": 413_651_686,
    "high_quality_500warcs-f32bb0": 1_879_608_841,
    "high_quality_1000warcs-bafd0c": 3_973_047_728,
    "high_quality_2000warcs-5d569a": 7_393_534_806,
    "high_quality_3000warcs-b9cfe2": 9_445_377_136,
    "med_quality_100warcs-1c23ed": 994_571_360,
    # random-300 draw (lpv1_300_done manifest) — the comparison point, distinct from the
    # scaling-curve med_quality_*warcs above (which use baseline_warcs_3000 prefixes).
    "med_quality_300warcs-b87e1b": 2_761_298_543,
    "low_quality_300warcs-2c2db2": 4_711_077_904,
    "med_quality_500warcs-da4724": 4_568_817_591,
    "med_quality_1000warcs-84becd": 9_626_306_890,
    "med_quality_2000warcs-37ac75": 18_079_574_033,
    "med_quality_3000warcs-73cd32": 26_931_419_795,
    "med_low_quality_100warcs-03c59e": 1_445_631_560,
    # fastpipe_v3 modernBERT-thresholded bands (keep top 100/80/60/40/20% by score), 10,364 WARCs.
    "fastpipe_v3_decon_10364warcs-77ee7f": 45_370_000_845,
    "fastpipe_v3_80_decon_10364warcs-143ee0": 37_966_939_668,
    "fastpipe_v3_60_decon_10364warcs-1f0b9a": 28_559_342_931,
    "fastpipe_v3_40_decon_10364warcs-54c951": 18_129_129_360,
    "fastpipe_v3_20_decon_10364warcs-141620": 8_658_430_656,
}
_SOURCE_BUCKET: str = "gs://marin-us-central2"
# BOS-fixed rebuilds were only tokenized on us-central1 (see rebuild_bos_fixed.py)
# so their d_obs must be read from this bucket if live verification is ever enabled.
_SOURCE_BUCKET_CENTRAL1: str = "gs://marin-us-central1"
_BOS_FIXED_CACHE_HASHES: set[str] = {
    "baseline_nemotron_full_bos_fixed-4b1ce7",
    "baseline_llm_curated_bos_fixed-d04ef8",
}


def _method(
    name: str,
    cache_hash: str,
    sampled_warcs: int = 3_000,
    reproduce_per_region: bool = False,
    skip_stats_read: bool = True,
    pin_region: str | None = None,
) -> CurationMethod:
    """Build a CurationMethod from a cache hash (the directory under tokenized/)."""
    if cache_hash not in _D_OBS_DEFAULTS:
        raise KeyError(f"No hardcoded D_obs for {cache_hash!r}. Add to _D_OBS_DEFAULTS.")
    d_obs = _D_OBS_DEFAULTS[cache_hash]
    if not skip_stats_read:
        # BOS-fixed caches only exist on us-central1; everything else is mirrored
        # from us-central2.
        source_bucket = _SOURCE_BUCKET_CENTRAL1 if cache_hash in _BOS_FIXED_CACHE_HASHES else _SOURCE_BUCKET
        source_stats_path = f"{source_bucket}/tokenized/{cache_hash}/"
        try:
            live = load_d_obs_from_stats(source_stats_path)
            if live != d_obs:
                logger.warning(
                    "Stale d_obs for %s: hardcoded=%d but live=%d.",
                    cache_hash,
                    d_obs,
                    live,
                )
                d_obs = live
        except Exception as e:
            logger.warning("Could not verify d_obs from %s (%s)", source_stats_path, e)
    return CurationMethod(
        name=name,
        tokenized_rel_path=f"tokenized/{cache_hash}/",
        d_obs_tokens=d_obs,
        sampled_warcs=sampled_warcs,
        reproduce_per_region=reproduce_per_region,
        pin_region=pin_region,
    )


def _grid_mix_method(name: str, base_cache_hash: str, grid_corpus: str, mixture_tag: str = "") -> GridMixCurationMethod:
    """An OLMIX-mixture arm over `grid_corpus`'s 24x5 grid.

    Reuses the BASE corpus's `d_obs` verbatim -- the grid cells partition exactly that corpus
    (verified: dclm 7.332B, high_quality 21.297B on both sides, same Llama-3.1 tokenizer), so
    `s` and the natural-epoch target match the base arm cell-for-cell. Anything else would make
    the mix-vs-natural comparison measure two things at once.

    `mixture_tag` selects the vendored solve; empty means the default lambda=0.05 file. The
    KL regularizer lambda sets how far the solve may pull away from the natural distribution,
    so a tagged arm differs from the default arm in that one number and nothing else.
    """
    if base_cache_hash not in _D_OBS_DEFAULTS:
        raise KeyError(f"No hardcoded D_obs for {base_cache_hash!r}. Add to _D_OBS_DEFAULTS.")
    return GridMixCurationMethod(
        name=name,
        tokenized_rel_path=f"tokenized/{base_cache_hash}/",
        d_obs_tokens=_D_OBS_DEFAULTS[base_cache_hash],
        sampled_warcs=EXPC_SAMPLED_WARCS,
        grid_corpus=grid_corpus,
        mixture_rel_path=f"experiments/data_mixing/mixtures/{grid_corpus}_R3e10_k20{mixture_tag}.json",
    )


METHODS: dict[str, CurationMethod] = {
    "dclm": _method("dclm", "baseline_dclm-23e9be"),
    "nemotron_org": _method("nemotron_org", "baseline_nemotron-c67de9"),
    "fineweb_edu": _method("fineweb_edu", "baseline_fineweb_edu-7a3bc5"),
    # Resiliparse: pinned to us-central1 for ExpC reuse from fixed-model and
    # consistency with llm_curated. `reproduce_per_region=True` is informational
    # — the cache could in principle be re-tokenized in another region — but
    # for ExpC we want a single canonical region to keep cross-experiment
    # dedup keys stable.
    "resiliparse": _method(
        "resiliparse",
        "baseline_resiliparse-7278c1",
        reproduce_per_region=True,
        # Cache merged-only mirrored to us-east5 on 2026-05-03 (254 GB, ~$5);
        # dropped pin_region so iris can dispatch to either region. Per-shard
        # build artifacts intentionally NOT copied (Levanter only reads
        # train/input_ids/{data,offsets} + train/shard_ledger.json at runtime;
        # cache.py:103).
    ),
    # BOS-fixed rebuilds — ONLY on us-central1 (not mirrored). Training that
    # uses these must pin to us-central1; pin_region enforces it at submit time.
    "nemotron_full_bos_fixed": _method(
        "nemotron_full_bos_fixed",
        "baseline_nemotron_full_bos_fixed-4b1ce7",
        pin_region="us-central1",
    ),
    "llm_curated_bos_fixed": _method(
        "llm_curated_bos_fixed",
        "baseline_llm_curated_bos_fixed-d04ef8",
        # Cache mirrored to us-east5 on 2026-04-28; dropped pin_region so iris
        # can dispatch to whichever region has v5p capacity. Original pin was
        # for when cache only existed in us-central1.
    ),
    # Fuzzy doc-deduped llm_curated (drop-in replacement for
    # llm_curated_bos_fixed; trains on 43.64B tokens vs 56.01B). Cache mirrored
    # to us-central1 / us-central2 / us-east5 (2026-05-05) so iris can dispatch
    # to whichever region has TPU capacity.
    "llm_curated_dedup": _method(
        "llm_curated_dedup",
        "baseline_llm_curated_deduped-c444e2",
    ),
    # DCLM-faithful curation pipeline (filter + bff dedup) applied to the
    # full 200-shard / 3000-WARC llm_curated extraction. Method name kept
    # distinct from the broken ``llm_curated_dclm_curated`` 30-shard attempt
    # (see scratch/broken_30shard_oneoff/). Cache will be mirrored to
    # us-central1 / us-central2 / us-east5 once tokenize completes.
    "llm_curated_dclm_filtered": _method(
        "llm_curated_dclm_filtered",
        "llm_curated_dclm_filtered_v1",
    ),
    # ExpC 10k re-extractions — placeholders. Replace cache_hash and the
    # corresponding _D_OBS_DEFAULTS entry once tokenization completes.
    # `_iter_valid_candidates` will raise if these are enumerated with d_obs=0.
    "dclm_10k": _method(
        "dclm_10k",
        "dclm_400m_1x_10k_dclm-3df0ba",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "nemotron_10k": _method(
        "nemotron_10k",
        "dclm_400m_1x_10k_nemotron_full-3dcb75",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    # CORE-v2-decontaminated variants of dclm_10k / nemotron_10k. NEW method names
    # (not a cache swap) so the non-decon runs stay intact and skip-if-done doesn't
    # skip these — run_name embeds the method name, not the cache hash. Caches are
    # lean-mirrored to us-east5; pin_region forces training there (v5p covers the
    # whole grid + co-located with the non-decon caches for a fair comparison).
    "dclm_10k_decon": _method(
        "dclm_10k_decon",
        "dclm_400m_1x_10k_dclm_decon-177776",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-east5",
    ),
    "nemotron_10k_decon": _method(
        "nemotron_10k_decon",
        "dclm_400m_1x_10k_nemotron_full_decon-e271a5",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-east5",
    ),
    # 10k natural-epoch methods. fineweb_edu_10k + fineweb_cc_10k are mirrored to all
    # 6 regions; high_quality_10k is mirrored to all EXCEPT us-central2 (2026-06-12).
    # So NO pin_region -> children float for capacity. high_quality lacks us-central2,
    # so the launcher MUST hard-constrain to the shared set via --allowed-regions
    # us-central1 us-east1 us-east5 us-west4 eu-west4 (see launch_10k_natural.py).
    "fineweb_edu_10k": _method(
        "fineweb_edu_10k",
        "dclm_400m_1x_10k_fineweb_edu-0a3143",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "high_quality_10k": _method(
        "high_quality_10k",
        "high_quality_decon_10364warcs-6451c8",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    # --- OLMIX-optimized mixture arms (registered 2026-08-01) ---
    # Same corpus, same tokenizer, same d_obs as the base arm above; the ONLY difference is the
    # sampling weights over the 24x5 topic/quality grid, fit at the locked target R=30B, k=20.
    # Grid cells live in us-east5 / us-central1 / europe-west4 / us-west4 but NOT us-central2,
    # so these MUST be launched with --allowed-regions excluding us-central2 (the v4 pool).
    "dclm_10k_mix": _grid_mix_method("dclm_10k_mix", "dclm_400m_1x_10k_dclm-3df0ba", "dclm_10k"),
    "high_quality_10k_mix": _grid_mix_method(
        "high_quality_10k_mix", "high_quality_decon_10364warcs-6451c8", "high_quality_10k"
    ),
    # --- Weaker-prior variants (registered 2026-08-03) ---
    # Same swarm, same fitted laws, same (R, k); only the KL regularizer changes, 0.05 -> 0.01.
    # Motivation: the lambda=0.05 mixes improved olmo_base_easy but did not transfer to DCLM
    # Core v2, and one hypothesis is that they stay too close to the natural distribution. At
    # lambda=0.01 the solve moves 38% (dclm) / 45% (hq) of the mixture mass off natural versus
    # 19% / 22% at the default, and reaches ~85-90% of the gain available at lambda=0.
    # Caveat to carry into any comparison: these were solved from a LATER fit instance than the
    # lambda=0.05 files (torch-LBFGS is not reproducible across worker environments), which adds
    # ~2% TV of fit noise on top of the lambda effect. See the mixture JSON's `fit_instance_note`.
    # Tagged `lambda0p01`, not `kl0p01`: a `k`-prefixed tag beside `_k20` reads as a second
    # repetition factor, which it is not.
    "dclm_10k_mix_lambda0p01": _grid_mix_method(
        "dclm_10k_mix_lambda0p01", "dclm_400m_1x_10k_dclm-3df0ba", "dclm_10k", mixture_tag="_lambda0p01"
    ),
    "high_quality_10k_mix_lambda0p01": _grid_mix_method(
        "high_quality_10k_mix_lambda0p01",
        "high_quality_decon_10364warcs-6451c8",
        "high_quality_10k",
        mixture_tag="_lambda0p01",
    ),
    # --- Core-v2-targeted mixture arms (registered 2026-08-05) ---
    # Same swarm, same (R=30B, k=20), same grid caches as the arms above. What changes is the
    # OBJECTIVE the per-task laws were fit against: DCLM Core v2's 22 tasks instead of the
    # 42-task OLMo Base-Easy bpb devset. This is the direct answer to the note on the
    # lambda0p01 arms above -- rather than hoping a bpb-optimal mixture transfers to Core v2,
    # these solve for Core v2 itself.
    #
    # Core v2 enters the fit as `1 - centered_accuracy`, i.e. an ERROR. `olmix_solve` minimises
    # a convex `sum(exp(t @ x))`; `cp.Maximize` of that is not DCP and ECOS refuses it, so the
    # target must be lower-is-better exactly as bpb is. argmin over mixtures == argmax of
    # centered accuracy.
    #
    # Three things to carry into any comparison:
    #   * IN-SAMPLE. Core v2 is the held-out set for the bpb objective (olmix_tasks builds the
    #     42-task devset by removing every Core v2 task). A mixture solved against Core v2 is
    #     optimised on the metric it will be judged by -- do not report it as a held-out win.
    #   * LOOSER LAWS. Accuracy is coarser and more quantised than bpb: average per-task fit
    #     correlation is 0.804 (dclm) / 0.738 (hq) against the bpb fits' 0.957.
    #   * OPPOSITE SHAPE. At lambda=0.05 the Core v2 solve stays CLOSER to natural than the bpb
    #     solve (TV 0.155 vs 0.190 dclm; 0.100 vs 0.222 hq) and far less peaked. The bpb devset
    #     is code/math-heavy and rewards concentration; Core v2 is commonsense QA + reading
    #     comprehension and rewards breadth.
    # Region constraint is identical to the arms above: grid cells are absent from us-central2.
    "dclm_10k_mix_corev2_lambda0p05": _grid_mix_method(
        "dclm_10k_mix_corev2_lambda0p05",
        "dclm_400m_1x_10k_dclm-3df0ba",
        "dclm_10k",
        mixture_tag="_corev2_lambda0p05",
    ),
    "dclm_10k_mix_corev2_lambda0p01": _grid_mix_method(
        "dclm_10k_mix_corev2_lambda0p01",
        "dclm_400m_1x_10k_dclm-3df0ba",
        "dclm_10k",
        mixture_tag="_corev2_lambda0p01",
    ),
    "high_quality_10k_mix_corev2_lambda0p05": _grid_mix_method(
        "high_quality_10k_mix_corev2_lambda0p05",
        "high_quality_decon_10364warcs-6451c8",
        "high_quality_10k",
        mixture_tag="_corev2_lambda0p05",
    ),
    "high_quality_10k_mix_corev2_lambda0p01": _grid_mix_method(
        "high_quality_10k_mix_corev2_lambda0p01",
        "high_quality_decon_10364warcs-6451c8",
        "high_quality_10k",
        mixture_tag="_corev2_lambda0p01",
    ),
    # --- BLEnD-objective arms (16 country tasks, everyday-life QA, gold bpb) ---
    # The one objective here that does NOT reward Science & Tech: it holds that topic at
    # 0.78-1.05x natural in all four fits while the bpb devset pushes it to 1.9-4.4x and Core v2
    # to 1.4-2.6x. Weight goes instead to Education & Jobs (2.6-4.2x in EVERY fit), plus Social
    # Life and History (dclm) or Fashion & Beauty and Travel (hq). TV(BLEnD, bpb) exceeds
    # TV(bpb, core_v2) in every fit, so this is a genuinely different direction rather than a
    # restatement of either -- it steers an everyday-life-vs-technical-prose TOPIC axis.
    #
    # Unlike the corev2 arms above, this objective is DISJOINT from both the bpb devset and Core
    # v2, so both remain genuinely held out for a model trained on these mixtures. Read the
    # per-file `fit_instance_note` first: gold-bpb scoring never sees BLEnD's cross-country
    # distractors, which is the construction that makes BLEnD a cultural test.
    # Region constraint is identical to the arms above: grid cells are absent from us-central2.
    "dclm_10k_mix_blend_lambda0p05": _grid_mix_method(
        "dclm_10k_mix_blend_lambda0p05",
        "dclm_400m_1x_10k_dclm-3df0ba",
        "dclm_10k",
        mixture_tag="_blend_lambda0p05",
    ),
    "dclm_10k_mix_blend_lambda0p01": _grid_mix_method(
        "dclm_10k_mix_blend_lambda0p01",
        "dclm_400m_1x_10k_dclm-3df0ba",
        "dclm_10k",
        mixture_tag="_blend_lambda0p01",
    ),
    "high_quality_10k_mix_blend_lambda0p05": _grid_mix_method(
        "high_quality_10k_mix_blend_lambda0p05",
        "high_quality_decon_10364warcs-6451c8",
        "high_quality_10k",
        mixture_tag="_blend_lambda0p05",
    ),
    "high_quality_10k_mix_blend_lambda0p01": _grid_mix_method(
        "high_quality_10k_mix_blend_lambda0p01",
        "high_quality_decon_10364warcs-6451c8",
        "high_quality_10k",
        mixture_tag="_blend_lambda0p01",
    ),
    # --- Dilution-ablation variants of high_quality ---
    # Base hq cache + both variant caches (hq_dense, hq_epoch_sub344) exist in BOTH
    # us-central1 and us-east5, so pin_region=None lets runs float across those two
    # (hard-constrained via --allowed-regions us-central1 us-east5 at launch — the
    # variant caches exist ONLY in those two regions). output_path is region-local
    # (run_curation_train_standalone.py:808), so no cross-region checkpoint egress.
    # hq_epoch: uniform subsample to DCLM's token count (composition unchanged) —
    # isolates the "fewer unique tokens / more epochs" confound.
    "hq_epoch": _method(
        "hq_epoch",
        "hq_epoch_sub344-f4dcd4",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    # hq_reweight: full hq cache upweighted with the dense expository partition so
    # dense char-share goes 12.3%→25% (=DCLM), at fixed token budget. Tests whether
    # concentrating fact-dense content closes the knowledge gap.
    "hq_reweight": ReweightedCurationMethod(
        name="hq_reweight",
        tokenized_rel_path="tokenized/high_quality_decon_10364warcs-6451c8/",
        d_obs_tokens=_D_OBS_DEFAULTS["high_quality_decon_10364warcs-6451c8"],
        sampled_warcs=EXPC_SAMPLED_WARCS,
        dense_rel_path="tokenized/hq_dense-7024eb/",
        dense_weight=0.144,
    ),
    "fineweb_cc_10k": _method(
        "fineweb_cc_10k",
        "fineweb_cc_10364warcs-ddfeda",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    # resiliparse_10k: tokenized in us-central2; lean copy (input_ids+ledger only,
    # no part-* build dirs) mirrored to us-east5 (2026-06-12, ~$12.6). NO pin ->
    # launch with --allowed-regions us-east5 us-central2 (the two it lives in).
    "resiliparse_10k": _method(
        "resiliparse_10k",
        "resiliparse_decon_10364warcs-beaaf5",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    # System-prompt-conditioned DCLM: [S][D] sequences trained as conditional
    # pretraining (p(D | S)). Cache lives ONLY in us-east5 (not mirrored), so
    # pin_region enforces us-east5 scheduling. Same 10k DCLM corpus underneath.
    "sysprompt_dclm": _method(
        "sysprompt_dclm",
        "sysprompt_dclm_qfull1-0ba2ee",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-east5",
    ),
    # 998M three-way system-prompt ablation on the DCLM-30B corpus @ 9e19.
    # Differ ONLY in data; optimizer HP are the frozen d1536@9e19 cell for all
    # three, and train_steps is one epoch of each arm's own cache.
    #   A  [S][D] prepended        (conditioned_text)
    #   B  token-matched baseline  (A's docs + extra, text)
    #   C  doc-matched baseline    (A's exact docs, text)
    # Caches exist ONLY in us-central1, so pin_region forces training there —
    # which is also where v5p-16 lives.
    "sysprompt30b_998m_9e19_A": _method(
        "sysprompt30b_998m_9e19_A",
        "sysprompt30b_998m_9e19_A-94f799",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-central1",
    ),
    "sysprompt30b_998m_9e19_B": _method(
        "sysprompt30b_998m_9e19_B",
        "sysprompt30b_998m_9e19_B-64d3a5",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-central1",
    ),
    "sysprompt30b_998m_9e19_C": _method(
        "sysprompt30b_998m_9e19_C",
        "sysprompt30b_998m_9e19_C-c94a3d",
        sampled_warcs=EXPC_SAMPLED_WARCS,
        pin_region="us-central1",
    ),
    # --- Random 3000-WARC sample (independent from the head-biased 3000-WARC
    #     methods; uniform draw seed 0, manifest baseline_warcs_3000_random.txt).
    #     Distinct IDs + cache hashes so these never collide with the existing
    #     3000-WARC runs. DCLM + Nemotron caches mirrored to all regions (float).
    #     resiliparse-dedup cache (432.8 GB): input_ids + ledger + stats copied
    #     central2->us-east5 (2026-05-31, ~$4.3, lean copy skipping vestigial
    #     part-dirs); floats us-central2/us-east5 like the head anchor -471baf to
    #     dodge us-central2 v4 starvation. (Full mirror incl eu-west4 was ~$69.)
    "dclm_random_3000": _method("dclm_random_3000", "baseline_dclm-cf177e", sampled_warcs=3000),
    "nemotron_full_random_3000": _method(
        "nemotron_full_random_3000", "baseline_nemotron_full-75f981", sampled_warcs=3000
    ),
    "resiliparse_random_dedup_3000": _method(
        "resiliparse_random_dedup_3000",
        "resiliparse_random_dedup_3000warcs-7de1e2",
        sampled_warcs=3000,
    ),
    # --- Random 100-WARC sample (10k pool, seed 0) — the unbiased counterpart to the
    #     head-biased dclm_100 / nemotron_full_100 / high_quality_100 (2013-only).
    #     Same 100 WARCs across all three. Caches mirrored to us-east5; pinned there.
    "dclm_random_100": _method(
        "dclm_random_100",
        "dclm_random_100warcs-350025",
        sampled_warcs=100,
        pin_region="us-east5",
    ),
    "high_quality_random_100": _method(
        "high_quality_random_100",
        "high_quality_random_100warcs-88f468",
        sampled_warcs=100,
        pin_region="us-east5",
    ),
    "nemotron_full_random_100": _method(
        "nemotron_full_random_100",
        "nemotron_full_random_100warcs-ade768",
        sampled_warcs=100,
        pin_region="us-east5",
    ),
    # Random 300/500 (nested seed-0 ladder). Caches mirrored to us-east5; pinned there.
    "dclm_random_300": _method(
        "dclm_random_300", "dclm_random_300warcs-cbd706", sampled_warcs=300, pin_region="us-east5"
    ),
    "nemotron_full_random_300": _method(
        "nemotron_full_random_300", "nemotron_full_random_300warcs-39307a", sampled_warcs=300, pin_region="us-east5"
    ),
    "high_quality_random_300": _method(
        "high_quality_random_300", "high_quality_random_300warcs-c6c5a5", sampled_warcs=300, pin_region="us-east5"
    ),
    # llm_pipeline_v1 = twin-pipeline two-stage extraction, decon'd (to match HQ),
    # same 300 seed-0 WARCs as the baselines. Data-rich (3.35B tok vs hq 639M).
    "llm_pipeline_v1_random_300": _method(
        "llm_pipeline_v1_random_300", "llm_pipeline_v1_decon_300warcs-b834c5", sampled_warcs=300, pin_region="us-east5"
    ),
    # llm_pipeline_v1_1 = markdown-extraction upgrade (mdx_d5 head/cont + guard suite) over the SAME 300
    # seed-0 random WARCs, decon'd to match HQ. Same filter as v1 → same 2.73M docs, 4.26B tok. Cache us-central1.
    "llm_pipeline_v1_1_random_300": _method(
        "llm_pipeline_v1_1_random_300",
        "llm_pipeline_v1_1_decon_300warcs-1563f5",
        sampled_warcs=300,
        pin_region="us-central1",
    ),
    # Same extractor at N=3000 (nested seed-0 ladder). Trains in the FIXED-MODEL sweep
    # (expFM_natural, 21 cells) — warc-scaling hard-fails at N=3000 since WARC_COUNTS
    # tops out at 2000. Cache lives only in us-central1, so pin_region enforces locality.
    "llm_pipeline_v1_1_random_3000": _method(
        "llm_pipeline_v1_1_random_3000",
        "llm_pipeline_v1_1_decon_3000warcs-aa7070",
        sampled_warcs=3000,
        pin_region="us-central1",
    ),
    # one-call llm extractor (llm_simple_v1) over the SAME 300 seed-0 random WARCs. Cache in us-central1.
    "llm_simple_v1_random_300": _method(
        "llm_simple_v1_random_300", "llm_simple_v1_decon_300warcs-b8a945", sampled_warcs=300, pin_region="us-central1"
    ),
    # med_quality LLM-extraction over the SAME 300 seed-0 random WARCs (no decon, matching the
    # quality-tier convention). Cache in us-central1 (consolidation lives there).
    "med_quality_random_300": _method(
        "med_quality_random_300", "med_quality_300warcs-b87e1b", sampled_warcs=300, pin_region="us-central1"
    ),
    # low_quality (LLM extraction, lowest quality tier) over the SAME 300 random WARCs, NO decon
    # (quality-tier convention). Deduped in us-east5 (the proven region for low_quality's dense
    # fuzzy-CC graph — us-central1 preemptible pool churns); cache + training pinned us-east5.
    "low_quality_random_300": _method(
        "low_quality_random_300", "low_quality_300warcs-2c2db2", sampled_warcs=300, pin_region="us-east5"
    ),
    # high_quality_v2 (new extractor) over the 300 random WARCs, decon'd. Cache in us-central1,
    # mirrored to us-east5 so the stuck 1e20 tail cells can train on us-east5-b v6e (v5p-32 there
    # is quota-tier-blocked; v6e has free capacity).
    "high_quality_v2_random_300": _method(
        "high_quality_v2_random_300", "high_quality_v2_decon_300warcs-3d73da", sampled_warcs=300, pin_region="us-east5"
    ),
    # fastpipe_v3 keep-top-X% bands (ModernBERT-prob), N=300, us-east5. Token-for-token quality/quantity sweep.
    "fastpipe_v3_100_random_300": _method(
        "fastpipe_v3_100_random_300", "fastpipe_v3_decon_300warcs-095914", sampled_warcs=300, pin_region="us-east5"
    ),
    "fastpipe_v3_80_random_300": _method(
        "fastpipe_v3_80_random_300", "fastpipe_v3_80_decon_300warcs-a2ea2c", sampled_warcs=300, pin_region="us-east5"
    ),
    "fastpipe_v3_60_random_300": _method(
        "fastpipe_v3_60_random_300", "fastpipe_v3_60_decon_300warcs-888150", sampled_warcs=300, pin_region="us-east5"
    ),
    "fastpipe_v3_40_random_300": _method(
        "fastpipe_v3_40_random_300", "fastpipe_v3_40_decon_300warcs-e667fb", sampled_warcs=300, pin_region="us-east5"
    ),
    "fastpipe_v3_20_random_300": _method(
        "fastpipe_v3_20_random_300", "fastpipe_v3_20_decon_300warcs-ff2322", sampled_warcs=300, pin_region="us-east5"
    ),
    "dclm_random_500": _method(
        "dclm_random_500", "dclm_random_500warcs-b8780e", sampled_warcs=500, pin_region="us-east5"
    ),
    "nemotron_full_random_500": _method(
        "nemotron_full_random_500", "nemotron_full_random_500warcs-3f7ddf", sampled_warcs=500, pin_region="us-east5"
    ),
    "high_quality_random_500": _method(
        "high_quality_random_500", "high_quality_random_500warcs-5cc14b", sampled_warcs=500, pin_region="us-east5"
    ),
    # N=1000 rung of the crossover ladder. hq pinned to us-central1 = co-located with its
    # 2.1B-token cache (largest), so zero mirror egress; dclm/nemotron mirror their smaller
    # us-central2 caches out to the v5p training regions.
    "dclm_random_1000": _method(
        "dclm_random_1000", "dclm_random_1000warcs-5a69c1", sampled_warcs=1000, pin_region="us-east5"
    ),
    "nemotron_full_random_1000": _method(
        "nemotron_full_random_1000", "nemotron_full_random_1000warcs-83bd90", sampled_warcs=1000, pin_region="us-east1"
    ),
    "high_quality_random_1000": _method(
        "high_quality_random_1000", "high_quality_random_1000warcs-2470d1", sampled_warcs=1000, pin_region="us-east5"
    ),
    # N=2000 rung. Same co-location strategy: hq stays us-central1 (4.2B cache local);
    # dclm/nemotron caches pre-copied from us-central2 to their v5p training regions.
    "dclm_random_2000": _method(
        "dclm_random_2000", "dclm_random_2000warcs-7ba7b6", sampled_warcs=2000, pin_region="us-east5"
    ),
    "nemotron_full_random_2000": _method(
        "nemotron_full_random_2000", "nemotron_full_random_2000warcs-7e5953", sampled_warcs=2000, pin_region="us-east1"
    ),
    "high_quality_random_2000": _method(
        "high_quality_random_2000", "high_quality_random_2000warcs-06a890", sampled_warcs=2000, pin_region="us-east5"
    ),
    # --- WARC-scaling sweep subsamples ---
    # dclm: mirrored to us-central1 / us-central2 / us-east1 / us-east5 — float
    # so children can claim v6e capacity in us-central2 / us-east5 when v5p is
    # contested in us-central1.
    "dclm_100": _method("dclm_100", "baseline_dclm_100warcs", sampled_warcs=100),
    "dclm_500": _method("dclm_500", "baseline_dclm_500warcs", sampled_warcs=500),
    "dclm_1000": _method("dclm_1000", "baseline_dclm_1000warcs", sampled_warcs=1000),
    "dclm_2000": _method("dclm_2000", "baseline_dclm_2000warcs", sampled_warcs=2000),
    # resiliparse: all 4 sizes mirrored to us-central1 + us-east5 — float to
    # both so children can pick up us-east5 capacity (especially v6e-4) when
    # us-central1 v5p is contested. One-time expense (~$46) to unblock
    # resiliparse_1000/2000 (was 0/30 + 0/23 stuck on us-central1).
    "resiliparse_100": _method("resiliparse_100", "baseline_resiliparse_100warcs", sampled_warcs=100),
    "resiliparse_500": _method("resiliparse_500", "baseline_resiliparse_500warcs", sampled_warcs=500),
    "resiliparse_1000": _method("resiliparse_1000", "baseline_resiliparse_1000warcs", sampled_warcs=1000),
    "resiliparse_2000": _method("resiliparse_2000", "baseline_resiliparse_2000warcs", sampled_warcs=2000),
    # resiliparse with per-N fuzzy dedup. N=100 smoke landed in eu-west4 and
    # was mirrored to 3 US regions; the N=3000 anchor cache `-471baf` lives
    # in us-central2 only (pinned). N=500/1000/2000 caches land in us-east5
    # and stay pinned there (no mirror per user direction — pin_region
    # enforces TPU scheduling there).
    "resiliparse_dedup_100": _method("resiliparse_dedup_100", "resiliparse_100warcs-7f8246", sampled_warcs=100),
    # N=3000 anchor for the resiliparse_dedup curve. Cache mirrored to
    # us-east5 on 2026-05-17 ($7.20, 360 GB) after FM children starved on
    # us-central2 v4 capacity; dropped pin_region so iris floats to either
    # us-central2 or us-east5.
    "resiliparse_dedup": _method(
        "resiliparse_dedup",
        "baseline_resiliparse_deduped-471baf",
    ),
    # N=300/500/1000/2000 entries — fill in cache_hash + d_obs once tokenize lands.
    "resiliparse_dedup_300": _method(
        "resiliparse_dedup_300",
        "resiliparse_dedup_300warcs-24d661",
        sampled_warcs=300,
        pin_region="us-east5",
    ),
    "resiliparse_dedup_500": _method(
        "resiliparse_dedup_500",
        "resiliparse_dedup_500warcs-a491f7",
        sampled_warcs=500,
        pin_region="us-east5",
    ),
    "resiliparse_dedup_1000": _method(
        "resiliparse_dedup_1000",
        "resiliparse_dedup_1000warcs-4384bb",
        sampled_warcs=1000,
        pin_region="us-east5",
    ),
    "resiliparse_dedup_2000": _method(
        "resiliparse_dedup_2000",
        "resiliparse_dedup_2000warcs-7e2171",
        sampled_warcs=2000,
        pin_region="us-east5",
    ),
    # nemotron_full BOS-fixed: mirrored to all 4 regions (us-central1/2 +
    # us-east1/5) — float for v6e access.
    "nemotron_full_100": _method("nemotron_full_100", "baseline_nemotron_full_bos_fixed_100warcs", sampled_warcs=100),
    "nemotron_full_500": _method("nemotron_full_500", "baseline_nemotron_full_bos_fixed_500warcs", sampled_warcs=500),
    "nemotron_full_1000": _method(
        "nemotron_full_1000", "baseline_nemotron_full_bos_fixed_1000warcs", sampled_warcs=1000
    ),
    "nemotron_full_2000": _method(
        "nemotron_full_2000", "baseline_nemotron_full_bos_fixed_2000warcs", sampled_warcs=2000
    ),
    # Nemotron-CC-HQ: quality=high subset of nemotron_full, both kind=actual +
    # kind=synthetic (matches Nvidia's "Nemotron-CC-HQ" definition).
    # Pinned to us-central1 — filter and tokenize both run there only.
    "nemotron_qhigh": _method(
        "nemotron_qhigh",
        "baseline_nemotron_qhigh-v1",
        pin_region="us-central1",
    ),
    # nemotron_qhigh_100/500: tokenized caches mirrored to us-central1/2 +
    # us-east1/5 (2026-05-12) to unblock TPU scheduling across US regions.
    # No pin_region so iris can dispatch to whichever region has capacity.
    "nemotron_qhigh_100": _method(
        "nemotron_qhigh_100",
        "baseline_nemotron_qhigh_100warcs",
        sampled_warcs=100,
    ),
    "nemotron_qhigh_500": _method(
        "nemotron_qhigh_500",
        "baseline_nemotron_qhigh_500warcs",
        sampled_warcs=500,
    ),
    "nemotron_qhigh_1000": _method(
        "nemotron_qhigh_1000",
        "baseline_nemotron_qhigh_1000warcs",
        sampled_warcs=1000,
        pin_region="us-central1",
    ),
    "nemotron_qhigh_2000": _method(
        "nemotron_qhigh_2000",
        "baseline_nemotron_qhigh_2000warcs",
        sampled_warcs=2000,
        pin_region="us-central1",
    ),
    # llm_curated BOS-fixed: all 4 sizes mirrored to us-central1 + us-east5,
    # float across both. One-time mirror cost paid 2026-05-01 to unblock the
    # remaining sizes that were starving on us-central1 capacity.
    "llm_curated_100": _method("llm_curated_100", "baseline_llm_curated_bos_fixed_100warcs", sampled_warcs=100),
    "llm_curated_500": _method("llm_curated_500", "baseline_llm_curated_bos_fixed_500warcs", sampled_warcs=500),
    "llm_curated_1000": _method("llm_curated_1000", "baseline_llm_curated_bos_fixed_1000warcs", sampled_warcs=1000),
    "llm_curated_2000": _method("llm_curated_2000", "baseline_llm_curated_bos_fixed_2000warcs", sampled_warcs=2000),
    # llm_curated_dclm_filtered WARC-scaling: filter + bff dedup applied to
    # per-N-WARC subset of llm_curated documents. Mirror set: us-central1,
    # us-central2, us-east5.
    "llm_curated_dclm_filtered_100": _method(
        "llm_curated_dclm_filtered_100",
        "baseline_llm_curated_dclm_filtered_100warcs",
        sampled_warcs=100,
    ),
    "llm_curated_dclm_filtered_500": _method(
        "llm_curated_dclm_filtered_500",
        "baseline_llm_curated_dclm_filtered_500warcs",
        sampled_warcs=500,
    ),
    "llm_curated_dclm_filtered_1000": _method(
        "llm_curated_dclm_filtered_1000",
        "baseline_llm_curated_dclm_filtered_1000warcs",
        sampled_warcs=1000,
    ),
    "llm_curated_dclm_filtered_2000": _method(
        "llm_curated_dclm_filtered_2000",
        "baseline_llm_curated_dclm_filtered_2000warcs",
        sampled_warcs=2000,
    ),
    "low_quality_100": _method("low_quality_100", "low_quality_100warcs-98f9ef", sampled_warcs=100),
    "low_quality_500": _method("low_quality_500", "low_quality_500warcs-8e47af", sampled_warcs=500),
    "low_quality_1000": _method("low_quality_1000", "low_quality_1000warcs-adbb79", sampled_warcs=1000),
    "low_quality_2000": _method("low_quality_2000", "low_quality_2000warcs-71e35e", sampled_warcs=2000),
    "high_quality_100": _method("high_quality_100", "high_quality_100warcs-e2ecfb", sampled_warcs=100),
    "high_quality_500": _method("high_quality_500", "high_quality_500warcs-f32bb0", sampled_warcs=500),
    "high_quality_1000": _method("high_quality_1000", "high_quality_1000warcs-bafd0c", sampled_warcs=1000),
    "high_quality_2000": _method("high_quality_2000", "high_quality_2000warcs-5d569a", sampled_warcs=2000),
    "high_quality_3000": _method("high_quality_3000", "high_quality_3000warcs-b9cfe2", sampled_warcs=3000),
    "med_quality_100": _method("med_quality_100", "med_quality_100warcs-1c23ed", sampled_warcs=100),
    "med_quality_500": _method("med_quality_500", "med_quality_500warcs-da4724", sampled_warcs=500),
    "med_quality_1000": _method("med_quality_1000", "med_quality_1000warcs-84becd", sampled_warcs=1000),
    "med_quality_2000": _method("med_quality_2000", "med_quality_2000warcs-37ac75", sampled_warcs=2000),
    "med_quality_3000": _method("med_quality_3000", "med_quality_3000warcs-73cd32", sampled_warcs=3000),
    "med_low_quality_100": _method("med_low_quality_100", "med_low_quality_100warcs-03c59e", sampled_warcs=100),
    # fastpipe_v3 modernBERT-thresholded bands (keep top X% of docs by score). Caches reconstructed
    # (byte-identical) in both us-east5 and us-central1, so no pin_region -> they float; restrict the
    # launch with --allowed-regions us-central1 us-east5. Natural epoching like dclm_10k etc.
    "fastpipe_v3_100": _method(
        "fastpipe_v3_100",
        "fastpipe_v3_decon_10364warcs-77ee7f",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "fastpipe_v3_80": _method(
        "fastpipe_v3_80",
        "fastpipe_v3_80_decon_10364warcs-143ee0",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "fastpipe_v3_60": _method(
        "fastpipe_v3_60",
        "fastpipe_v3_60_decon_10364warcs-1f0b9a",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "fastpipe_v3_40": _method(
        "fastpipe_v3_40",
        "fastpipe_v3_40_decon_10364warcs-54c951",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
    "fastpipe_v3_20": _method(
        "fastpipe_v3_20",
        "fastpipe_v3_20_decon_10364warcs-141620",
        sampled_warcs=EXPC_SAMPLED_WARCS,
    ),
}


# Methods that ExpC's `--methods all` should expand to. Excludes fineweb_edu
# (dropped from ExpC — strictly worse per upstream review) and the older 3k
# dclm/nemotron entries (superseded by *_10k for the data-constrained methods).
EXPC_METHOD_NAMES: tuple[str, ...] = (
    "dclm_10k",
    "nemotron_10k",
    "llm_curated_bos_fixed",
    "resiliparse",
)


# --- Experiment-tag formatter ------------------------------------------------

# (kind, t_target). kind ∈ {"A", "B", "C"}. t_target=None only valid for "A".
# resolve_experiments returns a list of these; enumerate_plans dispatches on kind.
ExperimentSpec = tuple[str, float | None]


def experiment_tag(t_target: float | None, kind: str = "B") -> str:
    """Format the experiment_tag string used in run names, WandB tags, tracker keys.

    `kind` distinguishes ExpB ("expB_T<N>T") from ExpC ("expC_T<N>T"); ignored
    when t_target is None (ExpA always emits "expA_natural"). Default "B"
    preserves the previous behavior for callers that don't specify a kind.
    """
    if t_target is None:
        return "expA_natural"
    if kind not in ("B", "C"):
        raise ValueError(f"Unsupported experiment kind {kind!r} for t_target={t_target}; expected 'B' or 'C'.")
    prefix = f"exp{kind}"
    trillions = t_target / 1e12
    if trillions == int(trillions):
        return f"{prefix}_T{int(trillions)}T"
    return f"{prefix}_T{trillions:.1f}T"


# --- PlannedRun: the coordinator → child contract ----------------------------


@dataclass(frozen=True)
class PlannedRun:
    """One planned training run. Serialized to/from CLI args at the coord/child boundary.

    The (model, optimizer) hyperparameters are passed explicitly so the child
    can rebuild a deterministic CandidateConfig without re-invoking the
    heuristic — which would risk drift if the heuristic is later edited.
    """

    method_name: str  # e.g. "dclm"
    experiment_tag: str  # e.g. "expA_natural" or "expB_T20T"
    budget: float  # FLOPs
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    batch_size: int
    train_steps: int
    learning_rate: float
    adam_lr: float
    epsilon: float
    beta1: float
    beta2: float
    t_exp: float  # tokens this run will train on
    t_target: float  # the target regime being simulated
    seq_len: int = SEQ_LEN
    tensor_parallel: int = 1
    z_loss_weight: float = 1e-7
    estimated_memory_bytes: int = 0
    v4_tpu: str = "v4-8"
    v5p_tpu: str = "v5p-8"
    v6e_tpu: str = ""  # empty when plan is too big for v6e single-vm slices
    cpu: float = 32.0
    memory_gb: int = 256
    """Container RAM limit. Was 128 GB, bumped to 256 after observed OOM on
    multi-host d4096-L40 smoke (peak 126 GB before kill). Smallest VM in our
    variants is v4-8 at ~400 GiB RAM — 256 GB leaves comfortable headroom
    while covering larger models. v5p-8 (448 GiB), v6e-4 (720 GiB) all fit.
    Larger plans (e.g. d4096+) might still push this; per-plan scaling TBD."""
    disk_gb: int = 50

    @property
    def run_name_core(self) -> str:
        # Stable identifier — used for both the iris job name and the output_path.
        return (
            f"curation-{self.method_name}-{self.experiment_tag}"
            f"-{self.budget:.0e}-d{self.hidden_dim}-L{self.num_layers}-B{self.batch_size}"
        )

    @property
    def run_key(self) -> str:
        """Stable key for the region tracker. Matches region_tracker.run_key_for(...) format."""
        return f"{self.method_name}__{self.experiment_tag}__{self.run_name_core}.region"

    def to_cli_args(self) -> list[str]:
        """Serialize to argv list for the standalone child script."""
        return [
            "--method",
            self.method_name,
            "--experiment-tag",
            self.experiment_tag,
            "--budget",
            f"{self.budget:.6e}",
            "--hidden-dim",
            str(self.hidden_dim),
            "--num-layers",
            str(self.num_layers),
            "--num-heads",
            str(self.num_heads),
            "--intermediate-dim",
            str(self.intermediate_dim),
            "--batch-size",
            str(self.batch_size),
            "--train-steps",
            str(self.train_steps),
            "--learning-rate",
            f"{self.learning_rate:.6e}",
            "--adam-lr",
            f"{self.adam_lr:.6e}",
            "--epsilon",
            f"{self.epsilon:.6e}",
            "--beta1",
            f"{self.beta1}",
            "--beta2",
            f"{self.beta2}",
            "--t-exp",
            f"{self.t_exp:.6e}",
            "--t-target",
            f"{self.t_target:.6e}",
            "--seq-len",
            str(self.seq_len),
            "--tensor-parallel",
            str(self.tensor_parallel),
            "--z-loss-weight",
            f"{self.z_loss_weight:.6e}",
        ]

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register the inverse of `to_cli_args` on a parser."""
        parser.add_argument("--method", required=True)
        parser.add_argument("--experiment-tag", required=True)
        parser.add_argument("--budget", type=float, required=True)
        parser.add_argument("--hidden-dim", type=int, required=True)
        parser.add_argument("--num-layers", type=int, required=True)
        parser.add_argument("--num-heads", type=int, required=True)
        parser.add_argument("--intermediate-dim", type=int, required=True)
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--train-steps", type=int, required=True)
        parser.add_argument("--learning-rate", type=float, required=True)
        parser.add_argument("--adam-lr", type=float, required=True)
        parser.add_argument("--epsilon", type=float, required=True)
        parser.add_argument("--beta1", type=float, required=True)
        parser.add_argument("--beta2", type=float, required=True)
        parser.add_argument("--t-exp", type=float, required=True)
        parser.add_argument("--t-target", type=float, required=True)
        parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
        parser.add_argument("--tensor-parallel", type=int, default=1)
        parser.add_argument("--z-loss-weight", type=float, default=1e-7)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> PlannedRun:
        """Inverse of `to_cli_args` — rebuild from parsed args (defaults set for irrelevant fields)."""
        return cls(
            method_name=ns.method,
            experiment_tag=ns.experiment_tag,
            budget=ns.budget,
            hidden_dim=ns.hidden_dim,
            num_layers=ns.num_layers,
            num_heads=ns.num_heads,
            intermediate_dim=ns.intermediate_dim,
            batch_size=ns.batch_size,
            train_steps=ns.train_steps,
            learning_rate=ns.learning_rate,
            adam_lr=ns.adam_lr,
            epsilon=ns.epsilon,
            beta1=ns.beta1,
            beta2=ns.beta2,
            t_exp=ns.t_exp,
            t_target=ns.t_target,
            seq_len=ns.seq_len,
            tensor_parallel=ns.tensor_parallel,
            z_loss_weight=ns.z_loss_weight,
            # Coordinator-side fields not needed in the child:
            estimated_memory_bytes=0,
            v4_tpu="",
            v5p_tpu="",
        )


# --- Enumeration -------------------------------------------------------------


def _iter_valid_candidates(
    method: CurationMethod,
    *,
    t_target: float | None,
    kind: str = "B",
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
    uniform_t_exp_cap: float | None = None,
) -> Iterator[tuple[float, CandidateConfig, int]]:
    """Yield (budget, candidate, target_budget) for each candidate that survives filtering.

    - Experiment A (kind="A", t_target=None): no ceiling, implicit target = T_exp * s.
    - Experiment B (kind="B", t_target set):  reject T_exp > ceiling; reject slice < floor.
    - Experiment C (kind="C", t_target set):  ceiling uses the data-rich branch
      (returns D_obs when target_epochs<1, T/s otherwise) AND a uniform cap
      (`uniform_t_exp_cap`) is applied across all methods for cross-method
      comparability. Min-slice floor is skipped for the data-rich case (the
      slicing path doesn't run there, so the floor is irrelevant).
    """
    if kind == "C" and method.d_obs_tokens <= 0:
        # Placeholder method (e.g. dclm_10k before its cache hash is filled in).
        # Refuse to enumerate to prevent silently emitting plans with bogus
        # slice math; surface the issue at the coordinator's dry-run.
        raise ValueError(
            f"Method {method.name!r} has d_obs_tokens={method.d_obs_tokens} — "
            f"likely an ExpC 10k cache placeholder. Update _D_OBS_DEFAULTS "
            f"and the cache_hash in METHODS once tokenization completes."
        )
    allow_data_rich = kind == "C"
    for budget in budgets:
        for cand in completed_adamh_heuristic.candidates_for_budget(budget, seq_len=seq_len):
            t_exp = cand.tokens
            if t_target is not None:
                ceiling = t_exp_ceiling(method, t_target, allow_data_rich=allow_data_rich)
                if uniform_t_exp_cap is not None:
                    ceiling = min(ceiling, uniform_t_exp_cap)
                if t_exp > ceiling:
                    continue
                # Min-slice floor only matters when slicing actually applies.
                # In the data-rich branch (target_epochs<1) the runner skips
                # slicing entirely, so a thin slice value is harmless.
                in_slicing_regime = t_target >= method.d_proj
                if in_slicing_regime and slice_tokens_for(method, t_exp, t_target) < min_slice_tokens:
                    continue
                target_budget = int(t_target)
            else:
                target_budget = int(implicit_target_exp_a(method, t_exp))
            yield budget, cand, target_budget


def _planned_run_from_candidate(
    method: CurationMethod,
    candidate: CandidateConfig,
    budget: float,
    target_budget: int,
    tag: str,
    seq_len: int = SEQ_LEN,
) -> PlannedRun:
    model = candidate.model_config
    opt = candidate.optimizer_config
    mem = completed_adamh_heuristic.estimate_memory_bytes(candidate)
    v4_tpu = pick_v4_type(mem)
    # Use delphi's canonical TP computation (completed_adamh._compute_tensor_parallel_size).
    tensor_parallel = _compute_tensor_parallel_size(v4_tpu, candidate.batch_size, model.hidden_dim)

    # v5p alternative: pick_v5p_type chooses the smallest v5p that fits the model's
    # memory. For large v4 plans (v4-32/v4-64), that v5p is often much smaller
    # (v5p-16/v5p-32) because v5p has 3x more HBM per chip. But the vm_count filter
    # in submit_one drops alternatives with mismatched vm_count — leaving multi-host
    # plans with NO fallback when v4 capacity is exhausted.
    #
    # Fix: when the memory-optimal v5p has a different vm_count than the v4 primary,
    # also try the v5p shape with MATCHING vm_count. More chips than needed, but it's
    # schedulable and v5p capacity is usually available when v4 is exhausted.
    v5p_tpu = pick_v5p_type(mem)
    v4_vm_count = get_tpu_topology(v4_tpu).vm_count
    if v4_vm_count > 1 and get_tpu_topology(v5p_tpu).vm_count != v4_vm_count:
        matching = f"v5p-{v4_vm_count * 8}"  # v5p-N: N = vm_count * 8 (cores)
        try:
            get_tpu_topology(matching)  # verify it's a valid shape
            v5p_tpu = matching
        except ValueError:
            pass  # no matching v5p shape exists; keep memory-optimal pick

    return PlannedRun(
        method_name=method.name,
        experiment_tag=tag,
        budget=budget,
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        intermediate_dim=model.intermediate_dim,
        batch_size=candidate.batch_size,
        train_steps=candidate.train_steps,
        learning_rate=opt.learning_rate,
        adam_lr=opt.adam_lr,
        epsilon=opt.epsilon,
        beta1=opt.beta1,
        beta2=opt.beta2,
        t_exp=candidate.tokens,
        t_target=float(target_budget),
        seq_len=seq_len,
        tensor_parallel=tensor_parallel,
        z_loss_weight=completed_adamh_heuristic.z_loss_weight,
        estimated_memory_bytes=mem,
        v4_tpu=v4_tpu,
        v5p_tpu=v5p_tpu,
        v6e_tpu=pick_v6e_type_single_vm(mem) or "",
    )


def _normalize_experiment_spec(item: float | None | ExperimentSpec) -> ExperimentSpec:
    """Accept either the legacy `float | None` form or an explicit ExperimentSpec.

    Legacy: None -> ("A", None), float -> ("B", float). Existing ExpA/ExpB
    callers can keep passing the bare float / None; ExpC callers pass tuples.
    """
    if isinstance(item, tuple):
        kind, t_target = item
        if kind not in ("A", "B", "C"):
            raise ValueError(f"Unknown experiment kind {kind!r}; expected A/B/C.")
        if kind == "A" and t_target is not None:
            raise ValueError("Experiment A must have t_target=None.")
        if kind != "A" and t_target is None:
            raise ValueError(f"Experiment {kind} requires t_target to be set.")
        return kind, t_target
    if item is None:
        return "A", None
    return "B", float(item)


def enumerate_plans(
    methods: list[CurationMethod],
    experiments: list[float | None] | list[ExperimentSpec],
    *,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> list[PlannedRun]:
    """Build the full list of `PlannedRun`s for the cartesian product of methods x experiments.

    `experiments` accepts mixed legacy and ExpC forms — see `_normalize_experiment_spec`.
    For ExpC, `_iter_valid_candidates` is invoked with the data-rich-aware
    ceiling and the `expc_uniform_t_exp_cap(t_target)` cross-method cap.
    """
    plans: list[PlannedRun] = []
    for method in methods:
        for raw in experiments:
            kind, t_target = _normalize_experiment_spec(raw)
            tag = experiment_tag(t_target, kind=kind if kind != "A" else "B")
            uniform_cap = expc_uniform_t_exp_cap(t_target) if (kind == "C" and t_target is not None) else None
            for budget, cand, target_budget in _iter_valid_candidates(
                method,
                t_target=t_target,
                kind=kind,
                budgets=budgets,
                min_slice_tokens=min_slice_tokens,
                seq_len=seq_len,
                uniform_t_exp_cap=uniform_cap,
            ):
                plans.append(_planned_run_from_candidate(method, cand, budget, target_budget, tag, seq_len))
    return plans


# --- Pure enumeration helpers (no submission) --------------------------------


def count_valid(
    method: CurationMethod,
    *,
    t_target: float | None,
    kind: str = "B",
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
    uniform_t_exp_cap: float | None = None,
) -> int:
    return sum(
        1
        for _ in _iter_valid_candidates(
            method,
            t_target=t_target,
            kind=kind,
            budgets=budgets,
            min_slice_tokens=min_slice_tokens,
            seq_len=seq_len,
            uniform_t_exp_cap=uniform_t_exp_cap,
        )
    )


def per_budget_counts(
    method: CurationMethod,
    *,
    t_target: float | None,
    kind: str = "B",
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
    uniform_t_exp_cap: float | None = None,
) -> dict[float, int]:
    counts: dict[float, int] = {b: 0 for b in budgets}
    for budget, _, _ in _iter_valid_candidates(
        method,
        t_target=t_target,
        kind=kind,
        budgets=budgets,
        min_slice_tokens=min_slice_tokens,
        seq_len=seq_len,
        uniform_t_exp_cap=uniform_t_exp_cap,
    ):
        counts[budget] += 1
    return counts


# --- CLI helpers (shared between coordinator and dry-run) --------------------


def resolve_methods(method_names: list[str], *, for_expc: bool = False) -> list[CurationMethod]:
    """Resolve --methods CLI to CurationMethod list.

    `for_expc=True`: "all" expands to EXPC_METHOD_NAMES (the four canonical ExpC
    methods: dclm_10k, nemotron_10k, llm_curated_bos_fixed, resiliparse).
    Default (`for_expc=False`): "all" expands to every registered method
    (preserves prior ExpA/ExpB launcher behavior).
    """
    if "all" in method_names:
        if for_expc:
            return [METHODS[n] for n in EXPC_METHOD_NAMES]
        return list(METHODS.values())
    missing = [m for m in method_names if m not in METHODS]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {list(METHODS)}")
    return [METHODS[m] for m in method_names]


def resolve_experiments(
    experiments: list[str],
    t_targets: list[float],
    *,
    t_target_c: float = DEFAULT_T_TARGET_C,
) -> list[ExperimentSpec]:
    """Resolve --experiments CLI flags to a list of ExperimentSpec.

    Returns tagged tuples (kind, t_target):
      - "A" / "all" → ("A", None)
      - "B" / "all" → ("B", t) for each t in t_targets
      - "C"         → ("C", t_target_c)

    Backward compat: launchers that expect the old `list[float | None]` form
    can pass through `enumerate_plans`, which accepts both via
    `_normalize_experiment_spec`. New ExpC callers should consume the tuple
    form directly so they can dispatch on kind (e.g. for region pinning).
    """
    resolved: list[ExperimentSpec] = []
    want_a = "A" in experiments or "all" in experiments
    want_b = "B" in experiments or "all" in experiments
    want_c = "C" in experiments  # "all" does NOT include C — C requires explicit opt-in
    if want_a:
        resolved.append(("A", None))
    if want_b:
        for t in t_targets:
            resolved.append(("B", float(t)))
    if want_c:
        resolved.append(("C", float(t_target_c)))
    return resolved


def print_dry_run(plans: list[PlannedRun]) -> None:
    """Pretty-print per-(method, experiment) counts and TPU pair distribution."""
    by_group: dict[tuple[str, str], list[PlannedRun]] = {}
    for p in plans:
        by_group.setdefault((p.method_name, p.experiment_tag), []).append(p)

    for (method, tag), group in sorted(by_group.items()):
        n = len(group)
        v4 = Counter(p.v4_tpu for p in group)
        v5p = Counter(p.v5p_tpu for p in group)
        v4_str = ", ".join(f"{k}:{v}" for k, v in sorted(v4.items()))
        v5p_str = ", ".join(f"{k}:{v}" for k, v in sorted(v5p.items()))
        print(f"{method:>15} | {tag:>14} | runs={n:>3} | v4=[{v4_str}]  v5p=[{v5p_str}]")
    print(f"\n  TOTAL: {len(plans)} runs")
