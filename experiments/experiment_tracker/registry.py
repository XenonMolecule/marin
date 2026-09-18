# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The experiment registry — THE hook for tracking a new experiment.

Add an `Experiment` to `EXPERIMENTS`: give it a stable id, a display name, a group
(sections on the dashboard, created on first use), and a zero-arg `probe` returning a
`probes.Reading`. Compose the primitives in `probes.py` (GCS counts, iris job states,
existence checklists, manual entries) or write any callable for bespoke experiments.
Probes run only on explicit refresh — keep each one to a handful of listings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from experiments.experiment_tracker.probes import (
    Reading,
    gcs_count,
    gcs_count_by,
    gcs_fs,
    manual,
)

ALL_BUCKETS = [
    "marin-us-east5",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
    "marin-us-west4",
    "marin-eu-west4",
]
# Focus set first (user 2026-09-12); hq + fastpipe_v1 deprioritized (pending evals stopped).
SWARM_CORPORA = {
    "lpv11 (true)": "lpv11_10k-",
    "fastpipe_v2": "lpv11_fastpipe_v2_10k-",
    "dclm": "dclm_10k-",
    "nemotron": "nemotron_full_10k-",
    "resiliparse": "resiliparse_10k-",
    "fineweb_cc": "fineweb_cc_10k-",
    "fineweb_edu": "fineweb_edu_10k-",
    "high_quality (deprio)": "high_quality_10k-",
    "fastpipe_v1 (deprio)": "lpv11_fastpipe_v1_10k-",
}
# Domain-target sweep v2 (post lambda-bug relaunch, 2026-09-14): per-corpus locked solves.
# dclm/nemotron = k20@100B l0.05; lpv11/rslp/fpv2 = k5@100B l0.05. Needles are FULL arm
# names so the killed lambda-bug generation and olmixexact arms can never be counted.
_V2 = [
    ("dclm_10k", "r100k20"),
    ("nemotron_full_10k", "r100k20"),
    ("lpv11_10k", "r100k5"),
    ("lpv11_fastpipe_v2_10k", "r100k5"),
    # resiliparse k5 killed 2026-09-14 (user: k20@100B is the way; rslp most bottlenecked).
    # k=20 twins of the big three, trained in parallel for the repetition-cap comparison.
    ("lpv11_10k", "r100k20"),
    ("resiliparse_10k", "r100k20"),
    ("lpv11_fastpipe_v2_10k", "r100k20"),
    ("fineweb_cc_10k", "r100k20"),
]
_V2_TARGETS = ("cultural", "mmlu_stem", "mmlu_humanities", "mmlu_social_sciences")
DOMAIN_V2_ARMS = {
    f"{corpus.removesuffix('_10k')} {t.removeprefix('mmlu_')} ({gen.removeprefix('r100')})": (
        f"{corpus}_mix_{t}_clp_{gen}_lambda0p05"
    )
    for corpus, gen in _V2
    for t in _V2_TARGETS
}

MIX_ARMS = {
    "dclm": "dclm_10k_mix_olmixexact",
    "old spec (hq)": "high_quality_10k_mix_olmixexact",
    "new spec (fastpipe_v1)": "lpv11_fastpipe_v1_10k_mix_olmixexact",
    "lpv11 (LLM extract)": "lpv11_10k_mix_olmixexact",
    "resiliparse": "resiliparse_10k_mix_olmixexact",
    "nemotron": "nemotron_full_10k_mix_olmixexact",
    "fineweb_cc": "fineweb_cc_10k_mix_olmixexact",
    "fineweb_edu": "fineweb_edu_10k_mix_olmixexact",
}
LPV11_FIT_PREFIX = "gs://marin-us-east5/metadata/olmix/lpv11_10k"
# The four choice_logprob solve dirs a corpus needs for full domain-mix coverage.
DOMAIN_FIT_DIRS = (
    "fit_mmlu_stem_clp_kl0p01",
    "fit_mmlu_humanities_clp_kl0p01",
    "fit_mmlu_social_sciences_clp_kl0p01",
    "fit_cultural_clp_kl0p01",
)
# Swarm-corpus ids as they appear under metadata/olmix/<corpus>/ (fit artifacts live in
# us-east5 with us-central1 as the fallback for corpora whose swarms ran there).
DOMAIN_MIX_CORPORA = {
    "lpv11 (true)": "lpv11_10k",
    "dclm": "dclm_10k",
    "resiliparse": "resiliparse_10k",
    "nemotron": "nemotron_full_10k",
    "high_quality": "high_quality_10k",
    "fastpipe_v1": "lpv11_fastpipe_v1_10k",
    "fastpipe_v2": "lpv11_fastpipe_v2_10k",
    "fineweb_cc": "fineweb_cc_10k",
    "fineweb_edu": "fineweb_edu_10k",
}


def domain_mix_progress() -> Reading:
    """Per-corpus count of solved domain mixes (x/4), one `ls` per corpus per bucket."""
    fs = gcs_fs()
    breakdown = []
    for label, corpus in DOMAIN_MIX_CORPORA.items():
        have: set[str] = set()
        for bucket in ("marin-us-east5", "marin-us-central1"):
            try:
                leaves = {p.rsplit("/", 1)[-1] for p in fs.ls(f"{bucket}/metadata/olmix/{corpus}", detail=False)}
            except FileNotFoundError:
                continue
            have |= {d for d in DOMAIN_FIT_DIRS if d in leaves}
        breakdown.append((label, len(have), len(DOMAIN_FIT_DIRS)))
    done = sum(d for _, d, _ in breakdown)
    return Reading(done=done, total=len(DOMAIN_FIT_DIRS) * len(breakdown), breakdown=breakdown).finalize()


@dataclass
class Experiment:
    id: str
    name: str
    group: str
    probe: Callable[[], Reading]
    unit: str = "runs"
    links: dict[str, str] = field(default_factory=dict)
    detail: str = ""


EXPERIMENTS: list[Experiment] = [
    # -- Domain-eval mixing (the paper's primary line) ------------------------------------
    Experiment(
        id="swarm_mmlu_evals",
        name="MMLU sl_verb evals over swarms",
        group="Domain mixing (MMLU + cultural)",
        unit="evals",
        detail="choice_logprob for all 57 subjects on every swarm proxy; feeds the category mixes.",
        probe=partial(
            gcs_count_by,
            ALL_BUCKETS,
            "metadata/olmix_swarm_mmlu/5shot",
            SWARM_CORPORA,
            per_group_total=363,
            unique_index=True,
            require_file="results.json",
        ),
    ),
    Experiment(
        id="swarm_cultural_evals",
        name="Cultural suite evals over swarms",
        group="Domain mixing (MMLU + cultural)",
        unit="evals",
        detail="BLEnD-MCQ + CulturalBench + FORK + NormAd (cultural_sl_verb), same proxies.",
        probe=partial(
            gcs_count_by,
            ALL_BUCKETS,
            "metadata/olmix_swarm_cultural/5shot",
            SWARM_CORPORA,
            per_group_total=363,
            unique_index=True,
            require_file="results.json",
        ),
    ),
    Experiment(
        id="domain_mixes",
        name="Domain mixes (4 per corpus)",
        group="Domain mixing (MMLU + cultural)",
        unit="mixes",
        detail="choice_logprob solves per corpus: STEM / humanities / social sciences / cultural.",
        probe=lambda: domain_mix_progress(),
    ),
    # -- Downstream mix training -----------------------------------------------------------
    Experiment(
        id="olmixexact_sweep",
        name="olmixexact 36-cell sweeps (8 arms)",
        group="Downstream mix training",
        unit="cells",
        detail="expFM fixed-model grids per mixture arm; per-arm breakdown below.",
        probe=partial(
            gcs_count_by,
            ["marin-us-central1"],
            "metadata/data_curation_10k_natural_results",
            MIX_ARMS,
            per_group_total=None,
        ),
    ),
    Experiment(
        id="olmixexact_bpb_evals",
        name="olmixexact OLMo-BPB evals (60-task waves)",
        group="Downstream mix training",
        unit="results",
        detail="Incremental expfm waves; feeds the paper head-to-head plots.",
        probe=partial(gcs_count, ALL_BUCKETS, "metadata/olmo_bpb_results", "olmixexact"),
    ),
    Experiment(
        id="domain_target_sweep_v2",
        name="Domain-target training runs v2 (d512 + d1536)",
        group="Downstream mix training",
        unit="cells",
        detail="4 targets x 20 cells per corpus on the locked lambda=0.05 solves "
        "(dclm/nemotron k20@100B, lpv11/rslp/fpv2 k5@100B; fwedu rides its natural arm). "
        "Counts only the *_r100k*_lambda0p05 generation, not the killed lambda-bug runs.",
        probe=partial(
            gcs_count_by,
            ["marin-us-central1"],
            "metadata/data_curation_10k_natural_results",
            DOMAIN_V2_ARMS,
            per_group_total=20,
        ),
    ),
    Experiment(
        id="domain_v2_model_evals",
        name="Domain-model evals (cultural + MMLU suites)",
        group="Downstream mix training",
        unit="evals",
        detail="results.json per trained v2 cell: cultural suite and 57-subject MMLU. "
        "Quarantined (stale/partial) results are excluded by construction.",
        probe=partial(
            gcs_count_by,
            ALL_BUCKETS,
            "metadata/cultural_target_models/5shot",
            {"cultural evals": "curation-"},
            per_group_total=None,
        ),
    ),
    Experiment(
        id="domain_v2_mmlu_evals",
        name="MMLU-model evals",
        group="Downstream mix training",
        unit="evals",
        detail="mmlu_target_models namespace; prioritized: nemotron + resiliparse humanities.",
        probe=partial(
            gcs_count_by,
            ALL_BUCKETS,
            "metadata/mmlu_target_models/5shot",
            {"mmlu evals": "curation-"},
            per_group_total=None,
        ),
    ),
    # -- Upstream corpora ------------------------------------------------------------------
    Experiment(
        id="fastpipe_v2_swarm",
        name="fastpipe_v2 proxy swarm",
        group="Upstream corpora",
        unit="proxies",
        detail="363 proxies must train+eval before fastpipe_v2 can get its own mixes.",
        probe=partial(
            gcs_count_by,
            ALL_BUCKETS,
            "metadata/olmix_swarm_results/lpv11_fastpipe_v2_10k",
            {"proxies": "olmix-"},
            per_group_total=363,
            unique_index=True,
        ),
    ),
    Experiment(
        id="fastpipe_v2_swarm_bpb",
        name="fastpipe_v2 swarm BPB evals",
        group="Upstream corpora",
        unit="evals",
        probe=partial(gcs_count, ALL_BUCKETS, "metadata/olmix_swarm_bpb", "lpv11_fastpipe_v2_10k", total=363),
    ),
    # -- Unrelated experiments (manual until wired) -----------------------------------------
    Experiment(
        id="hallucination_eval",
        name="lpv11 hallucination eval",
        group="Other paper experiments",
        unit="docs",
        detail="n-gram prefilter + Gemini judge; dev100 pass flagged 13/100 (leaked </think>, loops).",
        probe=partial(manual, done=100, total=100, status="manual", note="dev100 complete; full run not launched"),
    ),
    Experiment(
        id="corpus_diversity",
        name="Corpus diversity measures",
        group="Other paper experiments",
        probe=partial(manual, status="manual", note="placeholder — wire a probe when the experiment lands"),
    ),
]
