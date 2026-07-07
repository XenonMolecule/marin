# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-subcomponent loss-vs-tokens dashboard for the 10k-natural sweep.

The headline figures average loss across all Paloma / Uncheatable-Eval
datasets (a "macro loss"). This script instead renders one loss-vs-tokens
figure *per subcomponent* — each of the 16 Paloma datasets, each of the 7
Uncheatable-Eval datasets, and LIMA — so you can check whether a method
genuinely wins on every constituent, not just on the average.

For each subcomponent it also determines the **winning method** per model
size (lowest best-achieved loss) and an overall winner, then emits a
self-contained HTML dashboard with three tabs (Uncheatable / Paloma / LIMA).
Cards can be re-sorted by winning method or by how much "ours" (Spec-Driven
Extraction) beats the best competitor, so you can see which method dominates
the loss landscape.

The per-figure rendering reuses `plot_loss_vs_tokens_by_size.render_figure`
(which already handles arbitrary `eval/.../loss` metric keys), so the panels
match the paper figures in style. Images are base64-embedded, so the HTML is
a single portable file.

Usage:

    # Re-use already-pulled 10k summaries (fast inner loop):
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_subcomponents_dashboard.py

    # Refresh 10k summaries from gs:// first, then build:
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_subcomponents_dashboard.py --pull
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX as TENK_RESULTS_GS,
)
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    CODE_BPB,
    MATH_BPB,
    MT_MBPP_BPB,
    OLMO_BASE_EASY_BPB,
    QA_RC_MC_BPB,
)
from experiments.scaling_law_sweeps.plot_curation_isoflop import (
    PALOMA_DATASETS,
    UNCHEATABLE_EVAL_DATASETS,
    load_summaries,
)
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import (
    CHINCHILLA_TOKENS_PER_PARAM,
    METHOD_DISPLAY,
    METRIC_LABELS,
    TENK_METHOD_MAP,
    Point,
    apply_tenk_rename,
    collect_points,
    pull_gs_to_local,
    render_figure,
)

logger = logging.getLogger(__name__)

TENK_LOCAL_PREFIX = "scratch/fm10k_summaries/"
DEFAULT_OUTPUT_DIR = "scratch/plots/subcomponents_10k"

# The six distinct natural curation methods, by their `_10k` load names.
# apply_tenk_rename rewrites these to their base names (used by METHOD_DISPLAY).
DEFAULT_LOAD_METHODS: tuple[str, ...] = (
    "dclm_10k",
    "nemotron_10k",
    "high_quality_10k",
    "resiliparse_10k",
    "fineweb_edu_10k",
    "fineweb_cc_10k",
)
OURS_METHOD = "high_quality"  # base name after tenk-rename
DEFAULT_WIDTHS: tuple[int, ...] = (512, 1024, 1536, 2432, 3584)

# OLMo Base-Easy bits-per-byte results (a per-checkpoint loss metric) live in a separate,
# region-sharded location keyed by the same run_name, so they are joined in per run_name.
OLMO_BPB_LOCAL = "scratch/olmo_bpb_results/"
OLMO_BPB_BUCKETS: tuple[str, ...] = (
    "marin-us-east5",
    "marin-eu-west4",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
)
# Aggregate macro cards reproducing AI2's `olmo3:base_easy:*` suites (olmes
# oe_eval/configs/task_suites.py): each is a `macro` over its members, and a member that is
# itself a tuple is an inner macro (mt_mbpp over 17 languages; arc over challenge+easy;
# basic_skills over its splits) so each counts once. code_bpb and math_bpb are EXACT. qa_bpb
# is AI2's 21-task suite minus MMLU (absent from the in-loop bundle), with the 5 generation
# tasks (coqa/drop/jeopardy/naturalqs/squad) using generation bpb rather than AI2's gen2mc.
MINERVA_MATH_BPB: tuple[str, ...] = tuple(
    t for t in MATH_BPB if t.startswith("minerva_math_") and not t.startswith("minerva_math_500")
)
_BASIC_SKILLS_BPB: tuple[str, ...] = tuple(t for t in QA_RC_MC_BPB if t.startswith("basic_skills_"))
# AI2 olmo3:base_easy qa_bpb, MINUS mmlu (absent from the in-loop bundle). arc and basic_skills
# are inner macros (matching AI2); coqa/drop/jeopardy/naturalqs/squad use generation bpb here,
# not AI2's gen2mc formulation.
QA_BPB_MEMBERS: tuple = (
    ("arc_challenge/rc_5shot", "arc_easy/rc_5shot"),
    "csqa/rc_5shot",
    "hellaswag/rc_5shot",
    "winogrande/rc_5shot",
    "socialiqa/rc_5shot",
    "piqa/rc_5shot",
    "coqa/bpb_0shot",
    "drop/bpb_5shot",
    "jeopardy/bpb_5shot",
    "naturalqs_open/bpb_5shot",
    "squad/bpb_5shot",
    "sciq/rc_5shot",
    "qasper_yesno/rc_5shot",
    _BASIC_SKILLS_BPB,
    "lab_bench_dbqa/rc_3shot",
    "lab_bench_protocolqa/rc_3shot",
    "lambada/bpb_0shot",
    "medmcqa/rc_5shot",
    "medqa_en/rc_5shot",
    "sciriff_yesno/rc_5shot",
)
OLMO_BPB_CATEGORIES: tuple[tuple[str, str, tuple], ...] = (
    (
        "code_bpb",
        "code_bpb (AI2 olmo3:base_easy)",
        ("codex_humaneval/gold_bpb_3shot", "codex_mbpp/gold_bpb_3shot", MT_MBPP_BPB),
    ),
    ("math_bpb", "math_bpb (AI2 olmo3:base_easy)", MINERVA_MATH_BPB),
    ("qa_bpb", "qa_bpb (AI2 olmo3:base_easy, minus MMLU)", QA_BPB_MEMBERS),
)
# All OLMo bpb tasks rendered as individual cards: the 36 gold-continuation tasks + 20 MC-QA rc tasks.
OLMO_ALL_TASK_CARDS: tuple[str, ...] = OLMO_BASE_EASY_BPB + QA_RC_MC_BPB


@dataclass(frozen=True)
class Subcomponent:
    group: str  # "uncheatable" | "paloma" | "lima"
    key: str  # dataset id (or "lima")
    metric_key: str  # canonical eval/.../loss key
    display: str  # human-friendly label


def prettify(ds: str) -> str:
    """Turn a dataset id like `m2d2_s2orc_unsplit` into a readable label."""
    special = {
        "c4_en": "C4 (en)",
        "c4_100_domains": "C4 100-domains",
        "ptb": "Penn Treebank",
        "wikitext_103": "WikiText-103",
        "mc4": "mC4",
        "m2d2_s2orc_unsplit": "M2D2 S2ORC",
        "m2d2_wikipedia_unsplit": "M2D2 Wikipedia",
        "dolma-v1_5": "Dolma v1.5",
        "dolma_100_programing_languages": "Dolma 100 prog-langs",
        "dolma_100_subreddits": "Dolma 100 subreddits",
        "falcon-refinedweb": "Falcon RefinedWeb",
        "twitterAAE_HELM_fixed": "TwitterAAE (HELM)",
        "manosphere_meta_sep": "Manosphere",
        "4chan": "4chan",
        "gab": "Gab",
        "redpajama": "RedPajama",
        "arxiv_computer_science": "arXiv CS",
        "arxiv_physics": "arXiv Physics",
        "bbc_news": "BBC News",
        "github_cpp": "GitHub C++",
        "github_python": "GitHub Python",
        "ao3_english": "AO3 (English)",
        "wikipedia_english": "Wikipedia (English)",
    }
    return special.get(ds, ds.replace("_", " ").title())


# One-line factual descriptions + canonical source URL per subcomponent, keyed by
# the dataset id (Subcomponent.key). Researched and verified against the source
# projects/papers — NOT inferred from the dataset name. See PR/commit for the
# per-source citations. Uncheatable-Eval sources are freshly-crawled post-cutoff
# text (delphi's frozen snapshot in this sweep); Paloma bundles external corpora.
DATASET_INFO: dict[str, tuple[str, str]] = {
    # --- Uncheatable-Eval (recent, post-training-cutoff text) ---
    "ao3_english": (
        "Recently posted English fan-fiction from Archive of Our Own.",
        "https://archiveofourown.org",
    ),
    "arxiv_computer_science": (
        "Recently submitted arXiv Computer Science preprints.",
        "https://arxiv.org/list/cs/recent",
    ),
    "arxiv_physics": (
        "Recently submitted arXiv Physics preprints.",
        "https://arxiv.org/list/physics/recent",
    ),
    "bbc_news": (
        "Recent news articles scraped from BBC News.",
        "https://www.bbc.com/news",
    ),
    "github_cpp": (
        "Source code from recently updated C++ GitHub repositories.",
        "https://github.com",
    ),
    "github_python": (
        "Source code from recently updated Python GitHub repositories.",
        "https://github.com",
    ),
    "wikipedia_english": (
        "Recently created/edited English Wikipedia article text.",
        "https://en.wikipedia.org",
    ),
    # --- Paloma: web & social ---
    "4chan": (
        "Posts from 4chan's /pol/ imageboard (Papasavva et al. 2020, “Raiders of the Lost Kek”).",
        "https://arxiv.org/abs/2001.07487",
    ),
    "c4_en": (
        "English C4 — web text auto-filtered from an April-2019 Common Crawl scrape (T5).",
        "https://arxiv.org/abs/1910.10683",
    ),
    "c4_100_domains": (
        "Balanced samples from C4's top-100 web domains (Chronopoulou et al. 2021).",
        "https://arxiv.org/abs/2112.08786",
    ),
    "mc4": (
        "English slice of the multilingual C4 web corpus (mT5).",
        "https://arxiv.org/abs/2010.11934",
    ),
    "manosphere_meta_sep": (
        "Posts from ~9 “manosphere” forums (Ribeiro et al. 2020, Manosphere Corpus).",
        "https://arxiv.org/abs/2001.07600",
    ),
    "gab": (
        "2016\u20132018 posts from Gab, an alt-right Twitter alternative (Zannettou et al. 2018).",
        "https://arxiv.org/abs/1802.05287",
    ),
    "twitterAAE_HELM_fixed": (
        "Tweets in African-American- vs White-aligned English (Blodgett et al. 2016, via HELM).",
        "https://arxiv.org/abs/1608.08868",
    ),
    "falcon-refinedweb": (
        "Heavily filtered & deduplicated English Common Crawl web text (Falcon RefinedWeb).",
        "https://arxiv.org/abs/2306.01116",
    ),
    "redpajama": (
        "Open reproduction of the LLaMA pretraining mix (web + books, arXiv, Wikipedia, code).",
        "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T",
    ),
    # --- Paloma: curated & academic ---
    "dolma-v1_5": (
        "Held-out validation from Dolma v1.5, AI2's ~3T-token open pretraining corpus (OLMo).",
        "https://huggingface.co/datasets/allenai/dolma",
    ),
    "dolma_100_programing_languages": (
        "Balanced held-out text from Dolma's top-100 programming languages.",
        "https://huggingface.co/datasets/allenai/paloma",
    ),
    "dolma_100_subreddits": (
        "Balanced held-out text from Dolma's top-100 subreddits.",
        "https://huggingface.co/datasets/allenai/paloma",
    ),
    "m2d2_s2orc_unsplit": (
        "S2ORC academic papers from M2D2, organized by an arXiv-category hierarchy (Reid et al. 2022).",
        "https://arxiv.org/abs/2210.07370",
    ),
    "m2d2_wikipedia_unsplit": (
        "Wikipedia article text from M2D2, organized by the Wikipedia category ontology (Reid et al. 2022).",
        "https://arxiv.org/abs/2210.07370",
    ),
    "ptb": (
        "Penn Treebank — preprocessed Wall Street Journal news, the classic LM benchmark.",
        "https://catalog.ldc.upenn.edu/LDC99T42",
    ),
    "wikitext_103": (
        "WikiText-103 — ~100M tokens from verified “Good/Featured” Wikipedia articles (Merity et al. 2016).",
        "https://arxiv.org/abs/1609.07843",
    ),
    "lima": (
        "LIMA — 1,000 curated prompt\u2013response pairs for instruction tuning (Zhou et al. 2023).",
        "https://arxiv.org/abs/2305.11206",
    ),
}


# --- OLMo Base-Easy bpb descriptions -----------------------------------------
# Per-category macro blurbs + per-benchmark base descriptions; per-task DATASET_INFO
# entries are generated from these (base desc + shot count + gold-continuation note).
OLMO_MACRO_INFO: dict[str, str] = {
    "code_bpb": (
        "AI2 olmo3:base_easy code_bpb — macro over HumanEval (3-shot), MBPP (3-shot), and "
        "multilingual MBPP (inner macro over 17 languages). Gold-solution bits-per-byte. "
        "bpb = -log2 P(gold) / gold_bytes; lower is better."
    ),
    "math_bpb": (
        "AI2 olmo3:base_easy math_bpb — macro over the 7 MATH (Minerva) subject splits, "
        "gold-solution bits-per-byte. (GSM8K and MATH-500 are separate task cards, not in this macro.)"
    ),
    "qa_bpb": (
        "AI2 olmo3:base_easy qa_bpb — macro over 20 of its 21 rc:bpb QA tasks (arc, csqa, hellaswag, "
        "winogrande, socialiqa, piqa, sciq, qasper_yesno, basic_skills, lab_bench, medmcqa, medqa_en, "
        "sciriff_yesno as rc:bpb; coqa/drop/jeopardy/naturalqs/squad/lambada as generation bpb). "
        "TWO deviations from AI2: MMLU is omitted (absent from the in-loop bundle), and the 5 "
        "generation tasks use plain generation bpb rather than AI2's gen2mc formulation."
    ),
}

OLMO_BENCHMARK_INFO: dict[str, tuple[str, str]] = {
    "codex_humaneval": (
        "HumanEval hand-written Python function-completion problems (Chen et al. 2021).",
        "https://arxiv.org/abs/2107.03374",
    ),
    "codex_mbpp": (
        "MBPP crowd-sourced entry-level Python programming problems (Austin et al. 2021).",
        "https://arxiv.org/abs/2108.07732",
    ),
    "mt_mbpp": (
        "MBPP programming problems in {lang} (multilingual MBPP / MultiPL-E, Cassano et al. 2022).",
        "https://arxiv.org/abs/2208.08227",
    ),
    "gsm8k": (
        "GSM8K grade-school math word problems (Cobbe et al. 2021).",
        "https://arxiv.org/abs/2110.14168",
    ),
    "minerva_math": (
        "MATH competition mathematics, {subject} (Hendrycks et al. 2021; Minerva prompting).",
        "https://arxiv.org/abs/2103.03874",
    ),
    "coqa": ("CoQA conversational question answering (Reddy et al. 2019).", "https://arxiv.org/abs/1808.07042"),
    "drop": ("DROP discrete reasoning over paragraphs (Dua et al. 2019).", "https://arxiv.org/abs/1903.00161"),
    "jeopardy": ("Jeopardy! trivia question answering.", ""),
    "lambada": (
        "LAMBADA last-word prediction over narrative passages (Paperno et al. 2016).",
        "https://arxiv.org/abs/1606.06031",
    ),
    "naturalqs_open": (
        "Natural Questions open-domain QA (Kwiatkowski et al. 2019).",
        "https://aclanthology.org/Q19-1026/",
    ),
    "squad": ("SQuAD reading-comprehension QA (Rajpurkar et al. 2016).", "https://arxiv.org/abs/1606.05250"),
    "arc_challenge": (
        "ARC-Challenge grade-school science multiple-choice (Clark et al. 2018).",
        "https://arxiv.org/abs/1803.05457",
    ),
    "arc_easy": (
        "ARC-Easy grade-school science multiple-choice (Clark et al. 2018).",
        "https://arxiv.org/abs/1803.05457",
    ),
    "csqa": ("CommonsenseQA commonsense multiple-choice (Talmor et al. 2019).", "https://arxiv.org/abs/1811.00937"),
    "hellaswag": (
        "HellaSwag commonsense sentence completion (Zellers et al. 2019).",
        "https://arxiv.org/abs/1905.07830",
    ),
    "winogrande": (
        "WinoGrande Winograd-schema commonsense (Sakaguchi et al. 2019).",
        "https://arxiv.org/abs/1907.10641",
    ),
    "socialiqa": (
        "Social IQa social-commonsense multiple-choice (Sap et al. 2019).",
        "https://arxiv.org/abs/1904.09728",
    ),
    "piqa": ("PIQA physical-commonsense multiple-choice (Bisk et al. 2020).", "https://arxiv.org/abs/1911.11641"),
    "sciq": ("SciQ crowd-sourced science-exam multiple-choice (Welbl et al. 2017).", "https://arxiv.org/abs/1707.06209"),
    "qasper_yesno": (
        "Qasper yes/no questions over NLP papers (Dasigi et al. 2021).",
        "https://arxiv.org/abs/2105.03011",
    ),
    "basic_skills": ("Basic-skills probe — {subject} (AI2 OLMo in-loop basic-skills set).", ""),
    "lab_bench_dbqa": (
        "LAB-Bench DbQA — biology database QA (Laurent et al. 2024).",
        "https://arxiv.org/abs/2407.10362",
    ),
    "lab_bench_protocolqa": (
        "LAB-Bench ProtocolQA — lab-protocol QA (Laurent et al. 2024).",
        "https://arxiv.org/abs/2407.10362",
    ),
    "medmcqa": ("MedMCQA medical-entrance multiple-choice (Pal et al. 2022).", "https://arxiv.org/abs/2203.14371"),
    "medqa_en": ("MedQA (English) USMLE medical multiple-choice (Jin et al. 2020).", "https://arxiv.org/abs/2009.13081"),
    "sciriff_yesno": (
        "SciRIFF yes/no scientific-literature QA (Wadden et al. 2024).",
        "https://arxiv.org/abs/2406.07835",
    ),
}


def _olmo_shots(variant: str) -> int:
    m = re.search(r"(\d+)shot", variant)
    return int(m.group(1)) if m else 0


def build_olmo_dataset_info() -> dict[str, tuple[str, str]]:
    """DATASET_INFO entries for the 4 category macros + 36 per-task bpb cards."""
    info: dict[str, tuple[str, str]] = {c: (OLMO_MACRO_INFO[c], "") for c, _l, _t in OLMO_BPB_CATEGORIES}
    code_or_math = set(CODE_BPB + MT_MBPP_BPB + MATH_BPB)
    for task in OLMO_ALL_TASK_CARDS:
        base, variant = task.split("/")
        lang = subject = ""
        if base.startswith("mt_mbpp_"):
            bench, lang = "mt_mbpp", base[len("mt_mbpp_") :]
        elif base.startswith("minerva_math_"):
            bench = "minerva_math"
            sub = base[len("minerva_math_") :]
            subject = "the MATH-500 subset" if sub == "500" else sub.replace("_", " ")
        elif base.startswith("basic_skills_"):
            bench, subject = "basic_skills", base[len("basic_skills_") :].replace("_", " ")
        else:
            bench = base
        desc, url = OLMO_BENCHMARK_INFO.get(bench, (base.replace("_", " ").title(), ""))
        desc = desc.format(lang=lang.title(), subject=subject)
        kind = "solution" if task in code_or_math else "answer"
        info[task] = (f"{desc} Scored as bits-per-byte of the gold {kind} ({_olmo_shots(variant)}-shot).", url)
    return info


DATASET_INFO.update(build_olmo_dataset_info())


def _olmo_slug(task: str) -> str:
    """oe-eval task dir ('codex_humaneval/gold_bpb_0shot') -> flat metric-key segment."""
    return task.replace("/", "__")


def _olmo_metric_key(task_or_category: str) -> str:
    return f"eval/olmo_bpb/{task_or_category}/bpb"


def prettify_olmo_task(task: str) -> str:
    return task.replace("/", " · ").replace("_", " ")


def load_olmo_bpb(local_dir: str) -> dict[str, dict[str, float]]:
    """Read pulled OLMo bpb results -> {run_name: {task: bpb}}. Skips smoke/capped runs."""
    root = Path(local_dir)
    by_run: dict[str, dict[str, float]] = {}
    for res in sorted(root.glob("*/results.json")):
        run_name = res.parent.name
        if run_name.endswith("-smoke"):
            continue
        data = json.loads(res.read_text())
        if data.get("limit") is not None:
            continue  # only full (non-smoke) runs
        tasks = {t: v["bpb"] for t, v in data.get("tasks", {}).items() if "bpb" in v}
        if tasks:
            by_run[run_name] = tasks
    return by_run


def inject_olmo_bpb(summaries: list[dict], by_run: dict[str, dict[str, float]]) -> int:
    """Join OLMo bpb onto each summary's `eval` dict by run_name; also add category macros.

    Returns the number of summaries that received bpb keys.
    """
    n = 0
    for s in summaries:
        run_name = s.get("plan", {}).get("run_name")
        tasks = by_run.get(run_name) if run_name else None
        if not tasks:
            continue
        eval_d = s.setdefault("eval", {})
        for task, bpb in tasks.items():
            eval_d[_olmo_metric_key(_olmo_slug(task))] = bpb
        for cat_key, _label, members in OLMO_BPB_CATEGORIES:
            member_vals: list[float] = []
            for member in members:
                names = (member,) if isinstance(member, str) else member
                vals = [tasks[t] for t in names if t in tasks]
                if vals:  # inner macro (single task or, e.g., mt_mbpp over its languages)
                    member_vals.append(sum(vals) / len(vals))
            if member_vals:
                eval_d[_olmo_metric_key(cat_key)] = sum(member_vals) / len(member_vals)
        n += 1
    return n


def pull_olmo_bpb(local_dir: str) -> None:
    """Mirror each region bucket's olmo_bpb_results into local_dir (small, one-time)."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for bucket in OLMO_BPB_BUCKETS:
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"gs://{bucket}/metadata/olmo_bpb_results/*", local_dir],
            check=False,
        )


def build_subcomponents(olmo_available: bool = False) -> list[Subcomponent]:
    subs: list[Subcomponent] = []
    for ds in UNCHEATABLE_EVAL_DATASETS:
        subs.append(Subcomponent("uncheatable", ds, f"eval/uncheatable_eval/{ds}/loss", prettify(ds)))
    for ds in PALOMA_DATASETS:
        subs.append(Subcomponent("paloma", ds, f"eval/paloma/{ds}/loss", prettify(ds)))
    subs.append(Subcomponent("lima", "lima", "eval/lima/loss", "LIMA"))
    if olmo_available:
        for cat_key, label, _tasks in OLMO_BPB_CATEGORIES:
            subs.append(Subcomponent("olmo_summary", cat_key, _olmo_metric_key(cat_key), label))
        for task in OLMO_ALL_TASK_CARDS:
            subs.append(Subcomponent("olmo_bpb", task, _olmo_metric_key(_olmo_slug(task)), prettify_olmo_task(task)))
    return subs


# Two operating points at which we rank methods per subcomponent:
#   max_compute — loss at each method's deepest token budget (rightmost point).
#   chinchilla  — loss interpolated at 20 tokens/param, the compute-optimal point
#                 the figures already mark with the "1x" dotted guide.
MODE_MAX_COMPUTE = "max_compute"
MODE_CHINCHILLA = "chinchilla"
LABEL_MAX_COMPUTE = "Max compute"
LABEL_CHINCHILLA = "1\u00d7 Chinchilla"  # constant so the \u00d7-escape never lands in an f-string expr
MODES: tuple[tuple[str, str], ...] = (
    (MODE_MAX_COMPUTE, LABEL_MAX_COMPUTE),
    (MODE_CHINCHILLA, LABEL_CHINCHILLA),
)
# Only accept a Chinchilla-point loss if 20 tok/param lands within (or within this
# many log10-dex of) the method's actual token sweep — never extrapolate further.
_CHINCHILLA_EDGE_TOLERANCE_DEX = 0.1


def _loss_at_max_compute(pts: list[Point]) -> float | None:
    """Loss at the largest token budget (pts are sorted ascending by tokens)."""
    return pts[-1].loss if pts else None


def _loss_at_chinchilla(pts: list[Point], tokens_per_param: float) -> float | None:
    """Loss at 20 tok/param, linearly interpolated over log10(tokens).

    Returns None if the compute-optimal budget lies outside the method's swept
    token range by more than the edge tolerance (no extrapolation).
    """
    if not pts:
        return None
    target = tokens_per_param * pts[0].params
    xs = [math.log10(p.tokens) for p in pts]
    ys = [p.loss for p in pts]
    tx = math.log10(target)
    if tx <= xs[0]:
        return ys[0] if xs[0] - tx <= _CHINCHILLA_EDGE_TOLERANCE_DEX else None
    if tx >= xs[-1]:
        return ys[-1] if tx - xs[-1] <= _CHINCHILLA_EDGE_TOLERANCE_DEX else None
    for i in range(len(xs) - 1):
        if xs[i] <= tx <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            frac = (tx - xs[i]) / span if span else 0.0
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return None


def _mode_value(pts: list[Point], mode: str) -> float | None:
    if mode == MODE_MAX_COMPUTE:
        return _loss_at_max_compute(pts)
    return _loss_at_chinchilla(pts, CHINCHILLA_TOKENS_PER_PARAM)


@dataclass(frozen=True)
class SizeWinner:
    hidden_dim: int
    best_by_method: dict[str, float]  # base method name -> loss at this operating point
    winner: str  # base method name with lowest loss


@dataclass
class ModeStats:
    per_size: list[SizeWinner]
    overall_winner: str
    ours_size_wins: int
    n_sizes: int
    ours_margin: float  # mean over sizes of (best_other - ours)/best_other; >0 = ours better


@dataclass
class SubcomponentStats:
    sub: Subcomponent
    modes: dict[str, ModeStats]  # keyed by MODE_* id


def _summarize(per_size: list[SizeWinner]) -> ModeStats:
    """Overall winner + ours-stats from per-size rankings.

    Overall winner = most size-wins, tie-broken by lowest mean loss across the
    contested sizes.
    """
    margins: list[float] = []
    ours_wins = 0
    win_counts: dict[str, int] = {}
    loss_sums: dict[str, float] = {}
    loss_ns: dict[str, int] = {}
    for sw in per_size:
        if sw.winner == OURS_METHOD:
            ours_wins += 1
        if OURS_METHOD in sw.best_by_method and len(sw.best_by_method) > 1:
            ours = sw.best_by_method[OURS_METHOD]
            best_other = min(v for m, v in sw.best_by_method.items() if m != OURS_METHOD)
            if best_other > 0:
                margins.append((best_other - ours) / best_other)
        win_counts[sw.winner] = win_counts.get(sw.winner, 0) + 1
        for m, v in sw.best_by_method.items():
            loss_sums[m] = loss_sums.get(m, 0.0) + v
            loss_ns[m] = loss_ns.get(m, 0) + 1
    if win_counts:
        max_wins = max(win_counts.values())
        tied = [m for m, c in win_counts.items() if c == max_wins]
        overall = min(tied, key=lambda m: loss_sums[m] / loss_ns[m])
    else:
        overall = ""
    margin = sum(margins) / len(margins) if margins else float("nan")
    return ModeStats(per_size, overall, ours_wins, len(per_size), margin)


def compute_stats(
    summaries: list[dict],
    methods: list[str],
    widths: tuple[int, ...],
    sub: Subcomponent,
) -> SubcomponentStats:
    """Per-size method rankings at both operating points (max-compute & Chinchilla)."""
    pts_cache: dict[tuple[str, int], list[Point]] = {
        (m, w): collect_points(summaries, m, w, sub.metric_key) for w in widths for m in methods
    }
    modes: dict[str, ModeStats] = {}
    for mode_id, _label in MODES:
        per_size: list[SizeWinner] = []
        for w in widths:
            best_by_method: dict[str, float] = {}
            for m in methods:
                v = _mode_value(pts_cache[(m, w)], mode_id)
                if v is not None:
                    best_by_method[m] = v
            if len(best_by_method) < 2:
                continue  # not a real contest at this size
            winner = min(best_by_method, key=lambda m: best_by_method[m])
            per_size.append(SizeWinner(w, best_by_method, winner))
        modes[mode_id] = _summarize(per_size)
    return SubcomponentStats(sub=sub, modes=modes)


def render_all_figures(
    summaries: list[dict],
    methods: list[str],
    widths: tuple[int, ...],
    subs: list[Subcomponent],
    out_dir: Path,
) -> dict[str, Path]:
    """Render one loss-vs-tokens figure per subcomponent; return {metric_key: png_path}."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for sub in subs:
        # Give render_figure a nice y-axis label for this subcomponent.
        unit = "bpb" if sub.group == "olmo_bpb" else "loss"
        METRIC_LABELS[sub.metric_key] = f"{sub.display} — {unit}"
        png, _pdf = render_figure(
            summaries=summaries,
            methods=methods,
            hidden_sizes=list(widths),
            metric_key=sub.metric_key,
            ours_method=OURS_METHOD,
            out_dir=fig_dir,
        )
        paths[sub.metric_key] = png
        logger.info("rendered %s -> %s", sub.display, png.name)
    return paths


def method_label_color(method: str) -> tuple[str, str]:
    return METHOD_DISPLAY.get(method, (method, "#444444"))


def img_data_uri(png_path: Path) -> str:
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x * 100:+.1f}%"


GROUP_TABS = [
    ("uncheatable", "Uncheatable-Eval (7)"),
    ("paloma", "Paloma (16)"),
    ("lima", "LIMA"),
    ("olmo_summary", "OLMo bpb — summary (3)"),
    ("olmo_bpb", "OLMo bpb — all tasks (56)"),
]


def _mode_winner_line(mode_label: str, ms: ModeStats) -> str:
    """One header line: mode label → winning method + ours-wins + ours-margin."""
    win_label, win_color = method_label_color(ms.overall_winner)
    ours_all = ms.ours_size_wins == ms.n_sizes and ms.n_sizes > 0
    ours_badge = (
        '<span class="badge ours-all">ours wins all</span>'
        if ours_all
        else f'<span class="badge">ours {ms.ours_size_wins}/{ms.n_sizes}</span>'
    )
    return (
        '<div class="modewinner">'
        f'<span class="mode-tag">{html.escape(mode_label)}</span>'
        f'<span class="badge winner" style="--wc:{win_color}">{html.escape(win_label)}</span>'
        f"{ours_badge}"
        f'<span class="badge margin">{fmt_pct(ms.ours_margin)}</span>'
        "</div>"
    )


def _size_cell(sw: SizeWinner | None) -> str:
    """Two <td>s for one mode at one size: winner (dot+name) and ours loss."""
    if sw is None:
        return '<td class="na">—</td><td class="na">—</td>'
    wl, wc = method_label_color(sw.winner)
    ours_val = sw.best_by_method.get(OURS_METHOD)
    ours_txt = f"{ours_val:.3f}" if ours_val is not None else "—"
    return (
        f'<td><span class="dot" style="background:{wc}"></span>{html.escape(wl)}</td>' f'<td class="num">{ours_txt}</td>'
    )


def render_card(stats: SubcomponentStats, png_path: Path, has_samples: bool) -> str:
    sub = stats.sub
    mc = stats.modes[MODE_MAX_COMPUTE]
    ch = stats.modes[MODE_CHINCHILLA]
    mc_by_size = {sw.hidden_dim: sw for sw in mc.per_size}
    ch_by_size = {sw.hidden_dim: sw for sw in ch.per_size}

    all_sizes = sorted(set(mc_by_size) | set(ch_by_size))
    size_rows = "".join(
        f"<tr><td>d={w}</td>{_size_cell(mc_by_size.get(w))}{_size_cell(ch_by_size.get(w))}</tr>" for w in all_sizes
    )
    size_table = (
        '<table class="sizes"><thead>'
        '<tr><th rowspan="2">size</th><th colspan="2">Max compute</th>'
        '<th colspan="2">1\u00d7 Chinchilla</th></tr>'
        "<tr><th>winner</th><th>ours</th><th>winner</th><th>ours</th></tr>"
        f"</thead><tbody>{size_rows}</tbody></table>"
    )

    desc_html = ""
    info = DATASET_INFO.get(sub.key)
    if info is not None:
        desc, url = info
        link = f' <a href="{html.escape(url)}" target="_blank" rel="noopener">source ↗</a>' if url else ""
        desc_html = f'<p class="desc">{html.escape(desc)}{link}</p>'

    samples_btn = (
        f'<button class="samples-btn" data-key="{html.escape(sub.key)}" '
        f'data-title="{html.escape(sub.display)}">📄 View sample docs</button>'
        if has_samples
        else ""
    )

    # NaN margin (no "ours" data at that operating point) sorts last.
    def _margin_attr(ms: ModeStats) -> float:
        return ms.ours_margin if ms.ours_margin == ms.ours_margin else -999

    return f"""
    <div class="card" data-group="{sub.group}" data-name="{html.escape(sub.display)}"
         data-winner-maxc="{html.escape(method_label_color(mc.overall_winner)[0])}"
         data-winner-chin="{html.escape(method_label_color(ch.overall_winner)[0])}"
         data-margin-maxc="{_margin_attr(mc)}" data-margin-chin="{_margin_attr(ch)}">
      <div class="card-head">
        <div class="card-title">
          <h3>{html.escape(sub.display)}</h3>
          {desc_html}
        </div>
        <div class="badges">
          {_mode_winner_line(LABEL_MAX_COMPUTE, mc)}
          {_mode_winner_line(LABEL_CHINCHILLA, ch)}
          {samples_btn}
        </div>
      </div>
      <img loading="lazy" src="{img_data_uri(png_path)}" alt="{html.escape(sub.display)} loss vs tokens"/>
      {size_table}
    </div>"""


def render_summary_bar(all_stats: list[SubcomponentStats], group: str, mode_id: str, mode_label: str) -> str:
    """Tally overall winners across one group's subcomponents at one operating point."""
    group_stats = [s for s in all_stats if s.sub.group == group]
    tally: dict[str, int] = {}
    for s in group_stats:
        w = s.modes[mode_id].overall_winner
        tally[w] = tally.get(w, 0) + 1
    chips = []
    for m, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        label, color = method_label_color(m)
        chips.append(
            f'<span class="tally"><span class="dot" style="background:{color}"></span>'
            f"{html.escape(label)}: <b>{c}</b>/{len(group_stats)}</span>"
        )
    return (
        '<div class="summary"><span class="summary-tag">'
        f"{html.escape(mode_label)} winners</span>" + "".join(chips) + "</div>"
    )


def build_html(
    all_stats: list[SubcomponentStats],
    png_paths: dict[str, Path],
    samples: dict[str, dict],
) -> str:
    # Only show tabs for groups that actually have subcomponents (e.g. olmo_bpb is
    # present only when bpb results were joined in).
    present_groups = {s.sub.group for s in all_stats}
    tabs = [(g, lbl) for g, lbl in GROUP_TABS if g in present_groups]

    tab_buttons = "".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-tab="{g}">{html.escape(lbl)}</button>'
        for i, (g, lbl) in enumerate(tabs)
    )

    panels = []
    for i, (group, _lbl) in enumerate(tabs):
        cards = [
            render_card(s, png_paths[s.sub.metric_key], has_samples=s.sub.key in samples)
            for s in all_stats
            if s.sub.group == group
        ]
        summaries_html = "".join(
            render_summary_bar(all_stats, group, mode_id, mode_label) for mode_id, mode_label in MODES
        )
        panels.append(
            f'<section class="panel{" active" if i == 0 else ""}" data-panel="{group}">'
            f'<div class="summaries">{summaries_html}</div>'
            f'<div class="grid">{"".join(cards)}</div>'
            f"</section>"
        )

    # Escape `<` so no sample text (arXiv, code, wiki markup) containing `</script>`
    # or `<!--` can break out of the embedded JSON. `\\u003c` is valid inside a JS
    # string and renders back to `<`.
    samples_json = json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c")

    style = """
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #0f1115; color: #e6e6e6; }
    header { padding: 20px 28px 10px; border-bottom: 1px solid #262a33; position: sticky; top: 0;
             background: #0f1115; z-index: 10; }
    h1 { margin: 0 0 4px; font-size: 20px; }
    .sub { color: #9aa4b2; font-size: 13px; margin-bottom: 14px; }
    .tabs { display: flex; gap: 6px; }
    .tab { background: #1a1e26; color: #c7cdd6; border: 1px solid #2a2f3a; padding: 8px 14px;
           border-radius: 8px 8px 0 0; cursor: pointer; font-size: 14px; }
    .tab.active { background: #232936; color: #fff; border-bottom-color: #232936; }
    .controls { display: flex; gap: 10px; align-items: center; padding: 12px 28px; flex-wrap: wrap;
                border-bottom: 1px solid #262a33; }
    .controls label { font-size: 13px; color: #9aa4b2; }
    select { background: #1a1e26; color: #e6e6e6; border: 1px solid #2a2f3a; border-radius: 6px;
             padding: 6px 8px; font-size: 13px; }
    .panel { display: none; padding: 18px 28px 60px; }
    .panel.active { display: block; }
    .summaries { margin-bottom: 16px; }
    .summary { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
    .summary-tag { font-size: 12px; font-weight: 600; color: #7d8694; text-transform: uppercase;
                   letter-spacing: 0.04em; margin-right: 4px; }
    .tally { font-size: 13px; color: #c7cdd6; background: #171b22; padding: 6px 12px; border-radius: 999px; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;
           vertical-align: middle; }
    .grid { display: flex; flex-direction: column; gap: 22px; }
    .card { background: #141821; border: 1px solid #242a35; border-radius: 12px; padding: 16px 18px; }
    .card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
                 flex-wrap: wrap; margin-bottom: 10px; }
    .card-title { min-width: 260px; flex: 1; }
    .card-head h3 { margin: 0; font-size: 17px; }
    .desc { margin: 4px 0 0; font-size: 13px; color: #9aa4b2; line-height: 1.4; max-width: 720px; }
    .desc a { color: #6ea8fe; text-decoration: none; white-space: nowrap; }
    .desc a:hover { text-decoration: underline; }
    .badges { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
    .modewinner { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    .mode-tag { font-size: 11px; font-weight: 600; color: #7d8694; text-transform: uppercase;
                letter-spacing: 0.04em; min-width: 96px; text-align: right; }
    .badge { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: #1e2430; color: #c7cdd6; }
    .badge.winner { background: color-mix(in srgb, var(--wc) 22%, #1e2430); color: #fff;
                    border: 1px solid var(--wc); }
    .badge.ours-all { background: #143a24; color: #7fe6a4; border: 1px solid #2ca02c; }
    .badge.margin { background: #1a1e26; color: #9aa4b2; }
    .card img { width: 100%; height: auto; border-radius: 8px; background: #fff; }
    table.sizes { margin-top: 12px; border-collapse: collapse; font-size: 12px; width: auto; }
    table.sizes th, table.sizes td { text-align: left; padding: 3px 16px 3px 0; color: #b7bec9; }
    table.sizes thead th { color: #7d8694; font-weight: 600; border-bottom: 1px solid #2a2f3a; }
    table.sizes thead tr:first-child th { text-align: left; }
    table.sizes td.num { font-variant-numeric: tabular-nums; }
    table.sizes td.na { color: #556; }
    .samples-btn { font-size: 12px; padding: 4px 10px; border-radius: 999px; cursor: pointer;
                   background: #23314a; color: #cfe0ff; border: 1px solid #375080; }
    .samples-btn:hover { background: #2b3c5c; }
    /* Modal */
    .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6);
                     z-index: 100; padding: 40px 20px; overflow-y: auto; }
    .modal-overlay.open { display: block; }
    .modal { max-width: 900px; margin: 0 auto; background: #141821; border: 1px solid #2a3140;
             border-radius: 14px; padding: 22px 26px 30px; }
    .modal-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
                  position: sticky; top: -22px; background: #141821; padding-top: 4px; }
    .modal-head h2 { margin: 0 0 4px; font-size: 20px; }
    .modal-src { font-size: 12px; color: #7d8694; word-break: break-all; margin: 0; }
    .modal-close { background: none; border: none; color: #9aa4b2; font-size: 26px; cursor: pointer;
                   line-height: 1; padding: 0 4px; }
    .modal-close:hover { color: #fff; }
    .doc { margin-top: 16px; }
    .doc-label { font-size: 12px; color: #7d8694; margin-bottom: 4px; font-weight: 600; }
    .doc pre { margin: 0; background: #0d1017; border: 1px solid #232a36; border-radius: 8px;
               padding: 12px 14px; white-space: pre-wrap; word-break: break-word; font-size: 12.5px;
               line-height: 1.5; max-height: 340px; overflow-y: auto;
               font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #d6dce6; }
    .doc .trunc { color: #b78; font-style: italic; }
    .expand-btn { margin-top: 8px; font-size: 12px; padding: 4px 12px; border-radius: 6px;
                  cursor: pointer; background: #23314a; color: #cfe0ff; border: 1px solid #375080; }
    .expand-btn:hover { background: #2b3c5c; }
    @media (prefers-color-scheme: light) {
      body { background: #f6f7f9; color: #1a1d23; }
      header, .controls { background: #f6f7f9; border-color: #e2e5ea; }
      .card { background: #fff; border-color: #e2e5ea; }
      .sub, .tally, .badge.margin, .desc { color: #5b6470; }
      .desc a { color: #1a56db; }
      .tab { background: #eef0f3; color: #3a4048; border-color: #dcdfe4; }
      .tab.active { background: #fff; color: #000; }
      .tally { background: #eef0f3; }
      .samples-btn { background: #e7eefb; color: #1a56db; border-color: #b9cdf2; }
      .modal { background: #fff; border-color: #e2e5ea; }
      .modal-head { background: #fff; }
      .doc pre { background: #f4f6f9; border-color: #e2e5ea; color: #22262d; }
      .expand-btn { background: #e7eefb; color: #1a56db; border-color: #b9cdf2; }
    }
    """

    script = """
    // SAMPLES is embedded as a non-executable JSON block (below) and parsed here,
    // so no document text can break out into the JS source.
    const SAMPLES = JSON.parse(document.getElementById('samples-data').textContent);
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');
    tabs.forEach(t => t.addEventListener('click', () => {
      tabs.forEach(x => x.classList.remove('active'));
      panels.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.querySelector(`.panel[data-panel="${t.dataset.tab}"]`).classList.add('active');
    }));

    function sortCards(mode) {
      // Each mode picks a winner-label field and an ours-margin field so the two
      // operating points (max compute / 1x Chinchilla) can be sorted independently.
      const spec = {
        'winner-maxc':  {win: 'winnerMaxc', margin: 'marginMaxc'},
        'winner-chin':  {win: 'winnerChin', margin: 'marginChin'},
        'margin-maxc':  {margin: 'marginMaxc'},
        'margin-chin':  {margin: 'marginChin'},
      }[mode];
      panels.forEach(panel => {
        const grid = panel.querySelector('.grid');
        const cards = Array.from(grid.children);
        cards.sort((a, b) => {
          if (!spec) return a.dataset.name.localeCompare(b.dataset.name);  // alphabetical
          if (spec.win) {
            const wa = a.dataset[spec.win], wb = b.dataset[spec.win];
            if (wa !== wb) return wa.localeCompare(wb);
          }
          return parseFloat(b.dataset[spec.margin]) - parseFloat(a.dataset[spec.margin]);
        });
        cards.forEach(c => grid.appendChild(c));
      });
    }
    document.getElementById('sort').addEventListener('change', e => sortCards(e.target.value));
    sortCards('winner-maxc');

    // --- Sample-docs modal ---
    const overlay = document.getElementById('modal-overlay');
    const modalBody = document.getElementById('modal-body');
    const PREVIEW_CHARS = 1600;

    function makeDoc(text, charLen, i) {
      // text is the stored (possibly hard-capped) full document; charLen is its
      // true length. All text set via textContent — newlines are preserved
      // exactly and no markup can inject.
      const wrap = document.createElement('div');
      wrap.className = 'doc';
      const label = document.createElement('div');
      label.className = 'doc-label';
      label.textContent = `Document ${i + 1} · ${charLen.toLocaleString()} chars`;
      if (charLen > text.length) {  // stored text itself hit the extractor's hard cap
        const s = document.createElement('span');
        s.className = 'trunc';
        s.textContent = ` (showing first ${text.length.toLocaleString()})`;
        label.appendChild(s);
      }
      const pre = document.createElement('pre');
      const long = text.length > PREVIEW_CHARS;
      pre.textContent = long ? text.slice(0, PREVIEW_CHARS) : text;
      wrap.appendChild(label);
      wrap.appendChild(pre);
      if (long) {
        const btn = document.createElement('button');
        btn.className = 'expand-btn';
        btn.textContent = 'Expand ▾';
        let expanded = false;
        btn.addEventListener('click', () => {
          expanded = !expanded;
          pre.textContent = expanded ? text : text.slice(0, PREVIEW_CHARS);
          btn.textContent = expanded ? 'Collapse ▴' : 'Expand ▾';
        });
        wrap.appendChild(btn);
      }
      return wrap;
    }

    function openSamples(key, title) {
      const entry = SAMPLES[key];
      if (!entry) return;
      modalBody.innerHTML = '';
      const head = document.createElement('div');
      head.className = 'modal-head';
      const info = document.createElement('div');
      const h2 = document.createElement('h2');
      h2.textContent = title;
      const src = document.createElement('p');
      src.className = 'modal-src';
      const shardNote = entry.n_shards > 1
        ? ` · one doc from each of ${Math.min(entry.n_shown, entry.n_shards)} shards` : '';
      src.appendChild(document.createTextNode(
        `${entry.n_shown} real documents from the eval set${shardNote}`));
      src.appendChild(document.createElement('br'));
      src.appendChild(document.createTextNode(entry.source_glob));
      info.appendChild(h2);
      info.appendChild(src);
      const close = document.createElement('button');
      close.className = 'modal-close';
      close.setAttribute('aria-label', 'Close');
      close.innerHTML = '&times;';
      close.addEventListener('click', closeModal);
      head.appendChild(info);
      head.appendChild(close);
      modalBody.appendChild(head);
      entry.samples.forEach((text, i) => modalBody.appendChild(makeDoc(text, entry.char_len[i], i)));
      overlay.classList.add('open');
      overlay.scrollTop = 0;
    }
    function closeModal() { overlay.classList.remove('open'); }
    document.querySelectorAll('.samples-btn').forEach(b =>
      b.addEventListener('click', () => openSamples(b.dataset.key, b.dataset.title)));
    overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Subcomponent loss dashboard — 10k-natural sweep</title>
<style>{style}</style></head>
<body>
<header>
  <h1>Per-subcomponent loss-vs-tokens — 10k-natural sweep</h1>
  <div class="sub">Winner per model size at two operating points — <b>Max compute</b> (loss at each
  method's deepest token budget) and <b>1\u00d7 Chinchilla</b> (loss interpolated at 20 tokens/param).
  "ours" = Spec-Driven Extraction (high_quality).</div>
  <div class="tabs">{tab_buttons}</div>
</header>
<div class="controls">
  <label for="sort">Sort cards:</label>
  <select id="sort">
    <option value="winner-maxc">By winner — Max compute</option>
    <option value="winner-chin">By winner — 1\u00d7 Chinchilla</option>
    <option value="margin-maxc">Ours margin — Max compute (ours best first)</option>
    <option value="margin-chin">Ours margin — 1\u00d7 Chinchilla (ours best first)</option>
    <option value="name">Alphabetical</option>
  </select>
</div>
{"".join(panels)}
<div class="modal-overlay" id="modal-overlay"><div class="modal" id="modal-body"></div></div>
<script id="samples-data" type="application/json">{samples_json}</script>
<script>{script}</script>
</body></html>"""


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-prefix", default=TENK_LOCAL_PREFIX)
    parser.add_argument("--results-gs", default=TENK_RESULTS_GS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_LOAD_METHODS))
    parser.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    parser.add_argument("--pull", action="store_true", help="Refresh 10k summaries from gs:// first.")
    parser.add_argument("--olmo-prefix", default=OLMO_BPB_LOCAL, help="Local dir of pulled olmo_bpb results.")
    parser.add_argument(
        "--pull-olmo", action="store_true", help="Refresh olmo_bpb results from the region buckets first."
    )
    parser.add_argument(
        "--samples-json",
        default=None,
        help=(
            "Path to sample-docs JSON from extract_subcomponent_samples.py. "
            "Defaults to <output-dir>/samples.json; if absent, the 'view samples' "
            "buttons are omitted."
        ),
    )
    args = parser.parse_args(argv)

    if args.pull:
        pull_gs_to_local(args.results_gs, Path(args.results_prefix))
    if args.pull_olmo:
        pull_olmo_bpb(args.olmo_prefix)

    summaries = load_summaries(args.results_prefix, args.methods, suffix="")
    apply_tenk_rename(summaries)
    methods = [TENK_METHOD_MAP.get(m, m) for m in args.methods]

    # Join OLMo Base-Easy bpb (a loss metric) onto the summaries by run_name, if present.
    olmo_by_run = load_olmo_bpb(args.olmo_prefix)
    n_olmo = inject_olmo_bpb(summaries, olmo_by_run)
    logger.info("olmo_bpb: joined onto %d/%d summaries (%d runs had bpb)", n_olmo, len(summaries), len(olmo_by_run))

    subs = build_subcomponents(olmo_available=n_olmo > 0)
    out_dir = Path(args.output_dir)
    png_paths = render_all_figures(summaries, methods, tuple(args.widths), subs, out_dir)

    all_stats = [compute_stats(summaries, methods, tuple(args.widths), sub) for sub in subs]

    # Console summary: overall winner + ours-record at both operating points.
    for s in all_stats:
        mc = s.modes[MODE_MAX_COMPUTE]
        ch = s.modes[MODE_CHINCHILLA]
        logger.info(
            "%-22s [%s]  max-compute: %-28s ours %d/%d (%s)   1x-chinchilla: %-28s ours %d/%d (%s)",
            s.sub.display,
            s.sub.group,
            method_label_color(mc.overall_winner)[0],
            mc.ours_size_wins,
            mc.n_sizes,
            fmt_pct(mc.ours_margin),
            method_label_color(ch.overall_winner)[0],
            ch.ours_size_wins,
            ch.n_sizes,
            fmt_pct(ch.ours_margin),
        )

    samples_path = Path(args.samples_json) if args.samples_json else out_dir / "samples.json"
    if samples_path.exists():
        samples = json.loads(samples_path.read_text())
        logger.info("loaded sample docs for %d datasets from %s", len(samples), samples_path)
    else:
        samples = {}
        logger.warning(
            "no sample docs at %s (run extract_subcomponent_samples.py); omitting view-samples buttons",
            samples_path,
        )

    # OLMo bpb sample docs (context + gold continuation) live in a sibling file.
    olmo_samples_path = out_dir / "olmo_samples.json"
    if olmo_samples_path.exists():
        olmo_samples = json.loads(olmo_samples_path.read_text())
        samples.update(olmo_samples)
        logger.info("merged %d olmo_bpb sample sets from %s", len(olmo_samples), olmo_samples_path)

    html_str = build_html(all_stats, png_paths, samples)
    out_html = out_dir / "subcomponents_dashboard.html"
    out_html.write_text(html_str)
    size_mb = out_html.stat().st_size / 1e6
    logger.info("wrote %s (%.1f MB, %d subcomponents)", out_html, size_mb, len(subs))


if __name__ == "__main__":
    main()
