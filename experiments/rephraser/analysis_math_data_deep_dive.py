# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Math data deep dive — V1 automated analysis.

Reads ALL postprocessed extraction and resiliparse data from the math v2
pipeline. For each document, classifies structure, quality, domain, LaTeX
density, and content type. Outputs comprehensive stats and stratified samples
for V2 review.

This is a CPU-only job (no TPU needed). Outputs go to GCS:
  - math_deep_dive_v1_summary.json  — full aggregated stats
  - math_deep_dive_v1_samples.json  — representative docs per domain x quality
  - math_deep_dive_v1_report.md     — markdown V1 report

Launch:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
        -- python experiments/rephraser/analysis_math_data_deep_dive.py
"""

import gzip
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse

import fsspec

from experiments.rephraser.mathhelpforum_extraction_sft_v2_base import result as math_result
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SAMPLES_PER_BUCKET = 10  # docs per (domain, quality_tier) bucket
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
_LATEX_INLINE = re.compile(r"\$[^$]+\$")
_LATEX_DISPLAY = re.compile(r"\$\$[^$]+\$\$")
_LATEX_PARENS = re.compile(r"\\\([^)]+\\\)")
_LATEX_BRACKETS = re.compile(r"\\\[[^\]]+\\\]")
_HTML_TAGS = re.compile(r"<(?:div|span|p|br|table|tr|td|th|img|a |ul|ol|li|h[1-6])[^>]*>", re.IGNORECASE)
_HTML_ENTITIES = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);")
_QRA_QUESTION = re.compile(r"^#{1,3}\s*(?:Question|Problem)", re.MULTILINE | re.IGNORECASE)
_QRA_REASONING = re.compile(r"^#{1,3}\s*(?:Reasoning|Solution|Explanation)", re.MULTILINE | re.IGNORECASE)
_QRA_ANSWER = re.compile(r"^#{1,3}\s*(?:Answer)", re.MULTILINE | re.IGNORECASE)
_META_COMMENTARY = re.compile(
    r"(?:The user asked|Let me help|I cannot|I can't|As an AI|I apologize)",
    re.IGNORECASE,
)
_MATH_KEYWORDS = re.compile(
    r"\b(?:equation|theorem|proof|integral|derivative|matrix|polynomial|"
    r"algebra|calculus|geometry|probability|fraction|variable|coefficient|"
    r"quadratic|linear|exponential|logarithm|trigonometr|vector|"
    r"solve|simplif|factor|evaluat|comput|calculat)\b",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{")


def extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or ""
        # Strip www.
        if host.startswith("www."):
            host = host[4:]
        return host or "unknown"
    except Exception:
        return "unknown"


def classify_extraction_structure(text: str) -> str:
    """Classify the structure of an extraction document."""
    if len(text) < 200:
        return "minimal"

    has_html = bool(_HTML_TAGS.search(text)) or bool(_HTML_ENTITIES.search(text))
    if has_html and _HTML_TAGS.findall(text).__len__() > 10:
        return "garbled"

    has_q = bool(_QRA_QUESTION.search(text))
    has_r = bool(_QRA_REASONING.search(text))
    has_a = bool(_QRA_ANSWER.search(text))

    if has_q and has_r and has_a:
        # Check if answer section has actual content
        answer_match = _QRA_ANSWER.search(text)
        if answer_match:
            after_answer = text[answer_match.end():].strip()
            if len(after_answer) < 10 or "not provided" in after_answer.lower():
                return "qa_missing_answer"
        return "qa_complete"
    elif has_q and has_r:
        return "qa_missing_answer"
    elif has_q and has_a:
        return "qa_missing_reasoning"
    elif has_q:
        return "question_only"
    else:
        # No Q/R/A structure — could be tutorial or exercise list
        lines = text.strip().split("\n")
        numbered = sum(1 for l in lines if re.match(r"^\s*\d+[\.\)]\s", l))
        if numbered >= 3:
            return "exercise_list"
        return "tutorial"


def classify_resiliparse_structure(text: str) -> str:
    """Classify the structure of a resiliparse document."""
    if len(text) < 200:
        return "minimal"

    has_html = bool(_HTML_TAGS.search(text))
    if has_html and len(_HTML_TAGS.findall(text)) > 10:
        return "garbled"

    math_matches = len(_MATH_KEYWORDS.findall(text))
    latex_matches = len(_LATEX_INLINE.findall(text)) + len(_LATEX_DISPLAY.findall(text))

    if math_matches >= 5 or latex_matches >= 3:
        return "math_content"
    elif math_matches >= 1 or latex_matches >= 1:
        return "mixed_content"
    else:
        return "non_math"


def compute_quality_indicators(text: str) -> dict:
    """Compute quality indicators for a document."""
    char_count = len(text)
    word_count = len(text.split())

    # LaTeX
    latex_spans = (
        _LATEX_INLINE.findall(text)
        + _LATEX_DISPLAY.findall(text)
        + _LATEX_PARENS.findall(text)
        + _LATEX_BRACKETS.findall(text)
    )
    latex_chars = sum(len(s) for s in latex_spans)

    # HTML artifacts
    html_tag_count = len(_HTML_TAGS.findall(text))
    html_entity_count = len(_HTML_ENTITIES.findall(text))

    # Meta-commentary
    has_meta = bool(_META_COMMENTARY.search(text))

    # Math keyword density
    math_keyword_count = len(_MATH_KEYWORDS.findall(text))

    # Boxed answers
    has_boxed = bool(_BOXED.search(text))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "latex_count": len(latex_spans),
        "latex_density": latex_chars / max(char_count, 1),
        "html_tag_count": html_tag_count,
        "html_entity_count": html_entity_count,
        "has_meta_commentary": has_meta,
        "math_keyword_count": math_keyword_count,
        "math_keyword_density": math_keyword_count / max(word_count, 1),
        "has_boxed": has_boxed,
    }


def quality_tier(indicators: dict, structure: str) -> str:
    """Assign a quality tier: high, medium, low."""
    if structure in ("minimal", "garbled"):
        return "low"
    if indicators["has_meta_commentary"]:
        return "low"
    if indicators["html_tag_count"] > 5:
        return "low"

    if structure == "qa_complete" and indicators["latex_count"] >= 2:
        return "high"
    if structure == "math_content" and indicators["latex_count"] >= 2:
        return "high"
    if structure in ("tutorial", "mixed_content") and indicators["math_keyword_count"] >= 5:
        return "medium"
    if structure in ("qa_missing_answer", "qa_missing_reasoning", "question_only"):
        return "medium"
    if structure == "non_math":
        return "low"

    return "medium"


# ---------------------------------------------------------------------------
# Data analysis
# ---------------------------------------------------------------------------
@dataclass
class DomainStats:
    count: int = 0
    total_chars: int = 0
    total_words: int = 0
    structure_dist: Counter = field(default_factory=Counter)
    quality_dist: Counter = field(default_factory=Counter)
    latex_docs: int = 0
    html_artifact_docs: int = 0
    meta_commentary_docs: int = 0
    boxed_docs: int = 0
    total_latex_density: float = 0.0
    total_math_keyword_density: float = 0.0


def analyze_data_source(glob_path: str, data_type: str) -> tuple[dict, dict]:
    """Stream through all docs and compute per-domain stats + samples.

    Returns:
        (domain_stats, samples) where:
        - domain_stats: dict[domain_name, DomainStats]
        - samples: dict[(domain, quality_tier), list[doc_dicts]]
    """
    rng = random.Random(RANDOM_SEED)
    domain_stats: dict[str, DomainStats] = defaultdict(DomainStats)
    samples: dict[str, list] = defaultdict(list)  # key = "domain::tier"
    sample_counts: dict[str, int] = defaultdict(int)  # for reservoir sampling
    total_docs = 0
    total_errors = 0

    classify_fn = (
        classify_extraction_structure if data_type == "extraction" else classify_resiliparse_structure
    )

    logger.info(f"Analyzing {data_type} data from: {glob_path}")
    file_list = fsspec.open_files(glob_path)
    logger.info(f"Found {len(file_list)} files")

    for file_idx, open_file in enumerate(file_list):
        try:
            with open_file as f:
                # Handle .gz files
                if open_file.path.endswith(".gz"):
                    f = gzip.open(f, "rt", encoding="utf-8")
                for line in f:
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        total_errors += 1
                        continue

                    text = doc.get("text", "")
                    url = doc.get("url", doc.get("source_url", ""))
                    domain = extract_domain(url)

                    structure = classify_fn(text)
                    indicators = compute_quality_indicators(text)
                    tier = quality_tier(indicators, structure)

                    # Update stats
                    ds = domain_stats[domain]
                    ds.count += 1
                    ds.total_chars += indicators["char_count"]
                    ds.total_words += indicators["word_count"]
                    ds.structure_dist[structure] += 1
                    ds.quality_dist[tier] += 1
                    if indicators["latex_count"] > 0:
                        ds.latex_docs += 1
                    if indicators["html_tag_count"] > 0:
                        ds.html_artifact_docs += 1
                    if indicators["has_meta_commentary"]:
                        ds.meta_commentary_docs += 1
                    if indicators["has_boxed"]:
                        ds.boxed_docs += 1
                    ds.total_latex_density += indicators["latex_density"]
                    ds.total_math_keyword_density += indicators["math_keyword_density"]

                    total_docs += 1

                    # Reservoir sampling for representative docs
                    bucket = f"{domain}::{tier}"
                    sample_counts[bucket] += 1
                    if len(samples[bucket]) < MAX_SAMPLES_PER_BUCKET:
                        samples[bucket].append({
                            "url": url,
                            "domain": domain,
                            "structure": structure,
                            "quality_tier": tier,
                            "indicators": indicators,
                            "text_preview": text[:2000],
                            "text_length": len(text),
                        })
                    else:
                        j = rng.randint(0, sample_counts[bucket] - 1)
                        if j < MAX_SAMPLES_PER_BUCKET:
                            samples[bucket][j] = {
                                "url": url,
                                "domain": domain,
                                "structure": structure,
                                "quality_tier": tier,
                                "indicators": indicators,
                                "text_preview": text[:2000],
                                "text_length": len(text),
                            }

        except Exception as e:
            logger.warning(f"Error reading file {file_idx}: {e}")
            total_errors += 1

        if (file_idx + 1) % 50 == 0:
            logger.info(f"  Processed {file_idx + 1}/{len(file_list)} files, {total_docs} docs so far")

    logger.info(
        f"Done analyzing {data_type}: {total_docs} docs across {len(domain_stats)} domains "
        f"({total_errors} errors)"
    )
    return dict(domain_stats), dict(samples)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def stats_to_serializable(domain_stats: dict[str, DomainStats]) -> dict:
    """Convert DomainStats to JSON-serializable dict."""
    result = {}
    for domain, ds in sorted(domain_stats.items(), key=lambda x: -x[1].count):
        result[domain] = {
            "count": ds.count,
            "total_chars": ds.total_chars,
            "total_words": ds.total_words,
            "avg_chars": ds.total_chars / max(ds.count, 1),
            "avg_words": ds.total_words / max(ds.count, 1),
            "structure_dist": dict(ds.structure_dist),
            "quality_dist": dict(ds.quality_dist),
            "latex_docs": ds.latex_docs,
            "latex_pct": 100 * ds.latex_docs / max(ds.count, 1),
            "html_artifact_docs": ds.html_artifact_docs,
            "html_artifact_pct": 100 * ds.html_artifact_docs / max(ds.count, 1),
            "meta_commentary_docs": ds.meta_commentary_docs,
            "boxed_docs": ds.boxed_docs,
            "avg_latex_density": ds.total_latex_density / max(ds.count, 1),
            "avg_math_keyword_density": ds.total_math_keyword_density / max(ds.count, 1),
        }
    return result


def generate_v1_report(
    extraction_stats: dict[str, DomainStats],
    resiliparse_stats: dict[str, DomainStats],
    extraction_samples: dict,
    resiliparse_samples: dict,
) -> str:
    """Generate the V1 markdown report."""
    lines = []
    lines.append("# Math Data Deep Dive — V1 Automated Analysis")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")

    ext_total = sum(ds.count for ds in extraction_stats.values())
    res_total = sum(ds.count for ds in resiliparse_stats.values())
    lines.append(f"- **Extraction data**: {ext_total:,} documents across {len(extraction_stats)} domains")
    lines.append(f"- **Resiliparse data**: {res_total:,} documents across {len(resiliparse_stats)} domains")
    lines.append("")

    # Overall structure distribution
    lines.append("## 2. Extraction Data — Structure Distribution")
    lines.append("")
    ext_structure = Counter()
    ext_quality = Counter()
    for ds in extraction_stats.values():
        ext_structure += ds.structure_dist
        ext_quality += ds.quality_dist

    lines.append("| Structure | Count | % |")
    lines.append("|-----------|------:|---:|")
    for structure, count in ext_structure.most_common():
        lines.append(f"| {structure} | {count:,} | {100*count/max(ext_total,1):.1f}% |")
    lines.append("")

    lines.append("### Quality Tier Distribution")
    lines.append("")
    lines.append("| Tier | Count | % |")
    lines.append("|------|------:|---:|")
    for tier in ["high", "medium", "low"]:
        count = ext_quality.get(tier, 0)
        lines.append(f"| {tier} | {count:,} | {100*count/max(ext_total,1):.1f}% |")
    lines.append("")

    # Resiliparse structure
    lines.append("## 3. Resiliparse Data — Structure Distribution")
    lines.append("")
    res_structure = Counter()
    res_quality = Counter()
    for ds in resiliparse_stats.values():
        res_structure += ds.structure_dist
        res_quality += ds.quality_dist

    lines.append("| Structure | Count | % |")
    lines.append("|-----------|------:|---:|")
    for structure, count in res_structure.most_common():
        lines.append(f"| {structure} | {count:,} | {100*count/max(res_total,1):.1f}% |")
    lines.append("")

    lines.append("### Quality Tier Distribution")
    lines.append("")
    lines.append("| Tier | Count | % |")
    lines.append("|------|------:|---:|")
    for tier in ["high", "medium", "low"]:
        count = res_quality.get(tier, 0)
        lines.append(f"| {tier} | {count:,} | {100*count/max(res_total,1):.1f}% |")
    lines.append("")

    # Per-domain breakdown (extraction)
    lines.append("## 4. Extraction Data — Per-Domain Breakdown")
    lines.append("")
    lines.append("| Domain | Docs | Avg Chars | High% | Med% | Low% | LaTeX% | HTML Art% | Math KW Density |")
    lines.append("|--------|-----:|----------:|------:|-----:|-----:|-------:|----------:|----------------:|")

    for domain, ds in sorted(extraction_stats.items(), key=lambda x: -x[1].count):
        n = ds.count
        high_pct = 100 * ds.quality_dist.get("high", 0) / max(n, 1)
        med_pct = 100 * ds.quality_dist.get("medium", 0) / max(n, 1)
        low_pct = 100 * ds.quality_dist.get("low", 0) / max(n, 1)
        latex_pct = 100 * ds.latex_docs / max(n, 1)
        html_pct = 100 * ds.html_artifact_docs / max(n, 1)
        avg_chars = ds.total_chars / max(n, 1)
        mkd = ds.total_math_keyword_density / max(n, 1)
        lines.append(
            f"| {domain} | {n:,} | {avg_chars:.0f} | {high_pct:.1f} | {med_pct:.1f} | {low_pct:.1f} "
            f"| {latex_pct:.1f} | {html_pct:.1f} | {mkd:.4f} |"
        )
    lines.append("")

    # Per-domain breakdown (resiliparse)
    lines.append("## 5. Resiliparse Data — Per-Domain Breakdown")
    lines.append("")
    lines.append("| Domain | Docs | Avg Chars | High% | Med% | Low% | LaTeX% | HTML Art% | Math KW Density |")
    lines.append("|--------|-----:|----------:|------:|-----:|-----:|-------:|----------:|----------------:|")

    for domain, ds in sorted(resiliparse_stats.items(), key=lambda x: -x[1].count):
        n = ds.count
        high_pct = 100 * ds.quality_dist.get("high", 0) / max(n, 1)
        med_pct = 100 * ds.quality_dist.get("medium", 0) / max(n, 1)
        low_pct = 100 * ds.quality_dist.get("low", 0) / max(n, 1)
        latex_pct = 100 * ds.latex_docs / max(n, 1)
        html_pct = 100 * ds.html_artifact_docs / max(n, 1)
        avg_chars = ds.total_chars / max(n, 1)
        mkd = ds.total_math_keyword_density / max(n, 1)
        lines.append(
            f"| {domain} | {n:,} | {avg_chars:.0f} | {high_pct:.1f} | {med_pct:.1f} | {low_pct:.1f} "
            f"| {latex_pct:.1f} | {html_pct:.1f} | {mkd:.4f} |"
        )
    lines.append("")

    # Highlight concerning domains
    lines.append("## 6. Flagged Concerns")
    lines.append("")

    # Domains with >50% low quality
    lines.append("### Extraction — Domains with >50% Low Quality")
    lines.append("")
    for domain, ds in sorted(extraction_stats.items(), key=lambda x: -x[1].count):
        n = ds.count
        if n < 10:
            continue
        low_pct = 100 * ds.quality_dist.get("low", 0) / max(n, 1)
        if low_pct > 50:
            lines.append(f"- **{domain}**: {n:,} docs, {low_pct:.0f}% low quality")
    lines.append("")

    # Domains with >20% HTML artifacts
    lines.append("### Extraction — Domains with >20% HTML Artifacts")
    lines.append("")
    for domain, ds in sorted(extraction_stats.items(), key=lambda x: -x[1].count):
        n = ds.count
        if n < 10:
            continue
        html_pct = 100 * ds.html_artifact_docs / max(n, 1)
        if html_pct > 20:
            lines.append(f"- **{domain}**: {n:,} docs, {html_pct:.0f}% have HTML artifacts")
    lines.append("")

    # Domains with low math keyword density
    lines.append("### Extraction — Domains with Low Math Content")
    lines.append("")
    for domain, ds in sorted(extraction_stats.items(), key=lambda x: -x[1].count):
        n = ds.count
        if n < 10:
            continue
        mkd = ds.total_math_keyword_density / max(n, 1)
        if mkd < 0.005:
            lines.append(f"- **{domain}**: {n:,} docs, math keyword density {mkd:.4f}")
    lines.append("")

    # Sample previews
    lines.append("## 7. Sample Document Previews")
    lines.append("")

    # Show a few high and low quality extraction samples
    for data_type, all_samples in [("Extraction", extraction_samples), ("Resiliparse", resiliparse_samples)]:
        lines.append(f"### {data_type} — High Quality Samples")
        lines.append("")
        high_shown = 0
        for bucket_key, docs in sorted(all_samples.items()):
            if "::high" not in bucket_key:
                continue
            for doc in docs[:2]:
                if high_shown >= 5:
                    break
                lines.append(f"**Domain**: {doc['domain']} | **Structure**: {doc['structure']} | "
                           f"**Length**: {doc['text_length']:,} chars")
                lines.append(f"**URL**: {doc['url']}")
                lines.append("```")
                lines.append(doc["text_preview"][:500])
                lines.append("```")
                lines.append("")
                high_shown += 1
            if high_shown >= 5:
                break

        lines.append(f"### {data_type} — Low Quality Samples")
        lines.append("")
        low_shown = 0
        for bucket_key, docs in sorted(all_samples.items()):
            if "::low" not in bucket_key:
                continue
            for doc in docs[:2]:
                if low_shown >= 5:
                    break
                lines.append(f"**Domain**: {doc['domain']} | **Structure**: {doc['structure']} | "
                           f"**Length**: {doc['text_length']:,} chars")
                lines.append(f"**URL**: {doc['url']}")
                lines.append("```")
                lines.append(doc["text_preview"][:500])
                lines.append("```")
                lines.append("")
                low_shown += 1
            if low_shown >= 5:
                break

    lines.append("## 8. Data Composition Summary")
    lines.append("")
    lines.append("### Token Budget Estimate (Extraction)")
    lines.append("")
    total_chars = sum(ds.total_chars for ds in extraction_stats.values())
    lines.append(f"Total characters: {total_chars:,}")
    lines.append(f"Estimated tokens (chars/4): ~{total_chars//4:,}")
    lines.append("")

    # Top 10 domains by data volume
    lines.append("### Top 10 Domains by Volume (Extraction)")
    lines.append("")
    lines.append("| Rank | Domain | Docs | Chars | % of Total |")
    lines.append("|-----:|--------|-----:|------:|-----------:|")
    for rank, (domain, ds) in enumerate(
        sorted(extraction_stats.items(), key=lambda x: -x[1].total_chars)[:10], 1
    ):
        pct = 100 * ds.total_chars / max(total_chars, 1)
        lines.append(f"| {rank} | {domain} | {ds.count:,} | {ds.total_chars:,} | {pct:.1f}% |")
    lines.append("")

    lines.append("---")
    lines.append("*V1 report generated by automated analysis. V2 review with LLM quality assessment pending.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis function (runs as executor step)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeepDiveConfig:
    extraction_path: str
    resiliparse_path: str
    output_path: str


def run_deep_dive(config: DeepDiveConfig):
    """Run the full V1 automated analysis."""
    logger.info("Starting math data deep dive V1 analysis")
    logger.info(f"  Extraction: {config.extraction_path}")
    logger.info(f"  Resiliparse: {config.resiliparse_path}")

    # Analyze both data sources
    extraction_stats, extraction_samples = analyze_data_source(config.extraction_path, "extraction")
    resiliparse_stats, resiliparse_samples = analyze_data_source(config.resiliparse_path, "resiliparse")

    # Write summary JSON
    summary = {
        "extraction": stats_to_serializable(extraction_stats),
        "resiliparse": stats_to_serializable(resiliparse_stats),
        "extraction_total_docs": sum(ds.count for ds in extraction_stats.values()),
        "resiliparse_total_docs": sum(ds.count for ds in resiliparse_stats.values()),
    }
    summary_path = os.path.join(config.output_path, "math_deep_dive_v1_summary.json")
    with fsspec.open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Write samples JSON
    all_samples = {
        "extraction": extraction_samples,
        "resiliparse": resiliparse_samples,
    }
    samples_path = os.path.join(config.output_path, "math_deep_dive_v1_samples.json")
    with fsspec.open(samples_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    logger.info(f"Samples written to {samples_path}")

    # Generate and write V1 report
    report = generate_v1_report(extraction_stats, resiliparse_stats, extraction_samples, resiliparse_samples)
    report_path = os.path.join(config.output_path, "math_deep_dive_v1_report.md")
    with fsspec.open(report_path, "w") as f:
        f.write(report)
    logger.info(f"V1 report written to {report_path}")

    logger.info("Deep dive V1 analysis complete!")


# ---------------------------------------------------------------------------
# Executor step definition
# ---------------------------------------------------------------------------
# Resolve data paths from the math v2 base experiment
extraction_postprocess = math_result.extraction_branches[0].postprocess_step
resiliparse_processed = math_result.resiliparse_step

deep_dive_step = ExecutorStep(
    name="analysis/math_deep_dive_v1",
    description="Automated quality analysis of all math extraction + resiliparse training data.",
    fn=run_deep_dive,
    config=DeepDiveConfig(
        extraction_path=output_path_of(extraction_postprocess) / "*.jsonl.gz",
        resiliparse_path=output_path_of(resiliparse_processed) / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

if __name__ == "__main__":
    executor_main(
        steps=[deep_dive_step],
        description="Math data deep dive V1 — automated quality analysis",
    )
