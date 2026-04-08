# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive analysis of math SFT regression: why does math training hurt math eval?

Run: .venv/bin/python3 experiments/rephraser/analysis_math_regression.py

Reads locally-downloaded eval samples and training data from /tmp/math_regression_analysis/
"""

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/tmp/math_regression_analysis")
MODELS = {
    "baseline": DATA_DIR / "baseline",
    "sweep_best": DATA_DIR / "sweep_best",  # lr5e-7_bs16
    "ext_highreg": DATA_DIR / "ext_highreg",
}
TASKS = ["algebra", "prealgebra", "gsm8k"]


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------
def load_samples(path: Path, metric_filter: str | None = None) -> dict[int, dict]:
    """Load eval sample JSONL, indexed by doc_id."""
    samples = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec["doc_id"]
            if metric_filter and rec.get("filter") != metric_filter:
                continue
            samples[doc_id] = rec
    return samples


def load_training_data(data_dir: Path, max_docs: int = 2000) -> list[dict]:
    """Load postprocessed training data from gzipped JSONL shards."""
    docs = []
    for path in sorted(data_dir.glob("processed-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                docs.append(json.loads(line))
                if len(docs) >= max_docs:
                    return docs
    return docs


def get_response(sample: dict) -> str:
    """Extract the model's response text from a sample."""
    resps = sample.get("resps", [[]])
    if resps and resps[0]:
        return str(resps[0][0])
    return ""


def get_problem(sample: dict) -> str:
    """Extract the problem statement from a sample."""
    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        return doc.get("problem", doc.get("question", ""))
    return ""


def get_level(sample: dict) -> str:
    """Extract difficulty level from MATH problem."""
    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        return doc.get("level", "Unknown")
    return "Unknown"


def get_gold(sample: dict) -> str:
    """Extract gold answer."""
    return str(sample.get("target", ""))


# ---------------------------------------------------------------------------
# Phase 2a: Find regressions and improvements
# ---------------------------------------------------------------------------
def find_changes(baseline: dict, sft: dict, metric: str = "exact_match"):
    """Find problems where correctness changed between baseline and SFT."""
    regressions = []  # baseline right, SFT wrong
    improvements = []  # baseline wrong, SFT right
    common_ids = set(baseline.keys()) & set(sft.keys())

    for doc_id in sorted(common_ids):
        b = baseline[doc_id]
        s = sft[doc_id]
        b_correct = b.get(metric, 0)
        s_correct = s.get(metric, 0)

        if b_correct == 1 and s_correct == 0:
            regressions.append(
                {
                    "doc_id": doc_id,
                    "problem": get_problem(b),
                    "level": get_level(b),
                    "gold": get_gold(b),
                    "baseline_resp": get_response(b),
                    "sft_resp": get_response(s),
                    "baseline_filtered": b.get("filtered_resps", [""])[0] if b.get("filtered_resps") else "",
                    "sft_filtered": s.get("filtered_resps", [""])[0] if s.get("filtered_resps") else "",
                }
            )
        elif b_correct == 0 and s_correct == 1:
            improvements.append(
                {
                    "doc_id": doc_id,
                    "problem": get_problem(b),
                    "level": get_level(b),
                    "gold": get_gold(b),
                    "baseline_resp": get_response(b),
                    "sft_resp": get_response(s),
                    "baseline_filtered": b.get("filtered_resps", [""])[0] if b.get("filtered_resps") else "",
                    "sft_filtered": s.get("filtered_resps", [""])[0] if s.get("filtered_resps") else "",
                }
            )

    return regressions, improvements


# ---------------------------------------------------------------------------
# Phase 2b: Categorize failure modes
# ---------------------------------------------------------------------------
def has_repetition(text: str, chunk_size: int = 50, min_repeats: int = 3) -> bool:
    """Check if text contains degenerate repetition loops."""
    if len(text) < chunk_size * min_repeats:
        return False
    for i in range(0, len(text) - chunk_size):
        chunk = text[i : i + chunk_size]
        count = text.count(chunk)
        if count >= min_repeats:
            return True
    return False


def has_extraction_artifacts(text: str) -> bool:
    """Check for training-format artifacts in generation."""
    patterns = [
        r"## Question",
        r"## Reasoning",
        r"## Answer",
        r"\[\[ ## text ## \]\]",
        r"\[\[ ## completed ## \]\]",
    ]
    return any(re.search(p, text) for p in patterns)


def is_truncated(text: str) -> bool:
    """Check if response appears truncated (ends mid-sentence)."""
    text = text.strip()
    if not text:
        return True
    # Ends mid-word or mid-equation
    if text[-1] not in ".!?)}\n0123456789$\\":
        if len(text) > 500:  # Only flag for long responses
            return True
    return False


def categorize_regression(reg: dict) -> str:
    """Classify why the SFT model failed on a regression problem."""
    sft_resp = reg["sft_resp"]
    sft_filtered = str(reg.get("sft_filtered", ""))
    gold = reg["gold"]

    # Check repetition first (most distinctive)
    if has_repetition(sft_resp):
        return "repetition_loop"

    # Check for training format contamination
    if has_extraction_artifacts(sft_resp):
        return "extraction_artifacts"

    # Check truncation
    if is_truncated(sft_resp):
        return "truncation"

    # Check if answer was extracted but wrong
    if sft_filtered and sft_filtered.strip():
        # Has an answer, just wrong
        return "wrong_computation"

    # No answer extracted
    return "incomplete_reasoning"


# ---------------------------------------------------------------------------
# Phase 2c: Per-difficulty analysis
# ---------------------------------------------------------------------------
def accuracy_by_level(samples: dict, metric: str = "exact_match") -> dict[str, tuple[int, int]]:
    """Compute (correct, total) per difficulty level."""
    level_stats = defaultdict(lambda: [0, 0])
    for s in samples.values():
        level = get_level(s)
        level_stats[level][1] += 1
        if s.get(metric, 0) == 1:
            level_stats[level][0] += 1
    return dict(level_stats)


# ---------------------------------------------------------------------------
# Phase 3: Training data quality audit
# ---------------------------------------------------------------------------
def classify_training_doc(text: str) -> str:
    """Classify a training document by structure."""
    has_q = "## Question" in text
    has_r = "## Reasoning" in text
    has_a = "## Answer" in text

    if len(text) < 200:
        return "minimal"

    # Check for HTML artifacts
    html_patterns = ["<div", "<span", "<table", "&nbsp;", "&amp;", "<script"]
    if sum(1 for p in html_patterns if p in text) >= 2:
        return "garbled"

    if has_q and has_r and has_a:
        # Check if answer section has content
        answer_match = re.search(r"## Answer\s*\n(.*?)($|\n#)", text, re.DOTALL)
        if answer_match and len(answer_match.group(1).strip()) > 5:
            return "qa_complete"
        return "qa_no_answer"

    if has_q and has_r:
        return "qa_no_answer"

    if has_q and has_a:
        return "qa_no_reasoning"

    # No Q/R/A structure
    if any(kw in text.lower() for kw in ["exercise", "problem", "solve"]):
        if text.count("?") >= 3 or text.count("problem") >= 3:
            return "exercise_list"

    return "tutorial"


def audit_training_doc(text: str) -> dict:
    """Compute quality metrics for a training document."""
    return {
        "length": len(text),
        "has_latex": bool(re.search(r"\$[^$]+\$", text)),
        "has_display_math": bool(re.search(r"\$\$[^$]+\$\$", text)),
        "has_html_artifacts": bool(re.search(r"<(?:div|span|table|script|style|a href)", text)),
        "has_meta_commentary": bool(
            re.search(
                r"(?:the user asked|let me help|I'll solve|let me solve|as requested)",
                text,
                re.IGNORECASE,
            )
        ),
        "has_no_useful_content": "[NO_USEFUL_CONTENT]" in text,
        "has_think_tags": "<think>" in text,
        "has_dspy_markers": "[[ ##" in text,
        "num_equations": len(re.findall(r"\$[^$]+\$", text)),
        "num_headers": len(re.findall(r"^#{1,3} ", text, re.MULTILINE)),
        "structure": classify_training_doc(text),
    }


def estimate_token_budget(text: str) -> dict:
    """Estimate what fraction of tokens go to each component."""
    # Rough approximation: 1 token ≈ 4 chars
    total = len(text)
    if total == 0:
        return {"headers": 0, "question": 0, "reasoning": 0, "answer": 0, "other": 1.0}

    # Count header chars
    header_chars = sum(len(m.group(0)) for m in re.finditer(r"^#{1,3} .*$", text, re.MULTILINE))

    # Split by sections
    q_match = re.search(r"## Question\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    r_match = re.search(r"## Reasoning\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    a_match = re.search(r"## Answer\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)

    q_chars = len(q_match.group(1)) if q_match else 0
    r_chars = len(r_match.group(1)) if r_match else 0
    a_chars = len(a_match.group(1)) if a_match else 0
    other_chars = total - header_chars - q_chars - r_chars - a_chars

    return {
        "headers": header_chars / total,
        "question": q_chars / total,
        "reasoning": r_chars / total,
        "answer": a_chars / total,
        "other": max(0, other_chars / total),
    }


# ---------------------------------------------------------------------------
# Phase 5: Generation behavior shift
# ---------------------------------------------------------------------------
def generation_stats(samples: dict) -> dict:
    """Compute stats about model generation behavior."""
    lengths = []
    has_boxed = 0
    has_extraction_fmt = 0
    repetition_count = 0
    truncated = 0
    total = len(samples)

    for s in samples.values():
        resp = get_response(s)
        lengths.append(len(resp))

        if r"\boxed{" in resp:
            has_boxed += 1
        if has_extraction_artifacts(resp):
            has_extraction_fmt += 1
        if has_repetition(resp, chunk_size=40, min_repeats=3):
            repetition_count += 1
        if is_truncated(resp):
            truncated += 1

    lengths.sort()
    n = len(lengths)
    return {
        "count": total,
        "mean_len": sum(lengths) / n if n else 0,
        "median_len": lengths[n // 2] if n else 0,
        "p95_len": lengths[int(n * 0.95)] if n else 0,
        "max_len": max(lengths) if lengths else 0,
        "pct_boxed": has_boxed / total * 100 if total else 0,
        "pct_extraction_fmt": has_extraction_fmt / total * 100 if total else 0,
        "pct_repetition": repetition_count / total * 100 if total else 0,
        "pct_truncated": truncated / total * 100 if total else 0,
    }


def unique_ngram_ratio(text: str, n: int = 4) -> float:
    """Compute ratio of unique n-grams to total n-grams."""
    words = text.split()
    if len(words) < n:
        return 1.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_full_analysis():
    print("=" * 80)
    print("MATH SFT REGRESSION ANALYSIS")
    print("=" * 80)

    results = {}

    # -----------------------------------------------------------------------
    # Phase 2: Failure mode analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: FAILURE MODE ANALYSIS")
    print("=" * 60)

    for task in TASKS:
        print(f"\n--- {task.upper()} ---")
        baseline_path = MODELS["baseline"] / f"{task}.jsonl"
        sweep_path = MODELS["sweep_best"] / f"{task}.jsonl"

        if not baseline_path.exists() or not sweep_path.exists():
            print("  SKIP: missing files")
            continue

        baseline = load_samples(baseline_path)
        sweep = load_samples(sweep_path)
        metric = "exact_match"

        # Accuracy
        b_correct = sum(1 for s in baseline.values() if s.get(metric, 0) == 1)
        s_correct = sum(1 for s in sweep.values() if s.get(metric, 0) == 1)
        print(f"  Baseline accuracy: {b_correct}/{len(baseline)} = {100 * b_correct / len(baseline):.1f}%")
        print(f"  SFT accuracy:     {s_correct}/{len(sweep)} = {100 * s_correct / len(sweep):.1f}%")
        print(f"  Delta: {100 * (s_correct - b_correct) / len(baseline):+.1f}pp")

        # Regressions and improvements
        regressions, improvements = find_changes(baseline, sweep, metric)
        print(f"  Regressions: {len(regressions)} (baseline right → SFT wrong)")
        print(f"  Improvements: {len(improvements)} (baseline wrong → SFT right)")
        print(f"  Net: {len(improvements) - len(regressions):+d}")

        # Categorize regressions
        categories = Counter()
        for reg in regressions:
            cat = categorize_regression(reg)
            reg["category"] = cat
            categories[cat] += 1

        print("\n  Failure mode breakdown:")
        for cat, count in categories.most_common():
            print(f"    {cat:25s}: {count:4d} ({100 * count / len(regressions):.1f}%)")

        # Per-difficulty analysis (MATH tasks only, not GSM8K)
        if task != "gsm8k":
            print("\n  Per-difficulty accuracy:")
            b_levels = accuracy_by_level(baseline, metric)
            s_levels = accuracy_by_level(sweep, metric)
            for level in sorted(set(list(b_levels.keys()) + list(s_levels.keys()))):
                bc, bt = b_levels.get(level, (0, 0))
                sc, st = s_levels.get(level, (0, 0))
                b_pct = 100 * bc / bt if bt else 0
                s_pct = 100 * sc / st if st else 0
                print(
                    f"    {level:15s}: baseline={b_pct:5.1f}% ({bc}/{bt})  SFT={s_pct:5.1f}% ({sc}/{st})  Δ={s_pct - b_pct:+.1f}pp"
                )

        results[task] = {
            "regressions": regressions,
            "improvements": improvements,
            "categories": dict(categories),
            "baseline_acc": b_correct / len(baseline),
            "sft_acc": s_correct / len(sweep),
        }

    # -----------------------------------------------------------------------
    # Phase 3: Training data audit
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING DATA QUALITY AUDIT")
    print("=" * 60)

    training_docs = load_training_data(DATA_DIR / "training_data")
    print(f"\nLoaded {len(training_docs)} postprocessed training documents")

    # Structure classification
    structure_counts = Counter()
    audits = []
    for doc in training_docs:
        text = doc.get("text", "")
        audit = audit_training_doc(text)
        audits.append(audit)
        structure_counts[audit["structure"]] += 1

    print("\nStructure classification:")
    for struct, count in structure_counts.most_common():
        print(f"  {struct:20s}: {count:4d} ({100 * count / len(training_docs):.1f}%)")

    # Quality metrics
    n = len(audits)
    print("\nQuality metrics:")
    print(
        f"  Has LaTeX:           {sum(a['has_latex'] for a in audits):4d} ({100 * sum(a['has_latex'] for a in audits) / n:.1f}%)"
    )
    print(
        f"  Has display math:    {sum(a['has_display_math'] for a in audits):4d} ({100 * sum(a['has_display_math'] for a in audits) / n:.1f}%)"
    )
    print(
        f"  Has HTML artifacts:  {sum(a['has_html_artifacts'] for a in audits):4d} ({100 * sum(a['has_html_artifacts'] for a in audits) / n:.1f}%)"
    )
    print(
        f"  Has meta-commentary: {sum(a['has_meta_commentary'] for a in audits):4d} ({100 * sum(a['has_meta_commentary'] for a in audits) / n:.1f}%)"
    )
    print(f"  Has [NO_USEFUL]:     {sum(a['has_no_useful_content'] for a in audits):4d}")
    print(f"  Has <think> tags:    {sum(a['has_think_tags'] for a in audits):4d}")
    print(f"  Has DSPy markers:    {sum(a['has_dspy_markers'] for a in audits):4d}")

    # Length distribution
    lengths = [a["length"] for a in audits]
    lengths.sort()
    print("\n  Length distribution:")
    print(f"    Mean:   {sum(lengths) / n:,.0f} chars")
    print(f"    Median: {lengths[n // 2]:,d} chars")
    print(f"    p10:    {lengths[int(n * 0.1)]:,d} chars")
    print(f"    p90:    {lengths[int(n * 0.9)]:,d} chars")
    print(f"    Min:    {lengths[0]:,d} chars")
    print(f"    Max:    {lengths[-1]:,d} chars")

    # Token budget analysis (only for docs with Q/R/A structure)
    qa_docs = [doc for doc, a in zip(training_docs, audits) if a["structure"].startswith("qa_")]
    if qa_docs:
        budgets = [estimate_token_budget(doc["text"]) for doc in qa_docs]
        print(f"\n  Token budget (Q/R/A docs only, n={len(qa_docs)}):")
        for key in ["headers", "question", "reasoning", "answer", "other"]:
            vals = [b[key] for b in budgets]
            print(f"    {key:12s}: {100 * sum(vals) / len(vals):5.1f}% avg")

    # Domain distribution
    domain_counts = Counter()
    for doc in training_docs:
        url = doc.get("url", "")
        # Extract domain from URL
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        domain = match.group(1) if match else "unknown"
        domain_counts[domain] += 1

    print("\n  Top 15 source domains:")
    for domain, count in domain_counts.most_common(15):
        print(f"    {domain:35s}: {count:4d} ({100 * count / len(training_docs):.1f}%)")

    # -----------------------------------------------------------------------
    # Phase 5: Generation behavior shift
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5: GENERATION BEHAVIOR SHIFT")
    print("=" * 60)

    for task in TASKS:
        print(f"\n--- {task.upper()} ---")
        for model_name in ["baseline", "sweep_best"]:
            path = MODELS[model_name] / f"{task}.jsonl"
            if not path.exists():
                continue
            samples = load_samples(path)
            stats = generation_stats(samples)
            print(
                f"  {model_name:15s}: mean_len={stats['mean_len']:.0f}  median={stats['median_len']:.0f}  p95={stats['p95_len']:.0f}  "
                f"boxed={stats['pct_boxed']:.1f}%  extract_fmt={stats['pct_extraction_fmt']:.1f}%  "
                f"repetition={stats['pct_repetition']:.1f}%  truncated={stats['pct_truncated']:.1f}%"
            )

        # Unique n-gram analysis
        for model_name in ["baseline", "sweep_best"]:
            path = MODELS[model_name] / f"{task}.jsonl"
            if not path.exists():
                continue
            samples = load_samples(path)
            ratios = [unique_ngram_ratio(get_response(s)) for s in samples.values()]
            mean_ratio = sum(ratios) / len(ratios)
            low_diversity = sum(1 for r in ratios if r < 0.5)
            print(
                f"  {model_name:15s}: mean_4gram_diversity={mean_ratio:.3f}  low_diversity(<0.5)={low_diversity} ({100*low_diversity/len(ratios):.1f}%)"
            )

    # -----------------------------------------------------------------------
    # Print detailed examples for the report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DETAILED REGRESSION EXAMPLES")
    print("=" * 60)

    for task in TASKS:
        if task not in results:
            continue
        regressions = results[task]["regressions"]
        improvements = results[task]["improvements"]

        print(f"\n{'='*40} {task.upper()} REGRESSIONS {'='*40}")

        # Get examples from each failure category
        by_category = defaultdict(list)
        for reg in regressions:
            by_category[reg.get("category", "unknown")].append(reg)

        shown = 0
        for cat in [
            "wrong_computation",
            "repetition_loop",
            "truncation",
            "extraction_artifacts",
            "incomplete_reasoning",
        ]:
            if cat not in by_category or shown >= 8:
                break
            examples = by_category[cat][:2]
            for ex in examples:
                if shown >= 8:
                    break
                print(f"\n[REGRESSION #{shown + 1}] Category: {cat} | Level: {ex['level']} | doc_id: {ex['doc_id']}")
                print(f"Problem: {ex['problem'][:300]}")
                print(f"Gold: {ex['gold']}")
                print(f"Baseline answer (extracted): {str(ex['baseline_filtered'])[:200]}")
                print(f"SFT answer (extracted): {str(ex['sft_filtered'])[:200]}")
                print("--- Baseline response (first 500 chars) ---")
                print(ex["baseline_resp"][:500])
                print("--- SFT response (first 500 chars) ---")
                print(ex["sft_resp"][:500])
                shown += 1

        print(f"\n{'='*40} {task.upper()} IMPROVEMENTS {'='*40}")
        for i, imp in enumerate(improvements[:3]):
            print(f"\n[IMPROVEMENT #{i + 1}] Level: {imp['level']} | doc_id: {imp['doc_id']}")
            print(f"Problem: {imp['problem'][:300]}")
            print(f"Gold: {imp['gold']}")
            print(f"Baseline answer (extracted): {str(imp['baseline_filtered'])[:200]}")
            print(f"SFT answer (extracted): {str(imp['sft_filtered'])[:200]}")
            print("--- Baseline response (first 400 chars) ---")
            print(imp["baseline_resp"][:400])
            print("--- SFT response (first 400 chars) ---")
            print(imp["sft_resp"][:400])

    # -----------------------------------------------------------------------
    # Print training data examples
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING DATA EXAMPLES")
    print("=" * 60)

    # Good examples
    print("\n--- GOOD TRAINING DATA ---")
    good_count = 0
    for doc, audit in zip(training_docs, audits):
        if audit["structure"] == "qa_complete" and audit["has_latex"] and audit["length"] > 500 and good_count < 3:
            print(f"\n[GOOD #{good_count + 1}] url={doc.get('url', 'N/A')[:80]} len={audit['length']}")
            print(doc["text"][:1500])
            print("..." if len(doc["text"]) > 1500 else "")
            good_count += 1

    # Bad examples
    print("\n--- PROBLEMATIC TRAINING DATA ---")
    bad_count = 0
    for doc, audit in zip(training_docs, audits):
        text = doc.get("text", "")
        is_bad = False
        reason = ""

        if audit["structure"] == "garbled":
            is_bad = True
            reason = "HTML artifacts"
        elif audit["structure"] == "minimal":
            is_bad = True
            reason = "Too short"
        elif not audit["has_latex"] and audit["length"] > 200:
            is_bad = True
            reason = "No math (no LaTeX)"
        elif audit["has_meta_commentary"]:
            is_bad = True
            reason = "Meta-commentary"
        elif audit["structure"] == "tutorial" and audit["num_equations"] == 0:
            is_bad = True
            reason = "Tutorial with no equations"

        if is_bad and bad_count < 8:
            print(
                f"\n[BAD #{bad_count + 1}] Reason: {reason} | url={doc.get('url', 'N/A')[:80]} | len={audit['length']}"
            )
            print(text[:800])
            print("..." if len(text) > 800 else "")
            bad_count += 1

    # -----------------------------------------------------------------------
    # Format mismatch analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4: FORMAT MISMATCH ANALYSIS")
    print("=" * 60)

    # Extract a few-shot prompt from eval to show the format
    for task in ["algebra"]:
        path = MODELS["baseline"] / f"{task}.jsonl"
        if path.exists():
            samples = load_samples(path)
            first = list(samples.values())[0]
            args = first.get("arguments", [])
            if args:
                prompt = args[0] if isinstance(args[0], str) else str(args[0][0]) if args[0] else ""
                print(f"\n--- Eval prompt format ({task}, first 2000 chars) ---")
                print(prompt[:2000])

    # Compare with training format
    print("\n--- Training data format (first Q/R/A doc, first 1000 chars) ---")
    for doc, audit in zip(training_docs, audits):
        if audit["structure"] == "qa_complete":
            print(doc["text"][:1000])
            break

    # -----------------------------------------------------------------------
    # Save raw results for report generation
    # -----------------------------------------------------------------------
    output = {
        "task_results": {},
        "training_audit": {
            "structure_counts": dict(structure_counts),
            "total_docs": len(training_docs),
            "domain_counts": dict(domain_counts.most_common(20)),
        },
    }
    for task in TASKS:
        if task in results:
            output["task_results"][task] = {
                "baseline_acc": results[task]["baseline_acc"],
                "sft_acc": results[task]["sft_acc"],
                "n_regressions": len(results[task]["regressions"]),
                "n_improvements": len(results[task]["improvements"]),
                "categories": results[task]["categories"],
            }

    with open(DATA_DIR / "analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {DATA_DIR / 'analysis_results.json'}")


if __name__ == "__main__":
    run_full_analysis()
