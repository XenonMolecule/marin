#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-scale analysis of medical extraction vs resiliparse data.

Reads ALL shards from extraction raw output, postprocessed output, and
resiliparse output to produce comprehensive statistics about:
- Filter rates per domain
- Filter reasons (from model's <think> reasoning)
- Medical keyword density in kept vs filtered docs
- Text length distributions
- Side-by-side URL matching

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -- python experiments/rephraser/medical_data_analysis.py
"""

import gzip
import json
import logging
from collections import Counter, defaultdict
from urllib.parse import urlparse

import fsspec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTRACTION_RAW = "gs://marin-us-central1/documents/medical_extract_starter_ac9582fb-2631c9"
EXTRACTION_POST = "gs://marin-us-central1/processed/medical_extract_starter_ac9582fb-b46af4"
RESILIPARSE = "gs://marin-us-central1/processed/medical_resiliparse_text-9b0736"
OUTPUT_PATH = "gs://marin-us-central1/scratch/medical_data_analysis.json"

MEDICAL_KEYWORDS = [
    "diagnosis",
    "treatment",
    "symptoms",
    "medication",
    "drug",
    "dose",
    "dosage",
    "patient",
    "clinical",
    "therapy",
    "surgery",
    "condition",
    "disease",
    "disorder",
    "blood",
    "pain",
    "doctor",
    "nurse",
    "hospital",
    "prescription",
    "side effects",
    "chronic",
    "acute",
    "infection",
    "cancer",
    "diabetes",
    "heart",
    "lung",
    "liver",
    "kidney",
    "brain",
    "bone",
    "muscle",
    "nerve",
    "vitamin",
    "antibiotic",
    "vaccine",
    "allergy",
    "inflammation",
    "biopsy",
    "MRI",
    "cholesterol",
    "blood pressure",
    "glucose",
    "insulin",
    "hormone",
    "nursing",
    "NCLEX",
    "pharmacology",
    "anatomy",
    "physiology",
    "cardiology",
    "neurology",
    "pediatrics",
    "orthopedic",
    "radiology",
]


def medical_density(text):
    text_lower = text.lower()
    words = len(text_lower.split())
    if words == 0:
        return 0.0
    hits = sum(text_lower.count(kw) for kw in MEDICAL_KEYWORDS)
    return hits / words * 100


def get_domain(url):
    return urlparse(url).netloc


def classify_filter_reason(think):
    think = think.lower()
    if "not about medicine" in think or "not medical" in think or "not a medical" in think:
        return "not_medical"
    if "directory" in think or "index page" in think or "listing" in think or "sitemap" in think:
        return "directory/index"
    if "no longer active" in think or "closed" in think or "archived" in think:
        return "inactive_board"
    if "login" in think or "signup" in think or "search result" in think or "registration" in think:
        return "login/search"
    if "not in english" in think or "french" in think or "spanish" in think or "german" in think:
        return "not_english"
    if "who posted" in think or "who liked" in think or "likes page" in think:
        return "meta_page"
    if "education" in think or "school" in think or "career" in think or "job" in think:
        return "education/career"
    if "personal" in think and ("blog" in think or "story" in think or "life" in think):
        return "personal_blog"
    return "other"


def read_jsonl_gz(path):
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt") as gz:
            for line in gz:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def list_shards(base_path):
    fs, _, _paths = fsspec.get_fs_token_paths(base_path)
    all_files = fs.ls(base_path)
    return [f"gs://{p}" for p in all_files if p.endswith(".jsonl.gz")]


def main():
    logger.info("Starting full-scale medical data analysis")

    # ================================================================
    # Step 1: Build resiliparse index (URL -> text length + density)
    # ================================================================
    logger.info("Step 1: Indexing resiliparse data")
    resili_shards = list_shards(RESILIPARSE)
    logger.info(f"  Found {len(resili_shards)} resiliparse shards")

    resili_index = {}  # url -> {text_len, med_density}
    resili_domain_stats = Counter()
    resili_total = 0

    for i, shard in enumerate(resili_shards):
        if i % 100 == 0:
            logger.info(f"  Resiliparse shard {i}/{len(resili_shards)}")
        for r in read_jsonl_gz(shard):
            url = r.get("url", "")
            text = r.get("text", "")
            domain = get_domain(url)
            resili_index[url] = {
                "text_len": len(text),
                "med_density": medical_density(text),
            }
            resili_domain_stats[domain] += 1
            resili_total += 1

    logger.info(f"  Total resiliparse records: {resili_total}")

    # ================================================================
    # Step 2: Analyze extraction raw output (before postprocess)
    # ================================================================
    logger.info("Step 2: Analyzing extraction raw output")
    raw_shards = list_shards(EXTRACTION_RAW)
    logger.info(f"  Found {len(raw_shards)} raw extraction shards")

    domain_analysis = defaultdict(
        lambda: {
            "total": 0,
            "filtered": 0,
            "kept": 0,
            "filter_reasons": Counter(),
            "kept_med_densities": [],
            "filtered_med_densities": [],
            "kept_text_lens": [],
            "filtered_resili_lens": [],
            "kept_resili_lens": [],
            "filtered_extract_lens": [],
            "kept_extract_lens": [],
        }
    )

    raw_total = 0
    raw_filtered = 0
    all_filter_reasons = Counter()

    for i, shard in enumerate(raw_shards):
        if i % 100 == 0:
            logger.info(f"  Raw shard {i}/{len(raw_shards)}, processed {raw_total} records")
        for r in read_jsonl_gz(shard):
            url = r.get("url", "")
            text = r.get("generated_text", "")
            domain = get_domain(url)
            raw_total += 1

            # Parse think/output
            if "</think>" in text and "<think>" in text:
                actual = text.split("</think>")[-1].strip()
                think = text[text.index("<think>") + 7 : text.index("</think>")]
            elif "</think>" in text:
                actual = text.split("</think>")[-1].strip()
                think = text[: text.index("</think>")]
            else:
                actual = text
                think = ""

            is_filtered = "[NO_USEFUL_CONTENT]" in actual or len(actual.strip()) < 50
            da = domain_analysis[domain]
            da["total"] += 1

            # Get resiliparse info for this URL
            resili_info = resili_index.get(url)

            if is_filtered:
                raw_filtered += 1
                da["filtered"] += 1
                reason = classify_filter_reason(think)
                da["filter_reasons"][reason] += 1
                all_filter_reasons[reason] += 1

                if resili_info:
                    da["filtered_med_densities"].append(resili_info["med_density"])
                    da["filtered_resili_lens"].append(resili_info["text_len"])
            else:
                da["kept"] += 1
                da["kept_text_lens"].append(len(actual))

                if resili_info:
                    da["kept_med_densities"].append(resili_info["med_density"])
                    da["kept_resili_lens"].append(resili_info["text_len"])

    logger.info(f"  Total raw records: {raw_total}")
    logger.info(f"  Filtered: {raw_filtered} ({raw_filtered/raw_total*100:.1f}%)")

    # ================================================================
    # Step 3: Analyze extraction postprocessed
    # ================================================================
    logger.info("Step 3: Analyzing extraction postprocessed")
    post_shards = list_shards(EXTRACTION_POST)
    logger.info(f"  Found {len(post_shards)} postprocessed shards")

    extract_total = 0
    extract_domain_stats = Counter()
    extract_lengths = []

    for i, shard in enumerate(post_shards):
        if i % 200 == 0:
            logger.info(f"  Post shard {i}/{len(post_shards)}")
        for r in read_jsonl_gz(shard):
            url = r.get("url", "")
            text = r.get("text", "")
            domain = get_domain(url)
            extract_total += 1
            extract_domain_stats[domain] += 1
            extract_lengths.append(len(text))

    logger.info(f"  Total postprocessed records: {extract_total}")

    # ================================================================
    # Step 4: Compile results
    # ================================================================
    logger.info("Step 4: Compiling results")

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0

    results = {
        "summary": {
            "resiliparse_total": resili_total,
            "extraction_raw_total": raw_total,
            "extraction_postprocessed_total": extract_total,
            "overall_filter_rate": raw_filtered / raw_total if raw_total > 0 else 0,
            "filter_reasons": dict(all_filter_reasons.most_common()),
        },
        "per_domain": {},
    }

    for domain in sorted(domain_analysis, key=lambda d: -domain_analysis[d]["total"]):
        da = domain_analysis[domain]
        if da["total"] < 5:
            continue
        results["per_domain"][domain] = {
            "total": da["total"],
            "filtered": da["filtered"],
            "kept": da["kept"],
            "filter_rate": da["filtered"] / da["total"] if da["total"] > 0 else 0,
            "filter_reasons": dict(da["filter_reasons"].most_common()),
            "resiliparse_count": resili_domain_stats.get(domain, 0),
            "extract_count": extract_domain_stats.get(domain, 0),
            "avg_kept_med_density": safe_avg(da["kept_med_densities"]),
            "avg_filtered_med_density": safe_avg(da["filtered_med_densities"]),
            "avg_kept_resili_len": safe_avg(da["kept_resili_lens"]),
            "avg_filtered_resili_len": safe_avg(da["filtered_resili_lens"]),
            "avg_kept_text_len": safe_avg(da["kept_text_lens"]),
        }

    # ================================================================
    # Step 5: Print summary and save
    # ================================================================
    print("\n" + "=" * 90)
    print("FULL-SCALE MEDICAL DATA ANALYSIS")
    print("=" * 90)
    print(f"Resiliparse: {resili_total:,} records")
    print(f"Extraction raw: {raw_total:,} records")
    print(f"Extraction postprocessed: {extract_total:,} records")
    print(f"Overall filter rate: {raw_filtered/raw_total*100:.1f}%")

    print("\nFilter reasons (all domains):")
    for reason, count in all_filter_reasons.most_common():
        print(f"  {reason:<30} {count:>8} ({count/raw_filtered*100:.1f}%)")

    header = f"{'Domain':<35} {'Total':>7} {'Filt%':>6} {'ResiliN':>8} {'ExtractN':>9}"
    header += f" {'KeptDens':>9} {'FiltDens':>9} {'Problem?':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for domain, info in sorted(results["per_domain"].items(), key=lambda x: -x[1]["total"]):
        filt_pct = info["filter_rate"] * 100
        problem = ""
        if info["avg_filtered_med_density"] > 0 and info["avg_kept_med_density"] > 0:
            ratio = info["avg_filtered_med_density"] / info["avg_kept_med_density"]
            if ratio > 0.5:
                problem = f"YES ({ratio:.0%})"
        row = f"  {domain:<33} {info['total']:>7} {filt_pct:>5.1f}%"
        row += f" {info['resiliparse_count']:>8} {info['extract_count']:>9}"
        row += f" {info['avg_kept_med_density']:>8.2f} {info['avg_filtered_med_density']:>8.2f}"
        row += f" {problem:>10}"
        print(row)

    # Save full results
    with fsspec.open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
