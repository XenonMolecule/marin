---
license: odc-by
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.parquet"
language:
- en
task_categories:
- text-generation
pretty_name: high_quality (Common Crawl web text)
size_categories:
- 10B<n<100B
tags:
- common-crawl
- web
- pretraining
- text
- llm-extraction
---

# high_quality

A high-quality English web text corpus extracted from Common Crawl WARC files using an
LLM-based extraction and quality pipeline.

## Dataset Summary

`high_quality` is a pretraining-grade corpus of cleaned web documents. Raw Common Crawl
WARC records are passed through an LLM-based extractor that strips boilerplate and recovers
the main content, then filtered to retain only documents in the "high_quality" band,
deduplicated (exact + fuzzy), and decontaminated against common evaluation benchmarks.

- **Roughly 21 billion tokens.**
- Sourced from **10,364 Common Crawl WARC files** drawn from the DCLM 400m-1x pool.
- Distributed as **parquet shards**.
- Every row carries full provenance back to its original Common Crawl WARC record.

Exact post-join counts:

- Documents: **19,968,996**
- Tokens: **~23.4 billion** (estimated as characters / 4; ~21B by tokenizer count)
- Shards: **512** parquet files
- On-disk size: **59.18 GB**

## Provenance & Traceability

Each document retains the metadata needed to trace it back to the exact Common Crawl record
it was extracted from. This makes the corpus auditable: you can re-fetch the original raw
record, verify the extraction, or recover additional context (HTTP headers, original HTML).

Per-row provenance fields:

- `url` — the source page URL.
- `warc_file` — the Common Crawl WARC file path the document came from.
- `warc_record_id` — the WARC record UUID within that file.
- `snapshot` — the Common Crawl crawl snapshot, in `CC-MAIN-YYYY-WW` form.

To fetch an original record, locate `warc_file` within the corresponding `snapshot` on the
[Common Crawl](https://commoncrawl.org/) public data, then seek to the record identified by
`warc_record_id`. The `url` field lets you cross-check the recovered record against the page
it was crawled from.

**Coverage.** 99.99% of documents (19,967,190 of 19,968,996) carry full WARC provenance.
A small remainder — 1,806 documents (0.009%) — retain their `text` but have null provenance
fields: their exact extracted text could not be matched back to a source record during the
metadata join. They are kept for completeness; filter on a non-null `warc_file` if you need
only fully-traceable rows.

## How it was built

The pipeline runs the following stages:

1. **LLM-based extraction / cleaning.** Each raw Common Crawl WARC record is processed by an
   LLM-based extractor that removes navigation, boilerplate, and markup, and reconstructs the
   readable main text of the page.
2. **Quality filtering.** Extracted documents are scored and only those in the
   "high_quality" band are kept.
3. **Deduplication.** Exact duplicate removal followed by fuzzy near-duplicate removal using
   MinHash-LSH, to collapse documents that are byte-identical or close paraphrases/near-copies.
4. **Decontamination (CORE-v2).** Documents containing distinctive evaluation n-grams are
   dropped so the corpus is safe to train on without leaking common benchmarks (see below).

## Decontamination note

The corpus is decontaminated using CORE-v2: documents that contain distinctive n-grams from
common evaluation benchmarks are removed. The goal is to let downstream models train on this
data without contaminating those benchmarks. Decontamination reduces but does not provably
eliminate all overlap — n-gram filtering is a heuristic, and novel or reformatted eval items
may still slip through. Treat it as a strong best-effort safeguard, not a guarantee.

## Data fields

| Field             | Type  | Description                                                        |
| ----------------- | ----- | ------------------------------------------------------------------ |
| `text`            | `str` | The cleaned document text (LLM-extracted main content).            |
| `url`             | `str` | Source page URL.                                                   |
| `warc_record_id`  | `str` | WARC record UUID within the source WARC file.                      |
| `warc_file`       | `str` | Common Crawl WARC file path the document was extracted from.       |
| `snapshot`        | `str` | Common Crawl crawl snapshot (`CC-MAIN-YYYY-WW`).                   |

## Splits & size

The dataset is released as a single `train` split, sharded into parquet files.

| Split   | Documents    | Tokens   | Shards | Size       |
| ------- | ------------ | -------- | ------ | ---------- |
| `train` | 19,968,996   | ~23.4B   | 512    | 59.18 GB   |

## Licensing

The dataset compilation is released under the **Open Data Commons Attribution License
(ODC-BY 1.0)** — the same license used by comparable Common Crawl–derived corpora such as
C4 and FineWeb. ODC-BY covers the curation and packaging of this dataset; it does **not**
grant any rights over the underlying web content, which Common Crawl itself does not own.

Common Crawl is **not** distributed under a Creative Commons license. It is governed by the
[Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use), and the crawled pages
remain the property of their original authors. Accordingly, use of the documents in this
dataset is **also subject to the Common Crawl Terms of Use**, and individual documents
remain subject to the rights of their original publishers.

**Disclaimer.** This data is a derivative of publicly crawled web pages. Use is subject to the
Common Crawl Terms of Use, and downstream users are expected to respect `robots.txt` and the
Common Crawl ToU. The underlying content was authored by third parties; the dataset
maintainers do not claim ownership of the source text and make no representation about the
rights status of any individual document. If you are a content owner and want a document
removed, contact the maintainers.

## Limitations & biases

- **LLM-extraction artifacts.** The main text is reconstructed by an LLM, which can introduce
  errors: dropped or duplicated passages, hallucinated connective text, lost tables/formatting,
  or occasional misreading of page structure.
- **Web noise.** Despite quality filtering, the corpus reflects the open web: it contains
  factual errors, spam-adjacent content, dated information, and the demographic, topical, and
  linguistic skews of crawled English-language pages.
- **English only.** The corpus is filtered to English; it is not suitable as a multilingual
  resource.
- **Quality filtering is a heuristic.** The "high_quality" band is a model-scored judgment,
  not ground truth; some low-value documents are retained and some good documents are dropped.
- **Decontamination is best-effort** (see the decontamination note above).

## Citation

```bibtex
@misc{high_quality_cc_21b,
  title  = {high_quality: An LLM-extracted high-quality web text corpus from Common Crawl},
  author = {MichaelR207},
  year   = {2026},
  note   = {Derived from Common Crawl.}
}
```
