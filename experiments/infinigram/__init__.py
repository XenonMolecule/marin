# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and query infini-gram-mini FM-index indices over Marin document datasets.

infini-gram-mini (Xu et al. 2025, arXiv:2506.12229) is an FM-index that stores a
text corpus at ~0.44x its size and answers exact-match n-gram counting and
document-retrieval queries in seconds. This package wraps it so any Marin
extracted-document dataset can be turned into an index with a single call, and so
switching which index you query is a one-liner.

Layout:
    targets   -- the dataset registry (which gs:// document tiers to index)
    resolve   -- expand a registry entry into concrete, region-checked shard URLs
    stage     -- copy resolved shards to a local same-region dir the indexer reads
    build     -- invoke infini-gram-mini's indexing.py on the staged corpus
    upload    -- push the finished index to a canonical GCS location
    query     -- open an index (or several) and run count/find; also the smoke test
    pipeline  -- the end-to-end orchestrator + CLI (the body of an Iris job)
    launch_infinigram_iris -- Iris coordinator that submits one job per target
"""
