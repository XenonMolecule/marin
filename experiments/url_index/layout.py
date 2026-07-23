# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS layout for the URL index artifacts, one dir per (collection, dataset).

Each build emits three parquets in the dataset's own region (never cross-region):

* ``keys.parquet`` -- ``{dataset, rid_h, text_h, dom_h}`` UINT64 hashes for coverage.
* ``meta.parquet`` -- ``{dataset, url_key, domain, warc_record_id, snapshot,
  warc_file, text_len}`` -- the compact lookup-routing columns (no text), portable.
* ``text.parquet`` -- meta + ``text``, sorted ``(domain, url_key)``; stays in-region.

Plus ``stats.json`` with doc/provenance counts.
"""

from experiments.infinigram.targets import REGION_BUCKET, Collection

URL_INDEX_ROOT_TEMPLATE = "{bucket}/url_index/{collection}/{dataset}"

KEYS_NAME = "keys.parquet"
META_NAME = "meta.parquet"
TEXT_NAME = "text.parquet"
STATS_NAME = "stats.json"


def index_dir(region: str, collection: Collection, dataset: str) -> str:
    """Canonical GCS output dir for one dataset's URL-index artifacts."""
    return URL_INDEX_ROOT_TEMPLATE.format(
        bucket=REGION_BUCKET[region],
        collection=collection.value,
        dataset=dataset,
    )


def keys_path(region: str, collection: Collection, dataset: str) -> str:
    return f"{index_dir(region, collection, dataset)}/{KEYS_NAME}"


def meta_path(region: str, collection: Collection, dataset: str) -> str:
    return f"{index_dir(region, collection, dataset)}/{META_NAME}"


def text_path(region: str, collection: Collection, dataset: str) -> str:
    return f"{index_dir(region, collection, dataset)}/{TEXT_NAME}"


def stats_path(region: str, collection: Collection, dataset: str) -> str:
    return f"{index_dir(region, collection, dataset)}/{STATS_NAME}"
