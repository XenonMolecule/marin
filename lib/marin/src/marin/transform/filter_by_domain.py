# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter datasets by URL domain.

Drops or keeps JSONL records based on the domain in a URL field. Useful for
curating training data from multi-source crawls without re-running expensive
upstream steps (extraction, postprocessing).

Supports two modes:
- **allowlist**: keep only records whose domain is in ``allowed_domains``.
- **blocklist**: drop records whose domain is in ``blocked_domains``.

Domain matching extracts the registerable domain from the URL (e.g.
``forums.wolfram.com`` → ``wolfram.com`` when ``match_subdomains=True``,
or exact hostname match when ``match_subdomains=False``).

Example usage as an executor step::

    filtered = ExecutorStep(
        name="filtered/math_no_physics",
        fn=filter_by_domain,
        config=FilterByDomainConfig(
            input_path=postprocessed / "*.jsonl.gz",
            output_path=this_output_path(),
            blocked_domains=["physicsforums.com", "mathoverflow.net"],
        ),
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
        pip_dependency_groups=["cpu"],
    )
"""

import dataclasses
import gzip
import json
import logging
from collections.abc import Iterator
from urllib.parse import urlparse

import draccus
import fsspec
from zephyr import Dataset, ZephyrContext, load_jsonl

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FilterByDomainConfig:
    """Configuration for filtering JSONL records by URL domain.

    Specify exactly one of ``allowed_domains`` or ``blocked_domains``.

    Attributes:
        input_path: Glob pattern for input JSONL files.
        output_path: Directory to write filtered JSONL output.
        allowed_domains: If set, keep ONLY records matching these domains.
        blocked_domains: If set, DROP records matching these domains.
        url_column: Name of the field containing the URL.
        match_subdomains: If True, ``wolfram.com`` matches ``forums.wolfram.com``.
    """

    input_path: str
    output_path: str
    allowed_domains: list[str] = dataclasses.field(default_factory=list)
    blocked_domains: list[str] = dataclasses.field(default_factory=list)
    url_column: str = "url"
    match_subdomains: bool = True


def _extract_domain(url: str) -> str:
    """Extract hostname from a URL, lowercased."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _domain_matches(hostname: str, domain_set: set[str], match_subdomains: bool) -> bool:
    """Check if hostname matches any domain in the set."""
    if not hostname:
        return False
    hostname = hostname.lower()
    if hostname in domain_set:
        return True
    if match_subdomains:
        # Check if hostname is a subdomain of any domain in the set
        for domain in domain_set:
            if hostname.endswith("." + domain):
                return True
    return False


def _filter_records(*, config: FilterByDomainConfig, records: Iterator[dict]) -> Iterator[dict]:
    """Worker: yield records that pass the domain filter."""
    use_allowlist = len(config.allowed_domains) > 0
    domain_set = set(d.lower() for d in (config.allowed_domains if use_allowlist else config.blocked_domains))

    for record in records:
        url = record.get(config.url_column, "")
        hostname = _extract_domain(url)
        matches = _domain_matches(hostname, domain_set, config.match_subdomains)

        if use_allowlist and matches:
            yield record
        elif not use_allowlist and not matches:
            yield record


def filter_by_domain(config: FilterByDomainConfig):
    """Filter a JSONL dataset by URL domain."""
    if not config.allowed_domains and not config.blocked_domains:
        raise ValueError("Must specify either allowed_domains or blocked_domains")
    if config.allowed_domains and config.blocked_domains:
        raise ValueError("Specify only one of allowed_domains or blocked_domains, not both")

    mode = "allowlist" if config.allowed_domains else "blocklist"
    domains = config.allowed_domains or config.blocked_domains
    logger.info(f"Domain filter ({mode}): {domains}")

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(lambda records: _filter_records(config=config, records=records))
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="filter-by-domain") as ctx:
        ctx.execute(pipeline)

    # Count output and remove empty shards
    output_files = fsspec_glob(f"{config.output_path}/*.jsonl.gz")
    kept = 0
    empty_files: list[str] = []
    for path in output_files:
        shard_count = 0
        with fsspec.open(path, "rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                for _ in gz:
                    shard_count += 1
        if shard_count == 0:
            empty_files.append(path)
        kept += shard_count

    for path in empty_files:
        fs, _ = fsspec.core.url_to_fs(path)
        fs.rm(path)
        logger.info(f"Removed empty shard: {path}")
    if empty_files:
        logger.info(f"Removed {len(empty_files)} empty shard(s) out of {len(output_files)} total.")

    logger.info("=" * 70)
    logger.info(f"DOMAIN FILTER COMPLETE: kept {kept} records ({mode}: {domains})")
    logger.info("=" * 70)

    stats = {"kept": kept, "mode": mode, "domains": domains, "url_column": config.url_column}
    stats_path = f"{config.output_path}/filter_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote filter stats to {stats_path}")


@draccus.wrap()
def main(config: FilterByDomainConfig):
    filter_by_domain(config)


if __name__ == "__main__":
    main()
