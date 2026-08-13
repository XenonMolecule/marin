# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve an :class:`IndexTarget` into concrete, region-checked shard URLs.

This is the boundary that turns a registry entry (possibly a confighash prefix
for a still-processing dataset) into the exact list of gs:// document shards to
index, and enforces that every shard lives in the target's region -- the check
that prevents an index build from silently triggering cross-region egress.
"""

import logging
import re
from dataclasses import dataclass

from experiments.fsspec_paths import fsspec_glob, fsspec_size
from experiments.infinigram.targets import REGION_BUCKET, IndexTarget

logger = logging.getLogger(__name__)

# A confighash segment we resolve by listing: e.g. baseline_resiliparse-<hash>.
_CONFIGHASH_STAR = re.compile(r"^[A-Za-z0-9_.\-]+-\*$")

_BUCKET_RE = re.compile(r"^gs://([^/]+)/")


@dataclass(frozen=True)
class ResolvedTarget:
    """A target expanded to its concrete inputs."""

    target: IndexTarget
    shard_urls: tuple[str, ...]
    shard_bytes: tuple[int, ...]

    @property
    def shard_count(self) -> int:
        return len(self.shard_urls)

    @property
    def total_bytes(self) -> int:
        return sum(self.shard_bytes)


def _bucket_of(url: str) -> str:
    m = _BUCKET_RE.match(url)
    if not m:
        raise ValueError(f"not a gs:// URL: {url!r}")
    return m.group(1)


def _assert_in_region(urls: list[str], region: str) -> None:
    """Fail if any shard is outside the target region (cross-region egress guard)."""
    want = _bucket_of(f"{REGION_BUCKET[region]}/")
    bad = sorted({_bucket_of(u) for u in urls} - {want})
    if bad:
        raise ValueError(
            f"target region {region!r} expects bucket {want!r} but shards span {bad}; "
            "refusing to build across regions (egress cost)."
        )


def resolve_prefix(prefix_glob: str) -> list[str]:
    """Resolve a glob whose path contains a single ``<name>-*`` confighash segment.

    Asserts the ``-*`` segment matches exactly one directory, then globs the full
    pattern under it. Raises with the candidates on an ambiguous or empty match so
    a not-yet-landed dataset fails loudly rather than indexing nothing.
    """
    star_segments = [seg for seg in prefix_glob.split("/") if _CONFIGHASH_STAR.match(seg)]
    if len(star_segments) != 1:
        raise ValueError(
            f"prefix must contain exactly one '<name>-*' confighash segment, got {star_segments} in {prefix_glob!r}"
        )
    seg = star_segments[0]
    dir_pattern = prefix_glob.split(seg)[0] + seg
    dirs = sorted({d.rstrip("/") for d in fsspec_glob(dir_pattern.rstrip("/"))})
    if len(dirs) != 1:
        raise ValueError(f"confighash segment {seg!r} matched {len(dirs)} dirs, need exactly 1: {dirs}")
    shards = sorted(fsspec_glob(prefix_glob))
    if not shards:
        raise ValueError(f"no shards under resolved dir {dirs[0]!r} (pattern {prefix_glob!r})")
    return shards


def resolve_target(target: IndexTarget) -> ResolvedTarget:
    """Expand ``target.source`` into concrete shard URLs, region-checked, with sizes.

    Raises if the source resolves to no shards (e.g. a dataset still processing).
    """
    source = target.source
    shards: list[str] = []
    if source.prefix is not None:
        shards.extend(resolve_prefix(source.prefix))
    else:
        for pattern in source.globs:
            shards.extend(fsspec_glob(pattern))
    shards = sorted(set(shards))
    if not shards:
        raise ValueError(
            f"{target.name}: no shards for source {source}. "
            "Dataset may still be processing -- re-run once its documents land."
        )

    _assert_in_region(shards, target.region)
    sizes = tuple(fsspec_size(u) for u in shards)
    logger.info("%s resolved to %d shards, %.2f GiB", target.name, len(shards), sum(sizes) / 1024**3)
    return ResolvedTarget(target=target, shard_urls=tuple(shards), shard_bytes=sizes)
