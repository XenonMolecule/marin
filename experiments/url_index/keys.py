# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical keys for the URL index: url_key, registrable domain, and hashes.

Every dataset's documents are reduced to a small set of comparable keys so the
lookup and coverage tools agree on identity:

* ``url_key`` -- a normalized URL for exact interactive lookup (scheme dropped,
  host lowercased and de-``www``'d, fragment dropped, trailing slash trimmed,
  **query string kept verbatim** so ``?id=1`` and ``?id=2`` stay distinct).
* ``domain``  -- the registrable domain (eTLD+1) via the vendored Public Suffix
  List, so a query for ``google.com`` matches ``www.google.com``,
  ``maps.google.com`` etc.
* ``rid``     -- the normalized ``warc_record_id`` (source-page identity, the
  cross-pipeline coverage key).
* hashes ``rid_h`` / ``text_h`` / ``dom_h`` -- 64-bit ints for compact set math.
  ``text_h`` reuses :func:`experiments.infinigram.provenance.content_hash` so it
  matches the infinigram index byte-for-byte.
"""

import functools
import os
from urllib.parse import urlsplit

import xxhash

from experiments.infinigram.provenance import content_hash

_PSL_PATH = os.path.join(os.path.dirname(__file__), "public_suffix_list.dat")

# 64-bit mask to fold the 128-bit blake2b content hash down to a UINT64 for set math.
_U64_MASK = (1 << 64) - 1


def _normalize_record_id(record_id: str) -> str:
    """Strip the ``<urn:uuid:...>`` wrapper so ids join cleanly across pipelines.

    Mirrors ``experiments.baseline_collection.decode_warcs_clean._normalize_record_id``;
    reimplemented here to avoid importing that module's heavy WARC/HTTP deps.
    """
    rid = record_id.strip()
    if rid.startswith("<") and rid.endswith(">"):
        rid = rid[1:-1]
    if rid.startswith("urn:uuid:"):
        rid = rid[len("urn:uuid:") :]
    return rid


@functools.lru_cache(maxsize=1)
def _load_psl() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Parse the vendored PSL into (normal rules, wildcard rules, exception rules).

    Wildcard rules are stored without the leading ``*.`` label; exception rules
    without the leading ``!``. All are the suffix *below* the wildcard/exception
    marker, matched by label-suffix against a host.
    """
    normal: set[str] = set()
    wildcard: set[str] = set()
    exception: set[str] = set()
    with open(_PSL_PATH, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("!"):
                exception.add(line[1:])
            elif line.startswith("*."):
                wildcard.add(line[2:])
            else:
                normal.add(line)
    return frozenset(normal), frozenset(wildcard), frozenset(exception)


def _public_suffix(host: str) -> str:
    """The public suffix of ``host`` per the PSL algorithm (exceptions > wildcard > normal)."""
    normal, wildcard, exception = _load_psl()
    labels = host.split(".")
    # Exception rules win: the public suffix is everything after the first label
    # of the matched exception rule.
    for i in range(len(labels)):
        candidate = ".".join(labels[i:])
        if candidate in exception:
            return ".".join(labels[i + 1 :])
    # Longest matching normal or wildcard rule.
    for i in range(len(labels)):
        candidate = ".".join(labels[i:])
        if candidate in normal:
            return candidate
        # Wildcard: *.<parent> matches when labels[i+1:] equals a wildcard parent.
        parent = ".".join(labels[i + 1 :])
        if parent and parent in wildcard:
            return candidate
    # No rule matched: the whole last label is the suffix (PSL default rule "*").
    return labels[-1]


def registrable_domain(host: str) -> str:
    """eTLD+1 for ``host`` (e.g. ``maps.google.com`` -> ``google.com``, ``bbc.co.uk`` -> ``bbc.co.uk``).

    Returns ``host`` unchanged when it *is* a public suffix or has no label above it.
    """
    host = host.lower().strip(".")
    if not host:
        return ""
    suffix = _public_suffix(host)
    if host == suffix:
        return host
    suffix_labels = suffix.split(".")
    host_labels = host.split(".")
    take = len(suffix_labels) + 1
    if take > len(host_labels):
        return host
    return ".".join(host_labels[-take:])


def url_key(url: str) -> str:
    """Normalize a URL for exact lookup: drop scheme + fragment, lowercase/de-www host,
    trim trailing slash, keep path + query verbatim.

    Empty string for a url with no host (garbage/relative), so callers can skip it.
    """
    raw = url.strip()
    if "//" not in raw.split("?", 1)[0] and not raw.startswith("//"):
        # Bare host or host/path with no scheme -- give urlsplit a scheme to parse.
        raw = "//" + raw
    parts = urlsplit(raw, scheme="")
    host = parts.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    path = parts.path.rstrip("/") if parts.path != "/" else ""
    key = host + path
    if parts.query:
        key += "?" + parts.query
    return key


def domain_of(url: str) -> str:
    """Registrable domain for a URL string (empty when unparseable)."""
    raw = url.strip()
    if "//" not in raw.split("?", 1)[0] and not raw.startswith("//"):
        raw = "//" + raw
    host = urlsplit(raw, scheme="").hostname or ""
    return registrable_domain(host)


def u64(s: str) -> int:
    """Stable 64-bit hash of a string for set-math key columns."""
    return xxhash.xxh3_64_intdigest(s)


def text_hash_u64(text: str) -> int:
    """UINT64 fold of the infinigram blake2b content hash (same identity, compact)."""
    return int(content_hash(text), 16) & _U64_MASK
