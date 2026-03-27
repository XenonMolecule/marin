#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate that URL patterns return results from the Common Crawl CDX API.

Quick local check: for each source, queries CDX for just page 0 of a single recent
crawl index and reports whether records exist. Helps catch bad URLs before launching
a full download pipeline.

Usage:
    uv run python experiments/rephraser/validate_cdx_sources.py
"""

import json
import logging
import sys

import requests

logging.basicConfig(level=logging.WARNING)

# Recent crawl index to probe
PROBE_INDEX = "CC-MAIN-2024-10"
CDX_URL = f"https://index.commoncrawl.org/{PROBE_INDEX}-index"

# Sources to validate: (url_pattern, match_type)
SOURCES = [
    # === Original doc sites ===
    ("devdocs.io", "host"),
    ("developer.mozilla.org", "host"),
    ("www.tensorflow.org", "host"),
    ("api.flutter.dev", "host"),
    ("docs.nvidia.com", "host"),
    ("tutorial.ponylang.io", "host"),
    ("doc.qt.io", "host"),
    ("www.kernel.org/doc/Documentation", "prefix"),
    ("docs.swift.org/swift-book/documentation/the-swift-programming-language", "prefix"),
    ("www.typescriptlang.org/docs/handbook", "prefix"),
    ("www.newtonsoft.com/json/help/html", "prefix"),
    ("docs.oracle.com/javase/tutorial/java", "prefix"),
    ("qiskit.org/documentation", "prefix"),
    ("learn.microsoft.com/en-us/azure/quantum/user-guide", "prefix"),
    ("learn.microsoft.com/en-us/dotnet/csharp", "prefix"),
    ("huggingface.co/docs", "prefix"),
    ("llvm.org/docs", "prefix"),
    ("gcc.gnu.org/onlinedocs", "prefix"),
    ("www.mathworks.com/help/matlab", "prefix"),
    ("www.boost.org/doc", "prefix"),
    ("www.qemu.org/documentation", "prefix"),
    ("docs.zephir-lang.com/0.12/en/introduction", "prefix"),
    ("maxima.sourceforge.io/docs/manual/maxima_singlepage.html", "prefix"),
    # === Stack Exchange / Stack Overflow (code-heavy Q&A) ===
    ("stackoverflow.com", "host"),
    ("codereview.stackexchange.com", "host"),
    ("codegolf.stackexchange.com", "host"),
    ("cs.stackexchange.com", "host"),
    ("cstheory.stackexchange.com", "host"),
    ("math.stackexchange.com", "host"),
    ("stats.stackexchange.com", "host"),
    ("physics.stackexchange.com", "host"),
    ("unix.stackexchange.com", "host"),
    ("askubuntu.com", "host"),
    ("softwareengineering.stackexchange.com", "host"),
    ("dsp.stackexchange.com", "host"),
    ("ai.stackexchange.com", "host"),
    ("datascience.stackexchange.com", "host"),
    ("electronics.stackexchange.com", "host"),
    ("crypto.stackexchange.com", "host"),
    ("security.stackexchange.com", "host"),
    ("tex.stackexchange.com", "host"),
    ("mathematica.stackexchange.com", "host"),
    ("mathoverflow.net", "host"),
    # === Code-heavy language docs ===
    ("docs.python.org", "host"),
    ("doc.rust-lang.org", "host"),
    ("pkg.go.dev", "host"),
    ("en.cppreference.com", "host"),
    ("kotlinlang.org/docs", "prefix"),
    ("docs.scala-lang.org", "host"),
    ("docs.julialang.org", "host"),
    ("hexdocs.pm", "host"),
    ("www.php.net/manual", "prefix"),
    ("ruby-doc.org", "host"),
    ("docs.rs", "host"),
    # === Code tutorials / references ===
    ("rosettacode.org", "host"),
    ("www.geeksforgeeks.org", "host"),
    ("realpython.com", "host"),
    ("www.w3schools.com", "host"),
    ("www.tutorialspoint.com", "host"),
    # === Other technical ===
    ("readthedocs.io", "domain"),
    ("arxiv.org/abs", "prefix"),
]


def probe_single(url_pattern: str, match_type: str) -> tuple[int, int]:
    """Query CDX for just page 0 and count raw + filtered (200/html) records."""
    params = {
        "url": url_pattern,
        "output": "json",
        "matchType": match_type,
        "page": 0,
    }
    try:
        resp = requests.get(CDX_URL, params=params, timeout=30)
        if resp.status_code == 404:
            return 0, 0
        resp.raise_for_status()
    except Exception:
        return -1, -1  # error

    raw = 0
    filtered = 0
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            raw += 1
            if str(r.get("status", "")) == "200" and "text/html" in str(r.get("mime", "")).lower():
                filtered += 1
        except json.JSONDecodeError:
            raw += 1  # count it as raw even if we can't parse
    return raw, filtered


def main():
    print(f"Probing CDX index: {PROBE_INDEX} (page 0 only for speed)")
    print(f"{'Source':<65} {'Match':<8} {'Raw':>8} {'200/HTML':>8} {'Status'}")
    print("-" * 110)

    ok_count = 0
    fail_count = 0

    for url_pattern, match_type in SOURCES:
        raw, filtered = probe_single(url_pattern, match_type)
        if raw == -1:
            status = "ERROR"
            fail_count += 1
        elif filtered > 0:
            status = "OK"
            ok_count += 1
        elif raw > 0:
            status = "NO 200/HTML"
            fail_count += 1
        else:
            status = "NO RECORDS"
            fail_count += 1
        print(f"{url_pattern:<65} {match_type:<8} {raw:>8} {filtered:>8} {status}")

    print("-" * 110)
    print(f"Total: {ok_count} OK, {fail_count} failed/empty out of {len(SOURCES)} sources")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
