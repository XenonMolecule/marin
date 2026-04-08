# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Empirical CDX API rate limit test.

Tests increasing levels of concurrency against the CDX API to find
the practical parallelism ceiling before errors dominate.
"""

import concurrent.futures
import statistics
import time

import requests

CDX_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"

# Use a lightweight query that returns fast
TEST_PARAMS = {
    "url": "example.com",
    "output": "json",
    "matchType": "exact",
    "showNumPages": "true",
}

TIMEOUT = 30


def single_request(request_id: int) -> dict:
    """Make one CDX request and return timing + status."""
    t0 = time.monotonic()
    try:
        resp = requests.get(CDX_INDEX_URL, params=TEST_PARAMS, timeout=TIMEOUT)
        elapsed = time.monotonic() - t0
        return {
            "id": request_id,
            "status": resp.status_code,
            "elapsed": elapsed,
            "ok": resp.status_code == 200,
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "id": request_id,
            "status": str(e)[:80],
            "elapsed": elapsed,
            "ok": False,
        }


def test_concurrency(n_concurrent: int, n_total: int) -> dict:
    """Fire n_total requests with n_concurrent workers."""
    results = []
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(single_request, i) for i in range(n_total)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    wall_time = time.monotonic() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    latencies = [r["elapsed"] for r in ok]

    summary = {
        "concurrency": n_concurrent,
        "total": n_total,
        "ok": len(ok),
        "fail": len(fail),
        "wall_time": round(wall_time, 2),
        "qps": round(len(ok) / wall_time, 1) if wall_time > 0 else 0,
    }
    if latencies:
        summary["p50_ms"] = round(statistics.median(latencies) * 1000)
        summary["p90_ms"] = round(sorted(latencies)[int(len(latencies) * 0.9)] * 1000)
        summary["max_ms"] = round(max(latencies) * 1000)

    if fail:
        # Show unique error types
        error_types = {}
        for r in fail:
            key = str(r["status"])
            error_types[key] = error_types.get(key, 0) + 1
        summary["errors"] = error_types

    return summary


def main():
    # Warm up — single request to establish DNS/TLS
    print("Warming up...")
    warmup = single_request(0)
    print(f"  Warmup: status={warmup['status']}, latency={warmup['elapsed']:.2f}s")
    if not warmup["ok"]:
        print("  CDX API unreachable, aborting.")
        return

    print()
    print(
        f"{'Conc':>5} {'Total':>6} {'OK':>5} {'Fail':>5} {'Wall(s)':>8} {'QPS':>6} {'p50ms':>7} {'p90ms':>7} {'max_ms':>7} {'Errors'}"
    )
    print("-" * 90)

    configs = [
        # (concurrency, total_requests)
        (1, 5),
        (2, 10),
        (4, 16),
        (8, 24),
        (16, 32),
        (32, 32),
        (64, 64),
    ]

    for conc, total in configs:
        result = test_concurrency(conc, total)
        errors_str = str(result.get("errors", "")) if result.get("errors") else ""
        print(
            f"{result['concurrency']:>5} {result['total']:>6} {result['ok']:>5} {result['fail']:>5} "
            f"{result['wall_time']:>8} {result['qps']:>6} "
            f"{result.get('p50_ms', '-'):>7} {result.get('p90_ms', '-'):>7} {result.get('max_ms', '-'):>7} "
            f"{errors_str}"
        )
        # Brief pause between rounds to avoid contaminating results
        time.sleep(2)

    # Also test with a heavier query (domain match on a real site)
    print()
    print("=== Heavier query test (domain match, actual pagination) ===")
    heavy_params = {
        "url": "mathhelpforum.com",
        "output": "json",
        "matchType": "host",
        "page": 0,
    }

    def heavy_request(request_id: int) -> dict:
        t0 = time.monotonic()
        try:
            resp = requests.get(CDX_INDEX_URL, params=heavy_params, timeout=60)
            elapsed = time.monotonic() - t0
            lines = len(resp.text.strip().split("\n")) if resp.status_code == 200 else 0
            return {
                "id": request_id,
                "status": resp.status_code,
                "elapsed": elapsed,
                "ok": resp.status_code == 200,
                "lines": lines,
            }
        except Exception as e:
            return {"id": request_id, "status": str(e)[:80], "elapsed": time.monotonic() - t0, "ok": False, "lines": 0}

    for conc in [1, 4, 8, 16]:
        results = []
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(heavy_request, i) for i in range(conc)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        wall = time.monotonic() - t0
        ok = sum(1 for r in results if r["ok"])
        fail_codes = [str(r["status"]) for r in results if not r["ok"]]
        latencies = [r["elapsed"] for r in results if r["ok"]]
        p50 = round(statistics.median(latencies) * 1000) if latencies else "-"
        print(f"  conc={conc:>3}: {ok}/{len(results)} ok, wall={wall:.1f}s, p50={p50}ms, errors={fail_codes or 'none'}")
        time.sleep(2)


if __name__ == "__main__":
    main()
