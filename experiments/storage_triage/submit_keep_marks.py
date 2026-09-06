"""Submit keep marks to the gcs-usage mark & sweep ledger (https://gcs.oa.dev).

Reads mark_keep_prefixes.txt (marked keep + claimed as @me) and
mark_keep_communal.txt (marked keep, ownership left untouched) from this
directory and POSTs them to /api/actions in chunks.

Usage:
    export GCS_USAGE_TOKEN=...   # mint at https://gcs.oa.dev (user menu)
    python submit_keep_marks.py [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

URL = os.environ.get("GCS_USAGE_URL", "https://gcs.oa.dev")
MEMO = "purge preservation manifest 2026-08-26 (agent-assembled from KEEP_DELETE_MANIFEST + code/runbook sweeps)"
CHUNK = 100
HERE = Path(__file__).parent


def load(name: str) -> list[str]:
    path = HERE / name
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    for ln in lines:
        if not (ln.startswith("gs://marin-") and ln.endswith("/")):
            raise ValueError(f"bad prefix in {name}: {ln}")
    return lines


def post(actions: list[dict], token: str) -> None:
    req = urllib.request.Request(
        f"{URL}/api/actions",
        data=json.dumps(actions).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # the endpoint's WAF rejects the default Python-urllib user agent
            "User-Agent": "gcs-usage-submit/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(f"  POST {len(actions)} actions -> HTTP {resp.status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mine = load("mark_keep_prefixes.txt")
    communal = load("mark_keep_communal.txt")
    actions = [{"pattern": p, "keep": "keep", "owner": "@me", "memo": MEMO} for p in mine]
    actions += [{"pattern": p, "keep": "keep", "memo": MEMO + " [communal — not claiming ownership]"} for p in communal]
    print(f"{len(mine)} personal + {len(communal)} communal = {len(actions)} keep actions")

    if args.dry_run:
        print(json.dumps(actions[:3], indent=2), "\n... (dry run, nothing sent)")
        return

    token = os.environ.get("GCS_USAGE_TOKEN")
    if not token:
        sys.exit("GCS_USAGE_TOKEN is not set — mint one at https://gcs.oa.dev (user menu) and export it")
    for i in range(0, len(actions), CHUNK):
        post(actions[i : i + CHUNK], token)
    print("done — verify with: gcs-usage status <prefix> or GET /api/resolve")


if __name__ == "__main__":
    main()
