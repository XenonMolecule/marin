#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Debug script to verify checkpoint state for the rephraser sweep.

Checks the GCS output directory for spec 284df911 to confirm:
1. What files actually exist (and their extensions)
2. Whether the checkpoint scanner would find them
3. How many unique document IDs have been processed

Usage:
    python experiments/rephraser/debug_checkpoint.py
"""

import os

import fsspec

OUTPUT_PATH = "gs://marin-us-central1/documents/rephraser_spec_284df911-91bfe4"
CHECKPOINT_ID_COLUMN = "id"
CONFIGURED_FILETYPE = "jsonl.gz"  # What the config says


def fsspec_glob(pattern: str) -> list[str]:
    fs, path = fsspec.core.url_to_fs(pattern)
    matched = fs.glob(path)
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    return [f"{protocol}://{p}" for p in matched]


def main():
    print(f"Output path: {OUTPUT_PATH}")
    print()

    # 1. What does the checkpoint scanner look for?
    configured_pattern = os.path.join(OUTPUT_PATH, f"**/*.{CONFIGURED_FILETYPE}")
    print(f"[1] Checkpoint scanner glob pattern: {configured_pattern}")
    configured_files = fsspec_glob(configured_pattern)
    print(f"    Found: {len(configured_files)} files")
    if configured_files:
        for f in configured_files[:5]:
            print(f"      {f}")
        if len(configured_files) > 5:
            print(f"      ... and {len(configured_files) - 5} more")
    print()

    # 2. What files actually exist?
    all_pattern = os.path.join(OUTPUT_PATH, "**/*")
    print(f"[2] All files glob pattern: {all_pattern}")
    all_files = fsspec_glob(all_pattern)
    print(f"    Found: {len(all_files)} files total")

    # Group by extension
    ext_counts: dict[str, int] = {}
    for f in all_files:
        ext = "." + f.split(".", 1)[1] if "." in os.path.basename(f) else "(no ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("    By extension:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"      {ext}: {count}")
    if all_files:
        print("    Sample files:")
        for f in all_files[:5]:
            print(f"      {f}")
        if len(all_files) > 5:
            print(f"      ... and {len(all_files) - 5} more")
    print()

    # 3. Try with .json extension instead
    json_pattern = os.path.join(OUTPUT_PATH, "**/*.json")
    print(f"[3] Corrected glob pattern: {json_pattern}")
    json_files = fsspec_glob(json_pattern)
    print(f"    Found: {len(json_files)} files")
    print()

    # 4. Count unique IDs in the existing files
    if json_files:
        import json

        unique_ids: set[str] = set()
        total_records = 0
        for filepath in json_files:
            try:
                with fsspec.open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if CHECKPOINT_ID_COLUMN in record:
                            unique_ids.add(record[CHECKPOINT_ID_COLUMN])
                        total_records += 1
            except Exception as e:
                print(f"    Error reading {filepath}: {e}")

        print("[4] Records in existing .json files:")
        print(f"    Total records: {total_records}")
        print(f"    Unique IDs: {len(unique_ids)}")
        print()

    # 5. Diagnosis
    print("=" * 60)
    print("DIAGNOSIS:")
    if len(configured_files) == 0 and len(json_files) > 0:
        print(f"  BUG CONFIRMED: Checkpoint scanner looks for *.{CONFIGURED_FILETYPE}")
        print("  but ray.data.write_json() wrote *.json files.")
        print(f"  The {len(json_files)} output files from the previous run are")
        print("  invisible to the checkpoint recovery code.")
        print()
        print("  FIX: In rephraser_sweep.py, change the inference config to use")
        print('  filetype="json" instead of "jsonl.gz", OR set')
        print('  output_filetype_override="json".')
    elif len(configured_files) > 0:
        print("  Checkpoint files found with configured extension.")
        print("  Recovery should work — the issue is elsewhere.")
    else:
        print("  No output files found at all. The previous run may not")
        print("  have written any output, or the path hash changed.")


if __name__ == "__main__":
    main()
