# 10K WARC Cleanup Tracking

## Status: COMPLETED 2026-04-28

Deleted `gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/`
(10,364 `data-*.jsonl.gz` files, 7.985 TB) via
`gcloud storage rm -r` after building and self-verifying the manifest.

Surviving artifacts (all confirmed intact post-deletion):
- `filtered/dclm_400m_1x_10k_{dclm-8fd835, dclm_resharded-1fe977, nemotron_full-96bad9, fineweb_edu-0d49e9}/`
- `tokenized/dclm_400m_1x_10k_{dclm-3df0ba (7.33B tok), nemotron_full-3dcb75 (10.13B tok), fineweb_edu-0a3143 (2.35B tok)}/`
- `metadata/dclm_400m_1x_10k_warc_metadata-79158f/`
- `manifests/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz` (re-download verification)

To re-download: re-run `experiments/baseline_collection/pipeline_10k.py` (the
executor will repopulate the same `dclm_400m_1x_10k-ee2365/` prefix), then
`uv run python experiments/baseline_collection/verify_10k_manifest.py
--new-pool gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/`
to confirm byte-identical extraction (size + crc32c per file).

---

## (Original tracking notes follow, retained for context.)


## Background

We launched `experiments/baseline_collection/pipeline_10k.py` to extract Nemotron-CC v1
records for the full DCLM 400m-1x pool (10,363 WARCs). This downloaded ~10 TB of WARC
HTML to GCS as a temporary intermediate.

The 3000-WARC subset is preserved separately at the original `baseline_3000-<hash>/` path
and should NOT be deleted (the user keeps it for other things).

## What to delete

The 10K download is isolated to its own GCS prefix so cleanup is one command:

```bash
gcloud storage rm -r gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/
```

Wiping that prefix removes 7.99 TB of extracted WARC HTML (10,364
`data-<hash>.jsonl.gz` files). The downstream artifacts that we keep:

- DCLM filter output: `gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm-3df0ba/`
- Nemotron-full filter output: `gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9/`
- FineWeb-Edu filter output: `gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9/`
- Tokenized: `gs://marin-us-central2/tokenized/dclm_400m_1x_10k_{dclm,nemotron_full,fineweb_edu}-<HASH>/`

The intermediate per-WARC metadata at
`gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/` is small
(~500 MB for 10,364 WARCs at ~50 KB each); fine to keep.

## Pre-deletion manifest (for re-download verification)

Before any deletion, capture the full pool contents so a future re-download can
be verified 1:1 (path + size + crc32c per file):

- Builder: `experiments/baseline_collection/build_10k_manifest.py`
- Verifier: `experiments/baseline_collection/verify_10k_manifest.py`
- Manifest copy in repo: `experiments/baseline_collection/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz` (~0.6 MB)
- Manifest copy on GCS: `gs://marin-us-central2/manifests/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz`

Built 2026-04-28 against the live pool — all 10,364 source WARCs from
`experiments/distill/dclm_400m_1x.txt` matched exactly one `data-<hash>.jsonl.gz`
output (sha256(warc_path)[:12]). Self-test passed: 10364/10364 size+crc32c match.

To verify a future re-download::

    uv run python experiments/baseline_collection/verify_10k_manifest.py \
        --new-pool gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-<NEW_HASH>/

Source WARC list lives at `experiments/distill/dclm_400m_1x.txt` (10,364 lines).
Re-download is a re-run of `pipeline_10k.py` — pulls from CC's S3 (us-east-1),
extracts via the same `download_warcs._download_one_warc` code path. As long as
that extraction code is unchanged, byte-identical outputs are expected.

## When to delete

After:
1. The pipeline has reached the `tokenize_nemotron_10k_full` step (so we have what we need)
2. The user has confirmed they're satisfied with the output

DO NOT auto-delete. The user explicitly said "Don't delete them without asking me first,
but don't forget to remind me later!!!"

## Reminder mechanism

- Set a `/schedule` agent to ping the user in ~3 days asking whether to wipe.
- If the pipeline is still running, defer.
- If the user says go, run the `gcloud storage rm -r` above.

## Why we don't reuse the existing 3000

The 3000 download lives at a different hashed path (different config = different hash).
Reusing would require either (a) hardcoding the output path past the executor, breaking
reproducibility, or (b) pre-populating the new dir with `gcloud storage cp`. We chose
neither — duplicating ~3 TB of WARCs for the 3000 portion is cheaper than the engineering
risk, and the entire 10K dir gets wiped anyway.
