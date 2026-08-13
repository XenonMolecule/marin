# Draft message to rav — quality classifier artifact access

---

Hey rav — quick ask about the datakit quality classifier.

I'm building a quality × domain grid over our 10k-WARC curated corpora (DCLM, high_quality,
Nemotron-CC, FineWeb-Edu/CC, resiliparse) as the substrate for some data-mixing sweeps. Quality axis
is your 5 calibrated buckets, domain axis is WebOrganizer's 24 topics.

I want to use the pooled fast-transformer (`pooled_junkgate2`) rather than the older
`sonnet46-thr05` fastText that's mirrored to GCS. That's specifically because of the type-aware
rubric: since I'm crossing quality *with topic*, a quality score that correlates with domain would
confound the whole grid. The old model's own `metadata.json` makes the point — `cp/stackv2_code`
0.219 and `finepdfs/jpn_Jpan` 0.193 against `cp/pubmed` 0.725 — so with it, the (Software Dev.,
bottom bucket) cell fills up for reasons that have nothing to do with quality. Your rubric rewrite is
exactly the fix.

The problem is I can't reach the artifact. It's at:

    s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2/

We're running on GCP. `aws login` authenticates me as AWS account 818975787962, which gets
AccessDenied on that bucket, and I don't have CoreWeave/R2 object-storage keys.

Either of these would unblock me — whichever is less hassle for you:

1. **Copy it to GCS.** Four small files, a few MB total:
   - `pooled_junkgate2.eqx`
   - `pooled_junkgate2_remap.json`
   - `pooled_junkgate2_meta.json`
   - `calib_bme.json`

   Somewhere like `gs://marin-us-central2/datakit/quality_model/pooled_junkgate2/` would be natural —
   the older model already lives at
   `gs://marin-us-central2/datakit/llm-quality-classifier/model/sonnet46-thr05/`.

2. **CoreWeave object-storage read credentials** for US-EAST-02A and I'll pull it myself.

Optional bonus if it's easy: `s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet`.
Not needed to run, but having the labels means I could re-run `calibrate.py` or sanity-check
calibration against our corpora rather than trusting it transfers.

No rush on my end — the topic axis is 100% of the compute and it's unblocked, so I can run that
tonight and attach quality scores afterwards by id join.

Thanks!

---

## Context for us (not part of the message)

- Verified 2026-07-29: `aws sts get-caller-identity` → `arn:aws:iam::818975787962:root`;
  `aws s3 ls s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2/` → AccessDenied.
- No `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` in `.env` or any shell profile.
- `marin-us-east-02a` is CoreWeave AI Object Storage (`cwobject.com`), not the R2 endpoint the
  `coreweave.yaml` iris config uses (`74981a43…r2.cloudflarestorage.com`, bucket `marin-na`).
- Fallback already reachable: `gs://marin-us-central2/datakit/llm-quality-classifier/model/sonnet46-thr05/`
  (fastText, 103 MB, 5,052 Sonnet-4.6 labels, threshold 0.5). Emits `{id, score}` only — uncalibrated,
  no `quality_bucket`.
- Third option if neither lands: retrain ourselves. `rubric.py` is now in-tree, the model is 3M params
  and trains on ~5k labels in minutes, and we can generate oracle labels with Claude.
