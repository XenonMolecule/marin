# Worker Lifecycle — One Page

One worker = one Python process on one TPU slice. From boot to writing its
first batch:

1. **Boot.** Load vLLM (~3 min). Read the manifest, shuffle by
   `--shuffle-seed`.

2. **Skip what's already done.** Read the global `_completed/` registry
   (one GCS list call) → set of finished WARC hashes.

3. **Try to claim a WARC.**
   - Marked `_done` in any region → skip.
   - Has a fresh `_claimed` in any region → skip.
   - Otherwise PUT `_claimed` in the local regional bucket with
     `if_generation_match=0`. **First writer wins.** Everyone else in the
     same region gets HTTP 412 and moves on.

4. **Process the WARC, batch by batch.** Download from CommonCrawl,
   char-filter, then for each batch of 500 records:
   - Run vLLM. Write `batch_NNNN.jsonl.gz` + sidecars.
   - Refresh `_claimed` (heartbeat).
   - Yield if any *other* region has a fresh `_claimed` with an older
     `created_at` than ours.

5. **Finish.** Write `_done` with stats. Add a marker to the global
   `_completed/` registry.

6. **Endgame: steal mode.** Once most WARCs are claimed and forward progress
   stalls, the worker stops claiming whole WARCs and starts atomically
   claiming individual *batches* of in-progress WARCs from the back. Same
   primitive (`if_generation_match=0`), finer granularity, flattens the
   long tail.

That's the whole algorithm. **No scheduler, no queue, no coordinator
service** — workers self-coordinate through atomic GCS writes.
