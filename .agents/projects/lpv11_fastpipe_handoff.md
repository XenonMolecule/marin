# lpv11_fastpipe_v1 — session handoff (2026-08-11)

Resume point after a laptop restart. Iris jobs survive; everything session-local does not.

## Still running on the cluster (survives restart)

| job | state | what it is |
|---|---|---|
| `/michaelryan/fastcur-a-lpv11_fastpipe_v1-0` | RUNNING | Phase A smoke, us-east5, `--limit 2`. 1/2 pre-survivor parquet written. |

Check with:
```bash
set -a; source .env; set +a
uv run iris --cluster marin job list --prefix /michaelryan/fastcur-a-lpv11_fastpipe_v1-0 --json
gsutil ls gs://marin-us-east5/documents/fast_curation/lpv11_fastpipe_v1-2224e3e476/a_presurvivors/
```
(`iris job logs` does NOT work — finelog has been down all session. Use `iris job bug-report <name>`
and grep for "Job Summary" -A16.)

## Dies with the session (re-arm if wanted)

- All background bash waiters — nothing important, just pollers.
- The hourly WARC-check cron was **already deleted** (its work completed; lpv11 is at 100% coverage).
- Two subagents (resiliparse build, Phase C) — both **finished**, work is on disk.

## Next steps, in order

1. **Smoke Phase B** once Phase A finishes (this is the interesting one — pooled runs ahead of
   ModernBERT for the first time on real data):
   ```bash
   uv run python -m experiments.fast_curation.launch_tpu --spec lpv11_fastpipe_v1 \
     --region us-east5 --bucket gs://marin-us-east5 --mode v2b --num-workers 1 \
     --limit 2 --priority interactive
   ```
   Confirm in `timing_b/data-*.json`: `n_pooled_pass` < `n_presurvivors` (the cull) and `pooled_s`
   is a small fraction of `score_s`.
2. **Smoke Phase C** (resiliparse-rs) after B writes a keeplist — see `launch_cpu_c.py`.
3. **Fix the uncapped-HTML risk before any real run** — see below.

## KNOWN RISK, not yet fixed

`lpv11_fastpipe_v1` sets `justext_max_html_chars = 50_000_000`, and Phase C's Rust path reuses that
field as its size cap — i.e. effectively no cap. Multi-MB pages then get pickled into pool workers.
This is the exact failure that killed `score_resiliparse_rs` twice (`BrokenProcessPool`, which looks
like a crash but was memory); the fix there was capping at **2M chars in the parent** before dispatch.
Recommended: add a dedicated `resiliparse_rs_max_html_chars: int | None = None` to `PipelineSpec`
(None-default keeps every existing hash stable) and set it to 2_000_000. Doing this BEFORE any real
lpv11 data is produced is free; afterwards it forces a new namespace.

Two smaller gaps the Phase C agent flagged:
- No `resiliparse_rs_artifact_for(bucket)` resolver in `spec.py`; Phase C takes a
  `--resiliparse-artifact` flag instead. The artifact is currently us-east5-only, so multi-region
  Phase C needs it mirrored + the flag passed.
- No per-doc timeout on the Rust engine (a hang, as opposed to a segfault, would wedge a worker).
  Not observed over 100k docs.

## State that is DONE and verified

- `lpv11_fastpipe_v1` spec, hash **`2224e3e476`**: fastText lpv11 w640 (0.130) → pooled lpv11 10M
  (0.178) → ModernBERT lpv11 10M (0.410) → resiliparse-rs @ `850891b`.
- Phase B runs pooled before ModernBERT; pooled-dropped docs carry NaN `modernbert_prob` so Phase C's
  existing `>= threshold` filter excludes them with no special-casing.
- Phase C dispatches on `spec.extraction_engine`; the jusText path is byte-unchanged (fastpipe_v3 is
  LIVE with ~3,602 WARCs). 37 tests pass.
- All 3 lpv11 models mirrored to all 5 regions and verified through the spec's own resolvers.
- **Hash-stability bug fixed**: `_namespace_fields()` now omits None fields, so adding an optional
  field can't orphan a live corpus. `test_spec.py` pins `fastpipe_v3=da3893385e` and
  `lpv11_fastpipe_v1=2224e3e476`.
- `VERSIONS.md` records that v3's documented hash `6855733850` is WRONG — the real corpus
  (~3,602 WARCs) is under `da3893385e`; `6855733850` holds an abandoned 82-WARC stub.

## Uncommitted

Everything above is uncommitted in the working tree (branch `multi-spec-extraction`). New files:
`experiments/fast_curation/{test_spec.py,test_resiliparse_c.py}`,
`experiments/baseline_collection/{build_lpv11_target_columns.py,score_pooled_useful.py,score_resiliparse_rs.py,build_resiliparse_rs.py,lpv11_missing_5_warcs.txt}`.
