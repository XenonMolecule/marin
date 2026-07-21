# N=100 WARC-scaling models — DCLM Core v2 evals (overnight 2026-07-15)

**Status:** LAUNCHED + MONITORING. Owner: Michael Ryan (asleep; running autonomously).

## Task (per Michael)

Run DCLM Core v2 evals on ALL finished **N=100** WARC-scaling models. Scope locked:
- **N=100 ONLY** — NOT 500/1000/2000 ("Don't evaluate all 200/500/1k/2k").
- Prioritize **DCLM, Nemotron, HQ**; other extractors (resiliparse, llm_curated, …)
  are a follow-up "once those land".
- "if it is less than 100s that is my mistake" — the real count is 67, that's fine.

## What was found (all regions scanned; models FLOATED across regions)

67 finished models (DONE + verified `hf/` export), by method:
- `dclm_100`: 11
- `nemotron_full_100`: 12
- `nemotron_qhigh_100`: 11   (Nemotron-CC-HQ; counts as Nemotron)
- `high_quality_100`: 33     (HQ has the biggest N=100 grid)

Manifest: `experiments/core_eval_manifests/warc100_dclm_nemo_hq.txt` (.txt because
`*.csv` is gitignored → wouldn't bundle). Format run_name,region,output_path,hf_dir.
Region spread: us-central1 25, us-east5 24, eu-west4 13, us-east1 4, us-central2 1.
Builder script: scratchpad/build_warc100_manifest.sh (dedups run_name→region, checks hf/).

## Launch

Coordinator `warc100-core-eval` (iris, us-central1, interactive keep-alive) →
`launch_10k_manifest.py --manifest warc100_dclm_nemo_hq.txt --launch
--child-priority batch --wave-size 40`. Each child region-pins to its checkpoint's
region (eval cache `dclm_core_hf_cache/` confirmed present in ALL 6 regions → no Hub
calls, no cross-region reads). `--skip-existing` idempotent. Wave 1 = 40 submitted,
wave 2 (~27 more) after ~330s throttle.

Results land per-region at `gs://marin-<region>/metadata/data_curation_10k_core_results/
{run_name}_summary.json`; Core v2 score = `.dclm.Core_v2`.

## Monitoring

Monitor task `bcrl794v3` (persistent): counts warc100 Core-v2 summaries across ALL 6
regions, target 67; emits at 20/40/60 + completion, stall (3h), coordinator death,
systematic child failures (>=8 at once). Separate from the decon monitor `b3lys10l6`.

## CHECKPOINT RE-HOME to unblock stuck evals (2026-07-15, Michael OK'd if ~$6)

Evals stalled at 29/67: the 29 scored are exactly us-east5(24)+us-east1(4)+us-central2(1)
(regions with capacity); the 38 un-scored are ALL of us-central1(25)+eu-west4(13) — jammed
regions. VERIFIED cost to move those 38 (15.3GB) to the now-free us-east5:
**$0.45 realistic** (us-central1 10.4GB@$0.02 + eu-west4 4.8GB@$0.05) / $1.83 @ conservative
$0.12/GB flat — well under $6 and the $10 cap. Plain `gcloud storage cp` (NOT transfer svc).

Steps: rehome_warc100.sh rsyncs the 38 run dirs → us-east5 + writes re-homed manifest
`warc100_dclm_nemo_hq_eastpin.txt` (38 rows repinned us-east5, 29 unchanged). Then stop old
`warc100-core-eval` coordinator+children (kills 23 stale pending), relaunch over eastpin
manifest (skip-existing skips the 29 done → 38 run in free us-east5). Monitor unchanged
(counts summaries across ALL regions → 67).

## INCIDENT + FIX (2026-07-15): bunched relaunch failed, staggered fix

After re-homing 38 to us-east5 I relaunched with --wave-size 40 (all 38 cold-starting
at ONCE). Result: 15 hard-failed + 23 stuck "running" 1.5h with 0/38 completing (count
frozen at 29). ROOT CAUSE = bunched cold-starts hit the HF-Hub-rate-limit/preemption
wave the memory warns about ("Rate-limit hard = TPU-capacity bunching; let retries ride").
VERIFIED NOT a data problem: moved checkpoints are byte-identical to native us-east5 ones
that scored (safetensors + full hf/step file tree identical); manifest paths resolve fine.
A --wave-size 8 retry submitted 0 (transient coordinator GCS-read glitch; finelog was down).

FIX: stopped `warc100-core-eval-e5` (+children), relaunched `warc100-eval-staggered` over
the SAME eastpin manifest with --wave-size 5 --wave-delay 240 --name-suffix s1 (skip-existing
skips the 29 done). First wave = 5 submitted, all 5 RUNNING, 0 fail → staggering works.
~8 waves over ~30min submit the 38. Monitor `b0eal3i0i` tracks scored→66.

LESSON: for these Core evals ALWAYS use small waves (<=5-8), never wave-size 40. Finelog
was down throughout (logs unavailable) — used `iris job bug-report` + GCS integrity checks.

## To finish (when evals land)

Gather the 67 summaries (`.dclm.Core_v2`) across regions → per-method Core-vs-FLOPs
tables/curves for dclm_100 / nemotron_full_100 / nemotron_qhigh_100 / high_quality_100.
Then (follow-up) add other extractors' N=100 models.

## Note

Iris `job list` bare form is broken cluster-wide (`>5000 jobs`, MAX_LIST_JOBS_OFFSET);
ALWAYS use `--prefix /michaelryan/<job>`. Finelog (job logs) intermittently down →
use `iris job bug-report` for failure reasons.
