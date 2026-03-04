# Job Monitoring Loop

Monitor a Ray job, automatically restarting on failure.

## Before Starting

When the user asks you to start a monitoring loop, gather the required information first:

1. **job_id** - What is the Ray job ID? (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. **cluster** - Which cluster is it running on? (e.g., `us-east5-a`, `us-central2`)
3. **experiment** - What is the experiment script path? (e.g., `experiments/isoflop_sweep.py`)

Ask the user for any missing information before proceeding. Example:

> "I need a few details to start the monitoring loop:
> - Job ID?
> - Cluster?
> - Experiment script path?"

Once you have all three, write the state file and begin the loop.

## State File

Write to a local file (e.g., `monitoring_state.json` in the scratchpad):

```json
{
  "job_id": "<JOB_ID>",
  "cluster": "<CLUSTER>",
  "experiment": "<EXPERIMENT_PATH>",
  "restart_count": 0
}
```

## Loop

```
1. SLEEP
   sleep 570

2. CHECK
   ./scripts/ray/check_logs.sh <CLUSTER> <SUBMISSION_ID> [lines] [pattern]
   # Presets: "inference" (zephyr pipeline), "training" (loss/eval/steps),
   #          "errors", "all" (default), "raw" (unfiltered tail).
   # Lines: number for last N, or "0"/"all" for unlimited.
   # Or pass a custom regex as the pattern arg.

3. EVALUATE — be conservative, most issues are transient
   - If output contains "loss" lines → go to step 1 (HEALTHY)
   - If output contains errors OR no output → go to step 3.1 (VERIFY)

   3.1 VERIFY STATUS — before any restart, confirm the job is actually FAILED:
       list-jobs output is too large for stdout. Write to file and parse with python3:

       uv run scripts/ray/cluster.py --cluster <CLUSTER> list-jobs 2>/dev/null > /tmp/ray_jobs_check.json
       .venv/bin/python3 -c "
       import json
       with open('/tmp/ray_jobs_check.json') as f: data = json.load(f)
       for j in data:
           sid = j.get('submission_id','')
           if '<EXPERIMENT_KEYWORD>' in sid:
               print(j['status'], sid)
       "

       - If job status is RUNNING → go to step 1 (WAIT — empty logs can mean flaky
         dashboard or head node rotation, not a dead job)
       - If job status is FAILED → go to step 4 (RESTART)
       - If job is not found in the list → go to step 4 (RESTART)

4. STOP OLD JOB (CRITICAL - always do this before restarting)
   uv run scripts/ray/cluster.py --cluster <CLUSTER> stop-job <JOB_ID>

5. CHECK FOR DUPLICATES (before submitting a new job)
   Parse /tmp/ray_jobs_check.json for other RUNNING jobs matching the same experiment.
   If one exists → update state file to track that job instead, go to step 1.

6. RESTART
   uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> -- python <EXPERIMENT_PATH>

   - Capture new job_id from output
   - Update state file: job_id = <NEW_JOB_ID>, restart_count += 1
   - Go to step 1
```

## Fixing Small Bugs

When step 3 (EVALUATE) detects an error, before restarting:

1. **Analyze the error** in the logs:
   - Look for `Traceback`, `Error`, `Exception`
   - Identify the file and line number

2. **If it's a small fix** (typo, missing import, wrong variable name):
   - Read the relevant file
   - Make the fix with Edit tool
   - Proceed to step 4 (RESTART)

3. **If it's a complex issue** (architectural, unclear cause, requires investigation):
   - Do NOT attempt to fix automatically
   - Report to user and exit the loop

Examples of small fixes:
- `NameError: name 'foo' is not defined` → typo in variable name
- `ImportError: cannot import 'bar'` → missing or misspelled import
- `SyntaxError` → missing comma, bracket, colon
- `KeyError` → wrong dict key name (if obvious from context)

Examples of complex issues (do not auto-fix):
- OOM errors
- Distributed training failures
- Data loading issues
- Unclear stack traces spanning multiple files

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout)
- The loop is controlled at the agent level, not bash
- Track restart_count to detect flapping jobs
- State file allows resuming if context resets
- If the same error occurs after a fix attempt, do not retry - report to user
- Empty logs ≠ dead job — dashboard tunnels drop, head nodes rotate. Always verify with list-jobs.
- Duplicates waste compute and can corrupt checkpoints. Always stop old job + check for duplicates before submitting.
- Infra errors (OOM, node death, GCS errors, PENDING_NODE_ASSIGNMENT) are usually self-healing. Default to waiting.
- list-jobs output is huge — always redirect to file and parse with python3, never pipe directly.
- list-jobs may not return all jobs (API pagination/truncation). "Job not found" does NOT mean the job died — check `job-logs` for recent activity before restarting.
