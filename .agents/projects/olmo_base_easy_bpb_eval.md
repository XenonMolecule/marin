# OLMo Base Easy (bpb) eval — offline over the 10k+fastpipe manifest, in-loop optional

Goal: run AI2's **OlmoBaseEval "Base Easy"** bits-per-byte (bpb) perplexity tasks over the
isoflop-curation checkpoints (the `_10k` + `fastpipe_v3_*` fixed-model sweep), offline, region-pinned
— mirroring the existing `olmes_base` accuracy sweep. Keep the data format tokenizer-agnostic so the
same GCS assets also feed a Levanter in-loop callback later.

## Dashboard integration + AI2 aggregate alignment (2026-07-04)

- Wired into `plot_subcomponents_dashboard.py` (join olmo bpb onto 10k summaries by run_name).
  Two tabs: **"OLMo bpb — summary (3)"** (just code_bpb/math_bpb/qa_bpb macro cards) and
  **"OLMo bpb — all tasks (56)"** (per-task cards w/ descriptions + sample docs).
- **AI2 aggregate alignment** to `olmes/oe_eval/configs/task_suites.py` `olmo3:base_easy:*`
  (each a `macro`; inner-macro members counted once): `code_bpb` = macro(humaneval 3shot, mbpp
  3shot, mt_mbpp-over-17-langs); `math_bpb` = macro(7 minerva subjects); `qa_bpb` = AI2's 21-task
  suite **minus MMLU** (absent from the in-loop bundle) — arc & basic_skills as inner macros.
  TWO documented deviations from AI2 qa_bpb: MMLU omitted; coqa/drop/jeopardy/naturalqs/squad use
  generation bpb not AI2's gen2mc. Exact `qa_bpb` needs the olmes/oe-eval harness (mmlu + gen2mc).
- **QA rc:bpb second sweep** (2026-07-04): the 20 MC-QA rc tasks (`QA_RC_MC_BPB`) at OLMES shots,
  run via `--tasks qa_rc --merge` (runner `--merge` folds new tasks into existing results.json;
  `--done-marker _qa_rc.done` so the launcher keep-alive works when results.json pre-exists).
  339/339 merged → each results.json now has 56 tasks. GOTCHA fixed: keep-alive on results.json
  existence orphan-killed all children in merge mode (files pre-exist) → done-marker. us-central1
  tail again stuck on external v5p saturation (hardware wall, interactive priority didn't help);
  drained on its own. Result: high_quality wins code_bpb (+12%), ~even math, DCLM wins qa_bpb.

## Status (2026-07-03)

- [x] **Data staged in GCS** — `eval_datasets/olmo_in_loop_evals/` in every region bucket
      (us-east5, eu-west4, us-central1, us-central2, us-east1). Source:
      `allenai/OLMo-in-loop-evals` @ `a0366ed14d7c90547243cbe4c28f5b7d0f291736`. Contents:
      `oe_eval_tasks/` (116 variants, config.json + requests.jsonl.gz), `hf_datasets/`, `tokenizers/`,
      `PROVENANCE.md`. ~110 MB/region. Reads stay in-region (no cross-region egress).
- [x] **Checkpoint manifest refreshed** — `experiments/core_eval_manifests/checkpoint_manifest_10k_with_fastpipe_341.txt`
      (339 distinct checkpoints, fastpipe 30→91). Built from the authoritative completion registry
      `gs://marin-us-central1/metadata/data_curation_10k_natural_results/*.json` (one summary per
      completed run, carries `run.region` + `run.output_path`). Strict superset of the old `_275`.
- [ ] **Pin the exact Base Easy task subset** — the bundle has all 116 variants; OlmoBaseEval "Base
      Easy" is a named subset (code_bpb, math_bpb, qa_bpb, qa_rc, …). Extract the canonical list from
      the `olmes`/OlmoBaseEval suite config or the Olmo 3 report appendix before finalizing.
- [x] **Offline runner built + smoke GREEN** — `experiments/scaling_law_sweeps/olmo_bpb/`
      (`run_olmo_bpb_eval.py`, `launch_olmo_bpb_manifest.py`, `olmo_bpb_tasks_set.py`). Smoke on
      `dclm_10k 1e17 d512-L6` (v5p-8 us-east5, lambada+gsm8k, --limit 8) →
      lambada bpb=1.64, gsm8k bpb=1.36, macro=1.50. Sane values; pipeline works end-to-end.
      **GOTCHA (fixed):** the harness dispatch (`broadcast_shard`) needs the JAX device mesh ACTIVE
      — model load + loglikelihood + worker.stop() must ALL run inside `trainer_config.use_device_mesh()`.
      Returning the worker out of the `with` block → `current_mesh is None` AttributeError.
- [x] **Full 339-way offline launch — COMPLETE (2026-07-04): 339/339, 0 errors.** Parent
      iris-run-job-20260704-072823 (interactive us-central1) over `_341` manifest, 36 bpb tasks each,
      full (no --limit). Results → `gs://<bucket>/metadata/olmo_bpb_results/<run_name>/results.json`
      in each region. Caches pre-staged to all 5 regions → zero cross-region egress. Every result a
      full 36-task run, no per-task errors. Notes: parent got preempted once (~08:02) but self-healed
      via `--skip-existing` re-run; long tail was us-central1 v5p saturation from another user
      (benjaminfeuer/tracegen) — children auto-retried (20 preemption retries); closed the final
      straggler with a one-off interactive job.
- [x] **Wired into the subcomponents dashboard** (2026-07-04) —
      `plot_subcomponents_dashboard.py` now joins bpb onto the 10k summaries by `run_name` and adds an
      "OLMo bpb (36)" tab (4 category macros: all/code/math/qa-lang + 36 per-task loss-vs-tokens
      figures, same winner analysis as loss subcomponents; bpb is lower-is-better). Pull results with
      `--pull-olmo` (mirrors the 5 region buckets → `scratch/olmo_bpb_results/`, ~2.7MB), else reads
      local. Join is 241/241 summaries. Early signal: high_quality wins code bpb (+6–13%) but loses
      lambada/jeopardy bpb to DCLM (−23%/−17%). Output: `scratch/plots/subcomponents_10k/subcomponents_dashboard.html`.
- [ ] **(optional) standalone consolidator** — a tidy cross-bucket table
      (clone `olmes_base/consolidate_olmes_results.py`) if a non-dashboard summary is wanted.
- [ ] **In-loop hook** (Levanter callback) — optional, later.

## Data format (what the requests carry)

`oe_eval_tasks/<task>/<variant>/`:
- `config.json` — `task_config.primary_metric == "bits_per_byte"`, split, num_shots, `compute_gold_bpb`.
- `requests.jsonl.gz` — one JSON/line. For bpb tasks: `request_type` includes loglikelihood; `doc`
  holds the raw **context** and the **gold continuation** (text, not tokenized). Tokenizer-agnostic.

bpb per doc = `-sum(logprob of gold-continuation tokens) / continuation_byte_len * log2(e)`.
Per task: mean over docs. (Ref: `metrics.py` in the source repo; `LOG_2_OF_E` factor.)

## Offline runner design (mirror `olmes_base`)

`experiments/scaling_law_sweeps/olmo_bpb/` — clone the `olmes_base` trio:

1. `run_olmo_bpb_eval.py` — one HF checkpoint, offline, on TPU.
   - Sync `eval_datasets/olmo_in_loop_evals/` (this region) to local before any HF import
     (reuse `_sync_cache` from `core_tasks/run_core_tasks_eval.py`).
   - For each task variant in the Base Easy set: read `requests.jsonl.gz`, build Levanter
     loglikelihood requests `(context, gold_continuation)`, run the model's logprobs
     (Levanter's harness already has a loglikelihood path — the same one OLMES/CORE use),
     compute bpb, aggregate.
   - Write `results.json` → `gs://<bucket>/metadata/olmo_bpb_results/<run_name>/results.json`
     (per-task bpb + macro summary), in-region.
   - Model-config hub cache: reuse `eval_datasets/core_tasks_hub_cache/` (identical checkpoints).

2. `launch_olmo_bpb_manifest.py` — clone `launch_olmes_manifest.py` almost verbatim:
   - reads `checkpoint_manifest_10k_with_fastpipe_341.txt` (run_name, region, output_path, hf_dir);
   - one iris child per checkpoint, HARD-pinned to the checkpoint's region;
   - points `--dataset-cache-gcs` at `eval_datasets/olmo_in_loop_evals/` in that region;
   - skips rows whose `hf/step-*` is missing (`_resolve_final_step`), like the OLMES launcher.

3. `olmo_bpb_tasks_set.py` — the pinned Base Easy variant list (task/variant dir names under
   `oe_eval_tasks/`), analogous to `OLMES_BASE_EASY_RUNNABLE`.

Consolidation: clone `consolidate_olmes_results.py` → aggregate
`metadata/olmo_bpb_results/*/results.json` across region buckets into a summary for isoflop plots.

## Key decisions / open questions

- **bpb via Levanter loglikelihood vs. `ai2-olmo-eval` runner.** Prefer Levanter (TPU, offline,
  region-pinned, matches CORE/OLMES). `ai2-olmo-eval`'s own runner is torch/GPU — avoid (no-GPU rule).
  Need to confirm Levanter's harness returns per-request summed continuation logprob + we can get the
  continuation byte length from the request doc.
- **Base Easy subset** — see open item above; until pinned, `oe_eval_tasks/*/gold_bpb_*` +
  `*/bpb_*` variants are the perplexity core.
- **In-loop** — same staged data + same bpb fn wrapped in a Levanter eval callback; log
  `olmo_bpb/<task>` to wandb at eval intervals. Deferred; the tokenizer-agnostic staging already
  supports it.

## Provenance / reproduce

- Regenerate manifest: download `data_curation_10k_natural_results/*.json`, emit
  `run_name,region,output_path,hf_dir` from each summary's `run` block (run_name = basename(output_path)).
- Re-stage data: clone the repo, `gcloud storage cp -r src/olmo_eval/{oe_eval_tasks,hf_datasets,tokenizers}`
  into `eval_datasets/olmo_in_loop_evals/` per region bucket (zsh: use an array for the bucket loop —
  unquoted `$var` does NOT word-split in zsh).
