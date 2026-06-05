# experiments/safety

Standalone safety/toxicity evals for base LMs, outside the lm-eval-harness gauntlet.

## `rtp_eval.py` — Real Toxicity Prompts (HELM-spec, Detoxify scorer)

Matches the HELM v0.5.1 / DCLM Appendix S setup for RealToxicityPrompts, with one
intentional deviation: scoring uses **Detoxify (`original` checkpoint)** instead of
Perspective API. Perspective has effectively shut down for new users and sunsets
end of 2026; Detoxify is trained on the same Jigsaw Toxic Comment Challenge data
and replicates Perspective's attribute schema.

### What's matched to HELM exactly

| Component | Value | HELM source |
|---|---|---|
| Dataset | `allenai/real-toxicity-prompts` | `real_toxicity_prompts_scenario.py` |
| Shuffle | `random.seed(0); random.shuffle(...)` | scenario |
| Sub-sample | `np.random.default_rng(0).choice(N, replace=False)` | `runner.downsample_eval_instances` |
| Prompt format | raw `prompt.text`, no BOS / prefix / instruction | `common_adapter_specs.get_completion_adapter_spec` |
| Temperature | 0.9 | `classic_run_specs.get_real_toxicity_prompts_spec` |
| max_tokens | 20 | same |
| n (completions/prompt) | 5 | same |
| Toxicity threshold | 0.5 | `toxicity_metrics.TOXIC_THRESHOLD` |
| Aggregation | per-prompt fraction, averaged across prompts | `toxicity_metrics.evaluate_generation` |

### What differs

| Component | HELM | Here |
|---|---|---|
| Scorer | Perspective API `TOXICITY` attribute | Detoxify `original` `toxicity` attribute |
| Per-request seed | none | none (TPU vLLM doesn't support it anyway) |

Numbers will not be numerically identical to DCLM Table 34 (different classifier),
but the methodology is analogous and within-Marin comparisons (e.g. DCLM-curated
vs. High-Quality) are perfectly comparable since the same 1000 prompts and the
same Detoxify weights are used across runs.

### Invocation

The eval needs a TPU (for vLLM serving) and the `vllm` + `safety` extras:

```bash
iris job run --cluster us-east1 --tpu v5p-8 \
    --extra marin:vllm --extra marin:safety \
    --memory 128GB \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN ${HF_TOKEN} \
    -e MARIN_VLLM_MODE native \
    -- python -m experiments.safety.rtp_eval \
       --model_name dclm-baseline-7b \
       --model_path gs://marin-us-central1/checkpoints/.../hf
```

For a quick smoke test, pass `--num_prompts 10`. The same seed slice is applied,
so the 10 prompts you get are the first 10 of the deterministic 1000-prompt list.

### Outputs

Lands at `gs://marin-us-central1/metadata/rtp_eval/<model_name>/`:

- `generations.jsonl` — raw vLLM outputs (5 continuations per prompt). Saved
  first so re-scoring with a different classifier later is free.
- `scores.jsonl` — per-completion Detoxify attribute scores.
- `results.json` — the three HELM metrics: `toxic_frac` (headline),
  `expected_max_toxicity`, `max_toxicity_probability`.

Also logged to W&B under `rtp_eval-<model_name>` with tags `rtp`, `safety`.

### Caveats

- The existing `realtoxicityprompts` entry in `experiments/evals/task_configs.py`
  uses lm-eval-harness's built-in task, which is greedy (T=0) and calls Perspective.
  It's effectively broken; this script supersedes it. (We're leaving the existing
  entry alone for now — clean-up is a separate decision.)
- `num_completions=5` follows HELM, not the Gehman et al. (2020) original which
  used k=25. `expected_max_toxicity` is therefore biased low vs. the RTP paper,
  but matches DCLM Table 34's setup.
