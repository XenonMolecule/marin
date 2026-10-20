# System-prompt pretraining: more epochs and the 1B data-efficient run

Two follow-ups to the data-efficient 447M experiment
(`SYSPROMPT30B_DATA_EFFICIENT_447M.md`), from items 4 and 5 of the remaining-work list in
`docs/reports/trained_model_evaluation_2026_08.md` (jlcmoore/system-prompt-pretraining):

1. **Epochs.** Train the 447M doc-matched, mix50 and sysprompt arms for 2, 4 and 8 passes over
   the same 2.8M documents, with the optimizer re-derived for each horizon.
2. **1B.** Train the same three arms at 998M parameters on about 8B tokens, which keeps roughly
   eight tokens per parameter, with three training seeds each.

**Result.** Neither more epochs nor a larger model lets a prompted arm beat the doc-matched
control on prompt-free evaluation. Mix50 converges to the control: its gap shrinks from +0.003
bpb at one epoch to within noise at 2, 4 and 8 epochs, and at 998M its three-seed mean is
+0.0005 behind, well inside seed variation. Sysprompt stays 0.025 to 0.029 behind at both sizes
through 4 epochs; at 8 epochs it stops improving while the control keeps going, so its gap widens
to 0.044.

All 18 runs finished and were verified (completion marker and final-step HuggingFace export).
Prompted evaluation of these checkpoints has not been run; see Caveats.

## Setup common to both

Same corpus, prompts, prompt wrapper, tokenizer, architecture family and evaluation sets as the
data-efficient 447M models. Evaluation is bits per byte with no prompt at eval time, on the
standard held-out sets: `eval/bpb` is the headline, lima, paloma and uncheatable are reported
alongside. Lower is better. Prompted evaluation (blank, matched and random prompt on the frozen
200-document holdout and the paired cohorts) runs in the project's own scoring pipeline on SC.

| arm | directory name | training data |
|---|---|---|
| C | `doc-matched` | the documents with no prompts (control) |
| D | `mix50` | a prompt on a fixed half of the documents, chosen by hashing the document id |
| A | `sysprompt` | every document with its prompt |

All optimizer settings come from the delphi AdamH scaling heuristic for the model width and the
number of tokens processed. None were tuned for these runs: the data-efficient 447M sweep found
the heuristic's learning rate optimal for every arm. Fixed throughout: beta1 0.9, 10% warmup,
linear decay over the final 20%, gradient clip 0.1, z-loss 1e-7, sequence length 4096. All runs
trained on a single v5p-8 in us-central1.

## Epochs at 447M

Model: 447.2M parameters (hidden 1024, 11 layers). Data: the 2,812,816 documents of the
data-efficient 447M set; C has 3.600B tokens per pass, D 3.629B and A 3.659B.

Levanter shuffles a dataset once and replays the same order on later passes. These runs give
each pass its own shuffle: every epoch is a separate mixture component over the same cache,
shuffled with its own key, and a step schedule switches components at each epoch boundary.
Boundaries are aligned to Levanter's 2,048-sequence mixture blocks, so each pass drops at most
one block, about 0.2% of the documents, chosen at random per pass.

The heuristic keeps about 56k steps and scales the batch with the token horizon:

| epochs | tokens processed (C) | batch | steps (C) | lr | adam_lr | beta2 |
|---|---|---|---|---|---|---|
| 1 | 3.60B | 16 | 54,931 | 2.81e-3 | 2.71e-4 | 0.9999 |
| 2 | 7.20B | 32 | 54,912 | 3.22e-3 | 2.71e-4 | 0.9999 |
| 4 | 14.39B | 64 | 54,912 | 3.70e-3 | 2.71e-4 | 0.9999 |
| 8 | 28.79B | 128 | 54,912 | 4.26e-3 | 2.71e-4 | 0.9998 |

The 1-epoch row is the heuristic default point of the data-efficient sweep, not each arm's
selected checkpoint, so all four horizons use the same rule. Training compute is about
6 x 447M x tokens: 9.7e18, 1.9e19, 3.9e19 and 7.7e19 FLOPs for C at 1, 2, 4 and 8 epochs.

### Results

Bits per byte with no prompt at eval time; lower is better. One training seed per cell. The
1-epoch row is the data-efficient sweep's batch-16 default-learning-rate run for each arm.

| epochs | C doc-matched | D mix50 | A sysprompt | D − C | A − C |
|---|---|---|---|---|---|
| 1 | **1.1021** | 1.1051 | 1.1297 | +0.0030 | +0.0276 |
| 2 | **1.0660** | 1.0669 | 1.0910 | +0.0009 | +0.0250 |
| 4 | **1.0414** | 1.0417 | 1.0683 | +0.0003 | +0.0269 |
| 8 | **1.0256** | 1.0260 | 1.0696 | +0.0004 | +0.0440 |

Per eval set (lima / paloma / uncheatable):

| epochs | C | D | A |
|---|---|---|---|
| 1 | 0.9489 / 1.1062 / 1.0820 | 0.9520 / 1.1090 / 1.0874 | 0.9650 / 1.1353 / 1.0979 |
| 2 | 0.9187 / 1.0710 / 1.0371 | 0.9202 / 1.0719 / 1.0385 | 0.9365 / 1.0970 / 1.0547 |
| 4 | 0.8994 / 1.0471 / 1.0097 | 0.8991 / 1.0474 / 1.0100 | 0.9203 / 1.0755 / 1.0254 |
| 8 | 0.8883 / 1.0326 / 0.9959 | 0.8887 / 1.0332 / 0.9958 | 0.9176 / 1.0800 / 1.0235 |

- **Repetition keeps helping the control and mix50, with diminishing returns.** The control
  gains 0.036, 0.025 and 0.016 bpb from each doubling. Two passes over 3.6B tokens (1.0660) come
  close to one pass over the 447M ladder's 7.2B distinct tokens (1.0605).
- **Mix50 converges to the control and does not cross it.** The headline gap is under 0.001 from 2
  epochs on. At 2 epochs mix50 still trails by about 0.0015 on lima and uncheatable; at 4 and 8
  epochs every eval set differs by under 0.0005, in both directions, which is noise.
- **Sysprompt saturates.** Its headline bpb is slightly worse at 8 epochs than at 4 (1.0696 against
  1.0683), driven by paloma (+0.0045); lima and uncheatable still improve a little. Fully prompted
  training appears to start overfitting between 4 and 8 passes while the bare-text arms do not.

Processed tokens and training compute (6 × 447M × tokens processed):

| epochs | C tokens | D tokens | A tokens | compute (C) |
|---|---|---|---|---|
| 1 | 3.60B | 3.63B | 3.66B | 9.7e18 FLOPs |
| 2 | 7.20B | 7.25B | 7.31B | 1.9e19 FLOPs |
| 4 | 14.39B | 14.50B | 14.63B | 3.9e19 FLOPs |
| 8 | 28.79B | 28.99B | 29.26B | 7.7e19 FLOPs |


## 1B data-efficient run

Model: 998.0M parameters (hidden 1536, 16 layers, 12 heads, MLP 6144). Data: 6,202,688
documents, a prefix of the 998M ladder's document set; C has 7.776B tokens, D 7.841B and A 7.906B,
one epoch. That is 7.8 to 7.9 tokens per parameter, against about 15 for the 998M ladder models.

Optimizer (delphi cell 4.8e19 FLOPs): batch 32, about 60k steps (C 59,324, D 59,820, A 60,315),
lr 3.15e-3, adam_lr 2.60e-4, beta2 0.9999, epsilon 4.67e-8. Each arm was trained with seeds 0, 1
and 2. The seed sets model initialization and the data-shuffle key, so arms with the same seed
start from identical weights.

### Results

| arm | seed 0 | seed 1 | seed 2 | mean | seed SD | mean − C |
|---|---|---|---|---|---|---|
| C doc-matched | 1.0038 | 1.0068 | 1.0035 | **1.0047** | 0.0018 | |
| D mix50 | 1.0062 | 1.0042 | 1.0052 | 1.0052 | 0.0010 | +0.0005 |
| A sysprompt | 1.0372 | 1.0361 | 1.0278 | 1.0337 | 0.0051 | +0.0290 |

Per eval set, three-seed means (lima / paloma / uncheatable): C 0.8662 / 1.0110 / 0.9636;
D 0.8667 / 1.0114 / 0.9646; A 0.8875 / 1.0416 / 0.9793.

- **Mix50 and the control are indistinguishable.** Mix50 wins seed 1 and loses seeds 0 and 2. The
  mean difference, 0.0005, is a third of the control's own seed SD.
- **Sysprompt trails by 0.029**, about the same as at 447M one epoch (0.028) and smaller
  than the 0.039 at the 998M ladder's compute-optimal point. Seed 2 is notably better than seeds
  0 and 1 on every eval set, so sysprompt also has the largest seed variance.
- **Seeds matter at this scale.** The control moved 0.003 between seeds, the same size as the
  single-seed mix50 gaps reported for the 447M sweep. One-seed differences under about 0.004 should
  not be read as effects.

For reference, the 998M ladder models trained on 14.6B tokens score 0.9716 (C), 0.9717 (D) and
1.0102 (A). Training compute here is 6 × 998M × 7.78B ≈ 4.7e19 FLOPs for C.

## Caveats

- **Bare evaluation only.** These numbers measure prompt-free document modeling. The gains from a
  matched prompt, and the paired steering evaluations, come from scoring the checkpoints in the
  project pipeline on SC, which has not been run on them yet.
- **Batch size changes with epochs.** The heuristic holds steps near 56k and scales the batch with the
  token horizon, so batch and epoch count move together. That is the intended "learning-rate
  schedule for each horizon", but the epoch effect is not separated from the batch effect.
- **One seed for the epoch runs.** Given the 998M seed spread, the sub-0.001 mix50 differences at
  2 to 8 epochs are ties, not trends.
- **Interruptions.** The 998M seed-2 runs were evicted at about step 16k and the 8-epoch runs at
  about 50k; all resumed from checkpoints. The seed-0 and seed-1 998M result files were rewritten
  with identical values hours after those runs finished, consistent with a restarted task
  re-running its final evaluation.

## Where things are

Checkpoints: `gs://marin-us-central1/checkpoints/isoflop-curation/<run>/hf/step-<final>/`, with
run names `curation-sysprompt30b_447m_9e18_{A,C,D}_ep{2,4,8}-expFM_natural-{2,4,7}e+19-d1024-L11-B{32,64,128}`
and `curation-sysprompt30b_998m_4p8e19_{A,C,D}-expFM_natural-5e+19-d1536-L16-B32[-s1|-s2]`.
Result JSONs: `gs://marin-us-central1/metadata/sysprompt30b_{447m_9e18,998m_4p8e19}_results/`.
`pull_sysprompt30b_epochs_1b_to_sc.sh` copies all 18 to SC under
`dclm_sysprompts/models/{epochs_447m,data_efficient_998m}/` with md5 checks.

A reviewable sample of the training prompts with document excerpts is in
`dclm_sysprompts/models/examples/examples.md` on SC (full records in `examples.jsonl`), made by
`sample_sysprompt30b_examples.py` with seed 0.

