# MoralChoice steerability: system-prompt-pretrained vs plain DCLM

Comparison of two 447M models (same arch `d1024/L11`, same `2e+19` budget, same MoralChoice
loglik eval) that differ only in **pretraining data**:

| | run | step | checkpoint region | results |
|---|---|---|---|---|
| **DCLM** | `curation-dclm_10k-expFM_natural-2e+19-d1024-L11-B32` | 56002 | eu-west4 | `gs://marin-eu-west4/metadata/moralchoice_evals/dclm10k-2e19-step56002/` |
| **sysprompt** | `curation-sysprompt_dclm-expFM_natural-2e+19-d1024-L11-B32` | 59647 | us-east5 | `gs://marin-us-east5/metadata/moralchoice_evals/sysprompt-dclm10k-2e19-step59647/` |

The sysprompt model was pretrained on `[S][D]` data: a generated system prompt `S` per DCLM
document, rendered as `<|start_header_id|>system<|end_header_id|>\n\n{S}<|eot_id|>{D}`
(see `experiments/scaling_law_sweeps/tokenize_sysprompt_dclm.py`).

> ⚠️ **Major caveat up front (read before believing anything below).** Our MoralChoice steering
> preamble is a *plain prepended string*. The sysprompt model was trained to read instructions in
> the `<|start_header_id|>system<|end_header_id|>\n\n…<|eot_id|>` channel. **We are almost certainly
> understating its steerability** by not using its native format. Every number below is a lower
> bound on what native-format steering would show. See "Dig deeper" #1.

## Result 1 (robust): the `ab` channel goes from dead to alive

This is the part that is *not* a small effect — it's a structural change.

| neutral metric, high split | DCLM | sysprompt |
|---|---|---|
| `ab` first-position-bias | **1.000** (always picks "A") | **0.500** (no positional preference) |
| `ab` consistency (order-invariance) | 0.000 | 0.226 |
| `ab` mean \|steerability Δ\| | **0.000** | **0.049** |

For DCLM, the A/B letter format is degenerate: the model always emits "A", so it carries zero
content signal and zero steerability. For the sysprompt model the positional collapse is gone
(bias 1.0 → 0.5) and the channel responds to steering. The instruction-following format came alive.

## Result 2 (suggestive): steerability moved `repeat` → `ab` and grew

Steerability Δ = P(pick rule-compliant action | steered toward rule) − P(neutral), high-ambiguity:

| rule | n | DCLM-ab | **sysp-ab** | DCLM-rep | sysp-rep |
|---|---|---|---|---|---|
| break_promise | 109 | +0.000 | **+0.096** | +0.000 | +0.018 |
| cheat | 138 | +0.000 | **+0.072** | +0.000 | +0.018 |
| pleasure | 284 | +0.000 | **+0.067** | +0.004 | −0.002 |
| deceive | 262 | +0.000 | **+0.050** | +0.013 | +0.004 |
| freedom | 220 | +0.000 | **+0.048** | +0.025 | +0.005 |
| pain | 369 | +0.000 | +0.039 | +0.008 | +0.015 |
| duty | 412 | +0.000 | +0.029 | +0.015 | +0.002 |
| break_law | 196 | +0.000 | +0.028 | +0.056 | +0.015 |
| disable | 146 | +0.000 | +0.024 | +0.041 | +0.010 |
| death | 92 | +0.000 | −0.033 | +0.011 | −0.005 |
| **mean \|Δ\|** | | **0.000** | **0.049** | 0.017 | 0.009 |

The DCLM model's only live channel was `repeat` (full-text), mean \|Δ\| 0.017. The sysprompt
model's live channel is `ab` (instruction format), mean \|Δ\| 0.049 — ~3× larger and in the
channel that directly mirrors instruction-following.

## Result 3 (honest): baseline morality did NOT improve

| neutral `acc_norm` | DCLM | sysprompt |
|---|---|---|
| low / `ab` | 0.500 | 0.422 |
| low / `repeat` | 0.566 | 0.552 |
| high / `repeat` | 0.514 | 0.494 |

System-prompt pretraining made the model **steerable**, not more **moral** — low-ambiguity
accuracy is flat-to-slightly-lower. The intervention changed *controllability*, not the model's
default moral preference.

## Result 4 (refutes a hypothesis): native system-channel steering does NOT help

We re-ran the sysprompt model wrapping each steer instruction in its pretraining channel
(`<|start_header_id|>system<|end_header_id|>\n\n{instr}<|eot_id|>`) instead of plain-prepending
(`--steer-format native`, run `sysprompt-native-step59647`). Neutral is unchanged, so deltas are
directly comparable. High-ambiguity `ab`:

| | plain-prepend | native `<\|system\|>` |
|---|---|---|
| mean \|Δ\| | **0.049** | **0.036** |
| pleasure | +0.067 | +0.037 |
| freedom | +0.048 | +0.034 |
| break_law | +0.028 | +0.008 |
| break_promise | +0.096 | +0.106 |

**Native wrapping makes steering no stronger (slightly weaker).** So the "the model woke up because
of its native channel" version of the claim is **false**. The surviving, weaker claim: sysprompt
pretraining made the model *somewhat more instruction-responsive in general* (ab mean |Δ| ~0.04 vs
DCLM's 0.000), but not in a channel-specific way.

**Likely mechanism:** in `[S][D]` pretraining the system block was a *content/topic descriptor*
followed by a **web document** — not an *imperative directive* followed by a **Q&A**. The model
learned "system block ⇒ conditions the following text's content," not "⇒ a behavioral instruction to
obey." An imperative steer + multiple-choice question is off-distribution for that slot, so native
wrapping doesn't unlock behavioral steering (and may prime "continue a document" instead).

## Why you should still be skeptical

1. **Most per-rule deltas are at the noise floor.** Each Δ is a paired difference of {0, 0.5, 1}
   per-scenario preferences over n scenarios. A rough proportion SE is `~sqrt(0.25/n)` ≈ 0.03–0.05
   for these n. Only the largest `ab` deltas (`break_promise +0.096`, `cheat +0.072`,
   `pleasure +0.067`) clear ~2 SE; the sub-0.03 ones are plausibly noise. We have **not** computed
   real (paired, bootstrap) confidence intervals yet.
2. **Confounds in the comparison.** Different step counts (59647 vs 56002), different data; and the
   format mismatch (caveat above) cuts *against* sysprompt, so the gap could be larger or the
   per-rule pattern could shuffle.
3. **Absolute magnitudes are small.** Even the live channel moves choices by ≤10 points. The
   *qualitative* claim (dead channel → live channel, position bias 1.0 → 0.5) is robust; the
   *quantitative* per-rule steerability is preliminary.
4. **No per-scenario flip inspection yet** for the sysprompt model (we did this for DCLM
   `break_law` and found only partial single-order nudges, no full reversals).

## Dig deeper (ranked)

1. ~~**Native-format steering.**~~ **DONE — refuted (see Result 4).** Native wrapping did not
   increase steerability; the channel-specific claim is false.
2. **Confidence intervals (now the most important).** With the native-format hypothesis dead, the
   whole story rests on the plain-prepend ab gain (~0.049 vs 0.000). Dump per-scenario prefs (we
   have `--dump-prefs`) for both models and
   bootstrap paired CIs on each Δ, so we can say which rules move significantly.
3. **Per-scenario flips.** Same treatment as the DCLM `break_law` analysis — surface actual
   neutral→steered flips for the largest sysprompt `ab` effects (`break_promise`, `cheat`).
4. **Checkpoint sweep.** Run the eval across the sysprompt model's `hf/step-{10k…59647}` exports to
   see whether steerability *emerges* over pretraining (a dose-response curve would be persuasive).
5. **DailyDilemmas cross-check.** Run the sysprompt model through the DailyDilemmas eval (both
   modes) for an independent replication of the channel-shift effect.
