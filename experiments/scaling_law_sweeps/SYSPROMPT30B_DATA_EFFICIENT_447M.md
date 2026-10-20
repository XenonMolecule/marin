# Data-efficient 447M system-prompt models

Five 447M models, each the best of its own hyperparameter search, from a sweep that asked
whether conditioning on system prompts helps once the model is tuned in the data-efficient
regime. It does not: no conditioned model beats the doc-matched control on bare evaluation.

On SC:

```
/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/data_efficient_447m/
    doc-matched/  sysprompt/  sysprompt-masked/  mix50/  cooldown/
```

Each directory is a HuggingFace checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/data_efficient_447m/cooldown"
model = AutoModelForCausalLM.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)
```

## Setup

Model: Qwen3 architecture with Llama-3 rotary embeddings, hidden 1024, 11 layers, 8 heads,
MLP 4096, 447.2M parameters, Llama-3.1 tokenizer, sequence length 4096. Optimizer is AdamH,
which holds every linear weight matrix at its initial norm and so has no weight decay;
embeddings and norm gains use plain Adam at a separate `adam_lr`. Linear schedule, 10% warmup,
20% decay to zero, gradient clip 0.1, z-loss 1e-7.

Data: 2,812,816 DCLM-30B documents, about 3.6B tokens, trained for exactly one epoch. That is
8.2 tokens per parameter, against about 16 for the compute-optimal 447M ladder model. The
document set is a prefix of the ladder's 447M set, and every model below saw the same
documents.

The prompt format is `<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>{document}`.
Prompts average about 21 tokens with the wrapper, 1.6% of the conditioned token stream.

| directory | what the model trained on |
|---|---|
| `doc-matched` | the documents with no prompts (control) |
| `sysprompt` | every document with its prompt, loss on every token |
| `sysprompt-masked` | the same token stream as `sysprompt`, no loss on the prompt: trains p(D \| S) |
| `mix50` | a prompt on a fixed half of the documents, chosen by hashing the document id |
| `cooldown` | prompts on the first 80% of documents, then the last 20% bare; the switch lands at the start of the learning-rate decay |

## Best model per arm

Evaluation is bits per byte with no prompt at eval time. `eval/bpb` is the headline number; lower
is better. One seed each.

| directory | eval/bpb | vs control | lima | paloma | uncheatable | batch | steps | lr | adam_lr | beta2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `doc-matched` | **1.1021** | | 0.9489 | 1.1062 | 1.0820 | 16 | 54,931 | 2.81e-3 | 2.71e-4 | 0.9999 |
| `mix50` | 1.1044 | +0.002 | 0.9508 | 1.1081 | 1.0879 | 16 | 55,380 | 2.81e-3 | 2.71e-4 | 0.999 |
| `cooldown` | 1.1052 | +0.003 | 0.9498 | 1.1090 | 1.0884 | 32 | 27,811 | 3.97e-3 | 3.83e-4 | 0.999 |
| `sysprompt-masked` | 1.1165 | +0.014 | 0.9635 | 1.1207 | 1.0966 | 32 | 27,914 | 3.97e-3 | 3.83e-4 | 0.9999 |
| `sysprompt` | 1.1227 | +0.021 | 0.9652 | 1.1276 | 1.0973 | 32 | 27,914 | 3.97e-3 | 3.83e-4 | 0.9999 |

All five use beta1 0.9. Epsilon is 4.48e-8 at batch 16 and 3.17e-8 at batch 32. The learning
rates are the delphi scaling-heuristic values for this cell at that batch size; no model
preferred a scaled learning rate.

For `mix50` the top two settings are tied: batch 16 with beta2 0.999 scored 1.10438 and batch 32
at delphi defaults scored 1.10442. For `doc-matched`, batch 16 at 1.10210 and batch 32 at 1.10231
are tied the same way.

## The search

`doc-matched`, `sysprompt` and `mix50` each ran the same nine settings: the AdamH learning rate at
0.5x, 1x and 2x the delphi value for batch 16 and 32, plus beta2 0.95, beta2 0.999 and beta1 0.95
at the batch-16 delphi point.

| setting | sysprompt | doc-matched | mix50 |
|---|---|---|---|
| B16, lr 0.5x | 1.1401 | 1.1121 | 1.1142 |
| B16, lr 1x | 1.1297 | **1.1021** | 1.1051 |
| B16, lr 2x | 1.1461 | 1.1224 | 1.1226 |
| B32, lr 0.5x | 1.1331 | 1.1108 | 1.1137 |
| B32, lr 1x | **1.1227** | 1.1023 | 1.1044 |
| B32, lr 2x | 1.1412 | 1.1227 | 1.1242 |
| B16, lr 1x, beta2 0.95 | 1.1924 | 1.1635 | 1.1655 |
| B16, lr 1x, beta2 0.999 | 1.1289 | 1.1025 | **1.1044** |
| B16, lr 1x, beta1 0.95 | 1.1305 | 1.1088 | 1.1089 |

`sysprompt-masked` and `cooldown` were added afterwards and run only at `sysprompt`'s best point,
batch 32 with delphi values. `cooldown` also ran with beta2 0.999, which scored 1.1052 against
1.1070 at 0.9999.

## What it shows

- **No conditioned model wins on bare evaluation.** The best conditioned models, `mix50` and
  `cooldown`, sit 0.002 to 0.003 above the control. Matched-configuration noise is under 0.001,
  so that small gap is real.
- **Most of the `sysprompt` penalty is train/eval mismatch.** Masking the prompt tokens recovers
  0.006 of the 0.021 gap. Leaving the last 20% of training bare, as in `cooldown`, recovers 0.016.
- **The penalty is smaller in this regime but does not close.** `sysprompt` trails the control by
  0.021 here and by 0.039 at the compute-optimal 447M ladder point.
- **The delphi hyperparameters were already right.** Every arm's best learning rate was the
  delphi value. beta2 0.95 cost 0.06 everywhere; 0.999 matched 0.9999 except for `cooldown`, which
  it helped by 0.002, the expected direction for a mid-run data change. beta1 0.95 never helped.
  Only `sysprompt` gained from batch 32, by 0.007; the other arms moved less than 0.001.

## Caveats

- One seed per model. Differences under about 0.002 should not be read as rankings.
- All evaluation here is bare. The conditioned models have not yet been scored with a generated
  prompt prepended, which is the number that shows what the prompts buy when they are available.
- `sysprompt-masked` was built as a separate pretokenized cache. Its token ids match a direct
  Llama-3 encode; the `sysprompt` cache differs by 24k tokens of 3.66B because Marin's tokenizer
  splits documents longer than 10k characters before encoding.
