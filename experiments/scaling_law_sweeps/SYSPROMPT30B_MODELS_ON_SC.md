# System-prompt pretraining models on SC

Everything is under one directory:

```
/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/
```

Twenty models, all trained from scratch on the same DCLM-30B corpus with the
Llama-3.1 tokenizer, one epoch each. They differ only in whether documents had a
generated system prompt prepended:

| directory name | what the model saw during training |
|---|---|
| `sysprompt` | every document with a system prompt |
| `token-matched` | no prompts, extra documents to match `sysprompt`'s token count |
| `doc-matched` | no prompts, exactly the same documents as `sysprompt` |
| `mix50` | same documents, prompt on a fixed half of them |

Layout:

```
models/
  sysprompt/  token-matched/  doc-matched/  mix50/     998M models (14.9B tokens, 9e19 FLOPs)
  ladder/
    README.md                                          per-model tokens, budgets, and eval numbers
    210m/  358m/  447m/  669m/                         each holds the same four directories
```

Every leaf directory is a HuggingFace checkpoint (config, safetensors, tokenizer):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/ladder/447m/mix50"
model = AutoModelForCausalLM.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)
```

The prompt format the `sysprompt` and `mix50` models were trained with is
`<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>{document}`.
Training sizes: 210M on 3.0B tokens, 358M on 4.7B, 447M on 7.3B, 669M on 7.7B,
998M on 14.9B. Details and eval results are in `ladder/README.md`.
