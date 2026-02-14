# Download and Serve a Marin Checkpoint

This is a bit specific to Michael so don't merge this into Marin codebase without extra edits/review

How to pull a trained checkpoint from GCS, transfer it to a serving machine, and run inference with vLLM.

## 1. Find the checkpoint on GCS

List the HF-converted checkpoints for your run:

```bash
gcloud storage ls gs://marin-us-central1/checkpoints/<run-name>/hf/
```

Example:

```bash
gcloud storage ls gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-sft-875a72/hf/
```

Pick the step you want (e.g. `step-250`).

## 2. Download locally

```bash
mkdir -p ~/models/<model-name>
gcloud storage cp -r \
  "gs://marin-us-central1/checkpoints/<run-name>/hf/<step>/*" \
  ~/models/<model-name>/
```

## 3. Transfer to the serving machine

```bash
rsync -avP ~/models/<model-name>/ \
  <user>@<host>:<remote-path>/<model-name>/
```

## 4. Serve with vLLM

For a model using **Qwen's native tokenizer/chat template**:

```bash
vllm serve <model-path> --reasoning-parser qwen3
```

For a model using the **Marin custom tokenizer** (which requires a plugin for the `<think>` token handling):

```bash
vllm serve <model-path> \
  --reasoning-parser marin_think \
  --reasoning-parser-plugin <path-to>/marin_think_parser.py
```

## Concrete example (Qwen3-0.6B rephraser SFT)

```bash
# 1. List checkpoints
gcloud storage ls gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-sft-875a72/hf/

# 2. Download step-250
mkdir -p ~/models/qwen3-0.6b-rephraser-sft
gcloud storage cp -r \
  "gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-sft-875a72/hf/step-250/*" \
  ~/models/qwen3-0.6b-rephraser-sft/

# 3. Rsync to Stanford cluster
rsync -avP ~/models/qwen3-0.6b-rephraser-sft/ \
  mryan0@scdt.stanford.edu:/nlp/scr2/nlp/personal-rm/small-rephraser/small-rephraser/models/qwen3-0.6b-rephraser-sft-ckpt250/

# 4. Serve (Qwen tokenizer)
vllm serve small-rephraser/models/qwen3-0.6b-rephraser-sft-ckpt250/ --reasoning-parser qwen3
```
