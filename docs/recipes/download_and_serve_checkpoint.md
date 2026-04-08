# Download and Serve a Marin Checkpoint

This is a bit specific to Michael so don't merge this into Marin codebase without extra edits/review

How to pull a trained checkpoint from GCS, transfer it to a serving machine, and run inference with vLLM.

## 1. Find the checkpoint on GCS

List the HF-converted checkpoints for your run:

```bash
gcloud storage ls gs://marin-us-central1/checkpoints/<run-name>/hf/
```

```bash
gcloud storage ls gs://marin-us-central1/checkpoints/ | grep "rephraser"
```

Pick the step you want (e.g. `step-903` for the final step).

## 2. Download locally

```bash
mkdir -p ~/models/<model-name>
gcloud storage cp -r \
  "gs://marin-us-central1/checkpoints/<run-name>/hf/<step>/*" \
  ~/models/<model-name>/
```

## 3. Transfer to the serving machine

```bash
# Create remote directory first
ssh <user>@<host> "mkdir -p <remote-path>/<model-name>"

rsync -avP ~/models/<model-name>/ \
  <user>@<host>:<remote-path>/<model-name>/
```

### Batch transfer script

For multiple checkpoints, use `scripts/transfer_checkpoints_to_stanford.sh`.
Edit the `NAMES` and `GCS_PATHS` arrays, then run:

```bash
bash scripts/transfer_checkpoints_to_stanford.sh
```

## 4. Transfer the no-think chat template (for stripped models)

Stripped (no-thinking) models were trained without `<think>` blocks. They need
a custom chat template that doesn't emit any think tokens at generation time.

```bash
rsync -avP experiments/chat_templates/qwen3_no_think_serving.jinja \
  <user>@<host>:<remote-path>/qwen3_no_think_serving.jinja
```

## 5. Serve with vLLM

### Think models (trained with `<think>` reasoning traces)

Use `--reasoning-parser qwen3` to handle `<think>` token parsing:

```bash
vllm serve <model-path> --reasoning-parser qwen3
```

### Stripped models (trained WITHOUT `<think>` traces)

Use `--chat-template` with the no-think template. Do NOT use `--reasoning-parser`
or `enable_thinking` -- think tokens are out of distribution for these models.

```bash
vllm serve <model-path> \
  --chat-template <remote-path>/qwen3_no_think_serving.jinja
```

## Concrete examples (Kimi-distilled rephraser SFT)

### Serving

```bash
MODELS=/nlp/scr2/nlp/personal-rm/small-rephraser/small-rephraser/models
TEMPLATE=$MODELS/qwen3_no_think_serving.jinja

# Q3-0.6B Think
vllm serve $MODELS/qwen3-0.6b-rephraser-kimi-think-sft/ --reasoning-parser qwen3

# Q3-0.6B Stripped
vllm serve $MODELS/qwen3-0.6b-rephraser-kimi-stripped-sft/ --chat-template $TEMPLATE

# Q3-1.7B Think
vllm serve $MODELS/qwen3-1.7b-rephraser-kimi-think-sft/ --reasoning-parser qwen3

# Q3-1.7B Stripped
vllm serve $MODELS/qwen3-1.7b-rephraser-kimi-stripped-sft/ --chat-template $TEMPLATE
```

### GCS checkpoint paths

| Model | Variant | GCS Path |
|-------|---------|----------|
| Q3-0.6B | think | `gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-kimi-think-sft-f7321d/hf/step-903/` |
| Q3-0.6B | stripped | `gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-kimi-stripped-sft-2a6763/hf/step-903/` |
| Q3-1.7B | think | `gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-kimi-think-sft-777825/hf/step-903/` |
| Q3-1.7B | stripped | `gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-kimi-stripped-sft-5b3958/hf/step-903/` |
