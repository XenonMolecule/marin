# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Debug script: directly load SFT checkpoint and generate on GSM8K examples.

This bypasses the eval harness to see exactly what the model generates.
Tests multiple prompt formats and generation settings.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/debug_generation.py

Dry run (CPU, will be slow but works for debugging the script):
    HF_TOKEN=... uv run python experiments/rephraser/debug_generation.py --local
"""

import argparse
import json
import logging
import re

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoints to test
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "baseline": "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/hf/step-44758",
    "gsm8k_plaintext_sft": "gs://marin-us-central1/checkpoints/gsm8k-plaintext-sft-1e20-qwen3-a40a32/hf/step-10",
    "qra_plaintext_sft": (
        "gs://marin-us-central1/checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/hf/step-421"
    ),
    "resiliparse_sft": "gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/hf/step-954",
}

LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# ---------------------------------------------------------------------------
# GSM8K 8-shot prompt (same as eval harness uses)
# ---------------------------------------------------------------------------
FEWSHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8."""

# Test questions from GSM8K test set
TEST_QUESTIONS = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
        ),
        "answer": 18,
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
        ),
        "answer": 3,
    },
    {
        "question": (
            "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
        ),
        "answer": 70000,
    },
    {
        "question": (
            "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"
        ),
        "answer": 540,
    },
    {
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
        ),
        "answer": 20,
    },
]


def build_prompt(question: str) -> str:
    """Build 8-shot CoT prompt in plain text format."""
    return f"{FEWSHOT_EXAMPLES}\n\nQ: {question}\nA:"


def extract_answer_strict(text: str) -> str | None:
    """Extract answer using strict 'The answer is X' pattern."""
    match = re.search(r"The answer is (\-?[0-9\.\,]+)", text)
    return match.group(1).replace(",", "") if match else None


def extract_answer_flexible(text: str) -> str | None:
    """Extract answer using flexible number extraction (last number found)."""
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if matches:
        last = matches[-1]
        result = last[0] if last[0] else last[1]
        return result.replace(",", "").replace("$", "")
    return None


def run_debug(args):
    """Load models and generate on test examples."""
    from levanter.compat.hf_checkpoints import HFCheckpointConverter

    tokenizer = AutoTokenizer.from_pretrained(LLAMA3_TOKENIZER)
    output_path = args.output_path

    results = {}

    checkpoints_to_test = args.checkpoints.split(",") if args.checkpoints else list(CHECKPOINTS.keys())

    for ckpt_name in checkpoints_to_test:
        if ckpt_name not in CHECKPOINTS:
            logger.warning(f"Unknown checkpoint: {ckpt_name}")
            continue

        ckpt_path = CHECKPOINTS[ckpt_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading checkpoint: {ckpt_name}")
        logger.info(f"Path: {ckpt_path}")
        logger.info(f"{'='*60}")

        try:
            converter = HFCheckpointConverter.from_hf(ckpt_path)
            model_config = converter.LevConfigClass()
            model = converter.load_pretrained(model_config)
        except Exception as e:
            logger.error(f"Failed to load {ckpt_name}: {e}")
            results[ckpt_name] = {"error": str(e)}
            continue

        ckpt_results = []

        for i, test in enumerate(TEST_QUESTIONS):
            prompt = build_prompt(test["question"])
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            prompt_len = len(input_ids[0])

            logger.info(f"\n--- Test {i}: {test['question'][:80]}... ---")
            logger.info(f"  Prompt length: {prompt_len} tokens")
            logger.info(f"  Expected answer: {test['answer']}")

            # Generate with the model
            try:
                generated_ids = _generate_greedy(model, tokenizer, input_ids, max_new_tokens=256)
                generated_text = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"  Generation failed: {e}")
                generated_text = f"ERROR: {e}"

            strict = extract_answer_strict(generated_text)
            flexible = extract_answer_flexible(generated_text)

            logger.info(f"  Generated ({len(generated_text)} chars): {generated_text[:200]!r}")
            logger.info(f"  Strict extract: {strict}")
            logger.info(f"  Flexible extract: {flexible}")
            logger.info(f"  Correct (strict): {strict == str(test['answer']) if strict else False}")
            logger.info(f"  Correct (flexible): {flexible == str(test['answer']) if flexible else False}")

            ckpt_results.append(
                {
                    "question": test["question"],
                    "expected": test["answer"],
                    "generated": generated_text,
                    "strict_extract": strict,
                    "flexible_extract": flexible,
                    "prompt_tokens": prompt_len,
                }
            )

        results[ckpt_name] = ckpt_results

    # Save results
    if output_path:
        with fsspec.open(f"{output_path}/debug_generation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}/debug_generation_results.json")

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION DEBUG SUMMARY")
    print("=" * 80)
    for ckpt_name, ckpt_results in results.items():
        print(f"\n### {ckpt_name} ###")
        if isinstance(ckpt_results, dict) and "error" in ckpt_results:
            print(f"  ERROR: {ckpt_results['error']}")
            continue
        for r in ckpt_results:
            gen = r["generated"]
            is_empty = not gen.strip()
            correct_strict = r["strict_extract"] == str(r["expected"]) if r["strict_extract"] else False
            correct_flex = r["flexible_extract"] == str(r["expected"]) if r["flexible_extract"] else False
            print(f"  Q: {r['question'][:60]}...")
            print(f"    Expected: {r['expected']}")
            print(f"    Empty: {is_empty}")
            print(f"    Generated: {gen[:100]!r}")
            print(f"    Strict: {r['strict_extract']} ({'CORRECT' if correct_strict else 'WRONG'})")
            print(f"    Flexible: {r['flexible_extract']} ({'CORRECT' if correct_flex else 'WRONG'})")


def _generate_greedy(model, tokenizer, input_ids: np.ndarray, max_new_tokens: int = 256) -> np.ndarray:
    """Simple greedy generation using the model directly.

    This is a minimal implementation to debug what the model produces.
    """
    # For Levanter/JAX models, we need to use the model's generate method
    # This is a simplified version - the real eval harness has more sophisticated generation
    import haliax as hax
    from levanter.models.lm_model import LmHeadModel

    if not isinstance(model, LmHeadModel):
        raise ValueError(f"Expected LmHeadModel, got {type(model)}")

    eos_token_id = tokenizer.eos_token_id
    # Also stop at "Q:" which is the few-shot delimiter
    q_colon_ids = tokenizer.encode("Q:", add_special_tokens=False)

    current_ids = list(input_ids[0])

    for step in range(max_new_tokens):
        # Create input array
        input_array = jnp.array([current_ids])

        # Get next token logits
        # Use the model's __call__ to get logits
        Pos = model.Pos
        Vocab = model.Vocab

        # Truncate if needed
        if len(current_ids) > Pos.size:
            current_ids = current_ids[-Pos.size :]
            input_array = jnp.array([current_ids])

        tokens = hax.named(input_array[0, : len(current_ids)], Pos.resize(len(current_ids)))

        # Forward pass
        with jax.default_matmul_precision("bfloat16"):
            logits = model(tokens)

        # Get logits for last position
        last_logits = logits[model.Pos.resize(len(current_ids)), -1]
        # Convert to numpy
        last_logits_np = np.array(last_logits.array)
        next_token = int(np.argmax(last_logits_np))

        # Check for EOS
        if next_token == eos_token_id:
            break

        current_ids.append(next_token)

        # Check for "Q:" stop sequence
        if len(current_ids) >= len(q_colon_ids):
            if current_ids[-len(q_colon_ids) :] == q_colon_ids:
                # Remove the "Q:" tokens we just added
                current_ids = current_ids[: -len(q_colon_ids)]
                break

    return np.array(current_ids)


def _download_gcs_checkpoint(gcs_path: str) -> str:
    """Download a GCS checkpoint to a local temp directory."""
    import subprocess
    import tempfile

    local_dir = tempfile.mkdtemp(prefix="ckpt_")
    print(f"  Downloading {gcs_path} -> {local_dir}")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", local_dir],
        check=True,
        capture_output=True,
    )
    print("  Download complete")
    return local_dir


def run_debug_transformers(args):
    """Simpler debug using HuggingFace transformers directly (works on CPU too)."""
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(LLAMA3_TOKENIZER)

    checkpoints_to_test = args.checkpoints.split(",") if args.checkpoints else list(CHECKPOINTS.keys())
    results = {}

    for ckpt_name in checkpoints_to_test:
        if ckpt_name not in CHECKPOINTS:
            logger.warning(f"Unknown checkpoint: {ckpt_name}")
            continue

        ckpt_path = CHECKPOINTS[ckpt_name]
        print(f"\n{'='*60}")
        print(f"Loading: {ckpt_name} from {ckpt_path}")
        print(f"{'='*60}")

        try:
            # Download from GCS if needed
            if ckpt_path.startswith("gs://"):
                local_path = _download_gcs_checkpoint(ckpt_path)
            else:
                local_path = ckpt_path

            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            model.eval()
        except Exception as e:
            print(f"  FAILED to load: {e}")
            import traceback

            traceback.print_exc()
            results[ckpt_name] = {"error": str(e)}
            continue

        ckpt_results = []

        for i, test in enumerate(TEST_QUESTIONS):
            prompt = build_prompt(test["question"])
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]
            print(f"\n  Test {i}: {test['question'][:60]}... (prompt={prompt_len} tokens, expected={test['answer']})")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][prompt_len:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Show raw generation BEFORE stop-sequence stripping
            print(f"    Raw generation ({len(raw_text)} chars): {raw_text[:300]!r}")

            # Apply stop sequence stripping (same as eval harness)
            generated_text = raw_text
            for stop_seq in ["Q:", "</s>", "<|im_end|>"]:
                generated_text = generated_text.split(stop_seq)[0]

            after_strip = generated_text.strip()
            if not after_strip and raw_text.strip():
                # Generation was non-empty but stop sequence made it look empty
                print(f"    >>> STOP SEQUENCE caused empty! Raw starts with: {raw_text[:60]!r}")

            strict = extract_answer_strict(generated_text)
            flexible = extract_answer_flexible(generated_text)
            correct_strict = strict == str(test["answer"]) if strict else False
            correct_flex = flexible == str(test["answer"]) if flexible else False

            print(f"    After stop-seq ({len(generated_text)} chars): {generated_text[:200]!r}")
            print(f"    Strict: {strict} ({'CORRECT' if correct_strict else 'WRONG'})")
            print(f"    Flexible: {flexible} ({'CORRECT' if correct_flex else 'WRONG'})")

            ckpt_results.append(
                {
                    "question": test["question"],
                    "expected": test["answer"],
                    "raw_generation": raw_text,
                    "generated": generated_text,
                    "strict_extract": strict,
                    "flexible_extract": flexible,
                    "prompt_tokens": prompt_len,
                    "correct_strict": correct_strict,
                    "correct_flex": correct_flex,
                    "stop_seq_caused_empty": bool(not after_strip and raw_text.strip()),
                }
            )

        results[ckpt_name] = ckpt_results

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output_path = args.output_path or "/tmp/debug_generation"
    import os

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/debug_generation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION DEBUG SUMMARY")
    print("=" * 80)
    for ckpt_name, ckpt_results in results.items():
        print(f"\n### {ckpt_name} ###")
        if isinstance(ckpt_results, dict) and "error" in ckpt_results:
            print(f"  ERROR: {ckpt_results['error']}")
            continue
        non_empty = sum(1 for r in ckpt_results if r["generated"].strip())
        correct_s = sum(1 for r in ckpt_results if r.get("correct_strict", False))
        correct_f = sum(1 for r in ckpt_results if r.get("correct_flex", False))
        print(f"  Non-empty: {non_empty}/{len(ckpt_results)}")
        print(f"  Correct (strict): {correct_s}/{len(ckpt_results)}")
        print(f"  Correct (flexible): {correct_f}/{len(ckpt_results)}")
        for r in ckpt_results:
            gen = r["generated"]
            print(f"  Q: {r['question'][:50]}... => {gen[:80]!r}{'...' if len(gen) > 80 else ''}")

    print(f"\nResults saved to {output_path}/debug_generation_results.json")


def run_eval_with_full_logging(args):
    """Re-run eval harness with log_all=True to capture all 1319 samples."""
    import levanter.eval_harness as eval_harness
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    from levanter.distributed import RayConfig
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig

    ckpt_name = args.checkpoints or "gsm8k_plaintext_sft"
    ckpt_path = CHECKPOINTS[ckpt_name]

    print(f"Running full eval with all samples logged for: {ckpt_name}")
    print(f"Checkpoint: {ckpt_path}")

    import jmp

    trainer_config = TrainerConfig(
        tracker=WandbConfig(project="marin", tags=["debug", "full-logging"], name=f"debug-fulllog-{ckpt_name}"),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=1,
        ray=RayConfig(auto_start_cluster=False),
    )

    model_config = HFCheckpointConverter.from_hf(ckpt_path).LevConfigClass()

    from marin.evaluation.evaluation_config import EvalTaskConfig

    from experiments.evals.task_configs import convert_to_levanter_task_config

    tasks = convert_to_levanter_task_config(
        [
            EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
        ]
    )

    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=tasks,
            max_examples=None,
            log_samples=True,
            max_length=2048,
            apply_chat_template=False,
            confirm_run_unsafe_code=True,
            sample_logging=eval_harness.SampleLoggingConfig(log_all=True),
        ),
        tokenizer=ckpt_path,
        checkpoint_path=ckpt_path,
        checkpoint_is_hf=True,
        trainer=trainer_config,
        model=model_config,
    )

    results = eval_harness.run_eval_harness_main(eval_config)

    # Save full results
    output_path = args.output_path or "gs://marin-us-central1/debug/generation"
    with fsspec.open(f"{output_path}/full_eval_{ckpt_name}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    for task, metrics in results.get("results", {}).items():
        outputs = metrics.get("outputs", [])
        non_empty = sum(1 for o in outputs if o.get("generation", "").strip())
        print(f"\n{task}: {non_empty}/{len(outputs)} non-empty generations")

        strict = metrics.get("exact_match,strict-match", "N/A")
        flexible = metrics.get("exact_match,flexible-extract", "N/A")
        print(f"  strict={strict}, flexible={flexible}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Debug generation from SFT checkpoints")
    parser.add_argument("--local", action="store_true", help="Use transformers (CPU/GPU) instead of Levanter/JAX")
    parser.add_argument("--full_eval", action="store_true", help="Run full eval with all samples logged")
    parser.add_argument(
        "--checkpoints", type=str, default=None, help="Comma-separated checkpoint names to test (default: all)"
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output path for results")
    args = parser.parse_args()

    if args.full_eval:
        run_eval_with_full_logging(args)
    elif args.local:
        run_debug_transformers(args)
    else:
        run_debug(args)
