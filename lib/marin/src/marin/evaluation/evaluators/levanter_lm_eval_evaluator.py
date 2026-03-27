# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import os

import jmp
from iris.marin_fs import filesystem as marin_filesystem
import levanter
import levanter.eval_harness as eval_harness
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.evals.task_configs import convert_to_levanter_task_config
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator
from fray.v1.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


class LevanterLmEvalEvaluator(LevanterTpuEvaluator):
    """For `Evaluator`s that runs inference with Levanter's Lm Eval Harness on TPUs."""

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval", "tpu"],
            pip_packages=["statsmodels==0.14.4"],
            env_vars={
                "TOKENIZERS_PARALLELISM": "false",
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
                "HF_ALLOW_CODE_EVAL": "1",
            },
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): The tags to add to the wandb run.
        """
        # Eval Harness code: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/eval_harness.py
        # Run the harness with the model and the specified evals

        try:
            model_name_or_path: str = self.model_name_or_path(model)
            name = model.name + "_lmeval_" + "-".join([eval_task.name for eval_task in evals])
            logger.info(f"WandB Run Name: {name}")
            logger.info(f"Running eval harness on model: {model_name_or_path}")
            print("after wandb log")
            # NOTE(chris): Before, the batch size was 16, but this is too large for the 8B model.
            # In the future, we should make this user-configurable.
            #
            # Use bf16 for both params and compute to reduce per-chip HBM pressure.
            # With p=f32 the model weights take ~5.5GB per chip; bf16 halves that.
            trainer_config = TrainerConfig(
                tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
                mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
                per_device_eval_parallelism=1,
                ray=RayConfig(auto_start_cluster=False),
            )
            print("after trainer?")

            model_config = HFCheckpointConverter.from_hf(model_name_or_path).LevConfigClass()

            # convert to the config that Levanter's eval_harness expects
            tasks = convert_to_levanter_task_config(evals)
            logger.info(f"Tasks: {tasks}")

            model_path = model_name_or_path

            logger.info(f"Model path: {model_path}")
            logger.info(f"Model name: {model.name}")
            logger.info(f"model_name_or_path: {model_name_or_path}")

            # Build generation_kwargs, merging defaults with any user-provided params
            generation_kwargs = {"max_gen_toks": 1024, "temperature": 0.0, "n": 1, "seed": None}
            if model.generation_params:
                generation_kwargs.update(model.generation_params)

            print("starting harness")
            eval_config = eval_harness.EvalHarnessMainConfig(
                eval_harness=eval_harness.LmEvalHarnessConfig(
                    task_spec=tasks,
                    max_examples=max_eval_instances,
                    log_samples=True,
                    max_length=2048,
                    apply_chat_template=model.apply_chat_template,
                    confirm_run_unsafe_code=True,
                    sample_logging=eval_harness.SampleLoggingConfig(max_samples_per_benchmark=20),
                    generation_kwargs=generation_kwargs,
                ),
                tokenizer=model_path,  # levanter picks up the tokenizer from the model path
                checkpoint_path=model_path,
                checkpoint_is_hf=True,
                trainer=trainer_config,
                model=model_config,
            )

            # If apply_chat_template is enabled but the tokenizer doesn't have a
            # built-in chat template (e.g. Llama3 base tokenizer), inject the
            # Llama 3.1 chat template so eval prompts match the SFT training format.
            if model.apply_chat_template:
                tokenizer = eval_config.the_tokenizer
                if not getattr(tokenizer, "chat_template", None):
                    from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE

                    # Strip Levanter-specific {% generation %} tags that the HF
                    # tokenizer doesn't understand.
                    clean_template = LLAMA_3_1_CHAT_TEMPLATE.replace("{% generation %}", "").replace(
                        "{% endgeneration %}", ""
                    )
                    tokenizer.chat_template = clean_template
                    logger.info("Injected Llama 3.1 chat template into tokenizer for eval")

            results = eval_harness.run_eval_harness_main(eval_config)
            print("finished harness")

            try:
                # add a results.json to output path
                output_path = os.path.join(output_path, "results.json")

                logger.info(f"Uploading results to GCS: {output_path}")

                # write output JSON directly to output_path on GCS
                fs = marin_filesystem("gcs")
                with fs.open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=_json_default)

                levanter.tracker.current_tracker().finish()
                logger.info("Upload completed successfully.")

            except Exception as upload_error:
                logger.info(f"Failed to upload results to GCS: {upload_error}")

        except Exception as e:
            logger.error(f"Error running eval harness: {e}")
            raise e


def _json_default(value):
    """
    Provide a best-effort JSON serialization for objects returned by the eval harness.
    """
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    if isinstance(value, set):
        return list(value)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass

    return repr(value)
