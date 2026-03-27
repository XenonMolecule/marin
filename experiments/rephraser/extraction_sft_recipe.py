# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extraction SFT experiment recipe.

Reusable pipeline: URLs -> CDX query -> WARC download -> filter -> LLM extraction
-> postprocess -> tokenize -> train -> eval. Scale to new domains by specifying
URLs, extraction prompt(s), SFT model, and eval config.

All training uses plain text format (TextLmDatasetFormat). No chat templates.

Example usage:
    from experiments.rephraser.extraction_sft_recipe import *

    result = build_extraction_sft_experiment(
        domain="mathhelpforum",
        source=DomainSource(url_patterns=[UrlPattern("mathhelpforum.com")], html_data_override=some_step),
        extractions=[ExtractionSpec(name="qra", prompt=QRA_PROMPT, model=MODEL, model_tokenizer=TOK)],
        sft_model=SFTModelSpec(model_config=my_config, tokenizer="Qwen/Qwen3-0.6B"),
        eval_spec=EvalSpec(tasks=MATH_EVALS),
    )
    executor_main(steps=result.all_steps, description="My extraction SFT")
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import fsspec
import jmp

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.rephraser_cooldown import _read_token_count
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.download.commoncrawl.cdx_query import CDXQueryConfig, query_cdx
from marin.datakit.download.commoncrawl.cdx_query_columnar import query_cdx_columnar
from marin.datakit.download.commoncrawl.download_warc_records import WarcRecordDownloadConfig, download_warc_records
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.extract_text_from_html import ExtractTextConfig, extract_text_from_html
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt templates (DSPy field-delimited format)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

DEFAULT_USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
def _spec_hash(text: str) -> str:
    """Stable 8-char ID from spec content."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class UrlPattern:
    """A URL pattern with its CDX match type.

    The CDX API supports four match types:
      - "domain": all URLs under the domain (includes subdomains)
      - "host": exact hostname match
      - "prefix": all URLs starting with the pattern
      - "exact": only the exact URL
    """

    url: str
    match_type: str = "domain"


@dataclass(frozen=True)
class DomainSource:
    """Specifies where HTML data comes from.

    Each URL pattern carries its own CDX match_type. When multiple match types
    are present, the recipe builds separate CDX query + download steps per group
    and combines the results automatically.
    """

    url_patterns: list[UrlPattern]
    crawl_indices: list[str] | None = None
    # Skip CDX query + download and use pre-existing HTML data
    html_data_override: ExecutorStep | InputName | None = None
    # Use DuckDB columnar index instead of CDX HTTP API (much faster for bulk queries)
    use_columnar_cdx: bool = False
    columnar_cdx_workers: int = 10


@dataclass(frozen=True)
class SFTModelSpec:
    """Specifies the model to fine-tune."""

    model_config: LmConfig
    tokenizer: str  # e.g. "Qwen/Qwen3-0.6B"
    hf_model_name: str | None = None  # e.g. "Qwen/Qwen3-0.6B" for initialize_from_hf
    checkpoint_path: str | None = None  # GCS path for initialize_from_checkpoint_path
    pad_tokenizer_to_match_model: bool = False
    short_name: str | None = None  # Inferred from tokenizer if None

    @property
    def name(self) -> str:
        if self.short_name:
            return self.short_name
        # "Qwen/Qwen3-0.6B" -> "qwen3-0.6b"
        return self.tokenizer.split("/")[-1].lower()


@dataclass(frozen=True)
class ExtractionSpec:
    """Specifies an extraction prompt and inference configuration."""

    prompt: str
    model: str  # GCS path or HF name for the extraction model
    model_tokenizer: str  # For token length filtering
    name: str | None = None  # e.g. "qra", "general". Uses spec_hash[:8] if None.
    # Custom post-processed data — if provided, skip extraction + postprocess
    data_override: ExecutorStep | InputName | None = None
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    user_template_fmt: str = DEFAULT_USER_TEMPLATE_FMT
    max_context_tokens: int = 32768
    max_output_tokens: int = 4096
    tensor_parallel_size: int = 4
    tpu_type: str = "v5p-8"
    num_workers: int = 16
    records_per_shard: int = 500

    @property
    def spec_name(self) -> str:
        return self.name or _spec_hash(self.prompt)

    @property
    def spec_hash(self) -> str:
        return _spec_hash(self.prompt)


@dataclass(frozen=True)
class EvalSpec:
    """Specifies evaluation configuration."""

    tasks: list[EvalTaskConfig]
    engine_kwargs: dict = field(default_factory=lambda: {"max_model_len": 4096, "max_gen_toks": 1024})
    resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v5p-8"))


@dataclass(frozen=True)
class TrainHyperparams:
    """Training hyperparameters for single-epoch SFT."""

    batch_size: int = 64
    seq_len: int = 4096
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup: float = 0.03
    decay: float = 0.97
    lr_schedule: str = "cosine"
    max_grad_norm: float = 1.0
    train_tpu_type: str = "v5p-8"


@dataclass(frozen=True)
class BaselineDataset:
    """A pre-existing dataset to include as an SFT branch (e.g. GSM8K)."""

    name: str
    data_step: ExecutorStep | InputName
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ExtractionBranch:
    spec_name: str
    spec_hash: str
    extraction_step: ExecutorStep | None
    postprocess_step: ExecutorStep | None
    tokenize_step: ExecutorStep
    train_step: ExecutorStep
    eval_step: ExecutorStep


@dataclass
class RecipeOutputs:
    domain: str
    # Data acquisition steps (empty lists if html_data_override was used)
    cdx_query_steps: list[ExecutorStep]
    download_steps: list[ExecutorStep]
    combine_step: ExecutorStep | None
    # Shared processing
    filter_html_step: ExecutorStep | None
    resiliparse_step: ExecutorStep
    # Per-extraction-prompt branches
    extraction_branches: list[ExtractionBranch]
    # Shared branches (resiliparse + baselines): name -> (train_step, eval_step)
    shared_branches: dict[str, tuple[ExecutorStep, ExecutorStep]]
    # Baseline eval (no training)
    baseline_eval: ExecutorStep
    # Token count summary
    token_count_step: ExecutorStep

    @property
    def all_eval_steps(self) -> list[ExecutorStep]:
        evals = [self.baseline_eval]
        for branch in self.extraction_branches:
            evals.append(branch.eval_step)
        for _, eval_step in self.shared_branches.values():
            evals.append(eval_step)
        return evals

    @property
    def all_steps(self) -> list[ExecutorStep]:
        return [*self.all_eval_steps, self.token_count_step]


# ---------------------------------------------------------------------------
# Internal: single-epoch SFT training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SFTRunConfig:
    tokenized_path: str
    data_config: LMMixtureDatasetConfig
    output_path: str
    tags: tuple[str, ...]
    # Model spec
    model_config: LmConfig
    seq_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup: float
    decay: float
    lr_schedule: str
    max_grad_norm: float
    # Initialization
    hf_model_name: str | None
    checkpoint_path: str | None
    pad_tokenizer_to_match_model: bool
    # TPU type for training (default v5p-8 for backward compat)
    train_tpu_type: str = "v5p-8"


def _run_single_epoch_sft(config: _SFTRunConfig):
    """Train for exactly 1 epoch, computing num_train_steps from the tokenized cache."""
    total_tokens = _read_token_count(config.tokenized_path, split="train")
    num_train_steps = math.ceil(total_tokens / (config.batch_size * config.seq_len))

    logger.info(f"=== Single-Epoch SFT (tags: {config.tags}) ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {config.batch_size}, Seq len: {config.seq_len}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=list(config.tags),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=config.batch_size,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=min(100, num_train_steps))],
            ),
            allow_nondivisible_batch_size=True,
            initialize_from=None,
        ),
        train_seq_len=config.seq_len,
        model=config.model_config,
        optimizer=AdamConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup=config.warmup,
            decay=config.decay,
            lr_schedule=config.lr_schedule,
            max_grad_norm=config.max_grad_norm,
        ),
        hf_save_steps=num_train_steps,
    )

    # Set initialization source
    if config.hf_model_name:
        inner_config = dataclasses.replace(
            inner_config,
            initialize_from_hf=config.hf_model_name,
            pad_tokenizer_to_match_model=config.pad_tokenizer_to_match_model,
        )
    elif config.checkpoint_path:
        inner_config = dataclasses.replace(
            inner_config,
            initialize_from_checkpoint_path=config.checkpoint_path,
            initialize_from_hf=False,
        )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu(config.train_tpu_type),
        output_path=config.output_path,
    )

    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Internal: token count summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _TokenCountSummaryConfig:
    domain: str
    branch_paths: dict[str, str]  # branch_name -> tokenized_path
    output_path: str


def _write_token_count_summary(config: _TokenCountSummaryConfig):
    """Read token counts from all branches and write a JSON summary."""
    summary: dict[str, dict] = {}
    for branch_name, tokenized_path in config.branch_paths.items():
        try:
            total_tokens = _read_token_count(tokenized_path, split="train")
            summary[branch_name] = {
                "total_tokens": total_tokens,
                "tokenized_path": tokenized_path,
            }
        except Exception as e:
            logger.warning(f"Could not read token count for {branch_name}: {e}")
            summary[branch_name] = {
                "total_tokens": None,
                "tokenized_path": tokenized_path,
                "error": str(e),
            }

    result = {"domain": config.domain, "branches": summary}
    with fsspec.open(os.path.join(config.output_path, "token_counts.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Token count summary for {config.domain}:")
    for name, info in summary.items():
        tokens = info.get("total_tokens")
        if tokens is not None:
            logger.info(f"  {name}: {tokens:,} tokens")
        else:
            logger.info(f"  {name}: ERROR - {info.get('error', 'unknown')}")


# ---------------------------------------------------------------------------
# Internal: combine downloads from multiple CDX match_type groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _CombineDownloadsConfig:
    input_paths: list  # list of InputName — resolved to directory paths by executor
    output_path: str


def _combine_downloads(config: _CombineDownloadsConfig):
    """Merge JSONL.gz files from multiple download directories into one.

    When URL patterns use different CDX match_types, each group gets its own
    CDX query + WARC download step. This function copies all resulting JSONL.gz
    files into a single directory so downstream steps (filtering, extraction)
    see one unified dataset. Files are prefixed with a group index to prevent
    name collisions.
    """
    total = 0
    for group_idx, input_dir in enumerate(config.input_paths):
        source_files = fsspec.open_files(os.path.join(input_dir, "*.jsonl.gz"))
        for src_file in source_files:
            basename = os.path.basename(src_file.path)
            dest = os.path.join(config.output_path, f"g{group_idx}_{basename}")
            with src_file as fin, fsspec.open(dest, "wb") as fout:
                while chunk := fin.read(8 * 1024 * 1024):
                    fout.write(chunk)
            total += 1
        logger.info(f"Group {group_idx}: {len(source_files)} files from {input_dir}")
    logger.info(f"Combined {total} files from {len(config.input_paths)} groups -> {config.output_path}")


# ---------------------------------------------------------------------------
# Internal: build a single SFT branch (tokenize -> train -> eval)
# ---------------------------------------------------------------------------
def _build_sft_branch(
    domain: str,
    branch_name: str,
    data_source: ExecutorStep | InputName,
    sft_model: SFTModelSpec,
    eval_spec: EvalSpec,
    train_params: TrainHyperparams,
    tags: tuple[str, ...],
) -> tuple[ExecutorStep, ExecutorStep, ExecutorStep]:
    """Build tokenize -> train -> eval pipeline for a single data branch.

    Returns (tokenize_step, train_step, eval_step).
    """
    tokenized = default_tokenize(
        name=f"{domain}_{branch_name}_{sft_model.name}_sft",
        dataset=data_source / "**/*.jsonl.gz",
        tokenizer=sft_model.tokenizer,
        format=TextLmDatasetFormat(),
    )

    data_config = lm_data_config(training_set=tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/{domain}-{branch_name}-{sft_model.name}-sft",
        description=f"Single-epoch plaintext SFT: {domain}/{branch_name} ({sft_model.name}).",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=tags,
            model_config=sft_model.model_config,
            seq_len=train_params.seq_len,
            batch_size=train_params.batch_size,
            learning_rate=train_params.learning_rate,
            weight_decay=train_params.weight_decay,
            warmup=train_params.warmup,
            decay=train_params.decay,
            lr_schedule=train_params.lr_schedule,
            max_grad_norm=train_params.max_grad_norm,
            hf_model_name=sft_model.hf_model_name,
            checkpoint_path=sft_model.checkpoint_path,
            pad_tokenizer_to_match_model=sft_model.pad_tokenizer_to_match_model,
        ),
    )

    eval_step = evaluate_lm_evaluation_harness(
        model_name=f"{domain}-{branch_name}-{sft_model.name}-sft",
        model_path=output_path_of(train_step, "hf"),
        evals=eval_spec.tasks,
        engine_kwargs=eval_spec.engine_kwargs,
        resource_config=eval_spec.resource_config,
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )

    return tokenized, train_step, eval_step


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_extraction_sft_experiment(
    domain: str,
    source: DomainSource,
    extractions: list[ExtractionSpec],
    sft_model: SFTModelSpec,
    eval_spec: EvalSpec,
    train_params: TrainHyperparams = TrainHyperparams(),
    baseline_datasets: list[BaselineDataset] | None = None,
) -> RecipeOutputs:
    """Build the full extraction SFT experiment DAG.

    Args:
        domain: Domain name (e.g. "mathhelpforum"). Used for step naming.
        source: Where the HTML data comes from (URLs or override).
        extractions: List of extraction prompts to try.
        sft_model: Model to fine-tune.
        eval_spec: Evaluation configuration.
        train_params: Training hyperparameters.
        baseline_datasets: Optional pre-existing datasets to include as branches.

    Returns:
        RecipeOutputs with all steps in the DAG.
    """
    # ------------------------------------------------------------------
    # Step 1: Data acquisition (CDX query + WARC download, or override)
    # ------------------------------------------------------------------
    cdx_query_steps: list[ExecutorStep] = []
    download_steps: list[ExecutorStep] = []
    combine_step: ExecutorStep | None = None

    if source.html_data_override is not None:
        html_data = source.html_data_override
    else:
        # Group URL patterns by match_type so each CDX query uses one type.
        match_type_groups: dict[str, list[str]] = {}
        for pattern in source.url_patterns:
            match_type_groups.setdefault(pattern.match_type, []).append(pattern.url)

        for match_type in sorted(match_type_groups):
            urls = match_type_groups[match_type]
            # When there's only one group, use clean names (no suffix).
            suffix = f"_{match_type}" if len(match_type_groups) > 1 else ""

            if source.use_columnar_cdx:
                cdx_step = ExecutorStep(
                    name=f"cdx/{domain}{suffix}",
                    description=f"Query CDX columnar index for {domain} URLs (match_type={match_type}).",
                    fn=query_cdx_columnar,
                    config=CDXQueryConfig(
                        url_patterns=versioned(urls),
                        output_path=this_output_path(),
                        crawl_indices=source.crawl_indices,
                        match_type=match_type,
                    ),
                    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
                    pip_dependency_groups=["cpu"],
                    env_vars={"CDX_COLUMNAR_WORKERS": str(source.columnar_cdx_workers)},
                )
            else:
                cdx_step = ExecutorStep(
                    name=f"cdx/{domain}{suffix}",
                    description=f"Query CDX for {domain} URLs (match_type={match_type}).",
                    fn=query_cdx,
                    config=CDXQueryConfig(
                        url_patterns=versioned(urls),
                        output_path=this_output_path(),
                        crawl_indices=source.crawl_indices,
                        match_type=match_type,
                    ),
                    resources=ResourceConfig.with_cpu(cpu=2, ram="8g"),
                    pip_dependency_groups=["cpu"],
                )
            cdx_query_steps.append(cdx_step)

            dl_step = ExecutorStep(
                name=f"downloaded/{domain}{suffix}_html",
                description=f"Download {domain} HTML (match_type={match_type}).",
                fn=download_warc_records,
                config=WarcRecordDownloadConfig(
                    cdx_manifest_path=cdx_step / "cdx_manifest.json",
                    output_path=this_output_path(),
                ),
                resources=ResourceConfig.with_cpu(cpu=4, ram="64g"),
                pip_dependency_groups=["cpu"],
            )
            download_steps.append(dl_step)

        if len(download_steps) == 1:
            html_data = download_steps[0]
        else:
            # Multiple match_type groups — merge all downloads into one directory.
            combine_step = ExecutorStep(
                name=f"downloaded/{domain}_html",
                description=f"Combine {domain} HTML from {len(download_steps)} match_type groups.",
                fn=_combine_downloads,
                config=_CombineDownloadsConfig(
                    input_paths=[output_path_of(s) for s in download_steps],
                    output_path=this_output_path(),
                ),
                resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
                pip_dependency_groups=["cpu"],
            )
            html_data = combine_step

    # ------------------------------------------------------------------
    # Step 2: Filter HTML by token length (shared across extraction specs)
    # ------------------------------------------------------------------
    # Use the first extraction spec's tokenizer for filtering, since they
    # typically share the same model.
    filter_tokenizer = extractions[0].model_tokenizer
    max_context = extractions[0].max_context_tokens
    max_output = extractions[0].max_output_tokens

    filter_html_step: ExecutorStep | None = None
    if source.html_data_override is None:
        filter_html_step = ExecutorStep(
            name=f"filtered/{domain}_html",
            description=f"Filter {domain} HTML documents exceeding {max_context - max_output} tokens.",
            fn=filter_by_token_length,
            config=FilterByTokenLengthConfig(
                input_path=html_data / "*.jsonl.gz",
                output_path=this_output_path(),
                tokenizer=filter_tokenizer,
                text_column="html",
                max_tokens=max_context - max_output,
            ),
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
            pip_dependency_groups=["cpu"],
        )
        extraction_input = filter_html_step
    else:
        # When using html_data_override, the data is already consolidated.
        # We still need filtering for extraction but it may already be done.
        # Build the filter step over the override data.
        filter_html_step = ExecutorStep(
            name=f"filtered/{domain}_html",
            description=f"Filter {domain} HTML documents exceeding {max_context - max_output} tokens.",
            fn=filter_by_token_length,
            config=FilterByTokenLengthConfig(
                input_path=html_data / "*.jsonl.gz",
                output_path=this_output_path(),
                tokenizer=filter_tokenizer,
                text_column="html",
                max_tokens=max_context - max_output,
            ),
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
            pip_dependency_groups=["cpu"],
        )
        extraction_input = filter_html_step

    # ------------------------------------------------------------------
    # Step 3: Resiliparse text extraction (shared baseline)
    # ------------------------------------------------------------------
    resiliparse_step = ExecutorStep(
        name=f"processed/{domain}_resiliparse_text",
        description=f"Extract plain text from {domain} HTML using resiliparse.",
        fn=extract_text_from_html,
        config=ExtractTextConfig(
            input_path=html_data / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )

    # ------------------------------------------------------------------
    # Step 4: Per-extraction-spec branches
    # ------------------------------------------------------------------
    extraction_branches: list[ExtractionBranch] = []
    branch_tokenized_paths: dict[str, str] = {}

    for spec in extractions:
        sid = spec.spec_hash
        spec_name = spec.spec_name

        if spec.data_override is not None:
            # User provides custom post-processed data
            extract_step = None
            pp_step = None
            branch_data = spec.data_override
        else:
            # Run extraction + postprocess
            user_template = spec.user_template_fmt.format(spec=spec.prompt)
            extract_step = ExecutorStep(
                name=f"documents/{domain}_extract_{spec_name}_{sid}",
                description=f"Run extraction on {domain} HTML (spec {spec_name}).",
                fn=run_inference_v2,
                config=InferenceV2Config(
                    input_path=extraction_input / "*.jsonl.gz",
                    output_path=this_output_path(),
                    model_name=spec.model,
                    input_format="jsonl.gz",
                    output_format="jsonl.gz",
                    engine_kwargs={
                        "max_model_len": spec.max_context_tokens,
                        "enable_prefix_caching": True,
                    },
                    generation_kwargs={
                        "temperature": 0.0,
                        "max_tokens": spec.max_output_tokens,
                    },
                    system_message=spec.system_message,
                    template=user_template,
                    prompt_column="html",
                    apply_chat_template=True,
                    max_doc_tokens=spec.max_context_tokens - spec.max_output_tokens,
                    tensor_parallel_size=spec.tensor_parallel_size,
                    tpu_type=spec.tpu_type,
                    num_workers=spec.num_workers,
                    records_per_shard=spec.records_per_shard,
                ),
                pip_dependency_groups=["vllm"],
            )

            pp_step = ExecutorStep(
                name=f"processed/{domain}_extract_{spec_name}_{sid}",
                description=f"Post-process {domain} extraction output (spec {spec_name}).",
                fn=postprocess_extraction,
                config=PostProcessExtractionConfig(
                    input_path=extract_step / "*.jsonl.gz",
                    output_path=this_output_path(),
                ),
                resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
                pip_dependency_groups=["cpu"],
            )
            branch_data = pp_step

        # Build SFT branch: tokenize -> train -> eval
        branch_tags = (domain, spec_name, "extraction", "plaintext", "sft", sft_model.name, "single-epoch")
        tokenized, train_step, eval_step = _build_sft_branch(
            domain=domain,
            branch_name=f"extract-{spec_name}",
            data_source=branch_data,
            sft_model=sft_model,
            eval_spec=eval_spec,
            train_params=train_params,
            tags=branch_tags,
        )

        branch_tokenized_paths[f"extraction-{spec_name}"] = output_path_of(tokenized)

        extraction_branches.append(
            ExtractionBranch(
                spec_name=spec_name,
                spec_hash=sid,
                extraction_step=extract_step,
                postprocess_step=pp_step,
                tokenize_step=tokenized,
                train_step=train_step,
                eval_step=eval_step,
            )
        )

    # ------------------------------------------------------------------
    # Step 5: Resiliparse branch (tokenize -> train -> eval)
    # ------------------------------------------------------------------
    shared_branches: dict[str, tuple[ExecutorStep, ExecutorStep]] = {}

    resiliparse_tags = (domain, "resiliparse", "continued-pretraining", sft_model.name, "single-epoch")
    resiliparse_tokenized, resiliparse_train, resiliparse_eval = _build_sft_branch(
        domain=domain,
        branch_name="resiliparse",
        data_source=resiliparse_step,
        sft_model=sft_model,
        eval_spec=eval_spec,
        train_params=train_params,
        tags=resiliparse_tags,
    )
    shared_branches["resiliparse"] = (resiliparse_train, resiliparse_eval)
    branch_tokenized_paths["resiliparse"] = output_path_of(resiliparse_tokenized)

    # ------------------------------------------------------------------
    # Step 6: Baseline dataset branches
    # ------------------------------------------------------------------
    for baseline in baseline_datasets or []:
        baseline_tags = (
            domain,
            baseline.name,
            "baseline",
            "plaintext",
            "sft",
            sft_model.name,
            "single-epoch",
            *baseline.tags,
        )
        baseline_tokenized, baseline_train, baseline_eval_step = _build_sft_branch(
            domain=domain,
            branch_name=baseline.name,
            data_source=baseline.data_step,
            sft_model=sft_model,
            eval_spec=eval_spec,
            train_params=train_params,
            tags=baseline_tags,
        )
        shared_branches[baseline.name] = (baseline_train, baseline_eval_step)
        branch_tokenized_paths[baseline.name] = output_path_of(baseline_tokenized)

    # ------------------------------------------------------------------
    # Step 7: Baseline eval (no training, just eval the base model)
    # ------------------------------------------------------------------
    if sft_model.hf_model_name:
        baseline_model_path = sft_model.hf_model_name
    elif sft_model.checkpoint_path:
        baseline_model_path = sft_model.checkpoint_path
    else:
        baseline_model_path = sft_model.tokenizer

    baseline_eval = evaluate_lm_evaluation_harness(
        model_name=f"{domain}-{sft_model.name}-baseline",
        model_path=baseline_model_path,
        evals=eval_spec.tasks,
        engine_kwargs=eval_spec.engine_kwargs,
        resource_config=eval_spec.resource_config,
        apply_chat_template=False,
        discover_latest_checkpoint=False,
    )

    # ------------------------------------------------------------------
    # Step 8: Token count summary
    # ------------------------------------------------------------------
    token_count_step = ExecutorStep(
        name=f"summary/{domain}_{sft_model.name}_token_counts",
        description=f"Summarize token counts across all {domain} SFT branches.",
        fn=_write_token_count_summary,
        config=_TokenCountSummaryConfig(
            domain=domain,
            branch_paths=branch_tokenized_paths,
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
        pip_dependency_groups=["cpu"],
    )

    return RecipeOutputs(
        domain=domain,
        cdx_query_steps=cdx_query_steps,
        download_steps=download_steps,
        combine_step=combine_step,
        filter_html_step=filter_html_step,
        resiliparse_step=resiliparse_step,
        extraction_branches=extraction_branches,
        shared_branches=shared_branches,
        baseline_eval=baseline_eval,
        token_count_step=token_count_step,
    )
