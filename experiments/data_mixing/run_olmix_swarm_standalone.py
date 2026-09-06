# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One OLMIX swarm proxy run. Executes inside a single TPU iris job.

Reads its mixture from the swarm manifest by index, builds a region-local
``LMMixtureDatasetConfig`` over the non-zero cells, trains Levanter in-process, exports
one HF checkpoint at the end, and writes a results JSON + DONE marker.

This is the analogue of ``scaling_law_sweeps/run_curation_train_standalone.py`` for a
120-way grid mixture instead of a single-source corpus, and it deliberately borrows that
file's structure: region detection, region-local assertions, in-process ``train_lm.main``
(a nested iris submit would burn a second TPU), and a summary written only on trustworthy
completion so the coordinator retries otherwise.

Three guards run BEFORE any TPU work, because each one is a silent failure otherwise:

1. every component's ``cache_dir`` must be in the region-local bucket -- cross-region
   reads are the expensive mistake, and a store copied between regions keeps stale
   absolute paths in its artifact (``olmix_domains`` rebases; this re-checks);
2. no non-zero weight may fall below ``1/mixture_block_size`` -- Levanter truncates
   ``int(w * block)`` to zero samples and only ``warnings.warn``s, so the source would
   silently contribute nothing;
3. the mixture must have at least one component and sum to 1.

Usage (normally submitted by ``launch_olmix_swarm.py``; runnable by hand for debug):

    python -m experiments.data_mixing.run_olmix_swarm_standalone \\
        --manifest gs://marin-us-east5/metadata/olmix/dclm_10k/swarm_s42_K363.json \\
        --index 0 --budget 3e18
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LMMixtureDatasetConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import TrainLmOnPodConfig, _prepare_training_run

from experiments.data_mixing.olmix_plan import MIXTURE_BLOCK_SIZE, read_manifest, run_name
from experiments.scaling_law_sweeps import region_tracker
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from experiments.scaling_law_sweeps.curation_plan import METHODS  # noqa: F401 - registry import side effects
from experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep import _resolve_final_step

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PREFIX = "metadata/olmix_swarm_results/{corpus}"
CHECKPOINT_REL = "checkpoints/olmix-swarm/{run_name}"
DONE_MARKER = ".olmix_swarm_DONE"

# Proxy architecture: the d512 cell of the fixed-model natural sweep. See olmix_plan.
PROXY_HIDDEN_DIM = 512
PROXY_NUM_LAYERS = 6
PROXY_NUM_HEADS = 4
PROXY_INTERMEDIATE_DIM = 2048
PROXY_SEQ_LEN = 4096
PROXY_BATCH_SIZE = 32
PROXY_TRAIN_STEPS = 32_844
PROXY_Z_LOSS_WEIGHT = 1e-7


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True, help="gs:// path to the swarm manifest JSON.")
    p.add_argument("--index", type=int, required=True, help="Which mixture in the manifest to train.")
    p.add_argument("--batch-size", type=int, default=PROXY_BATCH_SIZE)
    p.add_argument("--train-steps", type=int, default=PROXY_TRAIN_STEPS)
    p.add_argument("--seq-len", type=int, default=PROXY_SEQ_LEN)
    p.add_argument("--mixture-block-size", type=int, default=MIXTURE_BLOCK_SIZE)
    p.add_argument("--results-prefix", default=None)
    p.add_argument(
        "--run-name-suffix",
        default="",
        help="Appended to the run name, so a smoke or debug run gets its own identity: its own "
        "checkpoint dir, DONE marker and results JSON. Required whenever --train-steps differs "
        "from the proxy config.",
    )
    p.add_argument("--wandb-project", default="marin")
    p.add_argument("--wandb-entity", default="marin-community")
    p.add_argument("--wandb-group", default="olmix-swarm")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tpu-type", default=None)
    p.add_argument("--dry-run", action="store_true", help="Build and validate configs, then exit without training.")
    return p.parse_args(argv)


def build_mixture(
    weights: dict[str, float],
    cache_dirs: dict[str, str],
    tokenizer: str,
    region: str,
    block_size: int,
) -> LMMixtureDatasetConfig:
    """Build the training mixture over the non-zero cells only.

    Zero-weight cells are omitted rather than given weight 0.0: with ~10 non-zero of 118
    that spares ~108 cache opens per run. (The marin convention of weight-0 validation
    components does not apply -- we have no validation components here; the swarm's
    objective is measured by a separate BPB eval pass over the exported checkpoint.)
    """
    if not weights:
        raise ValueError("mixture has no non-zero components")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"mixture weights sum to {total!r}, expected 1.0")

    floor = 1.0 / block_size
    below = {d: w for d, w in weights.items() if w < floor}
    if below:
        raise ValueError(
            f"{len(below)} non-zero weights fall below the block-size floor {floor:.3e} "
            f"(block_size={block_size}); Levanter would truncate them to zero samples per "
            f"block and only warn. Offenders: {sorted(below.items())[:5]}"
        )

    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    components: dict[str, DatasetComponent] = {}
    for name in sorted(weights):
        cache_dir = cache_dirs[name]
        if not cache_dir.startswith(expected_prefix):
            raise ValueError(
                f"component {name!r} has non-local cache_dir {cache_dir!r}; expected prefix "
                f"{expected_prefix!r}. Cross-region reads are forbidden -- run in the store's region."
            )
        components[name] = DatasetComponent(
            source=UrlDatasetSourceConfig(
                cache_dir=cache_dir,
                train_urls=[],
                validation_urls=[],
                format=TextLmDatasetFormat(),
                tags=[name],
            ),
            cache_dir=cache_dir,
            format=TextLmDatasetFormat(),
            tags=[name],
        )

    return LMMixtureDatasetConfig(
        tokenizer=tokenizer,
        components=components,
        train_weights={name: weights[name] for name in sorted(weights)},
        mixture_block_size=block_size,
        shuffle=True,
        permutation_type="feistel",
    )


def _model_config(hidden_dim: int, num_layers: int, num_heads: int, intermediate_dim: int, seq_len: int) -> Qwen3Config:
    return Qwen3Config(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _train_lm_config(args, mixture, output_path: str, tags: list[str]):
    from levanter.main import train_lm

    tokens = args.batch_size * args.train_steps * args.seq_len
    return train_lm.TrainLmConfig(
        data=mixture,
        trainer=TrainerConfig(
            tracker=WandbConfig(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, tags=tags),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            seed=args.seed,
            train_batch_size=args.batch_size,
            per_device_parallelism=-1,
            num_train_steps=args.train_steps,
            steps_per_eval=args.train_steps + 1,  # no in-loop eval: BPB is a separate pass
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
            # Rolling 15-min temp checkpoint for preemption recovery, no permanent
            # intermediates. `delete_old_temp_checkpoints=False` lets an iris retry resume
            # from the crashed attempt's temp instead of restarting at step 0.
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=15), keep=[], delete_old_temp_checkpoints=False
            ),
        ),
        train_seq_len=args.seq_len,
        model=_model_config(PROXY_HIDDEN_DIM, PROXY_NUM_LAYERS, PROXY_NUM_HEADS, PROXY_INTERMEDIATE_DIM, args.seq_len),
        optimizer=completed_adamh_heuristic.build_optimizer_config(args.batch_size, tokens),
        z_loss_weight=PROXY_Z_LOSS_WEIGHT,
        # ONE HF export, at the final step. The separate-eval-pass design needs a
        # checkpoint the BPB harness can load; `hf_save_steps == num_train_steps` gives
        # exactly one at the end rather than a rolling series.
        hf_save_path=f"{output_path}/hf",
        hf_save_steps=args.train_steps,
    )


def _write_summary(summary: dict, results_prefix: str, name: str) -> None:
    path = f"{results_prefix.rstrip('/')}/{name}.json"
    with fsspec.open(path, "w") as fh:
        fh.write(json.dumps(summary, indent=2))
    logger.info("wrote summary %s", path)


def assert_distinct_identity(train_steps: int, run_name_suffix: str) -> None:
    """A partial-length run must not borrow a swarm member's identity.

    Sharing the run name means sharing the checkpoint dir -- so the real member later *resumes*
    from a short run's optimizer state, shaped by an LR schedule that decayed to zero at the
    wrong step -- and sharing the results JSON, so the coordinator's skip-if-done retires that
    member for good. Both are silent; only the run name distinguishes them.
    """
    if train_steps != PROXY_TRAIN_STEPS and not run_name_suffix:
        raise ValueError(
            f"--train-steps {train_steps} != the proxy config's {PROXY_TRAIN_STEPS}, so this is not "
            f"a swarm run. Pass --run-name-suffix (e.g. -smoke) to give it its own checkpoint path, "
            f"DONE marker and results JSON."
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    assert_distinct_identity(args.train_steps, args.run_name_suffix)

    manifest = read_manifest(args.manifest)
    weights = manifest.row(args.index)
    name = run_name(manifest.corpus, manifest.seed, manifest.k, args.index, weights) + args.run_name_suffix

    region = region_tracker.detect_current_region() if not args.dry_run else manifest.region
    if region != manifest.region:
        raise RuntimeError(
            f"running in {region} but the manifest's cells are in {manifest.region}. "
            f"Cross-region reads are forbidden -- submit this run pinned to {manifest.region}."
        )
    bucket = region_tracker.REGION_TO_BUCKET[region]
    output_path = f"{bucket}/{CHECKPOINT_REL.format(run_name=name)}"
    results_prefix = args.results_prefix or f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=manifest.corpus)}"

    mixture = build_mixture(weights, manifest.cache_dirs, manifest.tokenizer, region, args.mixture_block_size)
    tokens = args.batch_size * args.train_steps * args.seq_len
    logger.info(
        "%s: %d/%d non-zero cells, %d tokens over %d steps (batch %d, block %d), region %s",
        name,
        len(weights),
        len(manifest.domains),
        tokens,
        args.train_steps,
        args.batch_size,
        args.mixture_block_size,
        region,
    )
    if args.dry_run:
        logger.info("dry run: configs built and validated, exiting before training")
        return

    tags = [
        "olmix-swarm",
        f"corpus={manifest.corpus}",
        f"index={args.index}",
        f"K={manifest.k}",
        f"seed={manifest.seed}",
        f"nnz={len(weights)}",
        f"d_model={PROXY_HIDDEN_DIM}",
        f"block_size={args.mixture_block_size}",
    ]
    pod_config = TrainLmOnPodConfig(
        train_config=_train_lm_config(args, mixture, output_path, tags),
        resources=ResourceConfig.with_tpu(args.tpu_type or os.environ.get("MARIN_TPU_TYPE", "v5p-8")),
        output_path=output_path,
        env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=16000"},
    )
    _prepared, train_config_ready, env = _prepare_training_run(pod_config)
    for k, v in env.items():
        os.environ[k] = v

    # In-process, not a nested iris submit: we are already on the TPU worker the
    # coordinator allocated. Levanter's trainer.initialize() handles jax.distributed.
    train_lm_module = importlib.import_module("levanter.main.train_lm")
    logger.info("launching levanter.main.train_lm.main() in-process")
    train_lm_module.main(train_config_ready)
    logger.info("training finished cleanly")

    _write_summary(
        {
            "run_name": name,
            "corpus": manifest.corpus,
            "region": region,
            "manifest": args.manifest,
            "index": args.index,
            "seed": manifest.seed,
            "k": manifest.k,
            "weights": weights,
            "n_nonzero": len(weights),
            "tokens_trained": tokens,
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "seq_len": args.seq_len,
            "mixture_block_size": args.mixture_block_size,
            "hidden_dim": PROXY_HIDDEN_DIM,
            "output_path": output_path,
            # Resolved from GCS, not computed from train_steps: levanter's final export lands
            # at the last *step index*, i.e. `step-{train_steps - 1}`, so a computed path is
            # off by one and points at a directory that does not exist.
            "hf_checkpoint": _resolve_final_step(f"{output_path}/hf/"),
        },
        results_prefix,
        name,
    )
    with fsspec.open(f"{output_path}/{DONE_MARKER}", "w") as fh:
        fh.write(json.dumps({"run_name": name, "index": args.index}))


if __name__ == "__main__":
    main()
