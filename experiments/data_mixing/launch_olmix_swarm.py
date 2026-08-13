# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive coordinator for one corpus's OLMIX swarm.

Samples all K mixtures once, writes the swarm manifest, then submits children in chunks
and **only scales up when the cluster is actually delivering** -- the pattern from
``baseline_collection/launch_adaptive.py``, reusing its ``_count_child_states``.

Why adaptive rather than a fixed in-flight cap: this sweep runs at *interactive* priority
to preempt the user's own batch queue (a deliberate inversion of the usual
parents-interactive/children-batch rule). Dumping 363 interactive v5p-8 jobs at once would
evict a large slice of that queue in one stroke. Submitting in chunks and waiting for
`pending == 0` releases pressure only as capacity genuinely frees up, at no cost in total
wall clock, and it never permanently gives up -- it backs off with a growing cooldown.

Each child is offered several device variants in one `device_variant_constraint`, so Iris
places it on whichever family has room. They must share (chips, vm_count) -- see
`TPU_SLICE_SHAPES` -- and `--tpu-variant` can widen the pool onto larger, idle slices
(v5p-16, v6e-8), which are gang-scheduled automatically and leave the training math
unchanged because the global batch simply shards wider.

Must run AS AN IRIS JOB, not locally: the parent has to outlive its children or Iris
orphan-kills them.

    iris --cluster marin job run --region us-east5 -e WANDB_API_KEY -e HF_TOKEN \\
        -- python -m experiments.data_mixing.launch_olmix_swarm \\
               --corpus dclm_10k --region us-east5 --launch
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from dataclasses import dataclass

import fsspec
import numpy as np
from iris.client.client import IrisClient, JobAlreadyExists
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device

from experiments.baseline_collection.launch_adaptive import _count_child_states
from experiments.data_mixing.dispatch_watchdog import DispatchState, DispatchWatchdog
from experiments.data_mixing.index_claims import try_claim
from experiments.data_mixing.olmix_domains import (
    domain_cache_dirs,
    domain_tokens,
    grid_tokenizer,
    load_grid_domains,
)
from experiments.data_mixing.olmix_plan import (
    MIXTURE_BLOCK_SIZE,
    SwarmManifest,
    manifest_path,
    read_manifest,
    run_name,
    write_manifest,
)
from experiments.data_mixing.olmix_sample import (
    DEFAULT_SAMPLE_MULTIPLIER,
    minimum_weight_for_m,
    natural_prior,
    sample_flat_dirichlet_swarm,
)
from experiments.data_mixing.run_olmix_swarm_standalone import DEFAULT_RESULTS_PREFIX
from experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep import _gcs_exists, _iris_client
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

SCRIPT = "experiments/data_mixing/run_olmix_swarm_standalone.py"

# K is FIXED at 3(120+1) for every corpus, never derived from m_eff -- holding swarm size
# constant keeps a cross-corpus comparison unconfounded by proxy compute. With m_eff <= 120
# this puts c = K/(m_eff+1) at 3.00-3.05, at or above OLMIX's recommended c >= 3.
SWARM_SIZE = 363

# olmix's literal strength range (every post-stage-0 config). Also the scale-free port of
# their 0.1-5 at m=24: per-domain concentration is strength * p_j, so preserving it means
# strength ~ m, and 0.1-5 at m=24 maps to ~0.5-25 at m=118.
STRENGTH_MIN = 1.0
STRENGTH_MAX = 20.0

# Slice shapes, as (chips, vm_count). Iris requires every variant offered in ONE
# device_variant_constraint to share BOTH numbers, or it mis-counts committed_tpu -- hence
# the homogeneity assert in `slice_shape` rather than a free-form variant list.
#
# Naming is not uniform across families and getting it wrong silently requests the wrong
# hardware: v5p counts TensorCores (v5p-8 = 4 chips), while v6e and v5e count chips
# directly (v6e-4 = 4 chips). All three families put 4 chips on a VM, so vm_count is
# chips // 4.
TPU_SLICE_SHAPES: dict[str, tuple[int, int]] = {
    "v5p-8": (4, 1),
    "v6e-4": (4, 1),
    "v5litepod-4": (4, 1),
    "v5p-16": (8, 2),
    "v6e-8": (8, 2),
    "v5p-32": (16, 4),
    "v6e-16": (16, 4),
    "v5litepod-16": (16, 4),
    "v5litepod-32": (32, 8),
    "v6e-32": (32, 8),
}

# The global batch is PROXY_BATCH_SIZE, so pure data parallelism cannot use more chips than
# that -- beyond it a device would get less than one example. 32 chips is therefore the hard
# ceiling, and it is already the inefficient end: 1 example/device is comms-dominated.
MAX_USEFUL_CHIPS = 32

# Default pool: the 4-chip single-VM slices reachable in the regions holding the grid caches
# (v5p in us-central1-a/us-east5-a, v6e in us-east5-b).
#
# `v4-8` was REMOVED after measurement: v4 exists only in us-central2-b, and every child is
# hard-pinned to its data's region, so the variant could never place. The cluster also had no
# v4-preemptible slices up at all -- just one reserved 2048-chip slice. It was pure noise in
# the constraint.
#
# Larger shapes are opt-in via --tpu-variant. Going to v5p-16 does NOT change the training
# math -- the global batch is fixed at PROXY_BATCH_SIZE and simply shards over 8 chips
# instead of 4, so runs remain directly comparable across slice sizes. That is what makes
# spending idle large slices safe for a controlled sweep.
TPU_VARIANTS = ("v5p-8", "v6e-4")
PRIMARY_TPU = "v5p-8"


def slice_shape(variants: tuple[str, ...]) -> tuple[int, int]:
    """(chips, vm_count) shared by every variant, or raise.

    Mixed shapes in one constraint make Iris mis-count committed TPU, so this refuses rather
    than letting a heterogeneous list through.
    """
    unknown = [v for v in variants if v not in TPU_SLICE_SHAPES]
    if unknown:
        raise ValueError(f"unknown TPU variant(s) {unknown}; known: {sorted(TPU_SLICE_SHAPES)}")
    shapes = {TPU_SLICE_SHAPES[v] for v in variants}
    if len(shapes) != 1:
        raise ValueError(
            f"variants {list(variants)} span shapes {sorted(shapes)}; Iris requires one "
            f"device_variant_constraint to be homogeneous in (chips, vm_count)"
        )
    chips, vms = shapes.pop()
    if chips > MAX_USEFUL_CHIPS:
        raise ValueError(
            f"variants {list(variants)} give {chips} chips, but the global batch is "
            f"{MAX_USEFUL_CHIPS}; a device would receive less than one example"
        )
    return chips, vms


CHILD_CPU = 32.0
CHILD_DISK = "50GB"

# Per-VM RAM differs by TPU family, and asking for more than a family HAS makes the job
# permanently unschedulable rather than merely slow -- the scheduler reports "no matching
# scaling group has enough per-VM capacity" and the coordinator dies on repeated failures.
# Measured against marin.yaml: v5p ct5p-hightpu-4t = 448 GiB, v6e ct6e-standard-4t = 720 GiB,
# v5e ct5lp-hightpu-4t = 192 GiB. The default 256GB silently excluded the ENTIRE v5e family
# (us-west4-a and europe-west4-b) from the reachable pool until this was measured.
# NOTE the v5e figure is ALLOCATABLE, not the VM spec: ct5lp-hightpu-4t advertises 192 GiB
# but the scheduler reports only ~116.7 GB available to a task, so 160GB was still rejected.
# Sized under the observed allocatable rather than the datasheet number.
CHILD_MEMORY_DEFAULT = "256GB"
CHILD_MEMORY_BY_FAMILY = {"v5litepod": "96GB"}


def child_memory(variants: tuple[str, ...]) -> str:
    """RAM to request, capped to what the smallest offered family actually provides."""
    families = {v.split("-")[0] for v in variants}
    limits = {CHILD_MEMORY_BY_FAMILY[f] for f in families if f in CHILD_MEMORY_BY_FAMILY}
    if not limits:
        return CHILD_MEMORY_DEFAULT
    if len(limits) > 1:
        raise ValueError(f"variants {list(variants)} span families with different RAM caps: {sorted(limits)}")
    return limits.pop()


# A submit that fails this many times in a row is a bug, not contention: stop retrying and
# let the coordinator die visibly rather than silently dispatching nothing.
MAX_CONSECUTIVE_SUBMIT_FAILURES = 5

# Pause between submit retries of the same index, so the five allowed attempts span seconds
# rather than hammering a transiently unhappy controller as fast as the loop can spin.
SUBMIT_RETRY_BACKOFF = 5.0

# How many times a run may lose its slice and still be retried. The retry ceiling MUST scale
# with host count: Iris counts TASK failures, and one gang death costs `vm_count` of them, so
# a flat ceiling abandons a 4-host run after a quarter as many actual failures as a 1-host
# run. That is what killed the first multi-host attempt -- 39 dclm and 47 hq indices were
# permanently stranded at a flat 10, which is only ~5 gang deaths for a 2-host slice.
#
# Retrying is cheap because a child resumes from its temp checkpoint (saved ~every 15 min)
# rather than restarting, so a lost slice costs minutes, not the 1.5-4h run. The failures
# are stochastic (mixed exit-1 disconnects and exit-139), not deterministic, so retries
# genuinely converge instead of repeating a doomed attempt.
GANG_DEATHS_TOLERATED = 20


def build_swarm_manifest(
    corpus: str,
    region: str,
    seed: int,
    k: int,
    clip: float | None,
    strength_min: float,
    strength_max: float,
    sample_multiplier: int,
    block_size: int,
    proxy_tokens: int,
) -> SwarmManifest:
    """Sample K mixtures over the corpus's cells and package them as a manifest.

    The swarm is constrained by **physical realizability only** -- `enable_bound=False`
    and an infinite repetition factor. The target run's availability caps (`k*N_j/R`)
    are NOT applied here: paper Table 4 measured that constraining the swarm scores 0.021
    BPB worse than constraining only the optimization, and `k`/`R` are solve-time knobs we
    want to sweep post-hoc from a single swarm.
    """
    domains = load_grid_domains(corpus, region)
    tokens = domain_tokens(domains)
    m = len(domains)
    clip = minimum_weight_for_m(m) if clip is None else clip

    floor = 1.0 / block_size
    if clip * 0.909 < floor:
        raise ValueError(
            f"clip {clip:.3e} is too small for block_size {block_size} (floor {floor:.3e}). "
            f"A surviving weight lands at 0.909-1.00x the clip, so it would truncate to zero "
            f"samples per block. Need clip >= {floor / 0.909:.3e}."
        )

    names, weights, repetitions = sample_flat_dirichlet_swarm(
        prior=natural_prior(tokens),
        tokens=tokens,
        num_samples_out=k,
        minimum_weight=clip,
        max_tokens=proxy_tokens,
        repetition_factor=float("inf"),
        enable_bound=False,
        seed=seed,
        min_strength=strength_min,
        max_strength=strength_max,
        temperature=1.0,
        sample_multiplier=sample_multiplier,
    )
    if names != [d.name for d in domains]:
        raise RuntimeError("sampler returned a different domain order than the domain set")

    manifest = SwarmManifest(
        corpus=corpus,
        region=region,
        seed=seed,
        domains=tuple(names),
        weights=tuple(tuple(float(x) for x in row) for row in weights),
        tokens=tokens,
        cache_dirs=domain_cache_dirs(domains),
        tokenizer=grid_tokenizer(corpus, region),
        sampler={
            "minimum_weight": clip,
            "min_strength": strength_min,
            "max_strength": strength_max,
            "temperature": 1.0,
            "sample_multiplier": sample_multiplier,
            "enable_bound": False,
            "repetition_factor": "inf",
            "proxy_max_tokens": proxy_tokens,
            "block_size": block_size,
            "prior": "natural",
        },
    )
    manifest.assert_trainable(block_size=block_size)

    nnz = (weights != 0).sum(axis=1)
    appeared = (weights != 0).sum(axis=0)
    logger.info(
        "%s: K=%d over m=%d (c=%.2f), clip=%.3e, nnz per mix min/median/max=%d/%d/%d, "
        "domains ever sampled=%d/%d, max proxy repetition=%.0fx",
        corpus,
        k,
        m,
        k / (m + 1),
        clip,
        nnz.min(),
        int(np.median(nnz)),
        nnz.max(),
        int((appeared > 0).sum()),
        m,
        float(np.max(repetitions)),
    )
    return manifest


def _child_env() -> dict[str, str]:
    """Environment forwarded to each child, mirroring ``launch_curation_sweep.submit_one``.

    The coordinator gets these via ``-e`` on its own ``iris job run``; children do NOT
    inherit them, so they must be passed explicitly. Omitting WANDB_API_KEY leaves every
    child unable to log, which makes progress invisible -- and WandB is the only usable
    progress signal while finelog is down.
    """
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError(
            "WANDB_API_KEY is required so children can report progress; pass it to the "
            'coordinator with `-e WANDB_API_KEY "$WANDB_API_KEY"`.'
        )
    env = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        # WandB's 90s default init has been seen to time out from TPU workers.
        "WANDB_INIT_TIMEOUT": "300",
        # Make the TPU compile cache slice-portable so a preempted child RESUMES rather
        # than recompiling from scratch; without it, preemption pressure burns the run.
        "LEVANTER_PORTABLE_TPU_CACHE": "1",
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def submit_child(
    client: IrisClient,
    manifest: SwarmManifest,
    manifest_gcs: str,
    index: int,
    child_priority_band: int,
    extra_args: list[str],
    tpu_variants: tuple[str, ...] = TPU_VARIANTS,
) -> str:
    """Submit one swarm child, hard-pinned to the manifest's region.

    ``replicas`` is derived from the slice shape rather than passed in: a multi-VM slice
    (v5p-16 and up) must be gang-scheduled across exactly its vm_count hosts, and letting a
    caller set that independently is a way to request a shape that cannot be satisfied.
    Levanter handles the multi-host side itself -- the child deliberately does not call
    ``jax.distributed.initialize``, because ``trainer.initialize()`` already does.
    """
    _, vm_count = slice_shape(tpu_variants)
    primary = tpu_variants[0] if PRIMARY_TPU not in tpu_variants else PRIMARY_TPU
    name = run_name(manifest.corpus, manifest.seed, manifest.k, index, manifest.row(index))
    cmd = [
        "python",
        SCRIPT,
        "--manifest",
        manifest_gcs,
        "--index",
        str(index),
        *extra_args,
    ]
    constraints = [
        # HARD region pin: the cells exist only in this region and cross-region reads are
        # the expensive mistake. `_assert_all_components_local`-style checks in the child
        # would catch a mismatch, but only after a TPU was allocated.
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=[manifest.region]),
        device_variant_constraint(list(tpu_variants)),
        preemptible_constraint(True),
    ]
    job_id = client.submit(
        name=name[:200],
        entrypoint=Entrypoint(command=cmd),
        # `extras=["tpu"]` is NOT optional: the base container ships no jax/libtpu, so a
        # child without it cannot train. The coordinator itself runs under `--extra cpu`;
        # extras are per-job, not inherited.
        environment=EnvironmentSpec(env_vars=_child_env(), extras=["tpu"]),
        resources=ResourceSpec(
            device=tpu_device(primary),
            cpu=CHILD_CPU,
            memory=child_memory(tpu_variants),
            disk=CHILD_DISK,
        ),
        replicas=vm_count,
        constraints=constraints,
        priority_band=child_priority_band,
        max_retries_preemption=100,
        max_retries_failure=GANG_DEATHS_TOLERATED * vm_count,
    )
    return str(job_id)


def completed_run_names(corpus: str) -> set[str]:
    """Every run_name already finished for this corpus, across ALL regions.

    Completion records are bucket-local, so a coordinator that checks only its own region
    cannot see a run finished elsewhere and re-dispatches it. That is not hypothetical: after
    ranges were moved between regions, 20 hq indices and 1 dclm index ended up with results in
    two buckets, and roughly 53 of every 57 completions in one measured hour were re-runs of
    already-finished work -- the swarm was mostly recomputing itself.

    One LIST per region rather than an existence check per index: 4 calls instead of 4*K, so
    the startup scan stays fast enough to run before every dispatch decision.
    """
    names: set[str] = set()
    for bucket in set(REGION_TO_BUCKET.values()):
        prefix = f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=corpus)}"
        try:
            fs, root = fsspec.core.url_to_fs(prefix)
            if not fs.exists(root):
                continue
            names |= {p.rsplit("/", 1)[-1][: -len(".json")] for p in fs.ls(root, detail=False) if p.endswith(".json")}
        except Exception as exc:  # a single unreachable region must not hide the others
            logger.warning("could not list completions in %s (%s); continuing", prefix, exc)
    return names


def already_done(manifest: SwarmManifest, index: int, results_prefix: str, done: set[str] | None = None) -> bool:
    """True if this index has finished ANYWHERE. `done` is the cross-region name set."""
    name = run_name(manifest.corpus, manifest.seed, manifest.k, index, manifest.row(index))
    if done is not None:
        return name in done
    return _gcs_exists(f"{results_prefix.rstrip('/')}/{name}.json")


@dataclass
class SwarmTarget:
    """One corpus's swarm: its manifest, where it lives, and where completions are recorded."""

    manifest: SwarmManifest
    manifest_gcs: str
    results_prefix: str
    index_start: int = 0
    index_end: int | None = None
    """Half-open ``[index_start, index_end)`` slice of the swarm this coordinator owns.

    A swarm can be split across regions because the manifest does not depend on region:
    the grid cells carry byte-identical token counts in us-central1 and us-east5 (verified
    for dclm at 7,331,583,923 and high_quality at 21,296,836,881), so the natural prior,
    the sampled weights and hence ``run_name`` are the same wherever it is built. Index
    ``i`` therefore denotes the SAME mixture in either region and the halves reassemble
    into one coherent design.

    What is *not* shared is the completion record: ``results_prefix`` is bucket-local, so
    a coordinator cannot see the other region's finished runs and ``already_done`` will not
    deduplicate across them. Two unbounded coordinators would re-run each other's indices
    and collide on Iris job names, which are global. Disjoint ranges are what make the
    split safe -- they must be assigned by the caller, not inferred.
    """

    def indices(self) -> range:
        end = self.manifest.k if self.index_end is None else self.index_end
        if not 0 <= self.index_start < end <= self.manifest.k:
            raise ValueError(
                f"{self.manifest.corpus}: index range [{self.index_start}, {end}) is not a "
                f"non-empty slice of [0, {self.manifest.k})"
            )
        return range(self.index_start, end)


def _interleave(targets: list[SwarmTarget]) -> list[tuple[SwarmTarget, int]]:
    """Round-robin the outstanding indices across corpora, one from each in turn.

    Submission ORDER is the whole ballgame when several corpora share a cluster. Iris ranks
    pending tasks by ``(priority_band, -depth, root_submitted_ms, submitted_ms)``, so two
    coordinators launched seconds apart are not peers: every child of the older root outranks
    every child of the younger one, no matter when it was submitted. A corpus that starts
    second gets *zero* slots until the first finishes all K runs -- observed live, hq pinned at
    0 running while dclm held 37. Driving both corpora from one coordinator puts every child
    under the same root at the same depth, so the tie breaks on submission time and an
    interleaved order gives each corpus an equal share of whatever capacity exists.
    """
    done_by_corpus = {t.manifest.corpus: completed_run_names(t.manifest.corpus) for t in targets}
    outstanding = [
        [
            (t, i)
            for i in t.indices()
            if not already_done(t.manifest, i, t.results_prefix, done_by_corpus[t.manifest.corpus])
        ]
        for t in targets
    ]
    for target, pending in zip(targets, outstanding, strict=True):
        logger.info(
            "%s: %d of %d runs outstanding in [%d, %d)",
            target.manifest.corpus,
            len(pending),
            target.manifest.k,
            target.indices().start,
            target.indices().stop,
        )
    return [item for group in itertools.zip_longest(*outstanding) for item in group if item is not None]


def run_adaptive_swarm(
    client: IrisClient,
    targets: list[SwarmTarget],
    initial_batch: int,
    chunk_size: int,
    check_interval: int,
    patience: int,
    child_priority_band: int,
    extra_args: list[str],
    tpu_variants: tuple[str, ...] = TPU_VARIANTS,
) -> None:
    """Submit children in chunks, scaling up only when nothing is pending."""
    todo = _interleave(targets)
    if not todo:
        logger.info("nothing to do")
        return

    coordinator_id = os.environ.get("IRIS_JOB_NAME") or f"local-{os.getpid()}"
    submitted: list[str] = []
    cursor = 0
    stalls = 0
    backoff = 1
    last_pending = 0

    # A wedged coordinator is invisible: Iris keeps reporting it RUNNING while its children
    # drain to zero. Only fires when there IS outstanding work and nothing is pending, i.e.
    # when this loop should be dispatching and is not. Needs --max-retries on the job to
    # self-heal; see dispatch_watchdog.
    watchdog = DispatchWatchdog(
        state=lambda: DispatchState(outstanding=cursor < len(todo), pending=last_pending),
    )
    watchdog.start()

    def _submit(n: int) -> int:
        """Submit up to n children, leaving a failed index in place to retry next chunk.

        The cursor advances only on success. Advancing it on failure looks harmless but is
        catastrophic: a systematic submit error (a bad ResourceSpec kwarg, say) burns every
        remaining index in a single call, and the coordinator then drops straight into its
        keep-alive loop reporting "all dispatched" with zero children alive. So a run of
        consecutive failures raises instead -- an unrecoverable bug must be loud.
        """
        nonlocal cursor
        sent = 0
        consecutive_failures = 0
        # Re-read completions before every chunk: `todo` was fixed at startup, so a run
        # finished since then by ANOTHER coordinator would otherwise be dispatched again.
        fresh = {c: completed_run_names(c) for c in {t.manifest.corpus for t, _ in todo}}
        while sent < n and cursor < len(todo):
            target, idx = todo[cursor]
            if already_done(target.manifest, idx, target.results_prefix, fresh[target.manifest.corpus]):
                logger.info("index %d finished elsewhere since startup; skipping", idx)
                cursor += 1
                watchdog.record_progress()
                continue
            name = run_name(
                target.manifest.corpus, target.manifest.seed, target.manifest.k, idx, target.manifest.row(idx)
            )
            # ATOMIC claim before submit. `already_done` above is a cheap filter, but it is
            # check-then-act and therefore racy: two coordinators can both see "not done" and
            # both submit. Only the claim decides. Losing is normal, not an error.
            if not try_claim(target.manifest.corpus, name, owner=coordinator_id):
                logger.info("index %d claimed by another coordinator; skipping", idx)
                cursor += 1
                watchdog.record_progress()
                continue
            try:
                submitted.append(
                    submit_child(
                        client,
                        target.manifest,
                        target.manifest_gcs,
                        idx,
                        child_priority_band,
                        extra_args,
                        tpu_variants,
                    )
                )
            except JobAlreadyExists:
                # A run of this name is already dispatched (an earlier coordinator's child, or
                # a hand-submitted one). Skip it: names are deterministic in the mixture, so
                # the collision IS the run. Not counted toward `sent` -- it adds no new load.
                logger.info("index %d already dispatched under its run name; skipping", idx)
                consecutive_failures = 0
                cursor += 1
                # A skip advances the cursor, so it is forward progress and must reset the
                # watchdog: the invariant is "time since progress", not "time since a
                # successful submit". Without this, a coordinator skipping a long run of
                # indices already dispatched elsewhere -- now the normal case, with three
                # coordinators owning adjacent ranges -- looks identical to a wedge. At ~5s
                # per round-trip a full 726-entry skip exceeds STALL_TIMEOUT and would abort
                # a coordinator that is working correctly.
                watchdog.record_progress()
                continue
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_SUBMIT_FAILURES:
                    raise
                # Deliberately NOT logger.exception: a coordinator was observed wedged for
                # 2h with its main thread parked in this handler while finelog was down, and
                # a full traceback is the largest payload this loop ever hands the logging
                # transport. A one-line message keeps the dispatch loop's liveness from
                # depending on the log backend being up.
                logger.warning("failed to submit index %d (%s: %s); will retry", idx, type(exc).__name__, exc)
                # Retrying the same index immediately just re-hits a transient cluster error
                # faster; back off so five attempts span seconds, not microseconds.
                time.sleep(SUBMIT_RETRY_BACKOFF)
                continue
            consecutive_failures = 0
            cursor += 1
            sent += 1
            watchdog.record_progress()
        return sent

    logger.info("submitting initial batch of %d", min(initial_batch, len(todo)))
    _submit(initial_batch)

    while cursor < len(todo):
        time.sleep(check_interval)
        running, pending, failed = _count_child_states(client, submitted)
        last_pending = pending
        watchdog.record_progress()
        logger.info(
            "%d running, %d pending, %d failed (%d submitted, %d/%d dispatched)",
            running,
            pending,
            failed,
            len(submitted),
            cursor,
            len(todo),
        )
        if pending > 0:
            stalls += 1
            if stalls >= patience:
                cooldown = min(check_interval * backoff, 1800)
                backoff = min(backoff * 2, 6)
                logger.info("stalled %d checks; backing off %ds (not giving up)", stalls, cooldown)
                time.sleep(cooldown)
                stalls = 0
            continue
        stalls = 0
        backoff = 1
        sent = _submit(chunk_size)
        logger.info("all submitted jobs running; dispatched %d more", sent)

    logger.info("all %d runs dispatched; holding to keep children alive", len(todo))
    while True:
        time.sleep(3600)
        running, pending, failed = _count_child_states(client, submitted)
        logger.info("%d running, %d pending, %d failed", running, pending, failed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="Repeatable. Several corpora MUST be driven by one coordinator, not one each: "
        "Iris ranks pending work by root job, so a second coordinator is starved outright.",
    )
    p.add_argument("--region", required=True, choices=sorted(REGION_TO_BUCKET))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swarm-size", type=int, default=SWARM_SIZE)
    p.add_argument("--clip", type=float, default=None, help="minimum_weight; default 1.2/m.")
    p.add_argument("--strength-min", type=float, default=STRENGTH_MIN)
    p.add_argument("--strength-max", type=float, default=STRENGTH_MAX)
    p.add_argument("--sample-multiplier", type=int, default=DEFAULT_SAMPLE_MULTIPLIER)
    p.add_argument("--block-size", type=int, default=MIXTURE_BLOCK_SIZE)
    p.add_argument("--proxy-tokens", type=int, default=32 * 32_844 * 4096)
    p.add_argument("--initial-batch", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--check-interval", type=int, default=300)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--child-priority", default="interactive", choices=sorted(PRIORITY_BAND_MAP))
    p.add_argument("--results-prefix", default=None)
    p.add_argument("--launch", action="store_true", help="Actually submit. Without it, sample and report only.")
    p.add_argument("--rewrite-manifest", action="store_true", help="Re-sample even if a manifest exists.")
    p.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="First swarm index this coordinator owns. Use with --index-end to split one "
        "swarm across regions; ranges MUST be disjoint, since completion records are "
        "bucket-local and cannot deduplicate across regions.",
    )
    p.add_argument("--index-end", type=int, default=None, help="One past the last index owned (default K).")
    p.add_argument(
        "--tpu-variant",
        action="append",
        default=None,
        choices=sorted(TPU_SLICE_SHAPES),
        help="Repeatable. Slice variants a child may land on; ALL must share (chips, vm_count). "
        "Defaults to the 4-chip single-VM pool. Larger shapes (v5p-16, v6e-8) are gang-scheduled "
        "automatically and do not change the training math -- the global batch just shards wider -- "
        "so they exist to spend otherwise-idle large slices.",
    )
    args, extra_args = p.parse_known_args()
    args.tpu_variant = tuple(args.tpu_variant) if args.tpu_variant else TPU_VARIANTS
    chips, vms = slice_shape(args.tpu_variant)
    logger.info("children request %s (%d chips, %d VM(s) gang-scheduled)", list(args.tpu_variant), chips, vms)

    bucket = REGION_TO_BUCKET[args.region]
    if args.results_prefix and len(args.corpus) > 1:
        raise ValueError("--results-prefix is single-corpus only; it would collide across corpora")

    targets: list[SwarmTarget] = []
    for corpus in args.corpus:
        mf_path = manifest_path(bucket, corpus, args.seed, args.swarm_size)
        results_prefix = args.results_prefix or f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=corpus)}"

        if _gcs_exists(mf_path) and not args.rewrite_manifest:
            logger.info("reusing existing manifest %s", mf_path)
            manifest = read_manifest(mf_path)
        else:
            manifest = build_swarm_manifest(
                corpus=corpus,
                region=args.region,
                seed=args.seed,
                k=args.swarm_size,
                clip=args.clip,
                strength_min=args.strength_min,
                strength_max=args.strength_max,
                sample_multiplier=args.sample_multiplier,
                block_size=args.block_size,
                proxy_tokens=args.proxy_tokens,
            )
            if args.launch:
                write_manifest(manifest, mf_path)
            else:
                logger.info("dry run: NOT writing manifest to %s", mf_path)

        logger.info(
            "corpus=%s region=%s K=%d m=%d results=%s",
            corpus,
            args.region,
            manifest.k,
            len(manifest.domains),
            results_prefix,
        )
        targets.append(
            SwarmTarget(
                manifest=manifest,
                manifest_gcs=mf_path,
                results_prefix=results_prefix,
                index_start=args.index_start,
                index_end=args.index_end,
            )
        )

    if not args.launch:
        _interleave(targets)
        logger.info("dry run complete; pass --launch to submit")
        return

    run_adaptive_swarm(
        client=_iris_client(),
        targets=targets,
        initial_batch=args.initial_batch,
        chunk_size=args.chunk_size,
        check_interval=args.check_interval,
        patience=args.patience,
        child_priority_band=PRIORITY_BAND_MAP[args.child_priority],
        extra_args=list(extra_args),
        tpu_variants=tuple(args.tpu_variant),
    )


if __name__ == "__main__":
    main()
