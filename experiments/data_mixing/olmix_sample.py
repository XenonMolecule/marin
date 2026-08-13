# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Swarm mixture sampling, ported from olmix's ``generate_weights_dirichlet``.

Step 1 of OlmixBase (Algorithm 1 of arXiv:2602.12237): draw ``K`` mixtures over
``m`` domains from a Dirichlet centred on the natural (token-count) distribution.

Faithfulness
------------
This is a port of ``olmix/generate/synthesize_mixture.py::generate_weights_dirichlet``
restricted to the **flat domain** case: every domain is a top-level
``SourceConfig`` with no ``topics``/``quality`` children and no fixed ``weight``.
That restriction is not a simplification of the method — it is the exact code path
olmix's own DCLM topic-mixing config takes (``configs/mixture_reuse_case_study/
generate/0_dclm.yaml`` declares 24 flat sources), and it is what our setting
reduces to. A corpus's 24x5 grid handed to olmix as one source with 24 topics each
carrying 5 quality buckets produces ``source_distribution == [1.0]`` and a single
120-leaf ``topic_distributions[source]``, i.e. mathematically the same flat
Dirichlet over cells, just driven by the topic-level knobs instead of the
source-level ones. We take the flat path because that is the one the golden
fixtures pin (see ``tests/data_mixing/test_olmix_sample_parity.py``).

Deliberately not ported: hierarchical source/topic/quality priors, ``manual_prior``
/ ``manual_topic_prior`` (mixture *reuse*, paper §4 — we do full recomputation),
fixed source weights, ``nonzero_weight``, and ``existing_mix_file``. Each raises
rather than silently doing nothing.

RNG parity
----------
``sample_flat_dirichlet_swarm`` reproduces the reference's *sequence* of RNG calls,
not merely its distribution, so the port can be pinned bit-for-bit. Two
consequences that look like dead code but are not:

* the per-domain ``np.random.dirichlet(np.array([1.0]) * strength, 1)`` draws. For a
  flat source the reference sets ``topic_priors[d] = np.array([1.0])``, so each draw
  is exactly ``[[1.0]]`` and contributes nothing to the mixture — but it consumes a
  gamma variate, and removing it desynchronises every subsequent draw.
* the interleaving: all 15 source-level draws happen first, then 15 per domain.

Do not "optimise" either away.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

# olmix ConfigDefaults (generate/synthesize_mixture.py).
DEFAULT_MIN_STRENGTH = 0.1
DEFAULT_MAX_STRENGTH = 5.0
DEFAULT_SAMPLE_MULTIPLIER = 10

# olmix hardcodes a 15-point strength grid; one point is chosen uniformly at random
# per accepted draw, which is what spreads the swarm across concentrations.
STRENGTH_GRID_POINTS = 15

# sort_and_deduplicate's atol, and the size at which it switches to the hashing variant.
DEDUP_ATOL = 1e-5
DEDUP_HASH_THRESHOLD = 10_000


def natural_prior(tokens: dict[str, int]) -> dict[str, float]:
    """The natural distribution p0: token counts normalized to sum to 1."""
    total = sum(tokens.values())
    if total <= 0:
        raise ValueError("total token count must be positive")
    return {name: count / total for name, count in tokens.items()}


# olmix's published `minimum_weight` values, for calibration. There is no value for a
# flat domain set larger than 24: every later stage of their case study recomputes only
# 2-7 dimensions because mixture reuse collapses the rest.
OLMIX_CLIP_TOPIC_LEVEL = 0.05  # 0_dclm.yaml, m=24; paper §A.3's sparse-swarm construction
OLMIX_CLIP_DEFAULT = 0.002  # ConfigDefaults.minimum_weight; examples/*.yaml at m=5
OLMIX_CLIP_PARTITION = 0.0001  # 5_partition_pdfs / 1_add_stackedu / partial_mixture_reuse


def minimum_weight_for_m(m: int, *, clip_to_mean_ratio: float = 1.2) -> float:
    """A clip that holds olmix's clip-to-mean-weight ratio fixed as ``m`` grows.

    **This is our construction, not olmix's, and it is not the only defensible
    choice.** It reproduces their topic-level 0.05 exactly at m=24 (where the mean
    weight 1/24 = 0.0417, so the clip sits at 1.2x the mean) and gives 0.01 at m=120.

    Read their published values before trusting it (:data:`OLMIX_CLIP_TOPIC_LEVEL` /
    :data:`OLMIX_CLIP_DEFAULT` / :data:`OLMIX_CLIP_PARTITION`): the clip *decreases*
    from 0.05 to 0.0001 as their domain set grows from 24 to 64, which no fixed
    multiple of the mean weight reproduces. It tracks the paper's RQ3 topic-vs-source
    axis instead -- sparse (0.05) for topics, effectively dense (0.0001) for sources
    -- and they never publish a value for a flat set larger than 24. Their closest
    analogue to a quality x topic grid is the Partition operator, which chooses dense.

    Measured on dclm's 118 non-empty cells, 1.2/m = 0.01 is nonetheless the right
    operating point, for a reason unrelated to the mean-weight analogy. What matters is
    whether the swarm matrix identifies each domain's coefficient: normalizing
    ``1/sigma_min(W - mean)`` against a ``Dirichlet(1)`` design of the same ``(K, m)``
    (olmix's own m=24 swarm scores 0.91x on that measure), at ``strength`` 1-20 the clip
    trades coverage against conditioning as

        0.05 -> 56% of domains ever sampled, 0.85x ideal
        0.01 -> 64%, 3.56x ideal          <-- chosen
        0.002 -> 65%, 36x ideal
        0.0005 -> 68%, 585x ideal

    so 0.01 buys 8 points of coverage over 0.05 cheaply, while 0.002 pays 10x worse
    conditioning for one more point. Keep ``strength`` at olmix's literal 1-20: it is the
    scale-free port of their 0.1-5 at m=24, since per-domain concentration is
    ``strength * p_j`` and preserving it means ``strength ∝ m`` (0.1-5 at m=24 maps to
    0.5-25 at m=118). Calibrate per corpus; a thinner tail moves these numbers.

    This value also sets the *quantization grid* -- the reference snaps surviving
    weights to multiples of it -- so it caps the number of non-zero domains at
    ``1/minimum_weight``, and it must stay above the training stack's own floor of
    ``1/mixture_block_size`` (see ``levanter.data.mixture``).
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    return clip_to_mean_ratio / m


def sort_and_deduplicate(
    samples: list[tuple[np.ndarray, np.ndarray]], threshold: float = DEDUP_ATOL
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop near-identical mixtures so the swarm does not retrain the same mix.

    Port of olmix's ``sort_and_deduplicate`` / ``sort_and_deduplicate_with_hash``.
    The reference switches to the hashing variant above 10k samples purely for
    speed, but the two are NOT equivalent (pairwise ``allclose`` vs. rounding to a
    grid), so the threshold is part of the behaviour and is reproduced here.
    """
    if len(samples) > DEDUP_HASH_THRESHOLD:
        unique: list[tuple[np.ndarray, np.ndarray]] = []
        seen: set[tuple[int, ...]] = set()
        for sample in samples:
            rounded = tuple(np.round(sample[0] / threshold).astype(int))
            if rounded not in seen:
                seen.add(rounded)
                unique.append(sample)
        return unique

    unique = []
    for sample in samples:
        if not any(np.allclose(sample[0], other[0], atol=threshold) for other in unique):
            unique.append(sample)
    return unique


def sample_flat_dirichlet_swarm(
    *,
    prior: dict[str, float],
    tokens: dict[str, int],
    num_samples_out: int,
    minimum_weight: float,
    max_tokens: int,
    repetition_factor: float,
    seed: int,
    min_strength: float = DEFAULT_MIN_STRENGTH,
    max_strength: float = DEFAULT_MAX_STRENGTH,
    temperature: float = 1.0,
    sample_multiplier: int = DEFAULT_SAMPLE_MULTIPLIER,
    enable_bound: bool = True,
    rng_parity: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw ``num_samples_out`` mixtures over the domains of ``prior``.

    Args:
        prior: natural distribution p0 over domains. **Iteration order defines the
            domain order** of every array returned here, and must match the column
            order used by the fit and the solver -- olmix keys priors, caps and the
            regression design matrix positionally off this same order.
        tokens: available tokens per domain, same keys as ``prior``.
        num_samples_out: K. The reference draws ``K * sample_multiplier`` candidates
            and samples K survivors, so it raises rather than returning fewer.
        minimum_weight: sparsity clip AND quantization grid (see
            :func:`minimum_weight_for_m`).
        max_tokens: the *proxy* run's token budget, used only by the repetition
            bookkeeping below -- not the target model's budget.
        repetition_factor: reject a draw needing more than this many epochs of any
            domain at ``max_tokens``. Pass ``float("inf")`` with
            ``enable_bound=False`` for a genuinely unconstrained swarm, which is
            what OlmixBase recommends w.r.t. the *target* budget (paper Table 4:
            constraining the swarm scored 0.0208 BPB worse than constraining only
            the optimization).
        temperature: only applied when < 1.0, matching the reference, so the
            default 1.0 is a no-op.
        enable_bound: apply per-domain availability caps
            ``min(tokens * repetition_factor / max_tokens, 1)`` during sampling.
        rng_parity: reproduce the reference's exact RNG call sequence, including its
            degenerate per-domain draws. Required to match the olmix goldens (which are
            m=24) and therefore the default. Costs ``m * 15`` extra Dirichlet calls per
            candidate -- ~2.2M calls for a single m=120 swarm -- so set False for
            calibration sweeps at large m, where there is no golden to match anyway.
            The sampled *distribution* is unchanged; only the stream differs, so
            results are not comparable across this flag.

    Returns:
        ``(domains, weights, repetitions)`` -- ``weights`` is ``(K, m)`` summing to
        1 per row, ``repetitions`` is ``(K, m)`` with the reference's
        ``max(1, ceil(needed/available * 1000)/1000)`` per domain.
    """
    if set(prior) != set(tokens):
        raise ValueError(
            f"prior and tokens must cover the same domains; symmetric difference: {set(prior) ^ set(tokens)}"
        )
    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError(f"minimum_weight must be in (0, 1], got {minimum_weight}")

    domains = list(prior)
    m = len(domains)
    prior_vec = np.array([prior[d] for d in domains], dtype=float)
    prior_vec = prior_vec / prior_vec.sum()
    token_vec = np.array([tokens[d] for d in domains], dtype=float)

    if temperature < 1.0:
        prior_vec = prior_vec**temperature
        prior_vec = prior_vec / prior_vec.sum()

    caps = np.minimum(token_vec * repetition_factor / max_tokens, 1.0) if enable_bound else None
    if caps is not None and caps.sum() < 1.0:
        raise ValueError(
            f"availability caps sum to {caps.sum():.4f} < 1: no mixture can satisfy them. "
            f"Raise repetition_factor ({repetition_factor}) or lower max_tokens ({max_tokens:,})."
        )

    # Match mk_mixtures: seed both RNGs immediately before sampling. `random` drives
    # the candidate choice and the final subsample; numpy drives the Dirichlet draws.
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (olmix RNG parity)

    strengths = np.logspace(np.log10(min_strength), np.log10(max_strength), STRENGTH_GRID_POINTS)
    degenerate_prior = np.array([1.0])

    collected: list[tuple[np.ndarray, np.ndarray]] = []
    n_no_candidates = 0
    n_bound_rejected = 0
    n_repetition_rejected = 0
    n_all_clipped = 0

    for _ in range(num_samples_out * sample_multiplier):
        # One draw per strength grid point, at the source level.
        source_samples = [np.random.dirichlet(prior_vec * strength, 1) for strength in strengths]
        # RNG-parity draws: degenerate for flat domains, but they advance the stream.
        if rng_parity:
            for _domain in domains:
                for strength in strengths:
                    np.random.dirichlet(degenerate_prior * strength, 1)

        candidates = [s.reshape(1, -1) for s in source_samples]

        if caps is not None:
            candidates = [c for c in candidates if bool(np.all((c[0] >= 0.0) & (c[0] <= caps)))]
        if not candidates:
            n_no_candidates += 1
            continue

        chosen = random.choice(candidates)

        # Clip below the minimum, renormalize, snap to the minimum-weight grid,
        # renormalize again. The snap is what makes the swarm's support coarse.
        chosen = np.where(chosen < minimum_weight, 0, chosen)

        # DIVERGENCE FROM THE REFERENCE, and a deliberate one. If every weight falls
        # below the clip, the reference divides by zero, produces an all-NaN mixture,
        # sails through its bounds re-check (NaN comparisons are False), and then dies
        # in `int(weight * max_tokens)` with "cannot convert float NaN to integer".
        # It never fires in their setting because a Dirichlet at concentration 0.1-5
        # over 24 domains is spiky enough that something always clears 0.05 -- but the
        # hazard grows with m (the mean weight 1/m sinks below the clip) and with
        # `strength` (draws become more uniform). At m=120 a *well-spread* draw is
        # entirely below 0.05. Reject the candidate instead of crashing.
        if not np.any(chosen > 0):
            n_all_clipped += 1
            continue

        chosen = chosen / np.sum(chosen).reshape(-1, 1)
        chosen = np.round(chosen / minimum_weight) * minimum_weight
        chosen = chosen / np.sum(chosen)

        # Renormalization can push a domain back over its cap.
        if caps is not None and bool(np.any((chosen[0] < 0.0) | (chosen[0] > caps))):
            n_bound_rejected += 1
            continue

        weights = chosen[0]
        repetitions = np.ones(weights.shape[0])
        reject = False
        for idx in range(m):
            required = int(weights[idx] * max_tokens)
            repetition = np.ceil(required / token_vec[idx] * 1000) / 1000
            if repetition > repetition_factor:
                reject = True
                break
            repetitions[idx] = max(1.0, repetition)
        if reject:
            n_repetition_rejected += 1
            continue

        collected.append((weights, repetitions))

    if not collected:
        raise ValueError(
            "No valid mixtures were sampled. Rejections: "
            f"{n_no_candidates} no-candidate, {n_bound_rejected} out-of-bounds, "
            f"{n_repetition_rejected} over-repetition, {n_all_clipped} fully-clipped. "
            f"A minimum_weight ({minimum_weight}) above the smallest availability cap "
            f"makes every draw infeasible; one above the mean weight (1/m = "
            f"{1.0 / m:.2e}) fully clips any well-spread draw."
        )

    deduped = sort_and_deduplicate(collected)
    if len(collected) < num_samples_out:
        raise ValueError(
            f"Collected only {len(collected)} mixtures, need {num_samples_out}. "
            f"Raise sample_multiplier (currently {sample_multiplier})."
        )
    if len(deduped) < num_samples_out:
        raise ValueError(
            f"Only {len(deduped)} of {len(collected)} sampled mixtures are distinct, "
            f"need {num_samples_out}. The minimum_weight grid ({minimum_weight}) is likely "
            "too coarse for this many domains."
        )

    selected = random.sample(deduped, num_samples_out)
    weights = np.stack([s[0] for s in selected], axis=0)
    repetitions = np.stack([s[1] for s in selected], axis=0)
    logger.info(
        "sampled %d mixtures over %d domains (rejected: %d no-candidate, %d out-of-bounds, "
        "%d over-repetition, %d fully-clipped); non-zero domains per mix: min=%d median=%d max=%d; "
        "domains never sampled: %d",
        num_samples_out,
        m,
        n_no_candidates,
        n_bound_rejected,
        n_repetition_rejected,
        n_all_clipped,
        (weights != 0).sum(axis=1).min(),
        int(np.median((weights != 0).sum(axis=1))),
        (weights != 0).sum(axis=1).max(),
        int(((weights != 0).sum(axis=0) == 0).sum()),
    )
    return domains, weights, repetitions
