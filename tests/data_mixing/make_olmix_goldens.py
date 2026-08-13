# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the olmix reference fixtures in ``tests/data_mixing/golden/``.

Our OLMIX-BASE port is only trustworthy if it reproduces the reference
implementation's numbers, so the parity tests compare against outputs captured
from allenai/olmix's own source. This script captures them.

It is NOT part of the test run: `tests/data_mixing/test_*.py` read the committed
JSON. Re-run this only when intentionally re-pinning against a new olmix commit.

Why the fixtures are captured rather than computed live:

* `pip install olmix` does not work — olmix pins
  `ai2-olmo-core @ git+…@add-inloop-evals`, a branch that no longer exists.
* Its module-level imports pull olmo_core / wandb / gcsfs / s3fs, none of which
  the functions we port actually touch. We stub those and import the real source.
* We call `generate_weights_dirichlet` directly instead of `mk_mixtures`, because
  the committed `configs/mixture_reuse_case_study/generate/0_dclm.yaml` cannot
  load: its `priors` block has 40 domains keyed `dclm:<topic>` while `data.sources`
  lists 24 keyed `<topic>`, so `mk_mixtures`'s `leaf_dist[source.name]` KeyErrors
  and its `domains` list is 40 long against a 24-long weight vector. The sampling
  function is the thing we are porting; their YAML plumbing is not.

Usage (needs the throwaway venv, NOT marin's):

    uv venv /tmp/olmix-ref --python 3.11
    uv pip install --python /tmp/olmix-ref/bin/python \
        numpy pandas tqdm torch cvxpy scikit-learn lightgbm scipy pyyaml pydantic matplotlib
    git clone --depth 1 https://github.com/allenai/olmix /tmp/olmix-src
    /tmp/olmix-ref/bin/python tests/data_mixing/make_olmix_goldens.py \
        --olmix-src /tmp/olmix-src --out tests/data_mixing/golden
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

# The 24 DCLM topic priors and token counts, copied verbatim from olmix's
# configs/mixture_reuse_case_study/generate/0_dclm.yaml (the `dclm:` entries of
# its `priors` block). relative_sizes there are shares of the full 40-domain
# dclm+stack-edu set, so they sum to ~0.85; we renormalize over the 24, which is
# what generate_weights_dirichlet does internally anyway.
DCLM_TOKEN_COUNTS: dict[str, int] = {
    "adult_content": 9986282062,
    "art_and_design": 10413621606,
    "crime_and_law": 25073396423,
    "education_and_jobs": 27219188653,
    "electronics_and_hardware": 11815005112,
    "entertainment": 65106484344,
    "fashion_and_beauty": 5490759782,
    "finance_and_business": 45733158679,
    "food_and_dining": 15612729259,
    "games": 33895620477,
    "health": 57992322702,
    "history_and_geography": 23735036428,
    "home_and_hobbies": 18703739024,
    "industrial": 6421534569,
    "literature": 53768218251,
    "politics": 90076592083,
    "religion": 40937951061,
    "science_math_and_technology": 62949325018,
    "social_life": 32236058743,
    "software": 15922527708,
    "software_development": 32921823893,
    "sports_and_fitness": 28997912992,
    "transportation": 13380851810,
    "travel_and_tourism": 8495229492,
}

# olmix's own proxy token budget: `max_tokens` in 0_dclm.yaml. Equals
# 20 tokens/param x 5x Chinchilla x 29,102,336 params (their olmo2_30m).
OLMIX_PROXY_MAX_TOKENS = 2910233600

# SwarmConfig defaults that 0_dclm.yaml leaves unset.
SEED = 42
VARIANTS = 128
MINIMUM_WEIGHT = 0.05
MIN_STRENGTH = 0.1
MAX_STRENGTH = 5.0
TEMPERATURE = 1.0
SWARM_REPETITION_FACTOR = 1.0


def _force_serial_fitting() -> None:
    """Make olmix's ``ScalingLaw.fit`` take its serial path.

    ``olmix/fit/law.py`` calls ``mp.set_start_method("fork")`` at import and fans the
    300 restarts over ``min(4, mp.cpu_count())`` workers. Forking a process that has
    already initialized torch deadlocks on macOS -- the pool hangs at 0% CPU forever.
    ``workers`` is not reachable through ``LogLinearRegressor.fit``, so we make
    ``cpu_count()`` report 1, which sends ``fit`` down its ``workers == 1`` branch.

    This does not change the result: both branches evaluate every initialization and
    keep the one with the lowest loss. Only the iteration order differs, and that
    matters solely for exact-tie breaking, which float losses do not produce.
    """
    import multiprocessing as mp

    mp.cpu_count = lambda: 1


def _install_import_stubs() -> None:
    """Stub the deps olmix imports at module level but never uses in our code paths."""

    class _NumpyDatasetDType(enum.Enum):
        uint16 = "uint16"
        uint32 = "uint32"

        def as_np_dtype(self):
            return np.dtype(self.value)

    olmo_core = types.ModuleType("olmo_core")
    aliases = types.ModuleType("olmo_core.aliases")
    aliases.PathOrStr = str
    dtypes = types.ModuleType("olmo_core.data.types")
    dtypes.NumpyDatasetDType = _NumpyDatasetDType
    data = types.ModuleType("olmo_core.data")
    data.types = dtypes
    io = types.ModuleType("olmo_core.io")
    io.get_file_size = lambda *a, **k: 0
    io.is_url = lambda p: "://" in str(p)
    io.normalize_path = lambda p: str(p)
    utils = types.ModuleType("olmo_core.utils")

    class OLMoEnvironmentError(RuntimeError):
        pass

    utils.OLMoEnvironmentError = OLMoEnvironmentError

    wandb = types.ModuleType("wandb")
    wandb_apis = types.ModuleType("wandb.apis")
    wandb_public = types.ModuleType("wandb.apis.public")
    wandb_public.Run = object
    wandb.apis = wandb_apis
    wandb_apis.public = wandb_public

    # synthesize_mixture evaluates `s3fs.S3FileSystem()` as a default argument at
    # import time, so the stubs need the class, not just the module.
    s3fs = types.ModuleType("s3fs")
    s3fs.S3FileSystem = type("S3FileSystem", (), {})
    gcsfs = types.ModuleType("gcsfs")
    gcsfs.GCSFileSystem = type("GCSFileSystem", (), {})

    for name, mod in [
        ("olmo_core", olmo_core),
        ("olmo_core.aliases", aliases),
        ("olmo_core.data", data),
        ("olmo_core.data.types", dtypes),
        ("olmo_core.io", io),
        ("olmo_core.utils", utils),
        ("wandb", wandb),
        ("wandb.apis", wandb_apis),
        ("wandb.apis.public", wandb_public),
        ("gcsfs", gcsfs),
        ("s3fs", s3fs),
    ]:
        sys.modules.setdefault(name, mod)


def _golden_swarm(
    gen_weights, source_config_cls, *, max_tokens: int, label: str, repetition_factor: float = SWARM_REPETITION_FACTOR
) -> dict:
    """Capture one `generate_weights_dirichlet` call over the 24 DCLM topics.

    Seeding replicates `mk_mixtures`, which does `random.seed(seed)` then
    `np.random.seed(seed)` immediately before calling the sampler.
    """
    names = sorted(DCLM_TOKEN_COUNTS)
    total = sum(DCLM_TOKEN_COUNTS.values())
    leaf_dist = {n: DCLM_TOKEN_COUNTS[n] / total for n in names}
    leaf_tokens = {n: DCLM_TOKEN_COUNTS[n] for n in names}
    sources = [source_config_cls(name=n, paths=["s3://stub"]) for n in names]

    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002 (olmix RNG parity)
    mixtures = gen_weights(
        sources=sources,
        leaf_dist=leaf_dist,
        minimum_source_weight=MINIMUM_WEIGHT,
        minimum_topic_weight=MINIMUM_WEIGHT,
        num_samples_out=VARIANTS,
        source_temperature=TEMPERATURE,
        topic_temperature=TEMPERATURE,
        min_source_strength=MIN_STRENGTH,
        max_source_strength=MAX_STRENGTH,
        min_topic_strength=MIN_STRENGTH,
        max_topic_strength=MAX_STRENGTH,
        max_tokens=max_tokens,
        leaf_tokens=leaf_tokens,
        repetition_factor=repetition_factor,
        manual_prior=None,
        manual_topic_prior=None,
        sample_multiplier=None,
        enable_bound=True,
        nonzero_weight=None,
        existing_mix_file=None,
    )
    weights = np.stack([m[0] for m in mixtures], axis=0)
    repetitions = np.stack([m[1] for m in mixtures], axis=0)
    return {
        "label": label,
        "domains": names,
        "leaf_tokens": leaf_tokens,
        "params": {
            "seed": SEED,
            "num_samples_out": VARIANTS,
            "minimum_weight": MINIMUM_WEIGHT,
            "min_strength": MIN_STRENGTH,
            "max_strength": MAX_STRENGTH,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "repetition_factor": repetition_factor,
            "enable_bound": True,
            "sample_multiplier": 10,
        },
        "weights": weights.tolist(),
        "repetitions": repetitions.tolist(),
        "nnz_per_mix": [int((w != 0).sum()) for w in weights],
    }


def _golden_fit(log_linear_cls) -> dict:
    """Capture LogLinearRegressor params for a fixed synthetic (X, Y).

    X is a Dirichlet design over 6 domains; Y is generated from a known
    log-linear law so the fit has a well-posed target.
    """
    rng = np.random.default_rng(0)
    n_domains, n_runs = 6, 40
    x = rng.dirichlet(np.ones(n_domains), n_runs)
    true_t = np.array([-1.2, -0.4, 0.15, -0.8, 0.05, -0.25])
    y = (np.exp(-0.7) + np.exp(x @ true_t)).reshape(-1, 1)

    reg = log_linear_cls()
    reg.fit(x, y, idx=0)
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "true_t": true_t.tolist(),
        "params": list(np.asarray(reg.model, dtype=float).tolist()),
        "predictions": np.asarray(reg.predict(x), dtype=float).tolist(),
    }


def _golden_solve(proposer_cls) -> dict:
    """Capture LogLinearExactProposer output, with and without binding caps."""
    n_tasks, n_domains = 5, 6
    rng = np.random.default_rng(1)
    log_c = rng.uniform(-1.0, 0.5, n_tasks)
    t = rng.uniform(-1.5, 0.3, (n_tasks, n_domains))

    class _FittedStub:
        """Stands in for a fitted Regressor: the proposer only reads `.model`."""

        def __init__(self, params):
            self.model = params

    predictors = [_FittedStub([log_c[i], *t[i]]) for i in range(n_tasks)]
    tokens = {f"d{j}": int(v) for j, v in enumerate([2e9, 8e9, 5e8, 3e10, 1e9, 6e9])}
    prior_total = sum(tokens.values())
    prior = {k: v / prior_total for k, v in tokens.items()}

    out = {
        "log_c": log_c.tolist(),
        "t": t.tolist(),
        "token_counts": tokens,
        "prior": prior,
        "kl_reg": 0.05,
        "cases": [],
    }
    # target_tokens=None/unconstrained, then a value where caps bind hard.
    for constrain, target_tokens, rep in [(False, None, 4.0), (True, 20_000_000_000, 4.0)]:
        x = proposer_cls().propose(
            predictor=predictors,
            prior_distributions=prior,
            token_counts=tokens,
            constrain_objective=constrain,
            kl_reg=0.05,
            obj_weights=None,
            target_tokens=target_tokens,
            repetition_factor=rep,
        )
        out["cases"].append(
            {
                "constrain_objective": constrain,
                "target_tokens": target_tokens,
                "repetition_factor": rep,
                "x": np.asarray(x, dtype=float).tolist(),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--olmix-src", required=True, help="Path to a clone of github.com/allenai/olmix")
    parser.add_argument("--out", required=True, help="Output dir for golden JSON")
    args = parser.parse_args()

    _install_import_stubs()
    _force_serial_fitting()
    sys.path.insert(0, os.path.abspath(args.olmix_src))

    import olmix
    from olmix.aliases import SourceConfig
    from olmix.fit.utils import LogLinearExactProposer, LogLinearRegressor
    from olmix.generate.synthesize_mixture import generate_weights_dirichlet

    commit = os.popen(f"git -C {args.olmix_src} rev-parse HEAD").read().strip()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {"olmix_commit": commit, "olmix_version": getattr(olmix, "__version__", "unknown")}

    swarms = [
        _golden_swarm(
            generate_weights_dirichlet,
            SourceConfig,
            max_tokens=OLMIX_PROXY_MAX_TOKENS,
            label="dclm24_proxy_budget_caps_inert",
        ),
        _golden_swarm(
            generate_weights_dirichlet,
            SourceConfig,
            # caps = min(4*N_j/max_tokens, 1) lands in [0.11, 1.0] here: binding for
            # most domains, yet still above minimum_weight so the 0.05 grid is
            # satisfiable. Push max_tokens much higher and every draw is rejected,
            # because a 0.05-grid weight cannot fit under a 0.014 cap.
            max_tokens=200_000_000_000,
            repetition_factor=4.0,
            label="dclm24_caps_bind_k4",
        ),
    ]
    _write(out_dir / "swarm_dclm24.json", {**provenance, "swarms": swarms})
    _write(out_dir / "fit_log_linear.json", {**provenance, **_golden_fit(LogLinearRegressor)})
    _write(out_dir / "solve_exact.json", {**provenance, **_golden_solve(LogLinearExactProposer)})
    print(f"wrote goldens for olmix@{commit} to {out_dir}")


def _write(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
