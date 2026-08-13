# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The log-linear data mixing law and its fitting loop.

Step 2 of OlmixBase: for each downstream task ``i``, fit

    f_i(p) = exp(log c_i) + exp(t_i . p)

where ``p`` is the mixture (raw proportions, not logs) and BPB is the untransformed
target. OlmixBase fits **one such model per task** and then averages the
*predictions* -- per-task granularity beat per-family and aggregated on both
regression fit and downstream BPB (paper RQ5, Table 3).

Ported from ``olmix/fit/law.py`` (itself adapted from Ye et al.'s mixinglaws repo)
and ``olmix/fit/utils.py::mixing_law`` / ``init_params_log_linear_law`` /
``LogLinearRegressor``. The numerics that matter and are reproduced exactly:

* **300 restarts** per task: 10 ``log_c`` seeds x 30 random ``t`` draws. Each draw
  initializes ``t`` negative for the task's own index and small-positive elsewhere.
* **summed Huber loss, delta=0.02** (not mean -- ``reduction="sum"``).
* **LBFGS**(lr=0.01, history_size=10, max_iter=20, line_search_fn="strong_wolfe").
* ``max_step=100`` outer steps, breaking the first time the loss fails to improve
  (the reference's ``eps=0.0`` compares ``abs(eval_loss - min_loss) < eps``, which
  is 0 exactly when nothing improved).
* ``valid_split=0``, so model selection scores on the training data itself (mean-
  reduction Huber), and the best restart is chosen by that score.
* seeds: ``np.random.seed(42)`` and ``random.seed(42)`` per regressor construction,
  which is what makes the restart draws deterministic.

Deliberately not ported: the reference's ``mp.Pool`` fan-out over restarts. It calls
``mp.set_start_method("fork")``, which deadlocks with an initialized torch on macOS,
and both branches of the reference select the same minimum-loss restart. We keep the
serial loop and expose ``max_workers`` as a no-op-free explicit argument instead.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)

# olmix LogLinearRegressor.__init__ seeds; the restart draws depend on them.
FIT_SEED = 42

# init_params_log_linear_law's grids.
LOG_C_GRID = (-2.0, 1.5, 10)  # np.linspace bounds and count
RESTARTS_PER_LOG_C = 30

# fit_scaling_laws / LogLinearRegressor.fit numerics.
HUBER_DELTA = 0.02
LBFGS_LR = 0.01
LBFGS_HISTORY = 10
LBFGS_MAX_ITER = 20
MAX_STEP = 100
EARLY_STOPPING_EPS = 0.0


def mixing_law(x: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """``exp(log c) + exp(x . t)``, with ``param = [log_c, t_1..t_m]``."""
    log_c = param[0]
    t = param[1:]
    return torch.exp(log_c) + torch.exp(torch.matmul(x, t))


def init_params_log_linear_law(idx: int, num_domains: int) -> Iterator[list[float]]:
    """Yield the 300 restart initializations for task ``idx``.

    ``idx`` is the task's column index, used by the reference as a *domain* index to
    seed one coefficient negative. That cross-use only lines up when the task and
    domain counts are similar, but it is what the reference does and it is only an
    initialization, so it cannot bias the fitted optimum -- only which local optimum
    a given restart falls into. Reproduced rather than "corrected" so our fits match.
    """
    for log_c in np.linspace(*LOG_C_GRID):
        for _ in range(RESTARTS_PER_LOG_C):
            t = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            yield [log_c, *t]


def _fit_one_restart(x: torch.Tensor, y: torch.Tensor, init_param: list[float]) -> tuple[float, np.ndarray]:
    """Run LBFGS from one initialization; return ``(selection_loss, params)``."""
    param = torch.nn.Parameter(torch.tensor(init_param, dtype=torch.float32))
    x = x.to(param)
    y = y.to(param)
    optimizer = torch.optim.LBFGS(
        [param],
        lr=LBFGS_LR,
        history_size=LBFGS_HISTORY,
        max_iter=LBFGS_MAX_ITER,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.huber_loss(y, mixing_law(x, param), delta=HUBER_DELTA, reduction="sum")
        loss.backward()
        return loss

    min_loss, best_param = 1e10, None
    for _step in range(MAX_STEP):
        optimizer.step(closure)
        with torch.no_grad():
            # valid_split=0 in the reference: selection scores on the train split,
            # with the DEFAULT (mean) reduction -- not the summed loss being optimized.
            eval_loss = torch.nn.functional.huber_loss(mixing_law(x, param), y, delta=HUBER_DELTA).item()
        improvement = abs(eval_loss - min_loss)
        if eval_loss <= min_loss:
            min_loss = eval_loss
            best_param = param.detach().clone()
        if improvement < EARLY_STOPPING_EPS or improvement == 0.0:
            break

    assert best_param is not None
    return min_loss, best_param.detach().cpu().numpy()


def fit_log_linear(x: np.ndarray, y: np.ndarray, idx: int) -> np.ndarray:
    """Fit ``f_idx(p) = exp(log c) + exp(t . p)`` to column ``idx`` of ``y``.

    Args:
        x: ``(K, m)`` mixture weights, one row per swarm run.
        y: ``(K, n_tasks)`` measured BPB.
        idx: which task column to fit.

    Returns:
        ``[log_c, t_1..t_m]`` -- the best of 300 restarts.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2-D, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y disagree on run count: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < x.shape[1] + 1:
        raise ValueError(
            f"{x.shape[0]} swarm runs cannot identify {x.shape[1] + 1} parameters; "
            f"the log-linear law needs at least m+1 = {x.shape[1] + 1} runs for a unique solution."
        )

    # Per-regressor seeding, as in LogLinearRegressor.__init__.
    np.random.seed(FIT_SEED)  # noqa: NPY002 (olmix RNG parity)
    random.seed(FIT_SEED)

    target = torch.tensor(y[:, idx], dtype=torch.float32)
    x_t = torch.tensor(x, dtype=torch.float32)

    min_loss, best = 1e10, None
    for init_param in init_params_log_linear_law(idx, num_domains=x.shape[1]):
        loss, params = _fit_one_restart(x_t, target, init_param)
        if loss < min_loss:
            min_loss, best = loss, params
    assert best is not None
    logger.debug("task %d: best restart loss %.6g", idx, min_loss)
    return best


def predict_log_linear(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a fitted law on mixtures ``x`` (``(K, m)`` -> ``(K,)``)."""
    return (
        mixing_law(torch.tensor(x, dtype=torch.float), torch.tensor(np.asarray(params), dtype=torch.float))
        .numpy()
        .astype(float)
    )
