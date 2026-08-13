# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fanning the per-task fits across processes must not change a single number.

The fit is the expensive step: 236 s per task measured on the real dclm swarm (318 runs,
75 live domains), so 42 tasks take ~2.75 h serially and the regression, not the sweep,
becomes the tail. Parallelising is only acceptable if it is *exact* -- a mixture proposal
that shifts depending on how many cores happened to be free would be worthless.

It is exact because ``fit_log_linear`` reseeds ``np.random`` and ``random`` to 42 on entry,
so a task's 300 restart draws depend on that task alone and never on evaluation order.
These tests pin that property rather than trusting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.data_mixing.olmix_fit import fit_all_tasks
from experiments.data_mixing.olmix_law import fit_log_linear

# One task costs ~8.7 s even on a tiny design -- the 300 restarts are fixed work, independent
# of data size -- and each pool worker additionally pays a torch import. So these are minutes,
# not milliseconds, and the default 60 s pytest-timeout is not survivable. Kept to 2 tasks to
# hold the cost down; the properties under test do not need more.
SLOW = pytest.mark.timeout(600)


@pytest.fixture
def design():
    """A small but honestly-shaped design: more runs than live domains, several tasks."""
    rng = np.random.default_rng(0)
    n_runs, n_domains, n_tasks = 24, 6, 2
    weights = rng.dirichlet(np.ones(n_domains), size=n_runs)
    true_t = rng.normal(scale=0.3, size=(n_tasks, n_domains))
    bpb = np.exp(-1.2) + np.exp(weights @ true_t.T)
    return weights, bpb, [f"task_{i}" for i in range(n_tasks)]


@SLOW
def test_parallel_matches_serial_exactly(design):
    """The property that matters: identical params, not merely close ones."""
    weights, bpb, names = design
    serial = fit_all_tasks(weights, bpb, names, num_workers=1)
    parallel = fit_all_tasks(weights, bpb, names, num_workers=3)
    np.testing.assert_array_equal(serial, parallel)


@SLOW
def test_worker_count_does_not_change_the_answer(design):
    """Results must not depend on how many cores were free."""
    weights, bpb, names = design
    baseline = fit_all_tasks(weights, bpb, names, num_workers=1)
    for workers in (2, 3):
        np.testing.assert_array_equal(baseline, fit_all_tasks(weights, bpb, names, num_workers=workers))


@SLOW
def test_row_order_follows_task_order(design):
    """Row j must stay task_names[j]; a pool that returned out of order would silently
    attach every task's coefficients to the wrong task and still 'look' fine."""
    weights, bpb, names = design
    params = fit_all_tasks(weights, bpb, names, num_workers=3)
    assert params.shape == (len(names), weights.shape[1] + 1)
    for j in range(len(names)):
        # Must compare against idx=j on the FULL matrix, not a sliced column fitted as idx=0:
        # `init_params_log_linear_law` seeds the coefficient at the task's own index negative,
        # so slicing changes the restart draws and would fail for a reason unrelated to order.
        np.testing.assert_array_equal(params[j], fit_log_linear(weights, bpb, idx=j))


@SLOW
def test_serial_remains_the_default(design):
    """Callers that never opt in keep the exact pre-existing behaviour."""
    weights, bpb, names = design
    np.testing.assert_array_equal(fit_all_tasks(weights, bpb, names), fit_all_tasks(weights, bpb, names, num_workers=1))
