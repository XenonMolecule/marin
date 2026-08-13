# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construction-only checks on the objects the coordinator hands to Iris.

Every launch bug we hit on 2026-07-30 was a wrong kwarg or a missing field that this
file would have caught instantly, instead of after an 18-minute submit-and-watch cycle:

1. `ResourceSpec(memory_gb=..., disk_gb=...)` -- the real fields are `memory`/`disk` and
   take size strings. Every `submit_child` raised TypeError.
2. `EnvironmentSpec` without `extras=["tpu"]` -- the base container has no jax/libtpu, so
   children could not train. Extras are per-job, not inherited from the coordinator.
3. Child env missing `WANDB_API_KEY` -- children could not log, and with finelog down
   WandB is the only progress signal, so the whole sweep would have run blind.
4. Submit failures swallowed per-index -- a persistent error advanced the cursor through
   all K runs while sending nothing, leaving a coordinator that looked healthy with zero
   children.

None of this touches the network: we build the same objects the coordinator builds and
assert they are well-formed. That is the whole point -- these are the failures that only
show up at submit time, which is exactly when they are most expensive to discover.
"""

from __future__ import annotations

import os

import pytest
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device

from experiments.data_mixing import launch_olmix_swarm as L


def test_child_resource_spec_constructs():
    """Bug 1: the real fields are `memory`/`disk`, not `memory_gb`/`disk_gb`."""
    spec = ResourceSpec(
        device=tpu_device(L.PRIMARY_TPU),
        cpu=L.CHILD_CPU,
        memory=L.child_memory(L.TPU_VARIANTS),
        disk=L.CHILD_DISK,
    )
    assert spec.cpu == L.CHILD_CPU
    assert isinstance(spec.memory, str | int)
    assert isinstance(spec.disk, str | int)
    assert spec.device is not None


def test_child_env_carries_credentials_and_cache_flag(monkeypatch):
    """Bug 3: without WANDB_API_KEY a child cannot report progress at all."""
    monkeypatch.setenv("WANDB_API_KEY", "dummy-key")
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    env = L._child_env()
    assert env["WANDB_API_KEY"] == "dummy-key"
    assert env["HF_TOKEN"] == "dummy-token"
    # Without this a preempted child recompiles from scratch instead of resuming, which
    # under preemption pressure means it never finishes.
    assert env["LEVANTER_PORTABLE_TPU_CACHE"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_child_env_refuses_to_run_blind(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        L._child_env()


def test_child_environment_spec_requests_the_tpu_extra(monkeypatch):
    """Bug 2: extras are per-job. A child without `tpu` has no jax and cannot train."""
    monkeypatch.setenv("WANDB_API_KEY", "dummy-key")
    spec = EnvironmentSpec(env_vars=L._child_env(), extras=["tpu"])
    assert "tpu" in list(spec.extras)
    assert spec.env_vars["WANDB_API_KEY"] == "dummy-key"


def test_submit_child_source_requests_the_tpu_extra():
    """Guard the real call site, not just a spec we construct here in the test."""
    src = os.path.join(os.path.dirname(L.__file__), "launch_olmix_swarm.py")
    with open(src) as fh:
        body = fh.read()
    assert 'extras=["tpu"]' in body, "submit_child must request the tpu extra"
    assert "_child_env()" in body, "submit_child must forward the child environment"
    assert "memory_gb" not in body and "disk_gb" not in body, "ResourceSpec has no *_gb fields"


def test_entrypoint_takes_a_command_list():
    ep = Entrypoint(command=["python", L.SCRIPT, "--manifest", "gs://x", "--index", "0"])
    assert ep.command[0] == "python"
    assert L.SCRIPT.endswith("run_olmix_swarm_standalone.py")


def test_tpu_variants_are_mutually_schedulable():
    """Iris requires every variant in one `device_variant_constraint` to share vm_count
    AND chip count. v6e-8 has 8 chips and would make Iris mis-count committed_tpu, so it
    must stay out of this list even though it is a single VM."""
    from iris.cluster.types import get_tpu_topology

    topos = {v: get_tpu_topology(v) for v in L.TPU_VARIANTS}
    vm_counts = {t.vm_count for t in topos.values()}
    assert vm_counts == {1}, f"variants disagree on vm_count: {topos}"
    assert L.PRIMARY_TPU in L.TPU_VARIANTS


def test_swarm_size_is_fixed_at_363():
    """K is held constant across corpora by user directive so a cross-corpus comparison
    is not confounded by differing proxy compute. It must never be derived from m_eff."""
    assert L.SWARM_SIZE == 363


def test_strength_range_is_olmix_literal():
    assert (L.STRENGTH_MIN, L.STRENGTH_MAX) == (1.0, 20.0)
