# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What ``launch_olmix_swarm.submit_child`` hands the Iris controller.

Every bug that stalled the first swarm launch was a wrong kwarg or a missing field in this
one call -- ``ResourceSpec(memory_gb=...)`` (no such parameter), an ``EnvironmentSpec``
without ``extras=["tpu"]`` (no libtpu in the base container), an env dict without
``WANDB_API_KEY`` (children train but report nothing). None of them are visible until a
child has been scheduled, so each cost a full launch cycle to find. Constructing the real
objects offline catches all three in milliseconds.

The submit itself is faked: these assert the *arguments*, not the controller's behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest
from iris.cluster.types import ResourceSpec

from experiments.data_mixing import launch_olmix_swarm
from experiments.data_mixing.olmix_plan import SwarmManifest, run_name

DOMAINS = ("c00_q0", "c00_q1", "c01_q0")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Dispatch now consults a cross-region completion set and an atomic claim registry, both
    of which live in GCS. Neither may be reached from a unit test -- stub them so these stay
    pure construction checks, and so a network hang can never masquerade as a slow test."""
    monkeypatch.setattr(launch_olmix_swarm, "completed_run_names", lambda corpus: set())
    monkeypatch.setattr(launch_olmix_swarm, "try_claim", lambda *a, **kw: True)


MANIFEST_GCS = "gs://marin-us-east5/metadata/olmix/dclm_10k/swarm_s42_K363.json"


class _RecordingClient:
    """Stands in for IrisClient, capturing exactly what submit_child passed."""

    def __init__(self):
        self.calls: list[dict] = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return f"/michaelryan/{kwargs['name']}"


@pytest.fixture
def manifest() -> SwarmManifest:
    return SwarmManifest(
        corpus="dclm_10k",
        region="us-east5",
        seed=42,
        domains=DOMAINS,
        weights=((0.5, 0.5, 0.0), (0.25, 0.25, 0.5)),
        tokens={d: 10_000_000 for d in DOMAINS},
        cache_dirs={d: f"gs://marin-us-east5/datakit/store/dclm_10k_gridv1/{d}" for d in DOMAINS},
    )


def _target(manifest: SwarmManifest) -> launch_olmix_swarm.SwarmTarget:
    return launch_olmix_swarm.SwarmTarget(
        manifest=manifest,
        manifest_gcs=MANIFEST_GCS,
        results_prefix=f"gs://marin-us-east5/metadata/olmix_swarm_results/{manifest.corpus}",
    )


@pytest.fixture
def submitted(manifest, monkeypatch) -> dict:
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    client = _RecordingClient()
    launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 1, 100, [])
    return client.calls[0]


def test_resources_are_constructible(submitted):
    """`ResourceSpec` takes `memory`/`disk`, not `memory_gb`/`disk_gb`. Passing the wrong
    kwarg raises TypeError at submit time, which the coordinator used to swallow."""
    resources = submitted["resources"]
    assert isinstance(resources, ResourceSpec)
    proto = resources.to_proto()
    assert proto.memory_bytes > 0
    assert proto.disk_bytes > 0
    assert proto.cpu_millicores >= ResourceSpec.MIN_ACCELERATOR_CPU_MILLICORES
    assert proto.HasField("device")


def test_child_requests_the_tpu_extra(submitted):
    """Extras are per-job. The coordinator runs under `--extra cpu`; a child that does not
    ask for `tpu` gets a container with no libtpu and dies on jax backend init."""
    assert "tpu" in submitted["environment"].extras


def test_child_env_carries_wandb_credentials(submitted):
    """WandB is the only progress signal while finelog is down, so a child that cannot log
    is indistinguishable from a child that is not running."""
    env = submitted["environment"].env_vars
    assert env["WANDB_API_KEY"] == "test-wandb-key"
    assert env["HF_TOKEN"] == "test-hf-token"
    assert env["LEVANTER_PORTABLE_TPU_CACHE"] == "1"


def test_missing_wandb_key_fails_before_submitting(manifest, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    client = _RecordingClient()
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 1, 100, [])
    assert client.calls == []


def test_child_is_pinned_to_the_manifest_region(submitted, manifest):
    values = {v.value for c in submitted["constraints"] for v in getattr(c, "values", [])}
    assert manifest.region in values
    assert set(launch_olmix_swarm.TPU_VARIANTS) <= set(values)


def test_child_name_and_command_identify_the_mixture(submitted, manifest):
    assert submitted["name"] == run_name(manifest.corpus, manifest.seed, manifest.k, 1, manifest.row(1))
    command = submitted["entrypoint"].command
    assert command[command.index("--index") + 1] == "1"
    assert command[command.index("--manifest") + 1] == MANIFEST_GCS


def test_failed_submits_do_not_consume_the_queue(manifest, monkeypatch):
    """A systematic submit failure must not advance the cursor. It used to: one `_submit(10)`
    burned all K indices, and the coordinator then slept forever reporting 'all dispatched'
    with zero children alive."""
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    attempted: list[int] = []

    def _always_fails(client, mf, mf_gcs, index, band, extra_args, tpu_variants):
        attempted.append(index)
        raise TypeError("simulated bad kwarg")

    monkeypatch.setattr(launch_olmix_swarm, "submit_child", _always_fails)
    monkeypatch.setattr(launch_olmix_swarm, "already_done", lambda *a, **kw: False)

    with pytest.raises(TypeError):
        launch_olmix_swarm.run_adaptive_swarm(
            client=_RecordingClient(),
            targets=[_target(manifest)],
            initial_batch=10,
            chunk_size=10,
            check_interval=0,
            patience=3,
            child_priority_band=100,
            extra_args=[],
        )
    # It gave up on the same index rather than walking the whole swarm.
    assert len(attempted) == launch_olmix_swarm.MAX_CONSECUTIVE_SUBMIT_FAILURES
    assert set(attempted) == {0}


def test_v5e_request_fits_its_smaller_vms(manifest, monkeypatch):
    """v5e VMs have 192 GiB; the 256GB default made every v5e request permanently
    unschedulable ("no matching scaling group has enough per-VM capacity") and killed the
    coordinator on repeated failures. That silently excluded the whole v5e family."""
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "t")
    client = _RecordingClient()
    launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 0, 1, [], tpu_variants=("v5litepod-4",))
    assert client.calls[0]["resources"].memory == "96GB"


def test_v5p_and_v6e_keep_the_larger_request(manifest, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "t")
    client = _RecordingClient()
    launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 0, 1, [], tpu_variants=("v5p-8", "v6e-4"))
    assert client.calls[0]["resources"].memory == "256GB"


def test_default_variants_are_four_chip_single_vm(manifest):
    """The default pool must stay gang-free: replicas is derived from vm_count, so a shape
    change here silently turns every child into a multi-host job."""
    assert launch_olmix_swarm.slice_shape(launch_olmix_swarm.TPU_VARIANTS) == (4, 1)


def test_v4_is_not_offered():
    """v4 lives only in us-central2-b and every child is region-pinned to its data, so the
    variant could never place -- it was noise in the constraint, not reachable capacity."""
    assert "v4-8" not in launch_olmix_swarm.TPU_VARIANTS


def test_mixed_shapes_are_rejected():
    """Iris mis-counts committed_tpu when one constraint spans shapes, so this must fail
    loudly at construction rather than produce a job that schedules wrongly."""
    with pytest.raises(ValueError, match="span shapes"):
        launch_olmix_swarm.slice_shape(("v5p-8", "v5p-16"))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown TPU variant"):
        launch_olmix_swarm.slice_shape(("v5p-9000",))


def test_large_slice_is_gang_scheduled_across_its_hosts(manifest, monkeypatch):
    """Opting into v5p-16 must request replicas=2. Levanter handles the multi-host init
    itself; what Iris needs is the right host count reserved together."""
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    client = _RecordingClient()
    launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 0, 1, [], tpu_variants=("v5p-16",))
    kwargs = client.calls[0]
    assert kwargs["replicas"] == 2
    assert "v5p-16" in str(kwargs["resources"].device)


def test_default_slice_requests_a_single_replica(submitted):
    assert submitted["replicas"] == 1


def test_corpora_are_interleaved_not_concatenated(manifest, monkeypatch):
    """Iris ranks pending work by root job, so submission order decides the split between
    corpora. Concatenating them would hand the whole cluster to the first one until it
    finished all K runs -- exactly the starvation observed with one coordinator per corpus."""
    monkeypatch.setattr(launch_olmix_swarm, "already_done", lambda *a, **kw: False)
    other = dataclasses.replace(manifest, corpus="high_quality_10k")
    order = launch_olmix_swarm._interleave([_target(manifest), _target(other)])

    corpora = [t.manifest.corpus for t, _ in order]
    assert corpora == ["dclm_10k", "high_quality_10k"] * manifest.k
    assert [i for _, i in order] == [0, 0, 1, 1]


def _sliced(manifest: SwarmManifest, start: int, end: int | None) -> launch_olmix_swarm.SwarmTarget:
    return dataclasses.replace(_target(manifest), index_start=start, index_end=end)


@pytest.fixture
def wide(manifest) -> SwarmManifest:
    """A 6-run swarm, so an index range has something to actually slice."""
    return dataclasses.replace(manifest, weights=tuple([(0.5, 0.5, 0.0), (0.25, 0.25, 0.5)] * 3))


def test_index_range_limits_a_coordinator_to_its_own_slice(wide, monkeypatch):
    monkeypatch.setattr(launch_olmix_swarm, "already_done", lambda *a, **kw: False)
    order = launch_olmix_swarm._interleave([_sliced(wide, 2, 5)])
    assert [i for _, i in order] == [2, 3, 4]


def test_split_ranges_partition_the_swarm_exactly(wide, monkeypatch):
    """The property that makes a cross-region split safe. Completion records are
    bucket-local, so `already_done` cannot see the other region: if the ranges overlap the
    same mixture is trained twice and the two coordinators collide on Iris job names, and
    if they leave a hole those indices are never trained and the swarm is silently short."""
    monkeypatch.setattr(launch_olmix_swarm, "already_done", lambda *a, **kw: False)
    east = [i for _, i in launch_olmix_swarm._interleave([_sliced(wide, 0, 4)])]
    central = [i for _, i in launch_olmix_swarm._interleave([_sliced(wide, 4, None)])]

    assert set(east) & set(central) == set()
    assert sorted(east + central) == list(range(wide.k))


def test_default_range_is_the_whole_swarm(wide, monkeypatch):
    monkeypatch.setattr(launch_olmix_swarm, "already_done", lambda *a, **kw: False)
    order = launch_olmix_swarm._interleave([_target(wide)])
    assert [i for _, i in order] == list(range(wide.k))


@pytest.mark.parametrize(
    ("start", "end"),
    [(4, 2), (0, 0), (-1, 3), (0, 99), (99, None)],
    ids=["inverted", "empty", "negative", "past_k", "start_past_k"],
)
def test_malformed_index_range_is_rejected(wide, start, end):
    """A silently-empty or over-long range would look like 'nothing to do' or would run
    indices the manifest does not have; both are worse than refusing to start."""
    with pytest.raises(ValueError, match="index range"):
        _sliced(wide, start, end).indices()


def test_interleave_skips_completed_runs(manifest, monkeypatch):
    done = {("dclm_10k", 0)}
    monkeypatch.setattr(
        launch_olmix_swarm,
        "already_done",
        lambda mf, idx, prefix, done_set=None: (mf.corpus, idx) in done,
    )
    other = dataclasses.replace(manifest, corpus="high_quality_10k")
    order = launch_olmix_swarm._interleave([_target(manifest), _target(other)])
    assert (manifest.corpus, 0) not in [(t.manifest.corpus, i) for t, i in order]
    assert len(order) == 2 * manifest.k - 1


def test_retry_ceiling_scales_with_host_count(manifest, monkeypatch):
    """Iris counts TASK failures, so one gang death costs vm_count of them. A flat ceiling
    abandons a 4-host run after a quarter as many real failures as a 1-host run -- that is
    what stranded 39 dclm and 47 hq indices on the first multi-host attempt."""
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "t")
    for variant, vms in (("v5p-8", 1), ("v5p-16", 2), ("v5p-32", 4), ("v6e-32", 8)):
        client = _RecordingClient()
        launch_olmix_swarm.submit_child(client, manifest, MANIFEST_GCS, 0, 1, [], tpu_variants=(variant,))
        got = client.calls[0]["max_retries_failure"]
        assert got == launch_olmix_swarm.GANG_DEATHS_TOLERATED * vms, f"{variant}: {got}"
        # Every shape tolerates the SAME number of gang deaths, which is the invariant.
        assert got // vms == launch_olmix_swarm.GANG_DEATHS_TOLERATED
