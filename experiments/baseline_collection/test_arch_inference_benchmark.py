# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for arch_inference_benchmark: one real end-to-end cell plus the gate/headline logic."""

import jax
import jmp
import numpy as np
import pytest
from jax.sharding import Mesh
from levanter.layers.attention import AttentionBackend

from experiments.baseline_collection.arch_inference_benchmark import (
    ARCHS,
    ArchResult,
    apply_gate,
    backend_for_ctx,
    bench_arch,
)


def test_backend_rule():
    assert backend_for_ctx(8192) == AttentionBackend.SPLASH
    assert backend_for_ctx(8191) == AttentionBackend.VANILLA
    assert backend_for_ctx(512) == AttentionBackend.VANILLA


def test_registry_has_gate_and_deployments():
    assert "mb-base" in ARCHS
    for name, spec in ARCHS.items():
        ctx, windows_per_doc = spec.deployment
        assert ctx > 0 and windows_per_doc >= 1.0, name


def test_bench_arch_end_to_end_cpu():
    """Real forward pass for the cheapest arch on CPU; deployment ctx unmeasured → proxy headline."""
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))
    result = bench_arch(
        "tiny2",
        ARCHS["tiny2"],
        ctxs=[64],
        batch_per_device=1,
        num_devices=len(devices),
        mesh=mesh,
        axis_mapping={"batch": "data"},
        policy=jmp.get_policy("p=f32,c=bfloat16"),
        warmup=1,
        iters=1,
    )
    (cell,) = result.cells
    assert cell.error is None
    assert cell.ms_per_forward > 0
    assert cell.seqs_per_sec_per_chip > 0
    assert result.params > 0
    assert result.headline_ctx == 64  # proxy: deployment ctx 8192 was not in the sweep
    assert result.effective_docs_per_sec_per_chip == pytest.approx(cell.seqs_per_sec_per_chip / result.windows_per_doc)


def _arch_result(name: str, eff: float | None) -> ArchResult:
    r = ArchResult(arch=name, deployment_ctx=8192, windows_per_doc=1.0)
    r.effective_docs_per_sec_per_chip = eff
    return r


def test_apply_gate_pass_fail_and_errors():
    results = [
        _arch_result("mb-base", 10.0),
        _arch_result("faster", 20.0),
        _arch_result("slower", 5.0),
        _arch_result("tied", 10.0),  # strictly faster required → tie FAILs
        _arch_result("oomed", None),
    ]
    apply_gate(results)
    by_name = {r.arch: r for r in results}
    assert by_name["mb-base"].gate == "GATE" and by_name["mb-base"].vs_gate == 1.0
    assert by_name["faster"].gate == "PASS" and by_name["faster"].vs_gate == 2.0
    assert by_name["slower"].gate == "FAIL" and by_name["slower"].vs_gate == 0.5
    assert by_name["tied"].gate == "FAIL"
    assert by_name["oomed"].gate == "FAIL" and by_name["oomed"].vs_gate is None


def test_apply_gate_without_reference():
    results = [_arch_result("tiny2", 30.0)]
    apply_gate(results)
    assert results[0].gate == "-"
