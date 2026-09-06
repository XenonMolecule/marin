# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU hardware utilities for memory estimation and slice selection.

This module provides utilities for estimating memory requirements and
selecting appropriate TPU slice sizes for training runs.
"""

import math
from dataclasses import dataclass

from fray.types import tpu_hbm_bytes_per_chip


@dataclass(frozen=True)
class TpuSpec:
    """Hardware specification for a TPU generation."""

    prefix: str
    """TPU generation prefix, e.g. "v5p" or "v4"."""

    cores_per_chip: int
    """Number of cores per chip."""

    core_options: tuple[int, ...]
    """Available core configurations (slice sizes), sorted ascending."""


# ---------------- TPU Hardware Specs ----------------

V5P_SPEC = TpuSpec(
    prefix="v5p",
    cores_per_chip=2,
    core_options=(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
)

V4_SPEC = TpuSpec(
    prefix="v4",
    cores_per_chip=2,
    core_options=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
)

# v5e (v5litepod) and v6e (Trillium) are single-core-per-chip parts named by chip
# count (e.g. "v6e-128" == 128 chips), unlike v5p/v4 which are named by core count
# (2 cores/chip). Both top out at a 256-chip single-slice topology. v5e has only
# 16 GiB/chip -- half of v4/v6e -- so the same model+batch needs proportionally
# more chips (or a smaller batch) to fit.
V5E_SPEC = TpuSpec(
    prefix="v5e",
    cores_per_chip=1,
    core_options=(1, 4, 8, 16, 32, 64, 128, 256),
)

V6E_SPEC = TpuSpec(
    prefix="v6e",
    cores_per_chip=1,
    core_options=(1, 4, 8, 16, 32, 64, 128, 256),
)


def pick_tpu_type(estimated_memory_bytes: int, spec: TpuSpec) -> str:
    """Select the smallest TPU slice that fits the estimated memory.

    Args:
        estimated_memory_bytes: Estimated memory requirement in bytes.
        spec: Hardware specification for the target TPU generation.

    Returns:
        TPU slice name, e.g., "v5p-8" or "v4-32".

    Raises:
        ValueError: If the model is too large for available slices.
    """
    chip_bytes = tpu_hbm_bytes_per_chip(spec.prefix)
    chips = math.ceil(estimated_memory_bytes / chip_bytes)
    cores_req = chips * spec.cores_per_chip

    valid = [c for c in spec.core_options if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available {spec.prefix} slices (need {cores_req} cores).")

    return f"{spec.prefix}-{min(valid)}"


def pick_v5p_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v5p slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V5P_SPEC)


def pick_v4_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v4 slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V4_SPEC)


def pick_v5e_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v5e slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V5E_SPEC)


def pick_v6e_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v6e slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V6E_SPEC)
