# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Make the JAX persistent compilation cache portable across TPU slices.

ROOT CAUSE (verified against jax 0.10.0 ``jax/_src/cache_key.py``):
the persistent-cache key hashes ``compile_options`` with
``strip_device_assignment=(backend.platform == "gpu")`` — i.e. the physical
device assignment is removed from the key *only on GPU*. On TPU the device
assignment stays in the key, so the key changes every time the job lands on a
different physical slice. Since every preemption gives a fresh slice, a
preempted training job NEVER hits the cache and pays the full cold compile
(~76 min for this 32k-seq x 152k-vocab graph) instead of a warm load (~7 min).

Empirically confirmed (CPU backend, same code path): for one computation,
slice {0..7} and slice {8..15} produce different keys with strip=False and the
SAME key with strip=True.

FIX: force ``strip_device_assignment=True`` in the cache-key computation. This
is SAFE for our workload — single-host (1 VM, 8 chips) data-parallel training,
where the compiled executable does not depend on *which* physical v6e-8/v5p-8 it
runs on. It is exactly what JAX already does for GPU. Do NOT use this for jobs
whose executable genuinely depends on the device assignment (e.g. heterogeneous
multi-host sharding where physical placement changes the program).

Must run in the *training* process (the Levanter child that compiles), before
any compilation. Wiring options:
  1. Preferred: in ``levanter/trainer.py`` right after it sets
     ``jax_compilation_cache_dir`` (~line 1034), call ``apply()`` gated by a
     TrainerConfig flag (e.g. ``portable_tpu_compilation_cache: bool``).
  2. Quick A/B: launch the train job with an entrypoint that calls ``apply()``
     before importing the trainer.

This module is import-safe and idempotent; ``apply()`` is a no-op if already
applied or if JAX internals have moved.
"""

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> bool:
    """Patch JAX so the persistent-cache key omits the TPU device assignment.

    Returns True if the patch is now active, False if it could not be applied.
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from jax._src import cache_key as ck
    except Exception as exc:  # JAX internals moved / not importable
        logger.warning("portable_tpu_cache: could not import jax cache_key: %s", exc)
        return False

    original = getattr(ck, "_hash_serialized_compile_options", None)
    if original is None:
        logger.warning("portable_tpu_cache: _hash_serialized_compile_options missing; cache layout changed")
        return False

    def _strip_device_assignment(hash_obj, compile_options_obj, strip_device_assignment=False):
        # Always strip, regardless of platform: portable across same-topology slices.
        return original(hash_obj, compile_options_obj, strip_device_assignment=True)

    ck._hash_serialized_compile_options = _strip_device_assignment
    _APPLIED = True
    logger.info("portable_tpu_cache: TPU persistent-cache key is now slice-portable (device assignment stripped).")
    return True


if __name__ == "__main__":
    # Self-test on whatever backend is available (CPU is fine): same computation
    # on two different device assignments must yield the same key once patched.
    import hashlib

    import numpy as np
    from jax._src import cache_key as ck
    from jax._src import compiler as jcomp

    def _opts(ids):
        return jcomp.get_compile_options(
            num_replicas=1, num_partitions=len(ids), device_assignment=np.array(ids).reshape(1, -1)
        )

    def _h(opts):
        h = hashlib.sha256()
        ck._hash_serialized_compile_options(h, opts)  # uses module-level (possibly patched) fn
        return h.hexdigest()[:16]

    a, b = _opts(range(0, 8)), _opts(range(8, 16))
    before = (_h(a), _h(b))
    apply()
    after = (_h(a), _h(b))
    print(f"before patch: A={before[0]} B={before[1]} equal={before[0] == before[1]}")
    print(f"after  patch: A={after[0]} B={after[1]} equal={after[0] == after[1]}")
    assert after[0] == after[1], "patch failed to make cache key slice-portable"
    print("OK: cache key is slice-portable after patch")
