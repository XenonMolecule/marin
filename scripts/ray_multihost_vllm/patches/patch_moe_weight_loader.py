# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix: Ensure JaxMoE.load_weights is called by JaxAutoWeightsLoader.

Root cause chain:
1. JaxAutoWeightsLoader loads via per-param weight_loader callbacks
2. FP8 MoE scale params don't have weight_loader → loaded with default
   (assign_and_shard_param) which puts into .value, not _weights_to_load
3. process_weights_after_loading reads _weights_to_load → finds None → returns False
4. MoE expert weights never fused/sharded → 240GB/chip OOM

Fix: Patch JaxAutoWeightsLoader._load_module to detect JaxMoE modules
and delegate to module.load_weights() which properly handles scales via
Fp8FusedMoEMethod.load_weights().
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py"
with open(PATH) as f:
    code = f.read()

# Find _load_module and add JaxMoE delegation
old = """    def _load_module(self, base_prefix: str, module: JaxModule,
                     weights: Iterable) -> Iterable:
        yield from super()._load_module(base_prefix, module, weights)"""

new = """    def _load_module(self, base_prefix: str, module: JaxModule,
                     weights: Iterable) -> Iterable:
        # FIX: If module is JaxMoE with quant_method, delegate to its
        # load_weights() which properly handles FP8 scale accumulation.
        from tpu_inference.layers.jax.moe.moe import JaxMoE as _JaxMoE
        if isinstance(module, _JaxMoE) and getattr(module, 'quant_method', None) is not None:
            # Collect all weights for this module prefix
            _module_weights = []
            _other_weights = []
            for name, weight in weights:
                if name.startswith(base_prefix) or (base_prefix and name.startswith(base_prefix.rstrip('.'))):
                    _module_weights.append((name, weight))
                else:
                    _other_weights.append((name, weight))
            # Call module's load_weights which delegates to quant_method
            if _module_weights:
                loaded = module.load_weights(iter(_module_weights))
                for name in loaded:
                    yield name
            # Pass through remaining weights
            yield from super()._load_module(base_prefix, module, iter(_other_weights))
            return
        yield from super()._load_module(base_prefix, module, weights)"""

if old in code:
    code = code.replace(old, new, 1)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED weight_utils.py: JaxMoE load_weights delegation in _load_module")
else:
    print("SKIP: _load_module pattern not found")
