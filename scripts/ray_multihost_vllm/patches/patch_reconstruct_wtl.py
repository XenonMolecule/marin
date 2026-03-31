# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix: Reconstruct _weights_to_load from .value for MoE FP8 params.

JaxAutoWeightsLoader loaded MoE weights directly into .value via
assign_and_shard_param, bypassing the _weights_to_load accumulation.
process_weights_after_loading reads _weights_to_load → all None → returns False.

Fix: After weight loading, split .value along expert dimension (axis 0)
to populate _weights_to_load, then call process_weights_after_loading.
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"
with open(PATH) as f:
    code = f.read()

old = """        logger.warning(f"FORCE_PWL: Processed {_pwl_count} modules")"""

new = """        # FIX: Reconstruct _weights_to_load from .value and call process_weights_after_loading
        import jax.numpy as _jnp
        if hasattr(self.model, 'layers'):
            for _i3, _l3 in enumerate(self.model.layers):
                if 'PPMissing' in type(_l3).__name__:
                    continue
                if not (hasattr(_l3, 'mlp') and hasattr(_l3.mlp, 'experts')):
                    continue
                _exp3 = _l3.mlp.experts
                _qm3 = getattr(_exp3, 'quant_method', None)
                if _qm3 is None or not hasattr(_qm3, 'process_weights_after_loading'):
                    continue
                # Reconstruct _weights_to_load from .value for ALL kernel+scale params
                _reconst = 0
                for _attr3 in dir(_exp3):
                    if not (_attr3.startswith('kernel_') and not _attr3.startswith('__')):
                        continue
                    _p3 = getattr(_exp3, _attr3, None)
                    if _p3 is None or not hasattr(_p3, '_weights_to_load'):
                        continue
                    _wtl3 = _p3._weights_to_load
                    if all(w is None for w in _wtl3) and hasattr(_p3, 'value'):
                        # Split .value along expert dimension (axis 0)
                        _v3 = _p3.value
                        _E3 = len(_wtl3)
                        if _v3.shape[0] == _E3:
                            for _eidx in range(_E3):
                                _wtl3[_eidx] = _v3[_eidx:_eidx+1]
                            _reconst += 1
                if _reconst > 0:
                    try:
                        _result3 = _qm3.process_weights_after_loading(_exp3)
                        if _result3:
                            _pwl_count += 1
                            logger.warning(f"FORCE_PWL_OK: layer {_i3} experts (reconstructed {_reconst} params)")
                    except Exception as _e3:
                        logger.warning(f"FORCE_PWL_ERR: layer {_i3}: {_e3}")
        logger.warning(f"FORCE_PWL: Processed {_pwl_count} modules")"""

if old in code:
    code = code.replace(old, new, 1)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED deepseek_v3.py: Reconstruct _weights_to_load from .value")
else:
    print("SKIP: pattern not found")
