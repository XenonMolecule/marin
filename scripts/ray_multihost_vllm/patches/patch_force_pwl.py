# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directly call process_weights_after_loading on known MoE modules.

We've confirmed:
- self.model.layers[i].mlp.experts has quant_method=Fp8FusedMoEMethod
- process_weights_after_loading needs to be called to fuse and shard weights
- The JaxAutoWeightsLoader doesn't reach these nested modules

Fix: Directly iterate self.model.layers and call pwl on .mlp.experts.
Also handle JaxEinsum modules in attention and shared experts.
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"
with open(PATH) as f:
    code = f.read()

old = """        loaded = loader.load_weights(weights)

        self.model.initialize_cache()"""

new = """        loaded = loader.load_weights(weights)

        # FIX: Directly call process_weights_after_loading on known modules.
        # JaxAutoWeightsLoader doesn't recurse deep enough to reach these.
        _pwl_count = 0
        if hasattr(self.model, 'layers'):
            for _i, _layer in enumerate(self.model.layers):
                # Skip PPMissingLayer
                if 'PPMissing' in type(_layer).__name__:
                    continue
                # MoE experts (JaxMoE/SharedFusedMoe with Fp8FusedMoEMethod)
                if hasattr(_layer, 'mlp') and hasattr(_layer.mlp, 'experts'):
                    _exp = _layer.mlp.experts
                    _qm = getattr(_exp, 'quant_method', None)
                    if _qm is not None and hasattr(_qm, 'process_weights_after_loading'):
                        try:
                            _result = _qm.process_weights_after_loading(_exp)
                            if _result:
                                _pwl_count += 1
                        except Exception as _e:
                            logger.warning(f"FORCE_PWL_ERR layer {_i} experts: {_e}")
                # Shared expert (JaxEinsum with Fp8LinearMethod)
                if hasattr(_layer, 'mlp') and hasattr(_layer.mlp, 'shared_experts'):
                    _se = _layer.mlp.shared_experts
                    for _attr in ['gate_proj', 'up_proj', 'down_proj']:
                        _proj = getattr(_se, _attr, None)
                        if _proj is not None:
                            _qm2 = getattr(_proj, 'quant_method', None)
                            if _qm2 is not None and hasattr(_qm2, 'process_weights_after_loading'):
                                try:
                                    _result2 = _qm2.process_weights_after_loading(_proj)
                                    if _result2:
                                        _pwl_count += 1
                                except Exception as _e2:
                                    logger.warning(f"FORCE_PWL_ERR layer {_i} shared {_attr}: {_e2}")
                # Attention projections (JaxEinsum with Fp8LinearMethod)
                if hasattr(_layer, 'self_attn'):
                    _attn = _layer.self_attn
                    for _attr in ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']:
                        _proj = getattr(_attn, _attr, None)
                        if _proj is not None:
                            _qm3 = getattr(_proj, 'quant_method', None)
                            if _qm3 is not None and hasattr(_qm3, 'process_weights_after_loading'):
                                try:
                                    _result3 = _qm3.process_weights_after_loading(_proj)
                                    if _result3:
                                        _pwl_count += 1
                                except Exception as _e3:
                                    logger.warning(f"FORCE_PWL_ERR layer {_i} attn {_attr}: {_e3}")
        logger.warning(f"FORCE_PWL: Processed {_pwl_count} modules")

        self.model.initialize_cache()"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED deepseek_v3.py: Direct process_weights_after_loading on known modules")
else:
    print("SKIP: pattern not found")
