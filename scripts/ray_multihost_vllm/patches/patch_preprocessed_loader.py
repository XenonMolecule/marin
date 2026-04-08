# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive loader patches for preprocessed K2-Instruct weights.

Consolidates ALL patches needed for loading preprocessed weights on multi-host
TPU via Ray. Apply with: PREPROCESSED_WEIGHTS=1 python3 patch_preprocessed_loader.py

Patches applied to:
  1. fp8.py — Fp8FusedMoEMethod (load_weights, process_weights_after_loading, create_weights_jax)
  2. fp8.py — Fp8BlockwiseLinearMethod (process_weights_after_loading, create_weights_jax)
  3. deepseek_v3.py — DeepseekV3MLA (pre-create k_up_proj/v_up_proj, skip MLAEinsum, skip_substrs)
  4. moe.py — JaxMoE._load_weights (preprocessed early return)
  5. weight_utils.py — default_weight_loader JAX compat, _load_param metadata lookup
"""

import os
import sys

PREPROCESSED = os.environ.get("PREPROCESSED_WEIGHTS", "0") == "1"
if not PREPROCESSED:
    print("SKIP: PREPROCESSED_WEIGHTS not set")
    sys.exit(0)

print("Applying comprehensive preprocessed loader patches...")

# =============================================================================
# 1. fp8.py — ALL patches
# =============================================================================

FP8_PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"

with open(FP8_PATH) as f:
    code = f.read()

patched_fp8 = False

# --- 1a. Fp8FusedMoEMethod.process_weights_after_loading ---
# After loading MoE weights to CPU in load_weights, place them on TPU here.
old_moe_pwl = """    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        \"\"\"
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        \"\"\""""

new_moe_pwl = """    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        \"\"\"
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        \"\"\"
        import os as _os
        if _os.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            # Preprocessed: weights stored as numpy in load_weights.
            # Now place on TPU with proper sharding.
            import numpy as _np3
            _pp_attrs = {
                "kernel_gating_upproj_EDF": layer.edf_sharding,
                "kernel_down_proj_EFD": layer.efd_sharding,
            }
            for _attr_name, _base_s in _pp_attrs.items():
                for _suffix in ["", "_" + self.weight_scale_name]:
                    _full = _attr_name + _suffix
                    _param = getattr(layer, _full, None)
                    if _param is not None and isinstance(_param, nnx.Param):
                        if _suffix:  # scale
                            _s = (_base_s[0],) + (None,) * (_param.value.ndim - 2) + (_base_s[-1],)
                        else:
                            _s = _base_s
                        _val_np = _np3.asarray(_param.value)
                        _param.value = shard_put(jnp.array(_val_np), shardings=_s, mesh=layer.mesh)
            logger.warning("PP_PROCESS: MoE %s — placed on TPU (numpy path)", layer.prefix)
            return True"""

if old_moe_pwl in code:
    code = code.replace(old_moe_pwl, new_moe_pwl)
    patched_fp8 = True
    print("  PATCHED: Fp8FusedMoEMethod.process_weights_after_loading → TPU placement")
else:
    print("  SKIP: MoE process_weights_after_loading (already patched or pattern changed)")

# --- 1b. Fp8FusedMoEMethod.load_weights ---
# Store preprocessed MoE weights as numpy (no TPU placement during loading).
old_load_weights = """        \"\"\"Load scale paramters and delegate the weight paramters to `original_load_weights_fn`\"\"\"

        # Remaining non-scale parameters will be loaded using original load_weights function.
        remaining_weights = dict()"""

new_load_weights = """        \"\"\"Load scale paramters and delegate the weight paramters to `original_load_weights_fn`\"\"\"
        import os as _os_lw
        if _os_lw.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            import numpy as np
            import ml_dtypes
            loaded = set()
            for name, tensor in weights:
                short = name.split(layer.prefix)[-1].lstrip(".")
                if isinstance(tensor, np.ndarray) and tensor.dtype == np.uint8:
                    tensor = tensor.view(ml_dtypes.float8_e4m3fn)
                # Store as numpy — TPU placement happens in process_weights_after_loading
                import numpy as _np
                np_tensor = _np.asarray(tensor)
                setattr(layer, short, nnx.Param(jnp.array(np_tensor)))
                loaded.add(short)
            # Clean up pre-processing attributes
            for attr in ["kernel_gating_EDF", "kernel_up_proj_EDF",
                         "kernel_gating_EDF_" + self.weight_scale_name,
                         "kernel_up_proj_EDF_" + self.weight_scale_name]:
                if hasattr(layer, attr):
                    delattr(layer, attr)
            logger.warning("PP_LOAD: MoE %s — loaded %d weights to CPU: %s", layer.prefix, len(loaded), list(loaded))
            return loaded

        # Remaining non-scale parameters will be loaded using original load_weights function.
        remaining_weights = dict()"""

if old_load_weights in code:
    code = code.replace(old_load_weights, new_load_weights)
    patched_fp8 = True
    print("  PATCHED: Fp8FusedMoEMethod.load_weights → CPU numpy storage")
else:
    print("  SKIP: MoE load_weights (already patched or pattern changed)")

# --- 1c. Fp8BlockwiseLinearMethod.create_weights_jax ---
# For preprocessed dense linears: 1D scale shape, identity permute.
# For preprocessed k_up_proj/v_up_proj: 3D scale shape, flexible loader.
# kv_b_proj excluded (uses original HF blockwise format).

# Non-batched path: 1D scale for preprocessed
old_scale_block = """        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        layer.weight_scale_inv = nnx.Param(
            kernel_init(
                rngs.params(),
                [(out_features + block_n - 1) // block_n,
                 (self.in_features + block_k - 1) // block_k],"""

new_scale_block = """        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        import os as _os_cw
        if _os_cw.environ.get("PREPROCESSED_WEIGHTS", "0") == "1" and "kv_b_proj" not in layer.prefix:
            # Preprocessed: scale is 1D per-channel (out_features,)
            _scale_shape = [out_features]
        else:
            _scale_shape = [(out_features + block_n - 1) // block_n,
                            (self.in_features + block_k - 1) // block_k]
        layer.weight_scale_inv = nnx.Param(
            kernel_init(
                rngs.params(),
                _scale_shape,"""

if old_scale_block in code:
    code = code.replace(old_scale_block, new_scale_block)
    patched_fp8 = True
    print("  PATCHED: create_weights_jax — 1D scale for preprocessed dense linears")
else:
    print("  SKIP: create_weights_jax scale shape (already patched or pattern changed)")

# Non-batched path: scale weight_loader permute
old_scale_loader = """            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(0, 1),
                param_name=layer.prefix + ".weight_scale_inv",
            ),
            eager_sharding=False)
        layer.weight_scale_inv.set_metadata('sharding', self.weight_sharding)"""

new_scale_loader = """            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=None if (_os_cw.environ.get("PREPROCESSED_WEIGHTS", "0") == "1" and "kv_b_proj" not in layer.prefix) else (0, 1),
                param_name=layer.prefix + ".weight_scale_inv",
            ),
            eager_sharding=False)
        layer.weight_scale_inv.set_metadata('sharding', self.weight_sharding)"""

if old_scale_loader in code:
    code = code.replace(old_scale_loader, new_scale_loader)
    patched_fp8 = True
    print("  PATCHED: create_weights_jax — scale permute_dims=None for preprocessed")
else:
    print("  SKIP: scale weight_loader permute (already patched or pattern changed)")

# Batched path: flexible scale for k_up_proj/v_up_proj
old_batched_scale = """            # Per-output-channel scale (1D, covers the free weight dim).
            layer.weight_scale_inv = nnx.Param(
                jnp.ones((out_features, ), dtype=layer.dtype),
                weight_loader=partial(
                    load_nnx_param_from_reshaped_torch,
                    permute_dims=None,
                    param_name=layer.prefix + ".weight_scale_inv",
                ),
                eager_sharding=False)
            layer.weight_scale_inv.set_metadata('sharding', ())
            return"""

new_batched_scale = """            # Per-output-channel scale
            import os as _os_batch
            if _os_batch.environ.get("PREPROCESSED_WEIGHTS", "0") == "1" and ("k_up_proj" in layer.prefix or "v_up_proj" in layer.prefix):
                # Preprocessed k/v scales: may be 2D (64,512) or 3D (64,1,512)
                _N = self.kernel_shape[1] if len(self.kernel_shape) >= 2 else 1
                _last = self.kernel_shape[0] if "k_up_proj" in layer.prefix else self.kernel_shape[2]
                _batch_scale_shape = (_N, 1, _last)
                def _kv_scale_loader(jax_param, torch_weight, param_name=layer.prefix + ".weight_scale_inv"):
                    import numpy as _np
                    w = torch_weight.numpy() if hasattr(torch_weight, 'numpy') else _np.asarray(torch_weight)
                    if w.ndim == 2:
                        w = w.reshape(w.shape[0], 1, w.shape[1])
                    from tpu_inference.models.jax.utils.weight_utils import assign_and_shard_param
                    assign_and_shard_param(jax_param, jnp.array(w), param_name)
                layer.weight_scale_inv = nnx.Param(
                    jnp.ones(_batch_scale_shape, dtype=layer.dtype),
                    weight_loader=_kv_scale_loader,
                    eager_sharding=False)
            else:
                layer.weight_scale_inv = nnx.Param(
                    jnp.ones((out_features, ), dtype=layer.dtype),
                    weight_loader=partial(
                        load_nnx_param_from_reshaped_torch,
                        permute_dims=None,
                        param_name=layer.prefix + ".weight_scale_inv",
                    ),
                    eager_sharding=False)
            layer.weight_scale_inv.set_metadata('sharding', ())
            return"""

if old_batched_scale in code:
    code = code.replace(old_batched_scale, new_batched_scale)
    patched_fp8 = True
    print("  PATCHED: create_weights_jax batched — flexible k/v scale loader")
else:
    print("  SKIP: batched scale (already patched or pattern changed)")

# --- 1d. Fp8BlockwiseLinearMethod.process_weights_after_loading ---
# No-op for preprocessed dense linears: just shard_put to TPU.
old_linear_pwl = """        assert self.quant_config.weight_block_size is not None

        if self.batch_features:"""

new_linear_pwl = """        assert self.quant_config.weight_block_size is not None

        import os as _os_pp
        if _os_pp.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            import numpy as _np2
            from jax.sharding import NamedSharding as _NS
            spec = layer.weight.get_metadata().get("sharding", ())
            if isinstance(spec, _NS):
                spec = spec.spec
            logger.warning("PP_PROCESS: Linear %s — weight=%s scale=%s", layer.prefix, layer.weight[...].shape, layer.weight_scale_inv[...].shape)
            w_np = _np2.asarray(layer.weight[...])
            layer.weight = nnx.Param(shard_put(jnp.array(w_np), shardings=spec))
            s_np = _np2.asarray(layer.weight_scale_inv[...])
            layer.weight_scale_inv = nnx.Param(shard_put(jnp.array(s_np), shardings=()))
            return True

        if self.batch_features:"""

if old_linear_pwl in code:
    code = code.replace(old_linear_pwl, new_linear_pwl)
    patched_fp8 = True
    print("  PATCHED: Fp8BlockwiseLinearMethod.process_weights_after_loading → shard_put no-op")
else:
    print("  SKIP: Linear process_weights_after_loading (already patched or pattern changed)")

if patched_fp8:
    with open(FP8_PATH, "w") as f:
        f.write(code)

# =============================================================================
# 2. moe.py — JaxMoE._load_weights preprocessed early return
# =============================================================================

MOE_PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/moe/moe.py"

try:
    with open(MOE_PATH) as f:
        moe_code = f.read()

    old_moe_lw = "    def _load_weights(self, weights: Iterable):"
    new_moe_lw = """    def _load_weights(self, weights: Iterable):
        import os as _os_moe_lw
        if _os_moe_lw.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            from flax import nnx
            from tpu_inference.models.jax.utils.weight_utils import shard_put
            import jax.numpy as jnp
            import ml_dtypes
            import numpy as np
            loaded = set()
            for name, tensor in weights:
                if isinstance(tensor, np.ndarray) and tensor.dtype == np.uint8:
                    tensor = tensor.view(ml_dtypes.float8_e4m3fn)
                parts = name.split(".")
                attr_name = parts[-1] if len(parts) == 1 else ".".join(parts)
                for candidate in [attr_name, parts[-1]]:
                    param = getattr(self, candidate, None)
                    if param is not None and isinstance(param, nnx.Param):
                        try:
                            sharding = param.get_metadata().get("sharding", ())
                            param.value = shard_put(jnp.array(tensor), sharding)
                            loaded.add(name)
                        except Exception:
                            pass
                        break
            return loaded"""

    if old_moe_lw in moe_code:
        moe_code = moe_code.replace(old_moe_lw, new_moe_lw)
        with open(MOE_PATH, "w") as f:
            f.write(moe_code)
        print("  PATCHED: JaxMoE._load_weights — preprocessed early return")
    else:
        print("  SKIP: _load_weights (already patched)")
except FileNotFoundError:
    print("  SKIP: moe.py not found")

# =============================================================================
# 3. deepseek_v3.py — MLA init, MLAEinsum skip, skip_substrs
# =============================================================================

DS_PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"

try:
    with open(DS_PATH) as f:
        ds_code = f.read()

    ds_patched = False

    # --- 3a. Pre-create k_up_proj/v_up_proj in MLA __post_init__ ---
    old_mla_init = """        self.kv_b_proj = MLAEinsum(
            mla_layer=self,
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
            prefix=self.prefix + ".kv_b_proj",
        )"""

    new_mla_init = """        self.kv_b_proj = MLAEinsum(
            mla_layer=self,
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
            prefix=self.prefix + ".kv_b_proj",
        )
        import os as _os_mla_init
        if _os_mla_init.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            # Pre-create k_up_proj and v_up_proj so weight loader can route to them
            self.k_up_proj = JaxEinsum(
                einsum_str="TNH,ANH->TNA",
                kernel_shape=(self.kv_lora_rank, self.N, self.qk_nope_head_dim),
                rngs=nnx.Rngs(0),
                prefix=self.prefix + ".k_up_proj",
                quant_config=self.quant_config,
            )
            self.v_up_proj = JaxEinsum(
                einsum_str="TNA,ANH->TNH",
                kernel_shape=(self.kv_lora_rank, self.N, self.v_head_dim),
                rngs=nnx.Rngs(0),
                prefix=self.prefix + ".v_up_proj",
                quant_config=self.quant_config,
            )"""

    if old_mla_init in ds_code:
        ds_code = ds_code.replace(old_mla_init, new_mla_init)
        ds_patched = True
        print("  PATCHED: DeepseekV3MLA.__post_init__ — pre-create k_up_proj/v_up_proj")
    else:
        print("  SKIP: MLA __post_init__ (already patched or pattern changed)")

    # --- 3b. Skip MLAEinsum.load_weights when preprocessed ---
    old_mla_lw = "    def load_weights(self, weights):\n        named_params = dict(self.named_parameters())"

    new_mla_lw = """    def load_weights(self, weights):
        import os as _os_mla_lw
        import logging as _mla_log
        _mla_log.warning("PP_MLA: MLAEinsum.load_weights called for %s, PREPROCESSED=%s",
                         self.prefix, _os_mla_lw.environ.get("PREPROCESSED_WEIGHTS", "0"))
        if _os_mla_lw.environ.get("PREPROCESSED_WEIGHTS", "0") == "1":
            # Skip kv_b_proj processing — k/v loaded as separate modules
            if hasattr(self, 'quant_method'):
                delattr(self, 'quant_method')
            if hasattr(self, 'weight'):
                delattr(self, 'weight')
            if hasattr(self, 'weight_scale_inv'):
                delattr(self, 'weight_scale_inv')
            _mla_log.warning("PP_MLA: Skipped kv_b_proj for %s (preprocessed)", self.prefix)
            return set()
        named_params = dict(self.named_parameters())"""

    if old_mla_lw in ds_code:
        ds_code = ds_code.replace(old_mla_lw, new_mla_lw)
        ds_patched = True
        print("  PATCHED: MLAEinsum.load_weights — skip when preprocessed")
    else:
        print("  SKIP: MLAEinsum.load_weights (already patched or pattern changed)")

    # --- 3c. Add kv_b_proj to skip_substrs ---
    old_skip = """            skip_substrs=[
                f"layers.{i}"
                for i in range(start_ignore_layer_num, end_ignore_layer_num)
            ],"""

    new_skip = """            skip_substrs=[
                f"layers.{i}"
                for i in range(start_ignore_layer_num, end_ignore_layer_num)
            ] + (["kv_b_proj"] if os.environ.get("PREPROCESSED_WEIGHTS", "0") == "1" else []),"""

    if old_skip in ds_code:
        ds_code = ds_code.replace(old_skip, new_skip)
        ds_patched = True
        print("  PATCHED: skip_substrs — added kv_b_proj")
    else:
        print("  SKIP: skip_substrs (already patched or pattern changed)")

    if ds_patched:
        with open(DS_PATH, "w") as f:
            f.write(ds_code)

except FileNotFoundError:
    print("  SKIP: deepseek_v3.py not found")

# =============================================================================
# 4. vllm weight_utils.py — JAX-compatible default_weight_loader
# =============================================================================

VWU_PATH = "/workspace/vllm/vllm/model_executor/model_loader/weight_utils.py"

try:
    with open(VWU_PATH) as f:
        vwu_code = f.read()

    old_dwl = """def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    \"\"\"Default weight loader.\"\"\"
    try:"""

    new_dwl = """def default_weight_loader(param, loaded_weight) -> None:
    \"\"\"Default weight loader.\"\"\"
    try:
        # JAX nnx.Param early return
        from flax import nnx as _nnx_check
        if isinstance(param, _nnx_check.Param):
            import numpy as _np
            import jax.numpy as _jnp
            w = loaded_weight
            if hasattr(w, 'numpy'):
                w = w.numpy()
            if not isinstance(w, _np.ndarray):
                w = _np.asarray(w)
            if w.dtype == _np.uint8:
                import ml_dtypes; w = w.view(ml_dtypes.float8_e4m3fn)
            from tpu_inference.models.jax.utils.weight_utils import assign_and_shard_param
            assign_and_shard_param(param, _jnp.array(w), "default_loader")
            return"""

    if old_dwl in vwu_code:
        vwu_code = vwu_code.replace(old_dwl, new_dwl)
        with open(VWU_PATH, "w") as f:
            f.write(vwu_code)
        print("  PATCHED: default_weight_loader — JAX nnx.Param early return")
    else:
        print("  SKIP: default_weight_loader (already patched or pattern changed)")
except FileNotFoundError:
    print("  SKIP: weight_utils.py not found")

# =============================================================================
# 5. vllm utils.py — _load_param metadata weight_loader lookup
# =============================================================================

VU_PATH = "/workspace/vllm/vllm/model_executor/models/utils.py"

try:
    with open(VU_PATH) as f:
        vu_code = f.read()

    old_lp = """            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)"""

    new_lp = """            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                _gm = getattr(param, 'get_metadata', None)
                if _gm is not None:
                    weight_loader = _gm("weight_loader", None)
            if weight_loader is None:
                weight_loader = default_weight_loader
            weight_loader(param, weight_data)"""

    if old_lp in vu_code:
        vu_code = vu_code.replace(old_lp, new_lp)
        with open(VU_PATH, "w") as f:
            f.write(vu_code)
        print("  PATCHED: _load_param — metadata weight_loader lookup")
    else:
        print("  SKIP: _load_param (already patched or pattern changed)")
except FileNotFoundError:
    print("  SKIP: utils.py not found")

print("\nPATCH COMPLETE. Set PREPROCESSED_WEIGHTS=1 to activate.")
