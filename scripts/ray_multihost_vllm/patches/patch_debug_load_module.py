# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Debug: trace _load_module post-processing to understand why
# process_weights_after_loading never runs

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py"
with open(PATH) as f:
    code = f.read()

# Add debug before the quant_method check
old = "        if (quant_method := getattr(module, 'quant_method', None)) is not None:"

new = """        with open("/tmp/pwl_trace.txt", "a") as _dbg:
            _qm = getattr(module, "quant_method", None)
            _dbg.write(f"LOAD_MODULE prefix={base_prefix} mod={type(module).__name__} qm={type(_qm).__name__ if _qm else None} done={self._process_weights_after_loading_per_module[base_prefix]}\\n")
        if (quant_method := getattr(module, 'quant_method', None)) is not None:"""

if old in code:
    code = code.replace(old, new, 1)

    # Also trace the actual call
    old2 = "            loaded = quant_method.process_weights_after_loading(module)"
    new2 = """            with open("/tmp/pwl_trace.txt", "a") as _dbg:
                _dbg.write(f"  CALLING pwl for {base_prefix} qm={type(quant_method).__name__}\\n")
            try:
                loaded = quant_method.process_weights_after_loading(module)
                with open("/tmp/pwl_trace.txt", "a") as _dbg:
                    _dbg.write(f"  RESULT loaded={loaded}\\n")
            except Exception as _e:
                with open("/tmp/pwl_trace.txt", "a") as _dbg:
                    _dbg.write(f"  ERROR: {_e}\\n")
                raise"""
    if old2 in code:
        code = code.replace(old2, new2, 1)

    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED: Added _load_module debug tracing")
else:
    print("SKIP: pattern not found")
