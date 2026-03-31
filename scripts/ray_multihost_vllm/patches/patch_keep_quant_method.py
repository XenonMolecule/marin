# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: Check why quant_method ends up None on Ray workers.

JaxMoE.__post_init__ sets quant_method from quant_config.get_quant_method().
If this returns None/falsy, quant_method = None. The question is whether
quant_config.get_quant_method() returns a valid method for FP8.

Add logging to see what happens during model construction.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/moe/moe.py"
with open(PATH) as f:
    code = f.read()

old = """        if self.quant_config is None:
            self.quant_method = None
        elif (quant_method :=
              self.quant_config.get_quant_method(self, prefix=self.prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self, rngs=rngs)
        else:
            self.quant_method = None"""

new = """        if self.quant_config is None:
            self.quant_method = None
            with open("/tmp/qm_debug.txt", "a") as _f:
                _f.write(f"JaxMoE {self.prefix}: quant_config=None -> quant_method=None\\n")
        elif (quant_method :=
              self.quant_config.get_quant_method(self, prefix=self.prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self, rngs=rngs)
            with open("/tmp/qm_debug.txt", "a") as _f:
                _f.write(f"JaxMoE {self.prefix}: quant_method={type(quant_method).__name__} KEPT\\n")
        else:
            self.quant_method = None
            with open("/tmp/qm_debug.txt", "a") as _f:
                _f.write(f"JaxMoE {self.prefix}: get_quant_method returned falsy -> quant_method=None (quant_config={type(self.quant_config).__name__})\\n")"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED moe.py: Added quant_method diagnostic logging")
else:
    print("SKIP: pattern not found")
