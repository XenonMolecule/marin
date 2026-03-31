# Trace: Is JaxMoE.load_weights() ever called during weight loading?
# Also trace JaxMoE._load_weights() and Fp8FusedMoEMethod.load_weights()

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/moe/moe.py"
with open(PATH) as f:
    code = f.read()

# Trace JaxMoE.load_weights entry
old1 = '    def load_weights(self, weights: Iterable):'
new1 = """    def load_weights(self, weights: Iterable):
        with open("/tmp/moe_load_trace.txt", "a") as _f:
            _f.write(f"JaxMoE.load_weights CALLED: prefix={self.prefix} qm={type(getattr(self, 'quant_method', None)).__name__}\\n")"""

if old1 in code:
    code = code.replace(old1, new1, 1)
    print("TRACE 1: JaxMoE.load_weights entry")

# Trace _load_weights entry
old2 = '    def _load_weights(self, weights: Iterable):'
new2 = """    def _load_weights(self, weights: Iterable):
        with open("/tmp/moe_load_trace.txt", "a") as _f:
            _f.write(f"JaxMoE._load_weights CALLED: prefix={self.prefix}\\n")"""

if old2 in code:
    code = code.replace(old2, new2, 1)
    print("TRACE 2: JaxMoE._load_weights entry")

with open(PATH, "w") as f:
    f.write(code)

# Also trace Fp8FusedMoEMethod.load_weights
PATH2 = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"
with open(PATH2) as f:
    code2 = f.read()

old3 = '    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,'
new3 = """    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,"""

# Just add a trace at the start of the method body
old3b = '        """Load scale paramters and delegate the weight paramters to `original_load_weights_fn`"""'
new3b = """        \"\"\"Load scale paramters and delegate the weight paramters to `original_load_weights_fn`\"\"\"
        with open("/tmp/moe_load_trace.txt", "a") as _f:
            _f.write(f"Fp8FusedMoEMethod.load_weights CALLED: layer.prefix={layer.prefix}\\n")"""

if old3b in code2:
    code2 = code2.replace(old3b, new3b, 1)
    with open(PATH2, "w") as f:
        f.write(code2)
    print("TRACE 3: Fp8FusedMoEMethod.load_weights entry")
else:
    print("SKIP: Fp8 load_weights pattern not found")
