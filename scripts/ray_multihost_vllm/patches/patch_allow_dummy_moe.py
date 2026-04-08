# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Allow dummy/random weights for MoE backends.
# The check spans lines 1133-1136 in deepseek_v3.py (multiline).

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"
with open(PATH) as f:
    code = f.read()

# The check is: if vllm_config.load_config.load_format == "dummy" and self.moe_backend in MoEBackend.fused_moe_backends(\n        ):
# Just find and neutralize it
import re

pattern = (
    r'if vllm_config\.load_config\.load_format == "dummy" and self\.moe_backend in MoEBackend\.fused_moe_backends\('
)
match = re.search(pattern, code)
if match:
    # Replace the condition with False
    start = match.start()
    # Find the full if statement (up to the raise)
    end_raise = code.find("raise ValueError(", start)
    if end_raise > 0:
        # Find end of the raise statement
        end_paren = code.find(")", end_raise + len("raise ValueError("))
        # Keep going past nested parens
        depth = 1
        i = end_raise + len("raise ValueError(")
        while i < len(code) and depth > 0:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        # Comment out the entire if block
        block = code[start:i]
        commented = "# DISABLED: " + block.replace("\n", "\n# ")
        code = code[:start] + commented + code[i:]
        with open(PATH, "w") as f:
            f.write(code)
        print("PATCHED: Disabled dummy MoE weight check (multiline)")
    else:
        print("SKIP: Found if but no raise")
else:
    # Maybe already patched
    if "DISABLED" in code and "dummy" in code:
        print("SKIP: Already disabled")
    else:
        print("SKIP: Pattern not found")
