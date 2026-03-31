# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Enable PP for DeepSeek V3 on the JAX path.

Remove DeepseekV3ForCausalLM from _PP_DISABLED_MODELS so the JAX model
handles pipeline parallelism natively (with the mesh fix).
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/common/model_loader.py"

with open(PATH) as f:
    code = f.read()

old = '{"DeepseekV3ForCausalLM", "Eagle3LlamaForCausalLM", "GptOssForCausalLM"}'
new = '{"Eagle3LlamaForCausalLM", "GptOssForCausalLM"}'

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED: removed DeepseekV3 from _PP_DISABLED_MODELS (JAX PP enabled)")
elif "DeepseekV3ForCausalLM" not in code or "PP_DISABLED" not in code:
    print("SKIP: DeepseekV3 already removed or pattern changed")
else:
    print("SKIP: unexpected state")
