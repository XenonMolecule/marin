# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix JaxModule.named_children() to find children in nnx.eval_shape models.

ROOT CAUSE: JaxModule.named_children() checks self.__dict__ for children.
For models created with nnx.eval_shape(), the children are stored in the
nnx graph state, not in __dict__. named_children() returns empty, so
AutoWeightsLoader._load_module can't recurse into child modules.

Result: MoE weights can't be matched to params (name mismatch) and are
silently dropped. The model stays abstract → 240GB/chip OOM.

Fix: Use vars() instead of __dict__, which includes both regular attrs
and nnx graph-managed attrs. Also check for nnx._pytree__nodes.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/__init__.py"
with open(PATH) as f:
    code = f.read()

old = """    def named_children(
            self) -> Iterator[tuple[str, "JaxModule | JaxModuleList"]]:
        \"\"\"Returns an iterator over immediate children modules.

        Yields:
            (string, Module | list): Tuple containing a name and child module
        \"\"\"
        for name, value in self.__dict__.items():
            if isinstance(value, JaxModule):
                yield name, value
            elif isinstance(value, list) or isinstance(value, nnx.List):
                yield name, JaxModuleList(value)"""

new = """    def named_children(
            self) -> Iterator[tuple[str, "JaxModule | JaxModuleList"]]:
        \"\"\"Returns an iterator over immediate children modules.

        Yields:
            (string, Module | list): Tuple containing a name and child module
        \"\"\"
        # FIX: Use vars() to find children in both regular __dict__ and
        # nnx graph state (needed for nnx.eval_shape abstract models).
        _seen = set()
        for name, value in vars(self).items():
            if name.startswith('_'):
                continue
            if isinstance(value, JaxModule):
                _seen.add(name)
                yield name, value
            elif isinstance(value, list) or isinstance(value, nnx.List):
                _seen.add(name)
                yield name, JaxModuleList(value)
        # Also check nnx pytree nodes for children not in vars()
        _nodes = getattr(self, '_pytree__nodes', {})
        if isinstance(_nodes, dict):
            for name, value in _nodes.items():
                if name in _seen or name.startswith('_'):
                    continue
                if isinstance(value, JaxModule):
                    yield name, value
                elif isinstance(value, list) or isinstance(value, nnx.List):
                    yield name, JaxModuleList(value)"""

if old in code:
    code = code.replace(old, new, 1)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED __init__.py: Fix named_children to find children in eval_shape models")
else:
    print("SKIP: pattern not found")
