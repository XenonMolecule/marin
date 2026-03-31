# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""REVERT: Switch back to 2D mesh (we'll adapt the model instead)."""

PATH = "/workspace/tpu_inference/tpu_inference/runner/tpu_runner.py"

with open(PATH) as f:
    code = f.read()

# Revert to 2D mesh
old = "self.mesh = self._create_new_model_mesh()"
new = "self.mesh = self._create_2d_mesh()"

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED: reverted to 2D mesh in tpu_runner")
elif new in code:
    print("SKIP: already using 2D mesh")
else:
    print("SKIP: pattern not found")
