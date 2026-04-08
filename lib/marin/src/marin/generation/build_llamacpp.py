# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build llama.cpp from source and upload the llama-server binary to GCS.

This module provides an ExecutorStep-compatible function that:
1. Clones llama.cpp at a pinned commit
2. Builds llama-server with native CPU optimizations
3. Uploads the binary to GCS for use by inference workers

The build only runs once — the executor's step-level caching ensures subsequent
runs skip this step entirely.

Example usage::

    build_step = ExecutorStep(
        name="tools/llamacpp-build-v3",
        fn=build_llamacpp,
        config=BuildLlamaCppConfig(output_path=this_output_path()),
        resources=ResourceConfig.with_cpu(cpu=8, ram="16g"),
        pip_dependency_groups=["cpu"],
    )
"""

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)

# Pin to a specific commit for reproducibility. Update this when you want a
# newer llama.cpp version (and bump the ExecutorStep name to invalidate cache).
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMACPP_COMMIT = "HEAD"  # Use HEAD for latest; pin a SHA for reproducibility


@dataclass(frozen=True)
class BuildLlamaCppConfig:
    """Configuration for building llama.cpp."""

    output_path: str
    """GCS directory where the built binary and metadata will be stored."""


def build_llamacpp(config: BuildLlamaCppConfig) -> None:
    """Clone llama.cpp, build llama-server, and upload to GCS."""
    workdir = "/tmp/llamacpp-build"
    os.makedirs(workdir, exist_ok=True)
    src_dir = os.path.join(workdir, "llama.cpp")

    # Step 1: Clone
    if not os.path.isdir(src_dir):
        logger.info("Cloning llama.cpp...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", LLAMACPP_REPO, src_dir],
            cwd=workdir,
        )
    else:
        logger.info("llama.cpp source already present, skipping clone.")

    # Step 2: Build
    binary_path = os.path.join(src_dir, "build", "bin", "llama-server")
    if not os.path.isfile(binary_path):
        logger.info("Building llama-server...")
        t0 = time.monotonic()
        nproc = str(os.cpu_count() or 4)
        # BUILD_SHARED_LIBS=OFF ensures the binary is self-contained and can run
        # on worker nodes without needing the .so files from the build directory.
        subprocess.check_call(
            [
                "cmake",
                "-B",
                "build",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=ON",
                "-DBUILD_SHARED_LIBS=OFF",
            ],
            cwd=src_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", f"-j{nproc}", "--target", "llama-server"],
            cwd=src_dir,
        )
        elapsed = time.monotonic() - t0
        logger.info("Build completed in %.1fs", elapsed)
    else:
        logger.info("llama-server binary already built, skipping.")

    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Build did not produce expected binary at {binary_path}")

    # Step 3: Get git info for metadata
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src_dir, text=True).strip()

    # Step 4: Upload binary to GCS
    dest_binary = os.path.join(config.output_path, "llama-server")
    logger.info("Uploading llama-server to %s", dest_binary)
    with open(binary_path, "rb") as local_f:
        with fsspec.open(dest_binary, "wb") as remote_f:
            remote_f.write(local_f.read())

    # Step 5: Write build metadata
    build_info = {
        "git_hash": git_hash,
        "git_repo": LLAMACPP_REPO,
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": platform.machine(),
        "platform": platform.platform(),
        "binary_size_bytes": os.path.getsize(binary_path),
    }
    info_path = os.path.join(config.output_path, "build_info.json")
    with fsspec.open(info_path, "w") as f:
        json.dump(build_info, f, indent=2)

    logger.info("Build complete: %s (git=%s, size=%d bytes)", dest_binary, git_hash[:8], build_info["binary_size_bytes"])
