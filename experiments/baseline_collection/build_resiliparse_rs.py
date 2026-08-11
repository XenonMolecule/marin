# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the fork's Rust main-content extractor for Linux x86_64 and publish it to GCS.

The extraction engine used by ``XenonMolecule/chatnoir-resiliparse`` is the Rust
crate ``resiliparse-extract-rs`` exposed to Python as ``resiliparse._extract_rs``.
It is not on PyPI, and building it needs rust + vcpkg + cmake + libclang. This
script does that build once on a Linux worker and publishes a *self-contained*
artifact so downstream jobs need no build toolchain at all.

Two subcommands, both meant to run as Iris CPU jobs in ``us-east5``::

    # 1. build + publish (needs network; installs its own toolchain under --work-dir)
    iris job run --cluster marin --region us-east5 --cpu 16 --memory 32GB --disk 60GB \\
        --enable-extra-resources --extra cpu --priority interactive --no-wait \\
        --job-name rp-rs-build -- bash -lc \\
        'python -m experiments.baseline_collection.build_resiliparse_rs build'

    # 2. verify the consumption contract on a clean worker (no toolchain)
    iris job run --cluster marin --region us-east5 --cpu 4 --memory 8GB \\
        --enable-extra-resources --extra cpu --priority interactive --no-wait \\
        --job-name rp-rs-verify -- bash -lc \\
        'python -m experiments.baseline_collection.build_resiliparse_rs verify'

lexbor is a shared library on the ``x64-linux`` vcpkg triplet, so the extension
module is linked with an ``$ORIGIN`` rpath and the lexbor ``.so`` files are
shipped next to it. A consumer only has to drop every published ``*.so`` into
``<resiliparse-py>/resiliparse/``.
"""

import argparse
import datetime
import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)

FORK_URL = "https://github.com/XenonMolecule/chatnoir-resiliparse.git"
FORK_BRANCH = "master"
CARGO_PACKAGE = "resiliparse-extract-rs"
BUILT_LIB_NAME = "libresiliparse_extract_rs.so"
EXTENSION_NAME = "_extract_rs.so"

GCS_ROOT = "gs://marin-us-east5/artifacts/resiliparse_rs"
LATEST_RELEASE = "latest"

VCPKG_URL = "https://github.com/microsoft/vcpkg.git"
VCPKG_TOOL_METADATA = "scripts/vcpkg-tool-metadata.txt"
VCPKG_TOOL_TAG_KEY = "VCPKG_TOOL_RELEASE_TAG="
VCPKG_TOOL_RELEASE_URL = "https://github.com/microsoft/vcpkg-tool/releases/download"
VCPKG_TOOL_ASSET = "vcpkg-glibc"
VCPKG_TRIPLET = "x64-linux"
# The extractor only links lexbor; the fork's manifest also lists lz4/re2/uchardet/zlib
# for the (unbuilt) fastwarc crates. Trimming avoids a multi-minute abseil build.
VCPKG_DEPENDENCIES = ["lexbor"]

RUSTUP_INSTALLER_URL = "https://sh.rustup.rs"
APT_BUILD_PACKAGES = ("cmake", "ninja-build", "libclang-dev", "pkg-config", "zip")
PIP_BUILD_PACKAGES = ("cmake", "ninja", "libclang")
# lexbor is a dylib on x64-linux, so bake a relative rpath and ship it alongside.
ORIGIN_LINK_FLAGS = "-C link-arg=-Wl,-rpath,$ORIGIN"

PY_PACKAGE_DIRNAME = "resiliparse-py"
PY_PACKAGE_TARBALL = "resiliparse_py.tar.gz"
BENCH_TARBALL = "bench_html.tar.gz"
MANIFEST_NAME = "manifest.json"
BUILD_LOG_NAME = "build_log.txt"
GOLDEN_HTML_DIR = "resiliparse-rs/tests/extract_golden"

PROBE_TOOLS = (
    "id",
    "uname",
    "cargo",
    "rustc",
    "rustup",
    "cmake",
    "ninja",
    "clang",
    "gcc",
    "g++",
    "git",
    "curl",
    "tar",
    "unzip",
    "zip",
    "gsutil",
    "apt-get",
    "pkg-config",
    "ldd",
    "patchelf",
)
BENCH_SECONDS = 5.0

CONTRACT_SNIPPET = (
    "from resiliparse._extract_rs import extract_plain_text; "
    "print(extract_plain_text('<html><body><p>hi</p></body></html>', "
    "main_content=True, preserve_formatting='markdown'))"
)


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a command, streaming its output into this process' stdout (which the job tees to GCS)."""
    shell = isinstance(cmd, str)
    logger.info("RUN %s (cwd=%s)", cmd if shell else " ".join(cmd), cwd)
    start = time.monotonic()
    subprocess.run(cmd, cwd=cwd, env=env, shell=shell, check=True)
    logger.info("OK   (%.1fs) %s", time.monotonic() - start, cmd if shell else " ".join(cmd))


def capture(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run a command and return its stdout."""
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True, capture_output=True).stdout


def probe_toolchain() -> None:
    """Log what the worker already provides, so failures are diagnosable from the GCS log alone."""
    logger.info("=== toolchain probe ===")
    logger.info("python: %s", sys.version.replace("\n", " "))
    logger.info("platform: %s %s", platform.platform(), platform.machine())
    logger.info("uid=%s euid=%s home=%s cwd=%s", os.getuid(), os.geteuid(), Path.home(), Path.cwd())
    for tool in PROBE_TOOLS:
        path = shutil.which(tool)
        if path is None:
            logger.info("  %-12s MISSING", tool)
            continue
        version = subprocess.run([tool, "--version"], check=False, text=True, capture_output=True)
        first_line = (version.stdout or version.stderr).strip().splitlines()
        logger.info("  %-12s %s | %s", tool, path, first_line[0] if first_line else "")
    logger.info("disk:\n%s", capture(["df", "-h"]))
    # noexec mounts matter twice over: the build execs binaries and Python dlopen()s the extension.
    noexec = [line for line in Path("/proc/mounts").read_text().splitlines() if "noexec" in line]
    logger.info("noexec mounts:\n%s", "\n".join(noexec))
    logger.info("=== end probe ===")


def ensure_rust(work_dir: Path) -> tuple[Path, dict[str, str]]:
    """Return cargo's bin directory plus the env overrides it needs.

    The worker image already ships rustup with a default toolchain, in which case the
    overrides must stay empty: pointing ``RUSTUP_HOME`` at a fresh directory hides that
    toolchain and rustup refuses to pick a cargo version.
    """
    existing = shutil.which("cargo")
    if existing is not None:
        return Path(existing).parent, {}
    overrides = {"CARGO_HOME": str(work_dir / "cargo"), "RUSTUP_HOME": str(work_dir / "rustup")}
    run(
        f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_INSTALLER_URL} | sh -s -- -y --no-modify-path --profile minimal",
        env=os.environ | overrides,
    )
    return work_dir / "cargo" / "bin", overrides


def apt_install_build_tools() -> Path | None:
    """Try to install cmake/libclang from Debian; return the ``libclang.so`` directory or None.

    The Iris CPU worker runs as root but without CAP_SETUID, so apt's privilege-dropping
    sandbox fails; ``APT::Sandbox::User=root`` disables it. apt is preferred over the pip
    wheels because ``libclang-dev`` also brings the clang builtin headers bindgen wants.
    """
    apt = ["apt-get", "-o", "APT::Sandbox::User=root", "-y", "-qq"]
    env = os.environ | {"DEBIAN_FRONTEND": "noninteractive"}
    for args in (["update"], ["install", "--no-install-recommends", *APT_BUILD_PACKAGES]):
        result = subprocess.run(apt + args, env=env, check=False)
        if result.returncode != 0:
            logger.warning("apt %s failed (rc=%s); falling back to pip wheels", args[0], result.returncode)
            return None
    matches = glob.glob("/usr/lib/llvm-*/lib/libclang.so*") + glob.glob("/usr/lib/*/libclang*.so*")
    return Path(sorted(matches)[0]).parent if matches else None


def pip_install_build_tools(work_dir: Path) -> tuple[Path, Path]:
    """Install cmake/ninja/libclang wheels into a private venv; return its bin dir and libclang dir."""
    venv_dir = work_dir / "toolvenv"
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
        run([str(venv_dir / "bin" / "pip"), "install", "--quiet", *PIP_BUILD_PACKAGES])
    matches = glob.glob(str(venv_dir / "lib" / "python*" / "site-packages" / "clang" / "native" / "libclang.so*"))
    if not matches:
        raise RuntimeError(f"libclang wheel did not provide libclang.so under {venv_dir}")
    return venv_dir / "bin", Path(matches[0]).parent


def ensure_native_build_tools(work_dir: Path) -> tuple[list[Path], Path]:
    """Make cmake and libclang available; return PATH additions and the ``libclang.so`` directory."""
    if shutil.which("cmake") is not None:
        installed = glob.glob("/usr/lib/llvm-*/lib/libclang.so*") + glob.glob("/usr/lib/*/libclang*.so*")
        if installed:
            return [], Path(sorted(installed)[0]).parent
    apt_libclang = apt_install_build_tools()
    if apt_libclang is not None:
        return [], apt_libclang
    tool_bin, libclang_dir = pip_install_build_tools(work_dir)
    return [tool_bin], libclang_dir


def ensure_vcpkg(work_dir: Path) -> Path:
    """Clone vcpkg and install the prebuilt tool binary; return VCPKG_ROOT.

    ``bootstrap-vcpkg.sh`` aborts unless ``zip`` is on PATH, even though on x86-64
    glibc all it then does is download a prebuilt binary. The worker image has no
    zip and no usable apt, so fetch that same release asset directly.
    """
    vcpkg_root = work_dir / "vcpkg"
    tool = vcpkg_root / "vcpkg"
    if tool.exists():
        return vcpkg_root
    if not vcpkg_root.exists():
        run(["git", "clone", "--depth", "1", VCPKG_URL, str(vcpkg_root)])
    metadata = (vcpkg_root / VCPKG_TOOL_METADATA).read_text().splitlines()
    tag = next(line.split("=", 1)[1].strip() for line in metadata if line.startswith(VCPKG_TOOL_TAG_KEY))
    url = f"{VCPKG_TOOL_RELEASE_URL}/{tag}/{VCPKG_TOOL_ASSET}"
    logger.info("downloading vcpkg tool %s", url)
    urllib.request.urlretrieve(url, tool)
    tool.chmod(0o755)
    return vcpkg_root


def clone_fork(work_dir: Path) -> Path:
    """Clone the fork at ``FORK_BRANCH`` and return the checkout path."""
    repo = work_dir / "chatnoir-resiliparse"
    if repo.exists():
        shutil.rmtree(repo)
    run(["git", "clone", "--quiet", "--branch", FORK_BRANCH, FORK_URL, str(repo)])
    return repo


def trim_vcpkg_manifest(repo: Path) -> None:
    """Restrict the vcpkg manifest to the extractor's only native dependency."""
    manifest = repo / "vcpkg.json"
    spec = json.loads(manifest.read_text())
    spec["dependencies"] = VCPKG_DEPENDENCIES
    manifest.write_text(json.dumps(spec, indent=2) + "\n")


def build_env(
    repo: Path,
    cargo_bin: Path,
    cargo_overrides: dict[str, str],
    tool_bins: list[Path],
    libclang_dir: Path,
    vcpkg_root: Path,
) -> dict[str, str]:
    """Assemble the environment cargo needs: toolchain on PATH, vcpkg root, libclang, $ORIGIN rpath."""
    env = os.environ | cargo_overrides
    env["PATH"] = os.pathsep.join([str(cargo_bin), *(str(p) for p in tool_bins), str(vcpkg_root), env["PATH"]])
    # The worker image exports CARGO_TARGET_DIR=/cargo/target; keep artifacts inside the checkout.
    env["CARGO_TARGET_DIR"] = str(repo / "target")
    env["VCPKG_ROOT"] = str(vcpkg_root)
    env["VCPKG_DEFAULT_TRIPLET"] = VCPKG_TRIPLET
    env["LIBCLANG_PATH"] = str(libclang_dir)
    env["RUSTFLAGS"] = ORIGIN_LINK_FLAGS
    env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    # Give bindgen gcc's builtin headers too: lexbor.h pulls in stddef.h/stdarg.h.
    gcc_include = capture(["gcc", "-print-file-name=include"]).strip()
    env["BINDGEN_EXTRA_CLANG_ARGS"] = f"-I{gcc_include}"
    return env


def runtime_libraries(so_path: Path, repo: Path) -> dict[str, Path]:
    """Map each build-tree dependency's SONAME to the file backing it.

    Keying on the SONAME matters: the loader asks for ``liblexbor.so.2``, while the file
    inside the vcpkg tree is ``liblexbor.so.2.4.0`` reached through a symlink. Shipping
    the real file under its own name would leave the consumer unable to resolve it.
    """
    output = capture(["ldd", str(so_path)])
    logger.info("ldd %s:\n%s", so_path, output)
    needed: dict[str, Path] = {}
    for line in output.splitlines():
        if "=>" not in line:
            continue
        soname, _, rest = line.partition("=>")
        target = rest.strip().split(" ")[0]
        if not target or not target.startswith(str(repo)):
            continue
        needed[soname.strip()] = Path(target).resolve()
    if not needed:
        raise RuntimeError(f"expected the extension to link a build-tree lexbor; ldd output was:\n{output}")
    return needed


def stage_artifact(repo: Path, stage: Path) -> list[str]:
    """Copy the built extension, its runtime libraries and the Python package into ``stage``.

    Returns the names of the extra ``.so`` files a consumer must ship alongside.
    """
    stage.mkdir(parents=True, exist_ok=True)
    built = repo / "target" / "release" / BUILT_LIB_NAME
    shutil.copy2(built, stage / EXTENSION_NAME)

    extra_names: list[str] = []
    for soname, lib in sorted(runtime_libraries(built, repo).items()):
        shutil.copy2(lib, stage / soname)
        extra_names.append(soname)

    with tarfile.open(stage / PY_PACKAGE_TARBALL, "w:gz") as tar:
        tar.add(repo / PY_PACKAGE_DIRNAME, arcname=PY_PACKAGE_DIRNAME)
    with tarfile.open(stage / BENCH_TARBALL, "w:gz") as tar:
        tar.add(repo / GOLDEN_HTML_DIR, arcname="extract_golden")
    return extra_names


def install_consumer_layout(stage: Path, target: Path) -> Path:
    """Reproduce exactly what a consumer does: untar the package, drop every .so into it."""
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(stage / PY_PACKAGE_TARBALL) as tar:
        tar.extractall(target, filter="data")
    package_root = target / PY_PACKAGE_DIRNAME
    for so_file in stage.glob("*.so*"):
        shutil.copy2(so_file, package_root / "resiliparse" / so_file.name)
    return package_root


def check_contract(package_root: Path) -> str:
    """Run the published consumption snippet in a fresh interpreter and return its output."""
    env = os.environ | {"PYTHONPATH": str(package_root)}
    result = subprocess.run(
        [sys.executable, "-c", CONTRACT_SNIPPET],
        cwd=package_root.parent,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    logger.info("contract snippet output:\n%s", result.stdout)
    return result.stdout


BENCH_SOURCE = """
import glob, json, os, sys, time
from resiliparse._extract_rs import extract_plain_text

docs = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.html"))):
    docs.append(open(path, encoding="utf-8", errors="replace").read())
budget = float(sys.argv[2])
n_bytes = sum(len(d) for d in docs)
count = 0
start = time.perf_counter()
while time.perf_counter() - start < budget:
    for doc in docs:
        extract_plain_text(doc, main_content=True, preserve_formatting="markdown")
        count += 1
elapsed = time.perf_counter() - start
print(json.dumps({
    "docs": count,
    "elapsed": elapsed,
    "docs_per_second_per_core": count / elapsed,
    "ms_per_doc": 1000.0 * elapsed / count,
    "mean_doc_bytes": n_bytes / len(docs),
}))
"""


def benchmark(package_root: Path, html_dir: Path) -> dict[str, float]:
    """Measure single-core markdown main-content throughput on the fork's golden pages."""
    env = os.environ | {"PYTHONPATH": str(package_root)}
    result = subprocess.run(
        [sys.executable, "-c", BENCH_SOURCE, str(html_dir), str(BENCH_SECONDS)],
        cwd=package_root.parent,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    logger.info("throughput: %.1f docs/s/core (%.2f ms/doc)", stats["docs_per_second_per_core"], stats["ms_per_doc"])
    return stats


def publish(stage: Path, release: str) -> list[str]:
    """Upload every staged file to the dated release prefix and refresh the ``latest`` pointer."""
    fs = fsspec.filesystem("gcs")
    published: list[str] = []
    for name in dict.fromkeys([release, LATEST_RELEASE]):
        prefix = f"{GCS_ROOT}/{name}"
        for path in sorted(stage.iterdir()):
            destination = f"{prefix}/{path.name}"
            fs.put(str(path), destination)
            published.append(destination)
            logger.info("published %s", destination)
    return published


def build_command(args: argparse.Namespace) -> None:
    """Install the toolchain, build the extension, verify it locally and publish it."""
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    probe_toolchain()

    cargo_bin, cargo_overrides = ensure_rust(work_dir)
    tool_bins, libclang_dir = ensure_native_build_tools(work_dir)
    vcpkg_root = ensure_vcpkg(work_dir)
    repo = clone_fork(work_dir)
    trim_vcpkg_manifest(repo)

    commit_sha = capture(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    logger.info("building %s @ %s (%s)", FORK_URL, commit_sha, FORK_BRANCH)

    env = build_env(repo, cargo_bin, cargo_overrides, tool_bins, libclang_dir, vcpkg_root)
    # Run vcpkg up front: the crates' build scripts swallow its stderr, and this primes
    # the binary cache so their own `vcpkg install` calls are cache hits.
    run(
        [
            str(vcpkg_root / "vcpkg"),
            "install",
            "--triplet",
            VCPKG_TRIPLET,
            "--x-install-root",
            str(repo / "vcpkg_installed"),
        ],
        cwd=repo,
        env=env,
    )
    run(["cargo", "build", "--release", "-p", CARGO_PACKAGE], cwd=repo, env=env)

    stage = work_dir / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    extra_libraries = stage_artifact(repo, stage)

    # build.rs also bakes an absolute rpath into the vcpkg tree. Delete it so the local
    # contract check can only succeed through the $ORIGIN rpath a consumer will rely on.
    shutil.rmtree(repo / "vcpkg_installed")
    check_root = install_consumer_layout(stage, work_dir / "contract_check")
    contract_output = check_contract(check_root)
    stats = benchmark(check_root, repo / GOLDEN_HTML_DIR)

    manifest = {
        "commit_sha": commit_sha,
        "branch": FORK_BRANCH,
        "repository": FORK_URL,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": f"{platform.system()}-{platform.machine()} glibc {platform.libc_ver()[1]}",
        "python_version": platform.python_version(),
        "extension": EXTENSION_NAME,
        "extra_shared_libraries": extra_libraries,
        "throughput": stats,
        "contract_output": contract_output,
        "notes": [
            "Consume with: put _extract_rs.so AND every other published *.so into "
            "<resiliparse-py>/resiliparse/, untar resiliparse_py.tar.gz, "
            "export PYTHONPATH=<untarred resiliparse-py dir>.",
            f"lexbor is a shared library on the vcpkg {VCPKG_TRIPLET} triplet, so the extension is "
            f"linked with RUSTFLAGS='{ORIGIN_LINK_FLAGS}'; the lexbor .so files must sit in the same "
            "directory as _extract_rs.so. No LD_LIBRARY_PATH or other env var is needed.",
            "Built against the fork's vcpkg manifest trimmed to lexbor only (lz4/re2/uchardet/zlib are "
            "fastwarc dependencies the extractor does not link).",
            "pyo3 extension modules are not abi3: use the same CPython minor version recorded in python_version.",
            "Do not unpack under /tmp on Iris CPU workers: it is mounted noexec, so dlopen() of the "
            "extension fails there. Anywhere on the root overlay (e.g. /root) works.",
            "bench_html.tar.gz holds the fork's golden pages used for the throughput number.",
        ],
    }
    (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.log_file:
        shutil.copy2(args.log_file, stage / BUILD_LOG_NAME)

    published = publish(stage, args.release)
    logger.info("published %d files:\n%s", len(published), "\n".join(published))


def verify_command(args: argparse.Namespace) -> None:
    """Download a published release and exercise the consumption contract with no build toolchain."""
    probe_toolchain()
    fs = fsspec.filesystem("gcs")
    prefix = f"{GCS_ROOT}/{args.release}"
    work_dir = Path(args.work_dir)
    stage = work_dir / "download"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for remote in fs.ls(prefix, detail=False):
        name = remote.rsplit("/", 1)[-1]
        fs.get(f"gs://{remote}" if not remote.startswith("gs://") else remote, str(stage / name))
        logger.info("downloaded %s", name)

    package_root = install_consumer_layout(stage, work_dir / "consumer")
    logger.info("PYTHONPATH=%s", package_root)
    logger.info("contract output: %r", check_contract(package_root))

    html_dir = work_dir / "bench"
    with tarfile.open(stage / BENCH_TARBALL) as tar:
        tar.extractall(html_dir, filter="data")
    stats = benchmark(package_root, html_dir / "extract_golden")
    logger.info("VERIFY OK: %s", json.dumps(stats))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    # Not /tmp: it is mounted noexec on the Iris CPU workers, which breaks both the
    # build (exec) and importing the finished extension (dlopen needs PROT_EXEC).
    parser.add_argument("--work-dir", default=str(Path.home() / "resiliparse_rs_build"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build the extension and publish it to GCS")
    build_parser.add_argument("--release", default=datetime.datetime.now(datetime.UTC).strftime("%Y%m%d"))
    build_parser.add_argument("--log-file", default=None, help="driver log to publish as build_log.txt")
    build_parser.set_defaults(func=build_command)

    verify_parser = subparsers.add_parser("verify", help="check the consumption contract on a clean worker")
    verify_parser.add_argument("--release", default=LATEST_RELEASE)
    verify_parser.set_defaults(func=verify_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
