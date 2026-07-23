#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Provision the infini-gram-mini runtime toolchain, then run the index pipeline.
#
# Iris jobs run the repo bundle in the marin env, which has no gcc-5 / sdsl
# toolchain and no way to attach a custom Docker image (EnvironmentSpec exposes
# only extras/pip_packages). So we bootstrap a micromamba env carrying the runtime
# libraries the prebuilt `cpp_indexing` binary links against, vendor the
# infini-gram-mini checkout, and hand off to the pipeline (which runs in the marin
# interpreter that is already active in the job).
#
# All args are forwarded verbatim to `python -m experiments.infinigram.pipeline`.
#
# Verify this end-to-end on ONE small target before fanning out (the toolchain is
# the highest-risk piece). See packaging.md.
set -euo pipefail

# The executable toolchain (micromamba binary, prebuilt cpp_indexing, compiled
# cpp_engine) must live on an exec-mounted filesystem; /tmp is often noexec in
# hardened containers, so default the work root to the job's working directory.
WORK_ROOT="${INFINIGRAM_WORK_ROOT:-${PWD}/.infinigram}"
mkdir -p "${WORK_ROOT}"

: "${INFINIGRAM_MINI_DIR:=${WORK_ROOT}/infini-gram-mini}"
MAMBA_ROOT="${MAMBA_ROOT:-${WORK_ROOT}/micromamba}"
MAMBA="${MAMBA_ROOT}/bin/micromamba"
ENV_PREFIX="${MAMBA_ROOT}/envs/infini-gram-mini"

# Pin the exact upstream commit once it is validated. Left as main until the
# smoke test confirms a working SHA; then replace and commit.
INFINIGRAM_MINI_REPO="${INFINIGRAM_MINI_REPO:-https://github.com/xuhaoxh/infini-gram-mini.git}"
INFINIGRAM_MINI_SHA="${INFINIGRAM_MINI_SHA:-main}"

# The smoke test (verify) needs the pybind engine; skip its compile when the
# pipeline is invoked with --no-verify.
COMPILE_ENGINE=1
for a in "$@"; do [[ "$a" == "--no-verify" ]] && COMPILE_ENGINE=0; done

log() { echo "[run_in_toolchain] $*" >&2; }

# 1. micromamba (referenced by full path — no shell hook / activate needed)
if [[ ! -x "${MAMBA}" ]]; then
  log "installing micromamba into ${MAMBA_ROOT}"
  mkdir -p "${MAMBA_ROOT}/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "${MAMBA_ROOT}" bin/micromamba
  chmod +x "${MAMBA}"
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"

# 2. toolchain env. The prebuilt cpp_indexing binary only needs a backward-
# compatible C++/GCC runtime at load time (recent libstdcxx-ng/libgcc-ng provide
# all older GLIBCXX/GCC symbols); pybind11 is added only when we compile the
# query engine for the smoke test. The README's ancient isl/mpc/mpfr pins are
# GCC *build* deps, not runtime deps, and are intentionally omitted.
# The query engine (engine/src/cpp_engine.cpp) is compiled against the repo's
# committed sdsl (sdsl/{include,lib}); that needs a C++ compiler (gxx) and
# pybind11. Indexing needs neither, so only add them when compiling the engine.
ENV_PKGS=(libstdcxx-ng libgcc-ng)
[[ "${COMPILE_ENGINE}" == "1" ]] && ENV_PKGS+=(gxx pybind11)
if [[ ! -d "${ENV_PREFIX}" ]]; then
  log "creating toolchain env at ${ENV_PREFIX}: ${ENV_PKGS[*]}"
  "${MAMBA}" create -y -p "${ENV_PREFIX}" -c conda-forge "${ENV_PKGS[@]}"
fi
# NOTE: do NOT prepend ${ENV_PREFIX}/bin to PATH — the compile deps pull a python
# into the env that would shadow marin's interpreter (which has fsspec/marin/
# zephyr). The compiler is invoked by full path, with PATH scoped to its subshell.
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# The prebuilt cpp_indexing (built with gcc-5) needs libcilkrts.so.5 (the Cilk
# runtime, removed from gcc>=8), which modern libgcc-ng lacks. Without it the final
# compression step silently fails -> a huge, unqueryable index. We vendor the .so
# (from psi4's gcc-5) so it ships in the job bundle -- no fragile runtime download.
if [[ ! -f "${ENV_PREFIX}/lib/libcilkrts.so.5" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VENDORED="${SCRIPT_DIR}/vendor/libcilkrts.so.5.0.0"
  if [[ -f "${VENDORED}" ]]; then
    cp "${VENDORED}" "${ENV_PREFIX}/lib/libcilkrts.so.5.0.0"
    ln -sf libcilkrts.so.5.0.0 "${ENV_PREFIX}/lib/libcilkrts.so.5"
    log "installed vendored libcilkrts.so.5"
  else
    log "WARN: vendored libcilkrts.so.5.0.0 not found at ${VENDORED}"
  fi
fi

# 3. vendored infini-gram-mini checkout
if [[ ! -d "${INFINIGRAM_MINI_DIR}/.git" ]]; then
  log "cloning infini-gram-mini @ ${INFINIGRAM_MINI_SHA}"
  git clone "${INFINIGRAM_MINI_REPO}" "${INFINIGRAM_MINI_DIR}"
  git -C "${INFINIGRAM_MINI_DIR}" checkout "${INFINIGRAM_MINI_SHA}"
fi
# indexing.py runs the final ./cpp_indexing compression with `2>/dev/null` AND
# never checks its return code, so a failed compression silently yields a huge,
# unqueryable (uncompressed) index. Un-suppress it so errors surface.
sed -i 's| 2>/dev/null||g' "${INFINIGRAM_MINI_DIR}/src/indexing.py" || true

# Committed sdsl shared libs must be loadable by the compiled cpp_engine.
export LD_LIBRARY_PATH="${INFINIGRAM_MINI_DIR}/sdsl/lib:${LD_LIBRARY_PATH}"

# 4. compile the query engine (engine/src/cpp_engine.cpp) against marin's python
#    -- the interpreter that imports it in the pipeline smoke test -- linking the
#    committed sdsl. pybind11 must be importable by that python.
if [[ "${COMPILE_ENGINE}" == "1" && -f "${INFINIGRAM_MINI_DIR}/engine/src/cpp_engine.cpp" ]]; then
  EXT="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
  ENGINE_SO="${INFINIGRAM_MINI_DIR}/engine/src/cpp_engine${EXT}"
  if [[ ! -f "${ENGINE_SO}" ]]; then
    python -m pip install --quiet pybind11 || uv pip install --quiet pybind11 || true
    PYBIND_INC="$(python -m pybind11 --includes)"
    CXX_BIN="$(ls "${ENV_PREFIX}"/bin/*-c++ 2>/dev/null | head -1 || command -v c++ || command -v g++)"
    log "compiling cpp_engine with ${CXX_BIN}"
    # PATH scoped to the compile so the conda toolchain finds its as/ld without
    # shadowing marin's python outside this subshell.
    ( export PATH="${ENV_PREFIX}/bin:${PATH}"
      cd "${INFINIGRAM_MINI_DIR}/engine" && "${CXX_BIN}" -std=c++17 -O3 -shared -fPIC \
        ${PYBIND_INC} src/cpp_engine.cpp -o "src/cpp_engine${EXT}" \
        -I../sdsl/include -L../sdsl/lib -lsdsl -ldivsufsort -ldivsufsort64 -pthread ) \
      || log "WARN: cpp_engine compile failed; querying/verify will not work"
  fi
fi

export INFINIGRAM_MINI_DIR

# Full round-trip self-test (faithful env: sdsl on LD_LIBRARY_PATH, engine built).
# Builds a tiny index, then loads the engine and counts -- always reports sizes +
# count to stderr and exits 1 so the diagnostic is visible in the job summary
# (task logs are unavailable on this cluster).
if [[ "${INFINIGRAM_SELFTEST:-0}" == "1" ]]; then
  ST="${WORK_ROOT}/selftest"; rm -rf "${ST}"; mkdir -p "${ST}/data"
  python - "${ST}/data/x.jsonl.gz" <<'PY'
import gzip, sys
with gzip.open(sys.argv[1], "wt") as f:
    for _ in range(5000):
        f.write('{"text":"the quick brown fox jumps over the lazy dog. the the united states of america."}\n')
PY
  ULIM=$(python -c 'import resource;print(resource.getrlimit(resource.RLIMIT_NOFILE)[1])')
  python "${INFINIGRAM_MINI_DIR}/src/indexing.py" --data_dir "${ST}/data" --save_dir "${ST}/idx" \
    --temp_dir "${ST}/tmp" --mem 4 --cpus 2 --ulimit "${ULIM}" >"${ST}/idx.log" 2>&1 || true
  # Write a full diagnostic to GCS (job task logs are unavailable on this cluster).
  INFINIGRAM_MINI_DIR="${INFINIGRAM_MINI_DIR}" python - "${ST}/idx" "${ST}/idx.log" <<'PY' || true
import json, os, sys, traceback
idx, logp = sys.argv[1], sys.argv[2]
rep = {"sizes": {}, "indexing_log_tail": open(logp).read()[-3000:] if os.path.exists(logp) else ""}
for n in os.listdir(idx) if os.path.isdir(idx) else []:
    rep["sizes"][n] = os.path.getsize(os.path.join(idx, n))
try:
    from experiments.infinigram.query import _load_engine_class
    eng = _load_engine_class()(index_dirs=[idx], load_to_ram=False, get_metadata=True)
    rep["count_the"] = eng.count("the")
    rep["find_the"] = str(eng.find("the"))[:500]
except Exception:
    rep["engine_error"] = traceback.format_exc()[-2000:]
import fsspec
with fsspec.open("gs://marin-us-central2/infinigram_selftest.json", "w") as f:
    json.dump(rep, f, indent=2)
print("wrote diagnostic to gs://marin-us-central2/infinigram_selftest.json")
PY
  echo "SELFTEST DONE" >&2
  exit 1
fi

log "handing off to pipeline: $*"
exec python -m experiments.infinigram.pipeline "$@"
