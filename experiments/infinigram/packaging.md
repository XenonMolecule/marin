# Packaging infini-gram-mini for Marin index builds

infini-gram-mini ([xuhaoxh/infini-gram-mini](https://github.com/xuhaoxh/infini-gram-mini),
Xu et al. 2025, arXiv:2506.12229) is not a pip package.

- **Indexing** runs `src/indexing.py`, which shells out to a **prebuilt**
  `src/cpp_indexing` binary (~575 KB, committed) needing only a backward-compatible
  libstdc++ at load time. It also raises `RLIMIT_NOFILE` (pass `--ulimit <hard>`)
  and calls `./cpp_indexing` relatively (run with `cwd=src/`).
- **Querying** uses `InfiniGramMiniEngine` in the separate **`engine/`** dir
  (`engine/src/engine.py`), backed by `engine/src/cpp_engine.cpp` (C++17) that must
  be compiled against the repo's **committed** sdsl (`sdsl/{include,lib}` ship in
  the repo — no sdsl build needed): `gxx` + `pybind11`, then
  `c++ ... $(python -m pybind11 --includes) src/cpp_engine.cpp -I../sdsl/include
  -L../sdsl/lib -lsdsl -ldivsufsort -ldivsufsort64`. Import it as
  `from src.engine import InfiniGramMiniEngine` with `<repo>/engine` on `sys.path`.
  `find()`→`{cnt, segment_by_shard}`, `count()`→`{count}`,
  `get_doc_by_rank(s, rank, needle_len, max_ctx_len)`→`{text, metadata:{...,
  metadata:{<record fields incl. url>}}}`.

## Why not a Docker image

Iris `EnvironmentSpec` exposes only `pip_packages`, `extras`, and `env_vars` —
there is no field to attach a custom base image. So instead of baking an image we
**bootstrap the runtime inside the job** via
[`run_in_toolchain.sh`](./run_in_toolchain.sh), which:

1. installs `micromamba` under a work root **on the job's working directory**
   (`$PWD/.infinigram`), not `/tmp` — `/tmp` is often mounted `noexec` in the
   container, which blocks executing the downloaded binary; the extracted
   micromamba is also `chmod +x`'d, and referenced by full path (no `shell hook`
   / `activate`, which don't work in a non-interactive job shell);
2. creates the toolchain env from conda-forge with just `libstdcxx-ng` +
   `libgcc-ng` (a recent libstdc++ is backward-compatible and supplies every
   older `GLIBCXX`/GCC symbol the prebuilt `cpp_indexing` needs) and puts its
   `lib/` on `LD_LIBRARY_PATH`. The README's ancient `isl=0.12.2` / `mpc=1.0.3` /
   `mpfr=3.1.4` pins are GCC *build* deps, are **not** co-installable on current
   conda-forge, and are intentionally omitted;
3. clones the repo into `INFINIGRAM_MINI_DIR` (`$WORK_ROOT/infini-gram-mini`);
4. compiles `cpp_engine` **only when verify is requested** (skipped under
   `--no-verify`), built against the marin `python` that will import it;
5. execs `python -m experiments.infinigram.pipeline "$@"` — the pipeline runs in
   the marin interpreter already active in the job (it imports `marin`/`zephyr`)
   while the toolchain env only supplies runtime `.so`s via `LD_LIBRARY_PATH`.

Notes from bringing this up on the cluster:
- The container has **no `gsutil`** — all GCS I/O goes through gcsfs/fsspec
  (`gcs_io.py`), never a shell CLI.
- `indexing.py` raises `RLIMIT_NOFILE` to 1048576, which a non-root process may
  not do; `build.py` passes the container's real hard limit via `--ulimit`.
- `indexing.py` calls `./cpp_indexing` by relative path, so it is run with
  `cwd=<repo>/src`.

## Pin the SHA before fan-out

`run_in_toolchain.sh` defaults `INFINIGRAM_MINI_SHA=main`. Once a green
end-to-end run confirms a working SHA, hardcode it and commit.

## Fallback: original infini-gram (pip wheel)

If the mini toolchain proves too fragile, switch to
[original infini-gram](https://github.com/liujch1998/infini-gram)
(`pip install infini_gram`) — a normal manylinux wheel needing **no** conda
toolchain and **no** `run_in_toolchain.sh`: set `EnvironmentSpec(pip_packages=
["infini_gram"])` and call its `indexing.py --data_dir/--save_dir/--tokenizer
<meta-llama/Meta-Llama-3.1-8B>/--cpus/--shard`. Trade-off: it indexes token ids
(needs a tokenizer, no free byte-level substring or metadata search) vs mini's
byte-level FM-index with searchable metadata at 0.44× size. Only `build.py`
changes; the registry, resolve, stage, upload, query, and launcher are unchanged
because staging already produces `.jsonl.gz` with a `text` field that both
indexers accept.
