# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replicate ReScraper's pool inference (cx-cmu/ReScraper, a Qwen3-0.6B fine-tune) over one
Common Crawl WARC on a TPU node.

Follows ReScraper's released-corpus pipeline:

1. Dripper (MinerU-HTML 1.0.0, greedy, 8,192 new tokens, trafilatura fallback) selects each page's
   main content. Pages with an empty ``main_html`` leave the page universe.
2. ReScraper renders the raw HTML with the MinerU-HTML text renderer, numbers the lines
   (``<lid:n>``), generates at T=1.0 / top-p 1.0 / 3,072 new tokens with thinking disabled, and its
   own executor applies the generated program.

Both models are 0.6B, so each phase runs data-parallel: one single-chip engine per TPU chip, each
in its own process, with the groups split across chips.

Departures from the paper's setup:

* ReScraper's pool is already English-only. Raw WARCs are not, so pages are first filtered with
  fastText language id on the rendered text.
* ``cairosvg`` is stubbed when libcairo is missing. The renderer imports it to rasterize inline
  SVG into image URLs, which the ``txt`` output format never emits.

Output: ``<output>/group_NNNN.jsonl.gz`` (one row per decoded HTML record, dropped pages included
with their ``status``), ``group_NNNN.stats.json``, and ``dripper/group_NNNN.jsonl.gz``. Groups
already on GCS are skipped, so a preempted job resumes.

    python -m experiments.baseline_collection.rescraper_extract_tpu --num-chips 4 --render-procs 14 \\
        --warc s3://commoncrawl/crawl-data/.../x.warc.gz --output gs://bucket/users/me/rescraper/tag

``--input-jsonl`` runs the same pipeline over an already-decoded page sample instead of a WARC.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import importlib
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import urllib.request
from enum import StrEnum
from multiprocessing import Pool
from typing import Any

import fsspec

from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc

logger = logging.getLogger(__name__)

RESCRAPER_HF_REPO = "cx-cmu/ReScraper"
RESCRAPER_HF_REVISION = "a15cfd6d4db69640985a8c55844aa2e27b615796"
RESCRAPER_MODEL_SUBDIR = "models/ReScraper"
RESCRAPER_PROMPT_FILE = "prompts/student_system_stage2.txt"
RESCRAPER_COMMIT = "fb23198e9fb5b663a6793073f4ed5732951c2dd4"
RESCRAPER_LIB_URL = f"https://raw.githubusercontent.com/cxcscmu/ReScraper/{RESCRAPER_COMMIT}/lib"
RESCRAPER_LIB_FILES = ("editops.py", "rescraper_ops.py")

DRIPPER_HF_REPO = "opendatalab/MinerU-HTML"
DRIPPER_HF_REVISION = "baa1ca47ff92f1b8316bf952b9837aded85b6f10"
# dripper 1.0.0, the last commit before the package became mineru_html 1.1.
DRIPPER_COMMIT = "367999651a83474ad36316d17296e87c4f29d6a1"
DRIPPER_SOURCE_URL = f"https://codeload.github.com/opendatalab/MinerU-HTML/tar.gz/{DRIPPER_COMMIT}"
DRIPPER_MAX_MODEL_LEN = 40960
DRIPPER_MAX_NEW_TOKENS = 8 * 1024

LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

MAX_MODEL_LEN = 32768
MAX_NEW_TOKENS = 3072
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS
# A prompt longer than this many characters cannot fit the token budget; skip tokenizing it.
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * 8
MIN_RENDERED_CHARS = 20
RENDER_TIMEOUT = 60
# DCLM's English threshold on fastText lid.176.
MIN_ENGLISH_PROB = 0.65
ENGLISH_LABEL = "__label__en"

_WS_RE = re.compile(r"\s+")


class Phase(StrEnum):
    COORDINATOR = "coordinator"
    DRIPPER = "dripper"
    RESCRAPER = "rescraper"


class RenderTimeout(Exception):
    pass


def _raise_render_timeout(signum, frame):
    raise RenderTimeout()


def _stub_cairosvg_if_unavailable() -> None:
    try:
        importlib.import_module("cairosvg")
    except (ImportError, OSError):
        stub = types.ModuleType("cairosvg")

        def svg2png(*args, **kwargs):
            raise ValueError("cairosvg unavailable")

        stub.svg2png = svg2png  # type: ignore[attr-defined]
        sys.modules["cairosvg"] = stub


def _init_render_worker() -> None:
    signal.signal(signal.SIGALRM, _raise_render_timeout)
    _stub_cairosvg_if_unavailable()


def webkit_txt(html: str) -> str:
    """ReScraper's ``lib/render.py::webkit_txt``."""
    from bs4 import BeautifulSoup  # noqa: PLC0415
    from webpage_converter.convert import convert_html_to_structured_data  # noqa: PLC0415

    raw = convert_html_to_structured_data(html, output_format="txt")
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = re.sub(r"\|+", " ", text)
    text = re.sub(r"^[\s\-\|]+$", "", text, flags=re.M)
    text = re.sub(r"\\([\$\*\[\]_`#])", r"\1", text)
    return "\n".join(line for line in (_WS_RE.sub(" ", x).strip() for x in text.split("\n")) if line)


def render(html: str) -> tuple[str | None, str]:
    """Returns ``(rendered text, status)``; text is None unless status is ``ok``."""
    signal.alarm(RENDER_TIMEOUT)
    try:
        return webkit_txt(html), "ok"
    except RenderTimeout:
        return None, "render_timeout"
    except Exception as e:  # the renderer raises arbitrary errors on malformed pages; record and move on
        return None, f"render_error:{type(e).__name__}"
    finally:
        signal.alarm(0)


def english_prob(lid_model, text: str) -> float:
    """``P(English)`` when English is the top label, else 0. Whitespace is collapsed: fastText takes one line."""
    # The low-level predict returns (prob, label) pairs and avoids the NumPy-2 failure in ``model.predict``.
    preds = lid_model.f.predict(_WS_RE.sub(" ", text), 1, 0.0, "strict")
    if not preds:
        return 0.0
    prob, label = preds[0]
    return float(prob) if label == ENGLISH_LABEL else 0.0


def _prepared_path(workdir: str, g: int) -> str:
    return os.path.join(workdir, "prepared", f"group_{g:04d}.jsonl")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_gcs_jsonl_gz(fs, path: str, rows: list[dict]) -> None:
    with fs.open(path, "wb") as f, gzip.open(f, "wt") as z:
        for row in rows:
            z.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_gcs_jsonl_gz(fs, path: str) -> list[dict]:
    with fs.open(path, "rb") as f, gzip.open(f, "rt") as z:
        return [json.loads(line) for line in z]


def _configure_engine_env() -> None:
    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _load_single_chip_engine(
    model_path: str, max_model_len: int, kv_cache_dtype: str = "auto", prefix_caching: bool = True
) -> Any:
    import jax  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"chip pinning failed: this process sees {len(devices)} devices: {devices}")
    logger.info("Loading engine for %s on %s", model_path, devices[0])
    t0 = time.monotonic()
    # vllm-tpu's "auto" picks an fp8 (e5m2) KV cache on v6e; the reference GPU stack keeps bf16.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        enable_prefix_caching=prefix_caching,
        kv_cache_dtype=kv_cache_dtype,
    )
    logger.info("Engine for %s loaded in %.1fs on %s", model_path, time.monotonic() - t0, devices[0])
    return llm


def _chip_groups(workdir: str, chip: int, num_chips: int) -> list[int]:
    groups = sorted(int(name[6:10]) for name in os.listdir(os.path.join(workdir, "prepared")))
    return [g for g in groups if g % num_chips == chip]


def run_coordinator(args: argparse.Namespace) -> None:
    fs = fsspec.filesystem("gcs")
    out_prefix = args.output.replace("gs://", "").rstrip("/")

    if args.input_rendered_jsonl:
        # Pre-rendered pages (e.g. ReScraper's released SFT inputs): no decode, render, language id or
        # Dripper; every page goes straight to the ReScraper phase.
        records = [
            {
                "doc_id": r["warc_record_id"],
                "url": r["url"],
                "warc_hash": r["warc_hash"],
                "html": "",
                "rendered": r["rendered"],
            }
            for r in _read_gcs_jsonl_gz(fs, args.input_rendered_jsonl.replace("gs://", ""))
        ]
    elif args.input_jsonl:
        records = [
            {"doc_id": r["warc_record_id"], "url": r["url"], "warc_hash": r["warc_hash"], "html": r["html"]}
            for r in _read_gcs_jsonl_gz(fs, args.input_jsonl.replace("gs://", ""))
        ]
    else:
        records = _decode_one_warc(args.warc, with_text_body=False)
    if args.limit_records:
        records = records[: args.limit_records]
    groups = [records[i : i + args.group_size] for i in range(0, len(records), args.group_size)]
    pending = [g for g in range(len(groups)) if not fs.exists(f"{out_prefix}/group_{g:04d}.jsonl.gz")]
    logger.info("%d records, %d groups, %d pending", len(records), len(groups), len(pending))
    if not pending:
        return

    workdir = tempfile.mkdtemp(prefix="rescraper_")
    os.makedirs(os.path.join(workdir, "prepared"))
    os.makedirs(os.path.join(workdir, "lib"))

    import fasttext  # noqa: PLC0415

    lid_path = os.path.join(workdir, "lid.176.ftz")
    urllib.request.urlretrieve(LID_MODEL_URL, lid_path)
    lid_model = fasttext.load_model(lid_path)

    t0 = time.monotonic()
    counts: collections.Counter[str] = collections.Counter()
    with Pool(args.render_procs, initializer=_init_render_worker) as pool:
        for g in pending:
            if args.input_rendered_jsonl:
                rendered = [(r["rendered"], "ok") for r in groups[g]]
            else:
                rendered = pool.map(render, [r["html"] for r in groups[g]], chunksize=16)
            with open(_prepared_path(workdir, g), "w", encoding="utf-8") as f:
                for rec, (text, status) in zip(groups[g], rendered, strict=True):
                    row = {k: rec[k] for k in ("doc_id", "url", "warc_hash", "html")}
                    row["status"] = status
                    if text is not None:
                        row["rendered"] = text
                        if len(text) < MIN_RENDERED_CHARS:
                            row["status"] = "render_too_short"
                        else:
                            row["english_prob"] = 1.0 if args.input_rendered_jsonl else english_prob(lid_model, text)
                            row["status"] = "english" if row["english_prob"] >= MIN_ENGLISH_PROB else "not_english"
                    counts[row["status"]] += 1
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    render_seconds = time.monotonic() - t0
    logger.info(
        "Rendered %d groups in %.0fs with %d procs: %s", len(pending), render_seconds, args.render_procs, dict(counts)
    )
    del records, groups

    for name in RESCRAPER_LIB_FILES:
        urllib.request.urlretrieve(f"{RESCRAPER_LIB_URL}/{name}", os.path.join(workdir, "lib", name))
    with urllib.request.urlopen(DRIPPER_SOURCE_URL) as response:
        with tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as tar:
            tar.extractall(workdir, filter="data")

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    dripper_model = snapshot_download(DRIPPER_HF_REPO, revision=DRIPPER_HF_REVISION)
    rescraper_snapshot = snapshot_download(
        RESCRAPER_HF_REPO,
        revision=RESCRAPER_HF_REVISION,
        allow_patterns=[f"{RESCRAPER_MODEL_SUBDIR}/*", RESCRAPER_PROMPT_FILE],
    )
    model_paths = {Phase.DRIPPER: dripper_model, Phase.RESCRAPER: rescraper_snapshot}

    phases = (Phase.RESCRAPER,) if args.input_rendered_jsonl else (Phase.DRIPPER, Phase.RESCRAPER)
    for phase in phases:
        t0 = time.monotonic()
        procs = []
        for chip in range(args.num_chips):
            env = dict(
                os.environ,
                TPU_VISIBLE_CHIPS=str(chip),
                TPU_VISIBLE_DEVICES=str(chip),
                TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
                TPU_PROCESS_BOUNDS="1,1,1",
            )
            cmd = [sys.executable, "-m", "experiments.baseline_collection.rescraper_extract_tpu",
                   "--phase", phase, "--chip", str(chip), "--num-chips", str(args.num_chips),
                   "--workdir", workdir, "--model-path", model_paths[phase], "--output", args.output,
                   "--render-seconds", str(round(render_seconds, 1))]  # fmt: skip
            if args.input_rendered_jsonl:
                cmd.append("--no-dripper")
            cmd += ["--kv-cache-dtype", args.kv_cache_dtype, "--temperature", str(args.temperature)]
            if args.no_prefix_caching:
                cmd.append("--no-prefix-caching")
            procs.append(subprocess.Popen(cmd, env=env))
        exit_codes = [p.wait() for p in procs]
        if any(exit_codes):
            raise RuntimeError(f"{phase} phase failed; per-chip exit codes {exit_codes}")
        logger.info("%s phase done in %.0fs on %d chips", phase, time.monotonic() - t0, args.num_chips)


class TpuDripperBackend:
    """Dripper ``InferenceBackend`` over an already-loaded vLLM engine, at Dripper's own sampling settings."""

    def __init__(self, llm: Any):
        from vllm import SamplingParams  # noqa: PLC0415

        self._llm = llm
        self._sampling = SamplingParams(top_k=1, top_p=0.95, temperature=0, max_tokens=DRIPPER_MAX_NEW_TOKENS)
        self.completion_tokens = 0

    def generate(self, prompt_list: list[str]) -> list[Any]:
        from dripper.inference.imp import ModelResponse  # noqa: PLC0415

        outputs = self._llm.generate(prompt_list, self._sampling, use_tqdm=False)
        self.completion_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
        return [ModelResponse(generated_text=o.outputs[0].text) for o in outputs]


def run_dripper(args: argparse.Namespace) -> None:
    fs = fsspec.filesystem("gcs")
    out_prefix = args.output.replace("gs://", "").rstrip("/")
    todo = [
        g
        for g in _chip_groups(args.workdir, args.chip, args.num_chips)
        if not fs.exists(f"{out_prefix}/dripper/group_{g:04d}.jsonl.gz")
    ]
    if not todo:
        return

    sys.path.insert(0, os.path.join(args.workdir, f"MinerU-HTML-{DRIPPER_COMMIT}"))
    from dripper.api import Dripper  # noqa: PLC0415

    dripper = Dripper(config={"model_path": args.model_path, "use_fall_back": True, "raise_errors": False})
    backend = TpuDripperBackend(_load_single_chip_engine(args.model_path, DRIPPER_MAX_MODEL_LEN))
    dripper._llm = backend

    for g in todo:
        rows = [r for r in _read_jsonl(_prepared_path(args.workdir, g)) if r["status"] == "english"]
        t0 = time.monotonic()
        tokens_before = backend.completion_tokens
        # One vLLM call per group. ReScraper's step 1 split a shard into 500-page / 5M-character
        # batches to bound GPU memory; decoding is greedy, so batching does not change the output.
        results = dripper.process([r["html"] for r in rows])
        out_rows = [
            {"doc_id": row["doc_id"], "main_html": result.main_html or ""}
            for row, result in zip(rows, results, strict=True)
        ]
        seconds = time.monotonic() - t0
        _write_gcs_jsonl_gz(fs, f"{out_prefix}/dripper/group_{g:04d}.jsonl.gz", out_rows)
        logger.info(
            "chip %d dripper group %d: %d pages, %d with main_html, %d completion tokens, %.0fs",
            args.chip, g, len(out_rows), sum(bool(r["main_html"].strip()) for r in out_rows),
            backend.completion_tokens - tokens_before, seconds,
        )  # fmt: skip


def run_rescraper(args: argparse.Namespace) -> None:
    fs = fsspec.filesystem("gcs")
    out_prefix = args.output.replace("gs://", "").rstrip("/")
    todo = [
        g
        for g in _chip_groups(args.workdir, args.chip, args.num_chips)
        if not fs.exists(f"{out_prefix}/group_{g:04d}.jsonl.gz")
    ]
    if not todo:
        return

    sys.path.insert(0, os.path.join(args.workdir, "lib"))
    editops = importlib.import_module("editops")
    rescraper_ops = importlib.import_module("rescraper_ops")
    with open(os.path.join(args.model_path, RESCRAPER_PROMPT_FILE), encoding="utf-8") as f:
        system_prompt = f.read().strip()

    llm = _load_single_chip_engine(
        os.path.join(args.model_path, RESCRAPER_MODEL_SUBDIR),
        MAX_MODEL_LEN,
        args.kv_cache_dtype,
        prefix_caching=not args.no_prefix_caching,
    )
    # vLLM's platform is resolved when the engine loads; importing SamplingParams before that fails on TPU.
    from vllm import SamplingParams  # noqa: PLC0415

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=MAX_NEW_TOKENS)

    for g in todo:
        t_start = time.monotonic()
        rows = _read_jsonl(_prepared_path(args.workdir, g))
        if args.no_dripper:
            main_html = {r["doc_id"]: "pre-rendered" for r in rows}
        else:
            main_html = {
                r["doc_id"]: r["main_html"]
                for r in _read_gcs_jsonl_gz(fs, f"{out_prefix}/dripper/group_{g:04d}.jsonl.gz")
            }
        generate: list[tuple[dict, str, list[int]]] = []
        for row in rows:
            row["html_chars"] = len(row.pop("html"))
            row["text"] = ""
            if row["status"] != "english":
                row.pop("rendered", None)
                continue
            row["main_html_chars"] = len(main_html[row["doc_id"]])
            if not main_html[row["doc_id"]].strip():
                row["status"] = "no_main_html"
                continue
            numbered = editops.number_lines(row["rendered"])
            row["status"] = "prompt_too_long"
            if len(numbered) > MAX_PROMPT_CHARS:
                continue
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": numbered}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            token_ids = tokenizer.encode(prompt)
            if len(token_ids) > MAX_PROMPT_TOKENS:
                continue
            row["status"] = "generated"
            row["prompt_tokens"] = len(token_ids)
            generate.append((row, numbered, token_ids))
        t_prep = time.monotonic() - t_start

        t_gen0 = time.monotonic()
        prompts = [{"prompt_token_ids": ids} for _, _, ids in generate]
        outputs = llm.generate(prompts, sampling, use_tqdm=False) if prompts else []
        t_gen = time.monotonic() - t_gen0

        for (row, numbered, _), out in zip(generate, outputs, strict=True):
            completion = out.outputs[0]
            row["raw_output"] = completion.text
            row["completion_tokens"] = len(completion.token_ids)
            row["finish_reason"] = completion.finish_reason
            try:
                tag, text = rescraper_ops.body_from_prediction_dfirst(numbered, completion.text)
            except Exception as e:  # upstream counts executor failures as parse_fail and drops the page
                row["status"] = f"parse_fail:{type(e).__name__}"
                continue
            row["e2e_tag"] = tag
            row["text"] = text or ""
            row["status"] = "kept" if tag != "<delete>" and row["text"].strip() else "deleted"

        counts: collections.Counter[str] = collections.Counter()
        for row in rows:
            counts[row["status"]] += 1
            if row.get("e2e_tag"):
                counts[f"tag{row['e2e_tag']}"] += 1
        stats = {
            "group": g,
            "chip": args.chip,
            "num_chips": args.num_chips,
            "records": len(rows),
            "generated": len(generate),
            "counts": dict(counts),
            "prompt_tokens": sum(len(ids) for _, _, ids in generate),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
            "kept_chars": sum(len(r["text"]) for r in rows),
            "seconds": {"prep": round(t_prep, 1), "generate": round(t_gen, 1)},
            "render_seconds_all_groups": args.render_seconds,
        }
        # Stats first: the group file is the done marker, so it must land last.
        with fs.open(f"{out_prefix}/group_{g:04d}.stats.json", "w") as f:
            json.dump(stats, f)
        _write_gcs_jsonl_gz(fs, f"{out_prefix}/group_{g:04d}.jsonl.gz", rows)
        logger.info("chip %d rescraper group %d: %s", args.chip, g, json.dumps(stats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Phase, choices=list(Phase), default=Phase.COORDINATOR)
    parser.add_argument("--output", required=True, help="gs:// directory for group outputs")
    parser.add_argument("--num-chips", type=int, required=True, help="TPU chips on this host; one engine each")
    coordinator = parser.add_argument_group("coordinator")
    coordinator.add_argument("--warc", help="s3://commoncrawl/... WARC path")
    coordinator.add_argument(
        "--input-jsonl", help="gs:// jsonl.gz of pages (warc_record_id, url, warc_hash, html) to run instead of a WARC"
    )
    coordinator.add_argument(
        "--input-rendered-jsonl",
        help="gs:// jsonl.gz of pre-rendered pages (warc_record_id, url, warc_hash, rendered); ReScraper phase only",
    )
    coordinator.add_argument("--group-size", type=int, default=1000, help="decoded records per output file")
    coordinator.add_argument("--limit-records", type=int, default=None, help="first N decoded records (smoke test)")
    coordinator.add_argument("--render-procs", type=int)
    chip = parser.add_argument_group("per-chip phases (spawned by the coordinator)")
    chip.add_argument("--chip", type=int)
    chip.add_argument("--workdir")
    chip.add_argument("--model-path")
    chip.add_argument("--render-seconds", type=float)
    chip.add_argument("--no-dripper", action="store_true", help="treat every page as having Dripper main content")
    parser.add_argument("--kv-cache-dtype", default="auto", help="vLLM kv_cache_dtype for the ReScraper engine")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="ReScraper sampling temperature (paper: 1.0; 0 = greedy)"
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="disable vLLM prefix caching for the ReScraper engine (tpu-inference#3735: cross-request KV corruption on v6e)",
    )
    args = parser.parse_args()
    tag = args.phase if args.phase == Phase.COORDINATOR else f"{args.phase} chip {args.chip}"
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [{tag}] %(message)s")

    if args.phase == Phase.COORDINATOR:
        inputs = [x for x in (args.warc, args.input_jsonl, args.input_rendered_jsonl) if x]
        if len(inputs) != 1 or not args.render_procs:
            parser.error(
                "the coordinator needs exactly one of --warc / --input-jsonl / --input-rendered-jsonl, and --render-procs"
            )
        run_coordinator(args)
        return
    _configure_engine_env()
    if args.phase == Phase.DRIPPER:
        run_dripper(args)
    else:
        run_rescraper(args)


if __name__ == "__main__":
    main()
