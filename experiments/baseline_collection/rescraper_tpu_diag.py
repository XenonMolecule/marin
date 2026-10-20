# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin down where ReScraper's TPU outputs diverge from the GPU reference.

On identical inputs, TPU greedy decoding disagrees with H200 greedy decoding on the *first generated
token* for most prompts over 4k tokens, so the difference is in the prefill forward pass. This job
runs the same prompts through several engine configurations, one per TPU chip, and a float32
HF-transformers CPU reference on the host, and records the first-token top-k logprobs plus a short
greedy continuation for each:

- ``default``:  tpu-inference JAX Qwen3 port (flax_nnx), chunked prefill at 8192 tokens, batched
- ``nochunk``:  JAX port, whole prompt prefilled in one step, one request at a time
- ``torchax``:  tpu-inference running vLLM's PyTorch Qwen3 via torchax (MODEL_IMPL_TYPE=vllm)
- ``torchax_nochunk``: torchax, one unchunked request at a time
- ``hf_cpu``:   transformers Qwen3ForCausalLM, float32, eager attention (ground truth)

Input is a jsonl.gz of ``{url, rendered, gpu_raw_output, tpu_raw_output, ...}`` rows; output is one
``<config>.json`` per configuration under ``--output``.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from enum import StrEnum
from typing import Any

import fsspec

from experiments.baseline_collection.rescraper_extract_tpu import (
    MAX_MODEL_LEN,
    RESCRAPER_HF_REPO,
    RESCRAPER_HF_REVISION,
    RESCRAPER_LIB_FILES,
    RESCRAPER_LIB_URL,
    RESCRAPER_MODEL_SUBDIR,
    RESCRAPER_PROMPT_FILE,
    _configure_engine_env,
)

logger = logging.getLogger(__name__)

NEW_TOKENS = 16
TOP_LOGPROBS = 8
HF_CPU_THREADS = 6


class EngineConfig(StrEnum):
    DEFAULT = "default"
    NOCHUNK = "nochunk"
    TORCHAX = "torchax"
    TORCHAX_NOCHUNK = "torchax_nochunk"
    DEFAULT_FP32 = "default_fp32"
    DEFAULT_Q_NDH = "default_q_ndh"
    TORCHAX_FP32 = "torchax_fp32"
    WEIGHTS_CHECK = "weights_check"
    DEFAULT_ROPEFIX = "default_ropefix"
    HF_CPU = "hf_cpu"


# Chip configurations, four per job; --configs picks the set.
CONFIG_SETS = {
    "split": (EngineConfig.DEFAULT, EngineConfig.NOCHUNK, EngineConfig.TORCHAX, EngineConfig.TORCHAX_NOCHUNK),
    # default_fp32: the JAX port with float32 weights/activations (precision vs structural bug);
    # default_q_ndh: the JAX port with LAYOUT_Q_PROJ_AS_NDH=1 (alternate q_proj kernel layout);
    # torchax_fp32: control for the float32 engine path.
    "precision": (
        EngineConfig.DEFAULT,
        EngineConfig.DEFAULT_FP32,
        EngineConfig.DEFAULT_Q_NDH,
        EngineConfig.TORCHAX_FP32,
    ),
    # weights_check: diff every parameter of the loaded JAX port against the safetensors.
    "weights": (EngineConfig.WEIGHTS_CHECK,),
    # default_ropefix: the JAX port with apply_rope_scaling neutralised for rope_type "default" (the
    # suspected bug: transformers 5 reports rope_scaling={"rope_type": "default", ...} for a config whose
    # JSON says null, and the port then applies Llama-3 scaling defaults to Qwen3).
    "ropefix": (EngineConfig.DEFAULT, EngineConfig.DEFAULT_ROPEFIX),
}
TORCHAX_CONFIGS = (EngineConfig.TORCHAX, EngineConfig.TORCHAX_NOCHUNK, EngineConfig.TORCHAX_FP32)
FP32_CONFIGS = (EngineConfig.DEFAULT_FP32, EngineConfig.TORCHAX_FP32)


def _read_jsonl_gz(path: str) -> list[dict]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompts(rows: list[dict], model_dir: str, lib_dir: str) -> list[dict]:
    """Tokenize each page exactly as the extraction job does (system prompt + numbered lines, thinking off).

    ``rendered`` may already carry the ``<lid:N>`` line numbers (the GPU trace sidecar stores the numbered
    input); only un-numbered text is numbered here.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    sys.path.insert(0, lib_dir)
    editops = importlib.import_module("editops")
    with open(os.path.join(model_dir, RESCRAPER_PROMPT_FILE), encoding="utf-8") as f:
        system_prompt = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, RESCRAPER_MODEL_SUBDIR))
    prompts = []
    for row in rows:
        user = row["rendered"] if row["rendered"].startswith("<lid:1>") else editops.number_lines(row["rendered"])
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append({"url": row["url"], "token_ids": tokenizer.encode(text)})
    return prompts


def rope_check(output: str) -> None:
    """Compare the JAX port's RoPE on this TPU with a float64 NumPy reference across positions 0..32k.

    Qwen3-0.6B geometry: head_dim 128, rope_theta 1e6, "split" (rotate-half) ordering.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from tpu_inference.layers.jax.rope_interface import apply_rope  # noqa: PLC0415

    head_dim, theta = 128, 1e6
    positions = np.arange(0, MAX_MODEL_LEN, dtype=np.int32)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((MAX_MODEL_LEN, 1, head_dim)).astype(np.float32)
    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    out_tpu = np.asarray(apply_rope(x_bf16, jnp.asarray(positions), head_dim, theta, None).astype(jnp.float32))
    # What the port actually runs under transformers 5: rope_scaling is the rope_parameters dict.
    as_loaded = {"rope_type": "default", "rope_theta": theta}
    out_scaled = np.asarray(apply_rope(x_bf16, jnp.asarray(positions), head_dim, theta, as_loaded).astype(jnp.float32))

    fraction = 2 * np.arange(0, head_dim // 2, dtype=np.float64) / head_dim
    timescale = 1.0 / (theta**fraction)
    ang = positions[:, None].astype(np.float64) * timescale[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    xb = np.asarray(x_bf16.astype(jnp.float32)).astype(np.float64)
    a, b = xb[..., : head_dim // 2], xb[..., head_dim // 2 :]
    ref = np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)
    ref_bf16 = np.asarray(jnp.asarray(ref, dtype=jnp.float32).astype(jnp.bfloat16).astype(jnp.float32))

    # Also the raw transcendental accuracy of XLA's float32 sin/cos on this device.
    ang32 = jnp.asarray(positions, dtype=jnp.float32)[:, None] * jnp.asarray(timescale, dtype=jnp.float32)[None, :]
    sin_tpu, cos_tpu = np.asarray(jnp.sin(ang32)).astype(np.float64), np.asarray(jnp.cos(ang32)).astype(np.float64)
    sin_ref, cos_ref = np.sin(ang), np.cos(ang)

    report = {"device": str(jax.devices()[0]), "buckets": {}}
    for lo, hi in [(0, 512), (512, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768)]:
        sl = slice(lo, hi)
        err = np.abs(out_tpu[sl] - ref[sl])
        report["buckets"][f"{lo}-{hi}"] = {
            "rope_max_abs_err": float(err.max()),
            "rope_mean_abs_err": float(err.mean()),
            "rope_frac_elements_not_bf16_equal": float((out_tpu[sl] != ref_bf16[sl]).mean()),
            "rope_with_default_dict_max_abs_err": float(np.abs(out_scaled[sl] - ref[sl]).max()),
            "rope_with_default_dict_mean_abs_err": float(np.abs(out_scaled[sl] - ref[sl]).mean()),
            "sin_max_abs_err": float(np.abs(sin_tpu[sl] - sin_ref[sl]).max()),
            "cos_max_abs_err": float(np.abs(cos_tpu[sl] - cos_ref[sl]).max()),
            "angle_max": float(ang[sl].max()),
        }
    _write(output, "rope_check", report)
    logger.info("rope_check: %s", json.dumps(report["buckets"], indent=1))


def _runner(llm: Any) -> Any:
    """The in-process TPUModelRunner behind an LLM (multiprocessing is off in this job)."""
    obj = llm.llm_engine
    for name in ("engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner"):
        if not hasattr(obj, name):
            raise AttributeError(f"{type(obj).__name__} has no {name}; attributes: {sorted(vars(obj))}")
        obj = getattr(obj, name)
    return obj


def weights_check(llm: Any, model_dir: str, output: str) -> None:
    """Diff the loaded flax_nnx parameters against the HF safetensors, using the loader's own maps."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from safetensors.numpy import load_file  # noqa: PLC0415

    state = _runner(llm).state
    loaded: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        key = ".".join(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))) for k in path)
        if hasattr(leaf, "shape"):
            loaded[key] = np.asarray(jax.device_get(leaf)).astype(np.float32)
    hf = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            hf.update(load_file(os.path.join(model_dir, f)))
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    n_heads, n_kv, hd, hidden = (
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
        cfg["hidden_size"],
    )

    def transform(name: str, w: np.ndarray) -> np.ndarray:
        w = w.astype(np.float32)
        if name.endswith("q_proj.weight"):
            return w.reshape(n_heads, hd, hidden).transpose(2, 0, 1)
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            return w.reshape(n_kv, hd, hidden).transpose(2, 0, 1)
        if name.endswith("o_proj.weight"):
            return w.reshape(hidden, n_heads, hd).transpose(1, 2, 0)
        if name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight", "lm_head.weight")):
            return w.T
        return w

    report = {"loaded_params": len(loaded), "hf_tensors": len(hf), "mismatch": [], "unmatched_hf": [], "matched": 0}
    used = set()
    for name, w in hf.items():
        want = transform(name, w)
        # loaded keys look like model.layers.0.self_attn.q_proj.kernel.value; match on the module path
        stem = name.rsplit(".", 1)[0]
        cands = [k for k in loaded if k.startswith(stem + ".") and loaded[k].shape == want.shape]
        if not cands:
            report["unmatched_hf"].append({"name": name, "shape": list(want.shape)})
            continue
        key = cands[0]
        used.add(key)
        err = float(np.abs(loaded[key] - want).max())
        if err > 1e-2:
            report["mismatch"].append(
                {"name": name, "loaded_as": key, "max_abs_err": err, "loaded_mean": float(loaded[key].mean())}
            )
        else:
            report["matched"] += 1
    report["loaded_but_not_in_hf"] = [
        {"key": k, "shape": list(v.shape), "mean": float(v.mean()), "std": float(v.std())}
        for k, v in loaded.items()
        if k not in used
    ]
    _write(output, EngineConfig.WEIGHTS_CHECK, report)
    logger.info(
        "weights_check: %s", json.dumps({k: v for k, v in report.items() if k != "loaded_but_not_in_hf"}, indent=1)
    )
    logger.info("loaded but not matched to any HF tensor: %s", json.dumps(report["loaded_but_not_in_hf"], indent=1))


def run_tpu_config(args: argparse.Namespace) -> None:
    _configure_engine_env()
    import jax  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415

    assert len(jax.devices()) == 1, jax.devices()
    if args.config == EngineConfig.DEFAULT:
        rope_check(args.output)
    if args.config == EngineConfig.DEFAULT_ROPEFIX:
        import tpu_inference.layers.jax.rope_interface as rope_interface  # noqa: PLC0415

        scaled = rope_interface.apply_rope_scaling

        def apply_rope_scaling_only_when_requested(freqs, rope_scaling):
            if rope_scaling.get("rope_type", "default") == "default":
                return freqs
            return scaled(freqs, rope_scaling)

        rope_interface.apply_rope_scaling = apply_rope_scaling_only_when_requested
    nochunk = args.config in (EngineConfig.NOCHUNK, EngineConfig.TORCHAX_NOCHUNK)
    engine_kwargs: dict[str, Any] = {"max_num_batched_tokens": MAX_MODEL_LEN, "max_num_seqs": 1} if nochunk else {}
    if args.config in FP32_CONFIGS:
        engine_kwargs["dtype"] = "float32"
    llm = LLM(
        model=os.path.join(args.model_path, RESCRAPER_MODEL_SUBDIR),
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=False,
        kv_cache_dtype="bfloat16",
        **engine_kwargs,
    )
    from vllm import SamplingParams  # noqa: PLC0415

    if args.config == EngineConfig.WEIGHTS_CHECK:
        weights_check(llm, os.path.join(args.model_path, RESCRAPER_MODEL_SUBDIR), args.output)
        return

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    inputs = [{"prompt_token_ids": p["token_ids"]} for p in prompts]
    results: dict[str, dict] = {p["url"]: {"prompt_tokens": len(p["token_ids"])} for p in prompts}

    # Greedy continuation first (robust), then first-token logprobs in a second pass.
    greedy = SamplingParams(temperature=0, max_tokens=NEW_TOKENS)
    for p, out in zip(prompts, llm.generate(inputs, greedy, use_tqdm=False), strict=True):
        results[p["url"]]["greedy_ids"] = list(out.outputs[0].token_ids)
        results[p["url"]]["greedy_text"] = out.outputs[0].text
    _write(args.output, args.config, results)

    with_lp = SamplingParams(temperature=0, max_tokens=1, logprobs=TOP_LOGPROBS)
    for p, out in zip(prompts, llm.generate(inputs, with_lp, use_tqdm=False), strict=True):
        lp = out.outputs[0].logprobs[0]
        results[p["url"]]["first_top"] = sorted(
            ((int(tid), float(v.logprob)) for tid, v in lp.items()), key=lambda x: -x[1]
        )
    _write(args.output, args.config, results)


def run_hf_cpu(prompts: list[dict], model_dir: str, output: str) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    torch.set_num_threads(HF_CPU_THREADS)  # leave cores for the four XLA compiles running alongside
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, RESCRAPER_MODEL_SUBDIR), torch_dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    results: dict[str, dict] = {}
    for i, p in enumerate(prompts):
        t0 = time.monotonic()
        ids = torch.tensor([p["token_ids"]])
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=NEW_TOKENS, do_sample=False, output_scores=True, return_dict_in_generate=True
            )
        logprobs = torch.log_softmax(out.scores[0][0], dim=-1)
        top = torch.topk(logprobs, TOP_LOGPROBS)
        results[p["url"]] = {
            "prompt_tokens": len(p["token_ids"]),
            "greedy_ids": out.sequences[0, ids.shape[1] :].tolist(),
            "first_top": [(int(t), float(v)) for v, t in zip(top.values, top.indices, strict=True)],
        }
        logger.info("hf_cpu %d/%d (%d tokens) in %.0fs", i + 1, len(prompts), len(p["token_ids"]), time.monotonic() - t0)
        _write(output, EngineConfig.HF_CPU, results)


def _write(output: str, config: str, results: dict) -> None:
    with fsspec.open(f"{output.rstrip('/')}/{config}.json", "wt", encoding="utf-8") as f:
        json.dump(results, f)


def run_coordinator(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    workdir = tempfile.mkdtemp(prefix="rescraper_diag_")
    lib_dir = os.path.join(workdir, "lib")
    os.makedirs(lib_dir)
    for name in RESCRAPER_LIB_FILES:
        urllib.request.urlretrieve(f"{RESCRAPER_LIB_URL}/{name}", os.path.join(lib_dir, name))
    model_dir = snapshot_download(
        RESCRAPER_HF_REPO,
        revision=RESCRAPER_HF_REVISION,
        allow_patterns=[f"{RESCRAPER_MODEL_SUBDIR}/*", RESCRAPER_PROMPT_FILE],
    )
    rows = _read_jsonl_gz(args.input)
    prompts = build_prompts(rows, model_dir, lib_dir)
    prompts_path = os.path.join(workdir, "prompts.json")
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    lengths = [len(p["token_ids"]) for p in prompts]
    logger.info("%d prompts, %d..%d tokens", len(prompts), min(lengths), max(lengths))

    procs = []
    for chip, config in enumerate(CONFIG_SETS[args.configs]):
        env = dict(
            os.environ,
            TPU_VISIBLE_CHIPS=str(chip),
            TPU_VISIBLE_DEVICES=str(chip),
            TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
            TPU_PROCESS_BOUNDS="1,1,1",
        )
        if config in TORCHAX_CONFIGS:
            env["MODEL_IMPL_TYPE"] = "vllm"
        else:
            env.pop("MODEL_IMPL_TYPE", None)
        if config == EngineConfig.DEFAULT_Q_NDH:
            env["LAYOUT_Q_PROJ_AS_NDH"] = "1"
        cmd = [
            sys.executable,
            "-m",
            "experiments.baseline_collection.rescraper_tpu_diag",
            "--config",
            config,
            "--prompts",
            prompts_path,
            "--model-path",
            model_dir,
            "--output",
            args.output,
        ]
        procs.append((config, subprocess.Popen(cmd, env=env)))

    if not args.skip_hf_cpu:
        run_hf_cpu(prompts, model_dir, args.output)

    for config, proc in procs:
        logger.info("%s exited %d", config, proc.wait())
    # Chip configs are independent diagnostics; a failed one is reported, not fatal.


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="gs:// diag prompts jsonl.gz (coordinator)")
    parser.add_argument("--output", required=True, help="gs:// directory")
    parser.add_argument("--configs", choices=sorted(CONFIG_SETS), default="split", help="which four chip configs to run")
    parser.add_argument("--skip-hf-cpu", action="store_true", help="reuse an earlier run's hf_cpu.json")
    parser.add_argument("--config", type=EngineConfig, help="chip process only")
    parser.add_argument("--prompts", help="chip process only: local prompts.json")
    parser.add_argument("--model-path", help="chip process only")
    args = parser.parse_args()
    if args.config:
        run_tpu_config(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
