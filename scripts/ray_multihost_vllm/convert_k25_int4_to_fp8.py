#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Kimi K2.5 INT4 (compressed-tensors) to FP8 (DeepseekV3ForCausalLM).

Takes the unsloth/Kimi-K2.5 model (INT4 quantized, KimiK25ForConditionalGeneration)
and produces FP8 weights compatible with the DeepseekV3ForCausalLM architecture,
matching the format of moonshotai/Kimi-K2-Instruct.

Steps:
  1. Load INT4 packed weights + group scales from unsloth/Kimi-K2.5
  2. Dequantize INT4 -> BF16 (unpack + apply group scales)
  3. Requantize BF16 -> FP8 e4m3 (per-tensor scaling)
  4. Strip 'language_model.' prefix from weight names
  5. Drop vision_tower and mm_projector weights
  6. Write new safetensors + config matching K2-Instruct format

Usage:
  python convert_k25_int4_to_fp8.py \
    --input gs://marin-us-central1/models/unsloth--Kimi-K2.5 \
    --output gs://marin-us-central1/models/kimi-k25-fp8 \
    --temp-dir /tmp/k25_convert

Requirements:
  pip install safetensors torch numpy compressed-tensors
  Needs ~100GB RAM (processes one safetensors shard at a time)
"""

import argparse
import json
import os
import subprocess

import torch
from safetensors.torch import load_file, save_file


def unpack_int4_symmetric(
    packed: torch.Tensor, scale: torch.Tensor, weight_shape: tuple[int, ...], group_size: int = 32
) -> torch.Tensor:
    """Unpack INT4 packed weights and dequantize to BF16.

    Args:
        packed: int32 tensor with 8 INT4 values per element
        scale: float32 per-group scales
        weight_shape: original (unpacked) weight shape
        group_size: quantization group size

    Returns:
        BF16 tensor with original weight shape
    """
    # Unpack 8 int4 values from each int32
    # packed shape: (out_features, in_features // 8) or similar
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    # Extract each 4-bit value
    unpacked = (packed.unsqueeze(-1) >> shifts) & 0xF
    # Sign-extend: values >= 8 are negative in 4-bit two's complement
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    # Reshape to flatten the pack dimension
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)

    # Trim to actual size if needed (last group might be padded)
    target_dim = weight_shape[-1] if len(weight_shape) > 0 else unpacked.shape[-1]
    if unpacked.shape[-1] > target_dim:
        unpacked = unpacked[..., :target_dim]

    # Apply group-wise scales
    # scale shape: (out_features, in_features // group_size)
    # unpacked shape: (out_features, in_features)
    unpacked_bf16 = unpacked.to(torch.bfloat16)
    if scale is not None and scale.numel() > 0:
        num_groups = scale.shape[-1]
        in_features = unpacked_bf16.shape[-1]
        actual_group_size = in_features // num_groups if num_groups > 0 else in_features

        grouped = unpacked_bf16.reshape(*unpacked_bf16.shape[:-1], num_groups, actual_group_size)
        scale_bf16 = scale.to(torch.bfloat16)
        scaled = grouped * scale_bf16.unsqueeze(-1)
        unpacked_bf16 = scaled.reshape(*unpacked_bf16.shape[:-1], in_features)

    return unpacked_bf16


def quantize_to_fp8(tensor: torch.Tensor, block_size: tuple[int, int] = (128, 128)) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 tensor to FP8 e4m3 with block-wise scales.

    K2-Instruct uses weight_block_size=[128, 128], meaning each 128x128 block
    has its own scale factor. For a [2048, 7168] weight, scales are [16, 56].

    Returns:
        (fp8_tensor, scale_inv) where scale_inv shape = [rows//block_r, cols//block_c]
    """
    max_fp8 = 448.0  # max value for e4m3
    block_r, block_c = block_size

    if tensor.ndim == 2:
        rows, cols = tensor.shape
        # Pad if needed (shouldn't be for these dimensions)
        pad_r = (block_r - rows % block_r) % block_r
        pad_c = (block_c - cols % block_c) % block_c
        if pad_r > 0 or pad_c > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_c, 0, pad_r))
            rows, cols = tensor.shape

        # Reshape into blocks
        n_blocks_r = rows // block_r
        n_blocks_c = cols // block_c
        blocked = tensor.reshape(n_blocks_r, block_r, n_blocks_c, block_c)
        blocked = blocked.permute(0, 2, 1, 3)  # [n_br, n_bc, block_r, block_c]

        # Per-block max abs
        amax = blocked.abs().amax(dim=(-2, -1))  # [n_br, n_bc]
        scale = (amax / max_fp8).clamp(min=1e-12)

        # Scale each block
        scaled = blocked / scale[:, :, None, None]
        # Reshape back
        scaled = scaled.permute(0, 2, 1, 3).reshape(rows, cols)

        # Remove padding
        if pad_r > 0:
            scaled = scaled[: rows - pad_r, :]
        if pad_c > 0:
            scaled = scaled[:, : cols - pad_c]

        fp8 = scaled.to(torch.float8_e4m3fn)
        scale_inv = (1.0 / scale).to(torch.float32)

        return fp8, scale_inv
    else:
        # For non-2D tensors (rare), fall back to per-tensor
        amax = tensor.abs().amax()
        scale = (amax / max_fp8).clamp(min=1e-12)
        fp8 = (tensor / scale).to(torch.float8_e4m3fn)
        scale_inv = (1.0 / scale).to(torch.float32)
        return fp8, scale_inv


def process_shard(
    input_path: str, shard_name: str, temp_dir: str, all_weight_shapes: dict
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Process a single safetensors shard: dequant INT4, requant FP8, rename.

    Returns:
        (output_tensors, processed_weight_bases) where processed_weight_bases
        tracks which base weights (without _packed/_scale/_shape suffix) were processed.
    """
    shard_path = os.path.join(temp_dir, "input", shard_name)
    if not os.path.exists(shard_path):
        # Download from GCS
        os.makedirs(os.path.join(temp_dir, "input"), exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", f"{input_path}/{shard_name}", shard_path], check=True)

    tensors = load_file(shard_path)
    output = {}
    processed_bases = set()

    for key, tensor in tensors.items():
        # Skip vision and projector weights
        if key.startswith("vision_tower.") or key.startswith("mm_projector."):
            continue

        # Strip language_model. prefix
        out_key = key
        if out_key.startswith("language_model."):
            out_key = out_key[len("language_model.") :]

        # Handle packed INT4 weights
        if key.endswith(".weight_packed"):
            base = key[: -len(".weight_packed")]
            if base in processed_bases:
                continue
            processed_bases.add(base)

            packed = tensor
            scale_key = base + ".weight_scale"
            shape_key = base + ".weight_shape"

            scale = tensors.get(scale_key)
            shape_tensor = tensors.get(shape_key)

            # Get original shape
            if shape_tensor is not None:
                orig_shape = tuple(shape_tensor.to(torch.int64).tolist())
            elif base in all_weight_shapes:
                orig_shape = all_weight_shapes[base]
            else:
                # Infer from packed dimensions
                orig_shape = (packed.shape[0], packed.shape[1] * 8)

            # Dequantize INT4 -> BF16
            bf16_weight = unpack_int4_symmetric(packed, scale, orig_shape)

            # Quantize BF16 -> FP8
            fp8_weight, scale_inv = quantize_to_fp8(bf16_weight)

            out_base = out_key[: -len(".weight_packed")] if out_key.endswith(".weight_packed") else out_key
            if out_base.startswith("language_model."):
                out_base = out_base[len("language_model.") :]

            output[out_base + ".weight"] = fp8_weight
            output[out_base + ".weight_scale_inv"] = scale_inv

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            # These are consumed by the _packed handler above
            continue

        else:
            # Non-quantized weight (attention, norms, embeddings, shared experts)
            # Keep as-is (already BF16)
            output[out_key] = tensor

    # Clean up downloaded shard
    if os.path.exists(shard_path):
        os.remove(shard_path)

    return output, processed_bases


def main():
    parser = argparse.ArgumentParser(description="Convert K2.5 INT4 to FP8")
    parser.add_argument("--input", required=True, help="GCS path to unsloth/Kimi-K2.5")
    parser.add_argument("--output", required=True, help="GCS path for FP8 output")
    parser.add_argument("--temp-dir", default="/tmp/k25_convert", help="Local temp directory")
    parser.add_argument("--group-size", type=int, default=32, help="INT4 group size")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.temp_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(args.temp_dir, "output"), exist_ok=True)

    # Download config and index
    print("Downloading config files...")
    for fname in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "tokenization_kimi.py",
        "tiktoken.model",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
        "generation_config.json",
    ]:
        src = f"{args.input}/{fname}"
        dst = os.path.join(args.temp_dir, "input", fname)
        try:
            subprocess.run(["gcloud", "storage", "cp", src, dst], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"  Warning: {fname} not found, skipping")

    # Load the safetensors index
    index_path = os.path.join(args.temp_dir, "input", "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Group weights by shard file
    shard_to_weights: dict[str, list[str]] = {}
    for weight_name, shard_name in weight_map.items():
        shard_to_weights.setdefault(shard_name, []).append(weight_name)

    # Collect weight shapes from shape tensors (we'll need these)
    all_weight_shapes = {}

    # Process each shard
    new_weight_map = {}
    total_shards = len(shard_to_weights)
    output_shard_idx = 0

    for shard_idx, (shard_name, weights) in enumerate(sorted(shard_to_weights.items())):
        print(f"\n[{shard_idx+1}/{total_shards}] Processing {shard_name} ({len(weights)} weights)...")

        output_tensors, processed_bases = process_shard(args.input, shard_name, args.temp_dir, all_weight_shapes)

        if not output_tensors:
            print("  Skipped (all vision/projector weights)")
            continue

        # Save output shard
        output_shard_idx += 1
        out_shard_name = f"model-{output_shard_idx:05d}-of-TOTAL.safetensors"
        out_shard_path = os.path.join(args.temp_dir, "output", out_shard_name)

        # Skip if already uploaded (resume support)
        check = subprocess.run(["gcloud", "storage", "ls", f"{args.output}/{out_shard_name}"], capture_output=True)
        if check.returncode == 0:
            print(f"  SKIP (already in GCS): {out_shard_name}")
            for tensor_name in output_tensors:
                new_weight_map[tensor_name] = out_shard_name
            continue

        save_file(output_tensors, out_shard_path)
        shard_size = os.path.getsize(out_shard_path) / 1e9
        print(f"  Wrote {out_shard_name} ({shard_size:.1f} GB, {len(output_tensors)} tensors)")

        # Upload to GCS
        subprocess.run(["gcloud", "storage", "cp", out_shard_path, f"{args.output}/{out_shard_name}"], check=True)
        os.remove(out_shard_path)

        # Update weight map
        for tensor_name in output_tensors:
            new_weight_map[tensor_name] = out_shard_name

    # Fix shard names (replace TOTAL with actual count)
    total_output_shards = output_shard_idx
    final_weight_map = {}
    for tensor_name, shard_name in new_weight_map.items():
        final_name = shard_name.replace("TOTAL", f"{total_output_shards:05d}")
        final_weight_map[tensor_name] = final_name
        # Rename on GCS if needed
        if final_name != shard_name:
            result = subprocess.run(
                ["gcloud", "storage", "mv", f"{args.output}/{shard_name}", f"{args.output}/{final_name}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  Warning: rename failed for {shard_name}: {result.stderr.strip()[:100]}")
            else:
                print(f"  Renamed {shard_name} -> {final_name}")

    # Write new index
    new_index = {
        "metadata": {"total_size": sum(1 for _ in final_weight_map)},
        "weight_map": final_weight_map,
    }
    index_out = os.path.join(args.temp_dir, "output", "model.safetensors.index.json")
    with open(index_out, "w") as f:
        json.dump(new_index, f, indent=2)
    subprocess.run(["gcloud", "storage", "cp", index_out, f"{args.output}/model.safetensors.index.json"], check=True)

    # Create new config.json matching K2-Instruct format
    config_path = os.path.join(args.temp_dir, "input", "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Extract text_config if this is a multimodal model
    if "text_config" in config:
        new_config = config["text_config"]
    else:
        new_config = config

    # Set architecture and quantization to match K2-Instruct
    new_config["architectures"] = ["DeepseekV3ForCausalLM"]
    new_config["model_type"] = "deepseek_v3"
    new_config["torch_dtype"] = "bfloat16"
    new_config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": None,  # per-tensor, not block
    }

    config_out = os.path.join(args.temp_dir, "output", "config.json")
    with open(config_out, "w") as f:
        json.dump(new_config, f, indent=2)
    subprocess.run(["gcloud", "storage", "cp", config_out, f"{args.output}/config.json"], check=True)

    # Copy tokenizer files
    for fname in [
        "tokenizer_config.json",
        "tokenization_kimi.py",
        "tiktoken.model",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
        "generation_config.json",
    ]:
        src = os.path.join(args.temp_dir, "input", fname)
        if os.path.exists(src):
            # Fix tokenizer_config.json vocab_file issue
            if fname == "tokenizer_config.json":
                with open(src) as f:
                    tc = json.load(f)
                if "vocab_file" in tc and tc["vocab_file"] is None:
                    del tc["vocab_file"]
                with open(src, "w") as f:
                    json.dump(tc, f, indent=2, ensure_ascii=False)

            subprocess.run(["gcloud", "storage", "cp", src, f"{args.output}/{fname}"], check=True)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"Output: {args.output}")
    print(f"Total output shards: {total_output_shards}")
    print(f"Total output weights: {len(final_weight_map)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
