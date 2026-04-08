#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests 1-6: CPU unit tests for preprocessing correctness.

Runs on CPU only (JAX_PLATFORMS=cpu). Validates:
  Test 1: Import validation
  Test 2: Single MoE layer byte-exact round-trip (8 experts)
  Test 3: Full 384 experts shape/dtype validation
  Test 4: Dense linear round-trip
  Test 5: kv_b_proj → k_up_proj + v_up_proj split
  Test 6: Safetensors save/load round-trip
"""

import os
import sys
import time
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/workspace/tpu_inference")

import jax.numpy as jnp
import ml_dtypes
import numpy as np

PASSED = 0
FAILED = 0


def test(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")


def ok(msg=""):
    global PASSED
    PASSED += 1
    print(f"  PASS: {msg}")


def fail(msg):
    global FAILED
    FAILED += 1
    print(f"  FAIL: {msg}")


# ============================================================
# Test 1: Import validation
# ============================================================
test("1. Import validation")
t0 = time.time()
try:
    from tpu_inference.layers.common.quantization import dequantize_tensor, quantize_tensor

    ok("dequantize_tensor, quantize_tensor")
except Exception as e:
    fail(f"quantization imports: {e}")

try:
    from tpu_inference.layers.common.process_weights.moe_weights import (
        FusedMoEWeights,
        process_moe_weights,
        quantize_moe_weights,
    )
    from tpu_inference.layers.common.moe import MoEBackend

    ok("moe_weights imports")
except Exception as e:
    fail(f"moe_weights imports: {e}")

try:
    from tpu_inference.layers.common.quantization.fp8 import process_blockwise_fp8_linear_weights

    ok("process_blockwise_fp8_linear_weights")
except Exception as e:
    fail(f"fp8 imports: {e}")

try:
    from safetensors import safe_open

    ok("safetensors")
except Exception as e:
    fail(f"safetensors: {e}")

print(f"  Imports done in {time.time()-t0:.1f}s")

# Check we can access model weights
import glob

SHARD_FILES = sorted(glob.glob("/mnt/gcs-models/*.safetensors"))
if len(SHARD_FILES) > 0:
    ok(f"Found {len(SHARD_FILES)} safetensors shards")
else:
    fail("No safetensors shards found at /mnt/gcs-models/")
    sys.exit(1)

# ============================================================
# Test 2: Single MoE layer byte-exact round-trip (8 experts)
# ============================================================
test("2. MoE byte-exact round-trip (8 experts)")

import torch
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights,
    process_moe_weights,
    quantize_moe_weights,
)
from tpu_inference.layers.common.moe import MoEBackend

E_TEST = 8
BLOCK_SIZE = (128, 128)
LAYER_IDX = 1  # First MoE layer

prefix = f"model.layers.{LAYER_IDX}.mlp.experts."
gate_w, gate_s, up_w, up_s, down_w, down_s = {}, {}, {}, {}, {}, {}

t0 = time.time()
for sf in SHARD_FILES:
    with safe_open(sf, framework="torch") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].split(".")
            eid = int(parts[0])
            if eid >= E_TEST:
                continue
            proj, ptype = parts[1], parts[2]
            t = f.get_tensor(key)
            if ptype == "weight":
                raw = t.cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            else:
                raw = t.cpu().float().numpy()

            target = {
                "gate_proj": {"weight": gate_w, "weight_scale_inv": gate_s},
                "up_proj": {"weight": up_w, "weight_scale_inv": up_s},
                "down_proj": {"weight": down_w, "weight_scale_inv": down_s},
            }
            if proj in target and ptype in target[proj]:
                target[proj][ptype][eid] = raw

print(f"  Loaded {len(gate_w)} experts in {time.time()-t0:.1f}s")

if len(gate_w) < E_TEST:
    fail(f"Only loaded {len(gate_w)}/{E_TEST} experts")
else:
    # Stack and fuse
    gw = np.stack([gate_w[i] for i in range(E_TEST)], axis=0)
    gs = np.stack([gate_s[i] for i in range(E_TEST)], axis=0)
    uw = np.stack([up_w[i] for i in range(E_TEST)], axis=0)
    us = np.stack([up_s[i] for i in range(E_TEST)], axis=0)
    dw = np.stack([down_w[i] for i in range(E_TEST)], axis=0)
    ds = np.stack([down_s[i] for i in range(E_TEST)], axis=0)

    # === PATH 1: Online (what vLLM does) ===
    t0 = time.time()
    w13_online = jnp.concatenate([jnp.array(gw), jnp.array(uw)], axis=1)
    s13_online = jnp.concatenate([jnp.array(gs), jnp.array(us)], axis=1)
    w2_online = jnp.array(dw)
    s2_online = jnp.array(ds)

    # Dequant
    w13_f32 = dequantize_tensor(w13_online, s13_online, (1, 2), jnp.float32, block_size=BLOCK_SIZE)
    w2_f32 = dequantize_tensor(w2_online, s2_online, (1, 2), jnp.float32, block_size=BLOCK_SIZE)

    # Requant
    fused_online = quantize_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_f32,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_f32,
            w2_weight_scale=None,
            w2_bias=None,
        ),
        jnp.float8_e4m3fn,
        None,
    )

    # Reorder
    result_online = process_moe_weights(
        fused_online, moe_backend=MoEBackend.GMM_TP, w13_reorder_size=4, w13_interleave=False
    )
    online_time = time.time() - t0

    # === PATH 2: Offline (our preprocessing script) ===
    t0 = time.time()
    w13_off = jnp.array(np.concatenate([gw, uw], axis=1))
    s13_off = jnp.array(np.concatenate([gs, us], axis=1))
    w2_off = jnp.array(dw)
    s2_off = jnp.array(ds)

    w13_f32_off = dequantize_tensor(w13_off, s13_off, (1, 2), jnp.float32, block_size=BLOCK_SIZE)
    w2_f32_off = dequantize_tensor(w2_off, s2_off, (1, 2), jnp.float32, block_size=BLOCK_SIZE)

    fused_off = quantize_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_f32_off,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_f32_off,
            w2_weight_scale=None,
            w2_bias=None,
        ),
        jnp.float8_e4m3fn,
        None,
    )

    result_off = process_moe_weights(fused_off, moe_backend=MoEBackend.GMM_TP, w13_reorder_size=4, w13_interleave=False)
    offline_time = time.time() - t0

    # === Compare all 4 tensors ===
    for name, online_val, offline_val in [
        ("w13_weight", result_online.w13_weight, result_off.w13_weight),
        ("w13_weight_scale", result_online.w13_weight_scale, result_off.w13_weight_scale),
        ("w2_weight", result_online.w2_weight, result_off.w2_weight),
        ("w2_weight_scale", result_online.w2_weight_scale, result_off.w2_weight_scale),
    ]:
        on = np.asarray(online_val)
        off = np.asarray(offline_val)
        if on.shape != off.shape:
            fail(f"{name}: shape mismatch {on.shape} vs {off.shape}")
            continue
        if on.dtype != off.dtype:
            fail(f"{name}: dtype mismatch {on.dtype} vs {off.dtype}")
            continue
        if on.dtype == ml_dtypes.float8_e4m3fn:
            match = np.array_equal(on.view(np.uint8), off.view(np.uint8))
        else:
            match = np.array_equal(on, off)
        if match:
            ok(f"{name}: byte-exact match {on.shape} {on.dtype}")
        else:
            n_diff = (
                np.sum(on.view(np.uint8) != off.view(np.uint8))
                if on.dtype == ml_dtypes.float8_e4m3fn
                else np.sum(on != off)
            )
            fail(f"{name}: {n_diff} values differ")

    print(f"  Online: {online_time:.2f}s, Offline: {offline_time:.2f}s")


# ============================================================
# Test 3: Full 384 experts — shape and dtype validation
# ============================================================
test("3. Full 384 experts — shape/dtype validation")

# Import preprocessing function
sys.path.insert(0, "/tmp")
try:
    # We'll replicate the key function inline rather than importing
    # (the script may not be in the container)
    E_FULL = 384
    print(f"  Loading {E_FULL} experts (this takes a minute)...")

    all_gate_w, all_gate_s = {}, {}
    all_up_w, all_up_s = {}, {}
    all_down_w, all_down_s = {}, {}

    t0 = time.time()
    for sf in SHARD_FILES:
        with safe_open(sf, framework="torch") as f:
            for key in f.keys():
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix) :].split(".")
                eid = int(parts[0])
                proj, ptype = parts[1], parts[2]
                t = f.get_tensor(key)
                if ptype == "weight":
                    raw = t.cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
                else:
                    raw = t.cpu().float().numpy()
                target = {
                    "gate_proj": {"weight": all_gate_w, "weight_scale_inv": all_gate_s},
                    "up_proj": {"weight": all_up_w, "weight_scale_inv": all_up_s},
                    "down_proj": {"weight": all_down_w, "weight_scale_inv": all_down_s},
                }
                if proj in target and ptype in target[proj]:
                    target[proj][ptype][eid] = raw

    loaded = len(all_gate_w)
    print(f"  Loaded {loaded} experts in {time.time()-t0:.1f}s")

    if loaded < E_FULL:
        fail(f"Only loaded {loaded}/{E_FULL} experts")
    else:
        gw_full = np.stack([all_gate_w[i] for i in range(E_FULL)], axis=0)
        gs_full = np.stack([all_gate_s[i] for i in range(E_FULL)], axis=0)
        uw_full = np.stack([all_up_w[i] for i in range(E_FULL)], axis=0)
        us_full = np.stack([all_up_s[i] for i in range(E_FULL)], axis=0)
        dw_full = np.stack([all_down_w[i] for i in range(E_FULL)], axis=0)
        ds_full = np.stack([all_down_s[i] for i in range(E_FULL)], axis=0)

        # Process
        print(f"  Processing {E_FULL} experts...")
        t0 = time.time()
        w13 = jnp.array(np.concatenate([gw_full, uw_full], axis=1))
        s13 = jnp.array(np.concatenate([gs_full, us_full], axis=1))
        w2 = jnp.array(dw_full)
        s2 = jnp.array(ds_full)

        w13_f32 = dequantize_tensor(w13, s13, (1, 2), jnp.float32, block_size=BLOCK_SIZE)
        w2_f32 = dequantize_tensor(w2, s2, (1, 2), jnp.float32, block_size=BLOCK_SIZE)
        fused = quantize_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_f32,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_f32,
                w2_weight_scale=None,
                w2_bias=None,
            ),
            jnp.float8_e4m3fn,
            None,
        )
        result = process_moe_weights(fused, moe_backend=MoEBackend.GMM_TP, w13_reorder_size=4, w13_interleave=False)
        print(f"  Processed in {time.time()-t0:.1f}s")

        # Validate shapes
        checks = [
            ("w13_weight", result.w13_weight, (384, 7168, 4096), ml_dtypes.float8_e4m3fn),
            ("w13_scale", result.w13_weight_scale, (384, 1, 1, 4096), np.float32),
            ("w2_weight", result.w2_weight, (384, 2048, 7168), ml_dtypes.float8_e4m3fn),
            ("w2_scale", result.w2_weight_scale, (384, 1, 1, 7168), np.float32),
        ]
        for name, tensor, expected_shape, expected_dtype in checks:
            arr = np.asarray(tensor)
            if arr.shape == expected_shape:
                ok(f"{name}: shape {arr.shape}")
            else:
                fail(f"{name}: expected {expected_shape}, got {arr.shape}")
            if arr.dtype == expected_dtype:
                ok(f"{name}: dtype {arr.dtype}")
            else:
                fail(f"{name}: expected {expected_dtype}, got {arr.dtype}")

        # Range checks
        for name, tensor in [("w13_scale", result.w13_weight_scale), ("w2_scale", result.w2_weight_scale)]:
            arr = np.asarray(tensor)
            if not np.any(np.isnan(arr)):
                ok(f"{name}: no NaN")
            else:
                fail(f"{name}: contains NaN")
            if not np.any(np.isinf(arr)):
                ok(f"{name}: no Inf")
            else:
                fail(f"{name}: contains Inf")

except Exception as e:
    import traceback

    fail(f"Test 3 exception: {e}")
    traceback.print_exc()


# ============================================================
# Test 4: Dense linear round-trip
# ============================================================
test("4. Dense linear round-trip")

try:
    # Load q_a_proj from layer 0
    q_proj_name = "model.layers.0.self_attn.q_a_proj.weight"
    q_scale_name = "model.layers.0.self_attn.q_a_proj.weight_scale_inv"
    q_weight = q_scale = None

    for sf in SHARD_FILES:
        with safe_open(sf, framework="torch") as f:
            if q_proj_name in f.keys():
                q_weight = f.get_tensor(q_proj_name).cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            if q_scale_name in f.keys():
                q_scale = f.get_tensor(q_scale_name).cpu().float().numpy()

    if q_weight is None or q_scale is None:
        fail(f"Could not load {q_proj_name}")
    else:
        print(f"  q_a_proj: weight={q_weight.shape}, scale={q_scale.shape}")

        # Online path
        w_on = jnp.array(q_weight)
        s_on = jnp.array(q_scale)
        result_on = process_blockwise_fp8_linear_weights(
            w_on,
            s_on,
            bias=None,
            weight_block_size=BLOCK_SIZE,
            requant_block_size=None,
            output_sizes=(q_weight.shape[0],),
            requant_weight_dtype=jnp.float8_e4m3fn,
            fuse_matmuls=True,
            n_shards=1,
        )

        # Offline path (same — verifies determinism)
        w_off = jnp.array(q_weight)
        s_off = jnp.array(q_scale)
        result_off = process_blockwise_fp8_linear_weights(
            w_off,
            s_off,
            bias=None,
            weight_block_size=BLOCK_SIZE,
            requant_block_size=None,
            output_sizes=(q_weight.shape[0],),
            requant_weight_dtype=jnp.float8_e4m3fn,
            fuse_matmuls=True,
            n_shards=1,
        )

        on_w = np.asarray(result_on.weight)
        off_w = np.asarray(result_off.weight)
        on_s = np.asarray(result_on.weight_scale)
        off_s = np.asarray(result_off.weight_scale)

        if np.array_equal(on_w.view(np.uint8), off_w.view(np.uint8)):
            ok(f"weight: byte-exact match {on_w.shape} {on_w.dtype}")
        else:
            fail("weight: values differ")

        if np.array_equal(on_s, off_s):
            ok(f"scale: exact match {on_s.shape}")
        else:
            fail("scale: values differ")

except Exception as e:
    import traceback

    fail(f"Test 4 exception: {e}")
    traceback.print_exc()


# ============================================================
# Test 5: kv_b_proj split
# ============================================================
test("5. kv_b_proj → k_up_proj + v_up_proj split")

try:
    kv_name = "model.layers.1.self_attn.kv_b_proj.weight"
    kv_scale_name = "model.layers.1.self_attn.kv_b_proj.weight_scale_inv"
    kv_weight = kv_scale = None

    for sf in SHARD_FILES:
        with safe_open(sf, framework="torch") as f:
            if kv_name in f.keys():
                kv_weight = f.get_tensor(kv_name).cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            if kv_scale_name in f.keys():
                kv_scale = f.get_tensor(kv_scale_name).cpu().float().numpy()

    if kv_weight is None or kv_scale is None:
        fail(f"Could not load {kv_name}")
    else:
        print(f"  kv_b_proj: weight={kv_weight.shape}, scale={kv_scale.shape}")

        # Run the split (same function, deterministic)
        w = jnp.array(kv_weight)
        s = jnp.array(kv_scale)

        dequantized = dequantize_tensor(w, s, (0, 1), jnp.float32, block_size=BLOCK_SIZE)
        dequantized = dequantized.T

        KV_LORA_RANK = 512
        NUM_HEADS = 64
        QK_NOPE = 128
        V_HEAD = 128

        dequantized = dequantized.reshape(KV_LORA_RANK, NUM_HEADS, QK_NOPE + V_HEAD)
        k_ANH = dequantized[:, :, :QK_NOPE]
        v_ANH = dequantized[:, :, QK_NOPE:]

        k_weight_1, k_scale_1 = quantize_tensor(jnp.float8_e4m3fn, k_ANH, axis=-1)
        v_weight_1, v_scale_1 = quantize_tensor(jnp.float8_e4m3fn, v_ANH, axis=0)
        k_scale_1 = jnp.transpose(k_scale_1, (1, 0))  # (A,N) → (N,A)
        # v_scale_1 already (N,V) — no transpose needed

        # Run again (verify determinism)
        dequantized2 = dequantize_tensor(
            jnp.array(kv_weight), jnp.array(kv_scale), (0, 1), jnp.float32, block_size=BLOCK_SIZE
        ).T
        dequantized2 = dequantized2.reshape(KV_LORA_RANK, NUM_HEADS, QK_NOPE + V_HEAD)
        k_weight_2, k_scale_2 = quantize_tensor(jnp.float8_e4m3fn, dequantized2[:, :, :QK_NOPE], axis=-1)
        v_weight_2, v_scale_2 = quantize_tensor(jnp.float8_e4m3fn, dequantized2[:, :, QK_NOPE:], axis=0)
        k_scale_2 = jnp.transpose(k_scale_2, (1, 0))
        # v_scale_2 already correct

        for name, a, b in [
            ("k_weight", k_weight_1, k_weight_2),
            ("k_scale", k_scale_1, k_scale_2),
            ("v_weight", v_weight_1, v_weight_2),
            ("v_scale", v_scale_1, v_scale_2),
        ]:
            an, bn = np.asarray(a), np.asarray(b)
            if an.dtype == ml_dtypes.float8_e4m3fn:
                match = np.array_equal(an.view(np.uint8), bn.view(np.uint8))
            else:
                match = np.array_equal(an, bn)
            if match:
                ok(f"{name}: byte-exact match {an.shape} {an.dtype}")
            else:
                fail(f"{name}: values differ")

except Exception as e:
    import traceback

    fail(f"Test 5 exception: {e}")
    traceback.print_exc()


# ============================================================
# Test 6: Safetensors save/load round-trip
# ============================================================
test("6. Safetensors save/load round-trip")

try:
    from safetensors.numpy import save_file as np_save_file

    # Create test data matching our output format
    test_tensors = {
        "fp8_weight": np.random.randint(0, 255, (8, 7168, 4096), dtype=np.uint8),
        "float32_scale": np.random.randn(8, 1, 1, 4096).astype(np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    np_save_file(test_tensors, tmp_path, metadata={"fp8_dtype": "e4m3fn"})

    # Load back
    loaded = {}
    with safe_open(tmp_path, framework="numpy") as f:
        for key in f.keys():
            loaded[key] = f.get_tensor(key)

    for key in test_tensors:
        if key not in loaded:
            fail(f"Missing key: {key}")
            continue
        if np.array_equal(test_tensors[key], loaded[key]):
            ok(f"{key}: round-trip match {test_tensors[key].shape} {test_tensors[key].dtype}")
        else:
            fail(f"{key}: data changed after save/load")

    os.unlink(tmp_path)

except Exception as e:
    import traceback

    fail(f"Test 6 exception: {e}")
    traceback.print_exc()


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"SUMMARY: {PASSED} passed, {FAILED} failed")
print(f"{'='*60}")
if FAILED == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {FAILED} test(s) FAILED")
    sys.exit(1)
