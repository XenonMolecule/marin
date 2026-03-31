# Ray Multi-Host vLLM on TPU: Deployment Guide for 235B+ Models

Lessons learned from deploying Qwen3-235B-A22B-Thinking on v5p-16 TPU (2 hosts, 8 chips).
Target audience: anyone replicating this or scaling to 1T-parameter models (Kimi K2.5/K2-Instruct).

## Architecture Overview

```
Host 0: [chip0, chip1, chip2, chip3] TP=4 -> PP stage 0 (layers 0-46)
Host 1: [chip0, chip1, chip2, chip3] TP=4 -> PP stage 1 (layers 47-93)
```

- **Pipeline Parallelism (PP)** across hosts -- each host runs a subset of layers
- **Tensor Parallelism (TP)** within each host -- weight matrices sharded across chips
- **Ray** manages worker placement; compiled DAG for forward pass
- PP communication via JAX transfer servers

## Hardware Sizing

| Model | Weights | Hardware | Config | Notes |
|-------|--------:|----------|--------|-------|
| Qwen3-235B MoE | 438 GB | v5p-16 (PP=2 TP=4) | **Recommended** | 91/96 GiB HBM used, plenty of KV cache |
| Qwen3-235B MoE | 438 GB | v6e-16 (PP=4 TP=4) | Tight | max_model_len=1024 only, 5x slower |
| Kimi K2.5 (~1T) | ~600 GB FP8 | v5p-32+ (PP=4 TP=4) | Untested | See scaling notes below |
| Kimi K2-Instruct (~1T) | ~600 GB FP8 | v5p-64 (PP=8 TP=4) | Untested | May need INT4->FP8 conversion |

**Rule: If the model fits on one host, DON'T use Ray PP.** Single-host TP is 2.6x faster (no pipeline bubble + cross-host transfer overhead).

## Gotchas & Hard-Won Lessons

### 1. Ray OOM Killer vs. Weight Loading (CRITICAL)

**Problem:** During weight loading, all 118 safetensors shards (~438 GB) are read into CPU RAM before being sharded to TPU HBM. Ray's default OOM threshold of 95% kills the worker process mid-load.

**Symptom:**
```
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node was 418.96GB / 440.83GB (0.950398), which exceeds the memory usage threshold of 0.95.
```

**Fix:** Disable Ray's OOM monitor at container startup:
```bash
docker run ... -e RAY_memory_monitor_refresh_ms=0 ... vllm/vllm-tpu:nightly
```

This is safe because:
- Weight loading is a transient peak -- once weights move to TPU HBM, CPU RAM usage drops dramatically
- The Linux kernel OOM killer is still active as a last resort
- TPU HBM management is handled by JAX, not Ray

**For 1T models:** This is even more critical. A 600 GB FP8 model will need ~600 GB of CPU RAM during loading. v5p hosts have ~441 GB RAM. You may need to use streaming weight loading (`--load-format runai_streamer` or `tpu_streaming_loader`) instead of loading all shards into CPU memory.

### 2. Iris-Managed TPUs Have Background Containers

**Problem:** TPUs allocated via Iris have 40-50+ `iris-task` containers running, consuming ~150 GB of CPU RAM. This reduces available RAM for weight loading and can trigger OOM.

**Impact on Qwen3-235B:**
- Fresh TPU: 407 GiB free --> weight loading succeeds
- Iris-loaded TPU: 220 GiB free --> weight loading OOM'd even with Ray threshold disabled

**Mitigation:**
- Check RAM before starting: `free -h` on the TPU host
- If RAM < 300 GiB free, the Iris containers are too heavy. Try a fresher TPU.
- **Do NOT kill iris-worker or iris-task containers** -- Iris manages their lifecycle.

**For 1T models:** You'll need even more free RAM. Consider dedicated standalone TPUs (not Iris-managed) for 1T inference.

### 3. TPUs Can Be Recycled Mid-Load

**Problem:** Iris-managed TPUs can be recycled (reallocated to other jobs) at any time. Our first attempt loaded to 37% (37/118 shards, ~16 min in) before the TPU disappeared:
```
ERROR: NOT_FOUND: Resource '...' was not found.
```

**Mitigation:**
- Use Iris dev TPU allocation (`scripts/iris/dev_tpu.py allocate`) to get a dedicated hold
- Monitor the Iris job status alongside vLLM loading
- For long loads (30-60 min), check TPU existence periodically

**For 1T models:** A 1T model load could take 60-120 min. Use on-demand (not preemptible) TPUs, or use GCS weight caching so subsequent loads are instant.

### 4. Docker Access Requires `sudo`, Not `sg docker`

**Problem:** Ahmed's `launch.sh` uses `sg docker 'docker ...'` but Iris-provisioned TPU VMs don't add SSH users to the `docker` group. All docker commands fail silently.

**Fix:** Replace `sg docker 'docker ...'` with `sudo docker ...` throughout.

### 5. gcsfuse Setup is Per-Host, Per-Reboot

**Problem:** gcsfuse must be installed and mounted on EVERY host in the TPU pod. After a TPU recycle, all gcsfuse state is lost.

**Setup (run on --worker=all):**
```bash
sudo gcsfuse --implicit-dirs -o allow_other \
  --only-dir 'models/path/to/model' \
  --file-cache-max-size-mb 0 \
  BUCKET_NAME /mnt/gcs-models
```

**Key flags:**
- `--only-dir`: Mount only the model subdirectory (avoids listing entire bucket)
- `--file-cache-max-size-mb 0`: Disable file cache (saves disk space)
- `-o allow_other`: Allow Docker containers to access the mount

### 6. Same-Region GCS is CRITICAL for Weight Loading

**Problem:** Cross-region gcsfuse throughput: ~50 MiB/s. Same-region: 1-10 Gbps.

| Scenario | 438 GB load time |
|----------|-----------------|
| Cross-region gcsfuse | ~2.5 hours |
| Same-region gcsfuse (cold) | ~45 min |
| Same-region gcsfuse (page cached) | ~2 min |

**Rule:** Always load weights from a GCS bucket in the same region as the TPU.

**For 1T models:** If weights aren't in the right region, download from HuggingFace directly to the TPU rather than cross-region GCS copy (avoids egress costs).

### 7. OS Page Cache Makes Reloads Instant

**Problem:** First weight load takes 30-60 min. But the OS page cache keeps the data in RAM.

**Rule:** NEVER `docker rm` your container. Use `docker exec` to restart services inside the container. `docker rm` destroys the page cache, forcing a full reload.

**For 1T models:** The page cache is even more valuable. Budget for the first load being slow, but design your workflow so you never need to reload from scratch.

### 8. Six Runtime Patches Are Required

The `vllm/vllm-tpu:nightly` image has bugs preventing Ray multi-host from working. These patches are applied at runtime by modifying Python files inside the container:

| # | Patch | What It Fixes |
|---|-------|---------------|
| 1 | `patch_ray_multihost_v2.py` | JAX sees all global devices --> XLA crashes. Fix: isolate each worker to see only local chips via `TPU_PROCESS_BOUNDS` |
| 2 | `patch_ray_sharding.py` | `shard_put` passes `None` to `general_device_put`. Fix: wrap in `NamedSharding` |
| 3 | `patch_ray_sharding_v3.py` | **MOST IMPORTANT.** `nnx.get_named_sharding` fails under Ray --> weights replicated --> OOM. Fix: hardcode TP partition specs by weight name |
| 4 | `patch_ray_mm.py` | `TPUModelRunner` missing `supports_mm_inputs`. Fix: add attribute |
| 5 | `patch_pp_parallel_state.py` | **CRITICAL FOR PP.** All workers get `rank=0` --> all create same layers. Fix: override PP group with actual TPU rank |
| 6 | `patch_kv_cache_local_names.py` | KV cache layer names reference wrong PP stage. Fix: re-register with local layer names |

**Key observation:** Patches 1-2 may already be fixed in newer nightly images (our runs showed "SKIP: already patched"). Patches 3, 4, 5 consistently needed applying.

**For 1T models (Kimi K2.5):**
- Patch 3 needs MODEL-SPECIFIC weight name patterns. Qwen3 patterns (`gate_proj`, `q_proj`, etc.) won't match DeepSeek/Kimi architecture. Ahmed's `patch_ray_sharding_v3.py` already includes some DeepSeek MLA patterns (`q_a_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`) -- verify these match K2.5's weight names.
- Patch 5 must handle PP>2 correctly (PP=4 or PP=8 for 1T models).
- Additional patches exist as untracked local files (not Ahmed's -- likely from a prior agent session): `patch_deepseek_v3_config.py`, `patch_pp_enable_deepseek.py`, `patch_int4_moe.py`, `patch_compressed_tensors.py`, `patch_5d_mesh.py`, `patch_mesh_axes.py`. These may be useful as starting points but have not been validated in a successful 1T run.

### 9. MoE Weight Padding Takes Significant Time

After safetensors shards are loaded, MoE expert weights must be padded to power-of-2 shapes for TPU efficiency. For Qwen3-235B (128 experts, 47 layers per PP stage), this takes ~10 minutes and produces verbose logs:
```
w13_weight_w1 shape after padding: (128, 4096, 1536)
w13_weight_w3 shape after padding: (128, 4096, 1536)
```

**For 1T models:** Kimi K2.5 has 256 experts. Expect this phase to take longer.

### 10. XLA Compilation Happens After Weight Loading

After weights are loaded and KV cache is allocated, vLLM precompiles XLA subgraphs for all padded input shapes (16, 32, 64, ... 2048 tokens). This takes ~5 minutes and produces int64 truncation warnings (benign):
```
UserWarning: Explicitly requested dtype int64 requested in astype is not available, and will be truncated to dtype int32.
```

### 11. Thinking Models Need max_tokens >= 4096

Qwen3-235B-Thinking uses ~3,000 tokens for reasoning chains. With `max_tokens=1024`, the model truncates before outputting `\boxed{}`, dropping accuracy from ~46% to ~12%.

**Recommendation:** Use `max_tokens=4096` minimum, `max_tokens=8192` for hard problems.

## Timeline: What to Expect

| Phase | Duration | Notes |
|-------|----------|-------|
| TPU allocation (Iris) | 1-5 min | Depends on cluster load |
| gcsfuse setup | 1-2 min | Including gcsfuse install |
| Docker + Ray cluster | 30 sec | Both containers + ray status |
| Patch application | 30 sec | 6 patches across 2 hosts |
| Weight loading (cold) | 30-45 min | 118 shards via gcsfuse |
| Weight loading (cached) | 1-2 min | OS page cache |
| MoE weight padding | ~10 min | 128 experts x 47 layers |
| XLA compilation | ~5 min | Precompile for all input shapes |
| **Total (cold start)** | **~55 min** | |
| **Total (warm reload)** | **~18 min** | |

## MATH-500 Benchmark Results

*(Results from our run -- Qwen3-235B-A22B-Thinking on v5p-16, PP=2 TP=4)*

**Initial run (20 problems, concurrency 16, max_tokens 4096):**

| Metric | Value |
|--------|-------|
| Accuracy | **100% (7/7 completed)** |
| Timeouts | 13/20 (300s default timeout too short for thinking model) |
| TTFT (p50) | 629 ms |
| TPOT (p50) | 95.5 ms |
| E2E (p50) | 149.6 s |
| Gen throughput | 25.1 tok/s (cold XLA) -> **191.7 tok/s (warmed up)** |
| Mean generated tokens | 1,614 per response |
| Format rate | 100% (all completed responses had `\boxed{}`) |

**Key takeaway:** The thinking model generates very long reasoning chains (~1,600 tokens average). The 300s per-request timeout in the benchmark script is too short for complex problems. With higher concurrency (64+), throughput scales significantly due to better HBM bandwidth utilization.

**Full run (50 problems, concurrency 64, max_tokens 4096):**

| Metric | Value |
|--------|-------|
| Accuracy | **95.24% (20/21 completed)** |
| Timeouts | 29/50 (300s default timeout -- thinking chains too long) |
| TTFT (p50) | 348.6 ms |
| TPOT (p50) | 88.0 ms |
| E2E (p50) | 120.1 s |
| Server-side throughput | **580-602 tok/s** (at concurrency 50) |
| Mean generated tokens | 1,478 per response |
| Format rate | 100% |

**Comparison to Ahmed's results (v5p-16 PP=2 TP=4):**
- Ahmed reported 1,127 tok/s at concurrency 256 -- our 580 tok/s at concurrency 50 is consistent (throughput scales with concurrency)
- Ahmed reported 46% accuracy with max_tokens=4096 on the full 500 problems
- Our 95% on completed problems is higher because only "easy" problems finish within 300s; the hard ones that bring accuracy down are the ones timing out
- To get the full 500-problem run, increase the per-request timeout beyond 300s in `benchmark_math500.py`

### 12. Benchmark Timeout Must Match Thinking Model Output Length

The `benchmark_math500.py` script has a hardcoded `aiohttp.ClientTimeout(total=300)` (300 seconds). Thinking models like Qwen3-235B-Thinking generate 1,500-4,000 token reasoning chains, which at ~90ms/token takes 135-360 seconds. Many complex problems timeout.

**Fix for future runs:** Increase the timeout in `benchmark_math500.py`:
```python
async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
```

Or use higher concurrency (256+) to increase effective throughput.

## Scaling to 1T Models (Kimi K2.5 / K2-Instruct)

### Key Differences from 235B

| Aspect | Qwen3-235B | Kimi K2.5 (~1T) |
|--------|-----------|-----------------|
| Weight size | 438 GB (BF16) | ~600 GB (FP8) |
| Experts | 128 per layer | 256 per layer |
| Layers | 94 | ~128 (estimated) |
| Architecture | Qwen3 MoE | DeepSeek V3 MLA |
| Min hardware | v5p-16 (PP=2) | v5p-32+ (PP=4) |

### Specific Concerns for 1T

1. **Weight format:** K2.5 may use INT4 quantization. Need `patch_int4_moe.py` and `patch_compressed_tensors.py`. There is a local (untracked) conversion script `convert_k25_int4_to_fp8.py` from a prior agent session -- not Ahmed's work, use with caution.

2. **CPU RAM during loading:** 600 GB of weights > 441 GB host RAM. Must use streaming loader or split loading across PP stages (load only your layers).

3. **PP=4+ complexity:** More PP stages = more KV cache boundary bugs. Patches 5 and 6 become more critical. Non-divisible layer counts (e.g., 128 layers / 4 PP = 32 per stage) are cleaner than 94/4.

4. **DeepSeek MLA architecture:** Multi-Latent Attention has different weight patterns than standard MHA. Patch 3 (`patch_ray_sharding_v3.py`) needs DeepSeek-specific entries:
   - `q_a_proj`, `q_b_proj` -> `P(None, 'model')`
   - `kv_a_proj_with_mqa` -> `P()` (replicate -- small)
   - `kv_b_proj` -> `P(None, 'model')`

5. **5D mesh topology:** DeepSeek V3 uses a 5D sharding mesh (`data`, `model`, `attn_dp`, `attn_dp_expert`, `expert`). Standard vLLM only creates a 2D mesh. Local (untracked) patches `patch_5d_mesh.py` and `patch_mesh_axes.py` attempt to address this but are unvalidated.

6. **Config overrides:** DeepSeek V3 has hardcoded params in vLLM that don't match K2.5 (different expert count, vocab size, head count). Local (untracked) patch `patch_deepseek_v3_config.py` attempts to fix this but needs K2.5-specific values verified against the actual model config.

### Recommended Approach for 1T

1. Start with Ahmed's `launch.sh` as a reference (the `launch_kimi_k2*.sh` scripts are untracked local variants from a prior agent session, not Ahmed's work)
2. Convert INT4 weights to FP8 if needed (the local `convert_k25_int4_to_fp8.py` is unvalidated)
3. Upload FP8 weights to same-region GCS
4. Allocate v5p-32 or v5p-64 (PP=4 or PP=8, TP=4)
5. Use ALL patches (core 6 + DeepSeek-specific)
6. Set `RAY_memory_monitor_refresh_ms=0`
7. Use streaming weight loader if CPU RAM is insufficient
8. Budget 2+ hours for first cold start

## Quick Reference: Launch Commands

```bash
# Allocate TPU via Iris
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml \
  --tpu-name my-session allocate --tpu-type v5p-16

# Get TPU name
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml \
  --tpu-name my-session status

# Launch (manual steps -- replace sg docker with sudo docker)
TPU_NAME="<from status output>"
ZONE="us-central1-a"
MODEL_PATH="gs://marin-us-central1/models/..."

# 1. gcsfuse on all hosts
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=all \
  --command="sudo gcsfuse --implicit-dirs -o allow_other --only-dir 'MODEL_SUBPATH' BUCKET /mnt/gcs-models"

# 2. Docker containers with OOM monitor disabled
# Head (worker 0):
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=0 \
  --command="sudo docker run -d --name ray-node --privileged --net=host --shm-size=16g \
    -e TPU_MULTIHOST_BACKEND=ray -e TPU_BACKEND_TYPE=jax -e RAY_DEDUP_LOGS=0 \
    -e RAY_memory_monitor_refresh_ms=0 \
    -v /dev/shm:/dev/shm -v /mnt/gcs-models:/mnt/gcs-models:ro \
    vllm/vllm-tpu:nightly bash -c 'ray start --head --port=6379 && sleep infinity'"

# Worker (worker 1):
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=1 \
  --command="sudo docker run -d --name ray-node --privileged --net=host --shm-size=16g \
    -e TPU_MULTIHOST_BACKEND=ray -e TPU_BACKEND_TYPE=jax -e RAY_DEDUP_LOGS=0 \
    -e RAY_memory_monitor_refresh_ms=0 \
    -v /dev/shm:/dev/shm -v /mnt/gcs-models:/mnt/gcs-models:ro \
    vllm/vllm-tpu:nightly bash -c 'ray start --address=HEAD_IP:6379 --block'"

# 3. Copy & apply patches
for p in patch_*.py; do
  gcloud compute tpus tpu-vm scp patches/$p $TPU_NAME:/tmp/$p --zone=$ZONE --project=hai-gcp-models --worker=all
done
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=all \
  --command="for p in /tmp/patch_*.py; do sudo docker cp \$p ray-node:/tmp/; sudo docker exec ray-node python \$p; done"

# 4. Launch vLLM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=0 \
  --command="sudo docker exec -d ray-node bash -c 'vllm serve /mnt/gcs-models \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --distributed-executor-backend ray --max-model-len 16384 \
    --gpu-memory-utilization 0.95 --port 8000 --trust-remote-code \
    > /tmp/vllm_serve.log 2>&1'"

# 5. Monitor
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=0 \
  --command="sudo docker exec ray-node tail -f /tmp/vllm_serve.log" | grep -v cpu_aot_loader

# 6. SSH tunnel + benchmark
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=hai-gcp-models --worker=0 \
  -- -L 8000:localhost:8000 -N &
python scripts/ray_multihost_vllm/benchmark_math500.py --server http://localhost:8000 \
  --concurrency 16 --max-tokens 4096 --output results.json
```
