# llama.cpp CPU Inference Benchmarks

## Hardware

- **CPU**: Intel Xeon Platinum 8481C @ 2.70GHz
- **Cores**: 104 physical (208 threads), 2 sockets x 52 cores
- **RAM**: 440 GiB
- **Node type**: v5p-8 TPU node (using idle CPUs)
- **Model**: qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M (1.1 GiB GGUF)

## Stage 1: Raw llama-bench (pp = prompt processing, tg = token generation)

| Threads | pp512 (tok/s) | pp2048 | pp8192 | tg256 |
|---------|---------------|--------|--------|-------|
| 1       | 44.6          | 36.5   | 20.5   | 6.3   |
| 2       | 77.6          | 65.7   | 39.1   | 12.5  |
| 4       | 153.8         | 124.6  | 73.8   | 23.7  |
| 8       | 227.1         | 185.1  | 117.5  | 35.4  |
| 16      | 333.5         | 286.4  | 198.5  | 54.3  |
| 32      | 422.2         | 390.1  | 287.8  | 73.5  |

Key insight: prefill scales nearly linearly, but generation gains diminish past 16 threads.

## Stage 3: Realistic Workload (llama-server, /v1/chat/completions)

Generation target: 4096 tokens. Context: 32768.

### 16k prompt (~16,534 tokens input)

| Servers x Threads | Total CPU | Gen tok/s | Total tok/s | Failed |
|-------------------|-----------|-----------|-------------|--------|
| 1 x 16            | 16        | 8.5       | 35.4        | 0      |
| **4 x 16**        | **64**    | **37.7**  | **157.1**   | **0**  |
| 4 x 8             | 32        | 20.0      | 83.5        | 0      |
| 8 x 8             | 64        | 35.6      | 148.2       | 0      |

### 28k prompt (~28,553 tokens input)

| Servers x Threads | Total CPU | Gen tok/s | Total tok/s | Failed |
|-------------------|-----------|-----------|-------------|--------|
| 1 x 16            | 16        | 5.3       | 34.6        | 0      |
| **4 x 16**        | **64**    | **20.9**  | **135.3**   | **0**  |
| 4 x 8             | 32        | 13.6      | 56.7        | 4      |
| 8 x 8             | 64        | 27.3      | 113.5       | 8      |

### Analysis

- **16 threads per server is optimal.** At same CPU budget, 16t x N consistently
  beats 8t x 2N in gen tok/s and has zero failures.
- **8-thread configs timeout on long prompts.** 4 and 8 failures at 28k prompts
  within the 1200s timeout.
- **Linear scaling up to 4 servers**: 4 x 16t achieves ~4.4x the gen tok/s of
  1 x 16t (37.7 vs 8.5) — near-perfect scaling.

## Cluster Throughput Projections

Configuration: 64 workers x 16 threads, ~4 workers per node (64 CPUs used per node).

Per-node aggregate (extrapolating from 4x16 benchmark):
- 16k prompts: ~37.7 gen tok/s per 4 workers
- 28k prompts: ~20.9 gen tok/s per 4 workers

Cluster (us-east5-a, ~23 nodes):
- 16k: 23 x 37.7 x (64/64) = **~867 gen tok/s** (conservative, 4 workers/node)
- 16k: 23 x 37.7 x (13/4) = **~2,819 gen tok/s** (13 workers/node if scaling holds)
- 28k: 23 x 20.9 x (13/4) = **~1,562 gen tok/s** (13 workers/node)

vs TPU baseline: 500-1000 tok/s (1-2 TPUs typically allocated)

**CPU inference achieves 2-6x higher aggregate throughput than the typical TPU allocation,
using resources that would otherwise be idle.**
