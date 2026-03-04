#!/bin/bash
# Benchmark llama-server continuous batching (--parallel N).
#
# Tests whether sending N concurrent requests to a SINGLE llama-server
# (with --parallel N) is faster than sending them sequentially.
# This directly tests the optimization proposed in inference_llamacpp.py.
#
# Current production config: 1 server per worker, 16 threads, sequential requests.
# Proposed optimization: 1 server per worker, 16 threads, --parallel N, concurrent requests.
#
# Usage (from SSH session on a TPU node):
#   bash /tmp/bench_parallel_batching.sh
#
# To copy to node and run:
#   gcloud compute tpus tpu-vm scp experiments/rephraser/bench_parallel_batching.sh <TPU_NAME>:/tmp/ --zone=<ZONE>
#   gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> -- "bash /tmp/bench_parallel_batching.sh"
#
# What this tests:
#   1. Sequential baseline: 1 server, 16 threads, send N requests one at a time
#   2. Parallel batching:   1 server, 16 threads, --parallel N, send N requests concurrently
#   3. Realistic case:      Same comparison with a 16k-token prompt + 4096 gen tokens
#
# The key question: does --parallel give us free throughput by overlapping
# prefill/generation across requests, or does it just split the same CPU budget?

set -euo pipefail

WORKDIR="/tmp/llama-cpp-bench"
MODEL_REPO="MichaelR207/qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M-GGUF"
MODEL_FILE="qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"
THREADS=16

echo "=== CPU Info ==="
lscpu | grep -E "^(Architecture|Model name|CPU\(s\)|Thread|Core|Socket|CPU MHz|CPU max)"
echo ""
echo "=== Memory ==="
free -h | head -2
echo ""

# ---------- 1. Build llama.cpp ----------
echo "=== Building llama.cpp ==="
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Ensure build dependencies are installed
if ! command -v cmake &>/dev/null; then
    echo "Installing cmake and build tools..."
    sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential
fi

if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    fi
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build build --config Release -j"$(nproc)" --target llama-server
    cd "$WORKDIR"
else
    echo "llama.cpp already built, skipping."
fi

SERVER="$WORKDIR/llama.cpp/build/bin/llama-server"

# ---------- 2. Download model ----------
echo ""
echo "=== Downloading model ==="
MODEL_PATH="$WORKDIR/$MODEL_FILE"
if [ ! -f "$MODEL_PATH" ]; then
    curl -L "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" -o "$MODEL_PATH"
else
    echo "Model already downloaded."
fi
echo "Model size: $(du -h "$MODEL_PATH" | cut -f1)"

# ---------- Server management ----------
SERVER_PID=""

cleanup_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

start_server() {
    local threads=$1
    local parallel=$2
    local ctx_size=$3
    local n_predict=$4

    cleanup_server

    local cmd=("$SERVER"
        --model "$MODEL_PATH"
        --host 127.0.0.1
        --port 8080
        --threads "$threads"
        --ctx-size "$ctx_size"
        --n-predict "$n_predict"
        --parallel "$parallel"
        --log-disable
    )

    echo "  Starting: ${cmd[*]}"
    "${cmd[@]}" > "$WORKDIR/server.log" 2>&1 &
    SERVER_PID=$!

    # Wait for healthy
    for attempt in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ERROR: Server died during startup. Last log:"
            tail -20 "$WORKDIR/server.log"
            return 1
        fi
        local status
        status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8080/health" 2>/dev/null)
        if [ "$status" = "200" ]; then
            # Verify model loaded with a tiny request
            local test_status
            test_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:8080/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null)
            if [ "$test_status" != "503" ]; then
                echo "  Server healthy (attempt $attempt)"
                return 0
            fi
        fi
        if [ $((attempt % 10)) -eq 0 ]; then
            echo "  Waiting for model load... (${attempt}s)"
        fi
        sleep 1
    done
    echo "  ERROR: Server not healthy after 120s"
    return 1
}

# Send N requests sequentially to the server, return aggregate stats
run_sequential() {
    local n=$1
    local prompt_file=$2
    local result_dir="$WORKDIR/results_seq_${n}"
    mkdir -p "$result_dir"

    local start_time
    start_time=$(date +%s%N)

    for i in $(seq 1 "$n"); do
        curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$prompt_file" \
            -o "$result_dir/response_${i}.json" \
            --max-time 1800
    done

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    collect_stats "$result_dir" "$n" "$elapsed_ms" "sequential"
}

# Send N requests concurrently to the server, return aggregate stats
run_concurrent() {
    local n=$1
    local prompt_file=$2
    local result_dir="$WORKDIR/results_par_${n}"
    mkdir -p "$result_dir"

    local start_time
    start_time=$(date +%s%N)

    local pids=()
    for i in $(seq 1 "$n"); do
        curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$prompt_file" \
            -o "$result_dir/response_${i}.json" \
            --max-time 1800 &
        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++)) || true
        fi
    done

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    collect_stats "$result_dir" "$n" "$elapsed_ms" "concurrent" "$failed"
}

collect_stats() {
    local result_dir=$1
    local n=$2
    local elapsed_ms=$3
    local mode=$4
    local failed=${5:-0}

    local total_gen_tokens=0
    local total_prompt_tokens=0
    for i in $(seq 1 "$n"); do
        local resp="$result_dir/response_${i}.json"
        if [ -f "$resp" ]; then
            local tokens
            tokens=$(python3 -c "
import json
try:
    d = json.load(open('$resp'))
    u = d.get('usage', {})
    print(u.get('completion_tokens', 0), u.get('prompt_tokens', 0))
except: print('0 0')
" 2>/dev/null)
            local gen_tok prompt_tok
            gen_tok=$(echo "$tokens" | cut -d' ' -f1)
            prompt_tok=$(echo "$tokens" | cut -d' ' -f2)
            total_gen_tokens=$((total_gen_tokens + gen_tok))
            total_prompt_tokens=$((total_prompt_tokens + prompt_tok))
        fi
    done

    local elapsed_s
    elapsed_s=$(echo "scale=1; $elapsed_ms / 1000" | bc 2>/dev/null || echo "?")
    local gen_tok_per_sec
    gen_tok_per_sec=$(echo "scale=1; $total_gen_tokens * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "N/A")
    local total_tok_per_sec
    total_tok_per_sec=$(echo "scale=1; ($total_gen_tokens + $total_prompt_tokens) * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "N/A")

    echo "  [$mode] ${n} requests | ${elapsed_s}s | prompt: ${total_prompt_tokens} | gen: ${total_gen_tokens} | gen tok/s: ${gen_tok_per_sec} | total tok/s: ${total_tok_per_sec} | failed: ${failed}"

    # Debug: show response if zero gen tokens
    if [ "$total_gen_tokens" -eq 0 ]; then
        local first="$result_dir/response_1.json"
        if [ -f "$first" ]; then
            echo "  DEBUG first response (300 chars):"
            head -c 300 "$first"
            echo ""
        fi
    fi
}

# ---------- Generate test prompts ----------
echo ""
echo "=== Generating test prompts ==="

# Short prompt (~1k tokens, 512 gen) for quick iteration
PROMPT_SHORT="$WORKDIR/prompt_short.json"
python3 -c '
import json
system_message = (
    "Your input fields are:\n"
    "1. \`html\` (str): \n2. \`extraction_spec\` (str):\n"
    "Your output fields are:\n1. \`text\` (str):\n"
    "All interactions will be structured in the following way.\n\n"
    "[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: "
    "Extract the main content text from a given HTML document."
)
html = """<!DOCTYPE html><html><head><title>Introduction to Machine Learning</title></head>
<body><nav><a href="/">Home</a> | <a href="/blog">Blog</a></nav>
<main><article><h1>Introduction to Machine Learning</h1>
<p>Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Instead of being explicitly programmed, these systems improve their
performance on a specific task through experience.</p>
<h2>Types of Machine Learning</h2>
<h3>Supervised Learning</h3>
<p>In supervised learning, the algorithm learns from labeled training data. Each example
consists of an input and a desired output. Common algorithms include linear regression,
decision trees, random forests, and neural networks.</p>
<h3>Unsupervised Learning</h3>
<p>Unsupervised learning involves finding patterns in data without pre-existing labels.
Common techniques include clustering, dimensionality reduction, and autoencoders.</p>
<h3>Reinforcement Learning</h3>
<p>In reinforcement learning, an agent learns to make decisions by taking actions in an
environment to maximize cumulative reward.</p>
<h2>Deep Learning</h2>
<p>Deep learning is a subset of machine learning based on artificial neural networks with
multiple layers. Key architectures include CNNs for images, RNNs for sequences, and
transformers for NLP. The transformer architecture has revolutionized the field.</p>
</article></main>
<footer><p>Copyright 2024. All rights reserved.</p></footer></body></html>"""
spec = "Extract the main content from the HTML into clean Markdown. Remove boilerplate."
user = f"[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{spec}\n\nRespond with [[ ## text ## ]] then [[ ## completed ## ]]."
req = {"model":"qwen3","messages":[{"role":"system","content":system_message},{"role":"user","content":user}],"max_tokens":512,"temperature":0.0}
print(json.dumps(req))
' > "$PROMPT_SHORT"
echo "Short prompt: ~1k tokens, 512 gen tokens"

# Realistic prompt (~16k tokens, 4096 gen) matching production workload
PROMPT_REALISTIC="$WORKDIR/prompt_realistic.json"
export WORKDIR
python3 -c '
import json, os

system_message = (
    "Your input fields are:\n"
    "1. \`html\` (str): \n2. \`extraction_spec\` (str):\n"
    "Your output fields are:\n1. \`text\` (str):\n"
    "All interactions will be structured in the following way.\n\n"
    "[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: "
    "Extract the main content text from a given HTML document."
)

spec = """Extract the main content from the provided HTML into clean Markdown.
First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if not useful.
If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads.
- Preserve all technical content exactly."""

topics = [
    ("Machine Learning", [
        ("Supervised Learning", "In supervised learning, the algorithm learns from labeled training data. The training set consists of input-output pairs, and the algorithm learns a mapping function. Common approaches include linear regression for continuous outputs, logistic regression for classification, decision trees that partition the feature space, random forests that ensemble multiple trees, support vector machines that find optimal hyperplanes, and neural networks that learn hierarchical representations. The choice of algorithm depends on the data, dataset size, and computational resources."),
        ("Unsupervised Learning", "Unsupervised learning finds patterns in data without labeled examples. Clustering algorithms like k-means, hierarchical clustering, and DBSCAN group similar points. Dimensionality reduction techniques such as PCA, t-SNE, and UMAP project high-dimensional data into lower dimensions. Autoencoders learn compressed representations through neural networks. Generative models like GANs and VAEs learn to generate new data samples."),
        ("Reinforcement Learning", "Reinforcement learning involves an agent that learns sequential decisions by interacting with an environment. The agent receives rewards based on actions and learns a policy maximizing cumulative reward. Key concepts include Markov Decision Processes, value functions, policy gradient methods, and actor-critic architectures."),
    ]),
    ("Deep Learning", [
        ("CNNs", "Convolutional neural networks process grid-like data such as images. They use convolutional layers with learnable filters, pooling layers for spatial reduction, and fully connected layers for classification. Key innovations include residual connections (ResNet), inception modules, dense connections (DenseNet), and attention mechanisms."),
        ("Transformers", "The transformer architecture relies entirely on self-attention mechanisms. Multi-head attention allows jointly attending to information from different subspaces. Positional encodings inject sequence order. Encoder-decoder structure enables translation, while encoder-only (BERT) and decoder-only (GPT) variants excel at understanding and generation."),
        ("Generative Models", "GANs consist of a generator and discriminator trained in competition. VAEs combine probabilistic inference with neural networks. Diffusion models generate data by reversing a noise process. Large language models demonstrate remarkable few-shot learning."),
    ]),
    ("NLP", [
        ("Text Representation", "Words can be represented as dense vectors through Word2Vec, GloVe, and FastText. Contextual embeddings from BERT and GPT capture context-dependent meaning. Subword tokenization methods like BPE handle rare words."),
        ("Language Understanding", "NER identifies entities in text. Sentiment analysis determines emotional tone. QA systems extract answers from context. Relation extraction identifies semantic relationships."),
        ("Text Generation", "Autoregressive models generate text token by token. Beam search and nucleus sampling control diversity. RAG combines parametric knowledge with retrieved documents. RLHF aligns outputs with human preferences."),
    ]),
    ("Computer Vision", [
        ("Object Detection", "Two-stage detectors like Faster R-CNN propose regions then classify them. Single-stage detectors like YOLO predict boxes directly. Anchor-free methods detect objects as keypoints. FPNs handle multiple scales."),
        ("Segmentation", "Semantic segmentation labels every pixel. Instance segmentation distinguishes objects. U-Net uses skip connections. DeepLab employs atrous convolutions. Mask R-CNN extends detection with segmentation."),
    ]),
    ("Training", [
        ("Optimization", "SGD updates with mini-batches. Momentum accelerates convergence. Adam combines momentum with adaptive rates. Cosine annealing and warm restarts improve dynamics. Gradient clipping prevents explosions."),
        ("Regularization", "Dropout zeros activations randomly. Weight decay adds L2 penalty. Batch norm normalizes inputs. Layer norm is preferred in transformers. Label smoothing prevents overconfidence."),
        ("Distributed", "Data parallelism replicates models across devices. Model parallelism partitions models. Pipeline parallelism overlaps micro-batches. ZeRO partitions optimizer states. Mixed precision uses FP16 with FP32 accumulation."),
    ]),
]

target_chars = 16000 * 4  # ~16k tokens
parts = ["<!DOCTYPE html><html><head><title>AI Guide</title></head><body>"]
parts.append("<nav><a href=\"/\">Home</a> | <a href=\"/blog\">Blog</a></nav><main><article>")
current = sum(len(p) for p in parts)
rnd = 0
while current < target_chars:
    for topic, sections in topics:
        if current >= target_chars:
            break
        rnd += 1
        parts.append(f"<h2>{topic} (Part {rnd})</h2>")
        for title, text in sections:
            if current >= target_chars:
                break
            parts.append(f"<h3>{title}</h3><p>{text}</p>")
            if rnd % 3 == 0:
                parts.append(f"<div class=\"note\"><p><strong>Note:</strong> {title} is essential.</p></div>")
            current = sum(len(p) for p in parts)
parts.append("</article></main><footer><p>Copyright 2024.</p></footer></body></html>")
html = "".join(parts)

user = f"[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{spec}\n\nRespond with [[ ## text ## ]] then [[ ## completed ## ]]."
req = {"model":"qwen3","messages":[{"role":"system","content":system_message},{"role":"user","content":user}],"max_tokens":4096,"temperature":0.0}

workdir = os.environ.get("WORKDIR", "/tmp/llama-cpp-bench")
path = os.path.join(workdir, "prompt_realistic.json")
with open(path, "w") as f:
    json.dump(req, f)
print(f"Realistic prompt: ~{len(json.dumps(req)) // 4} estimated tokens ({len(json.dumps(req))} chars)")
'
echo ""

# ============================================================================
echo "========================================================"
echo "=== TEST 1: SHORT PROMPT (~1k tokens, 512 gen)       ==="
echo "========================================================"
echo ""
echo "Comparing sequential vs concurrent requests on a single server."
echo "Each test sends 4 requests total. Server has 16 threads."
echo ""

N_REQUESTS=4

# --- Baseline: --parallel 1 (default), sequential requests ---
echo "=== Baseline: --parallel 1, sequential ==="
start_server "$THREADS" 1 4096 512
run_sequential "$N_REQUESTS" "$PROMPT_SHORT"
cleanup_server
sleep 2

# --- Parallel: --parallel 2, concurrent requests ---
echo ""
echo "=== --parallel 2, concurrent ==="
start_server "$THREADS" 2 4096 512
run_concurrent 2 "$PROMPT_SHORT"
cleanup_server
sleep 2

# --- Parallel: --parallel 4, concurrent requests ---
echo ""
echo "=== --parallel 4, concurrent ==="
start_server "$THREADS" 4 4096 512
run_concurrent "$N_REQUESTS" "$PROMPT_SHORT"
cleanup_server
sleep 2

# --- Parallel: --parallel 8, concurrent requests ---
echo ""
echo "=== --parallel 8, concurrent ==="
start_server "$THREADS" 8 8192 512
run_concurrent 8 "$PROMPT_SHORT"
cleanup_server
sleep 2

# --- Parallel: --parallel 16, concurrent requests ---
echo ""
echo "=== --parallel 16, 16 concurrent ==="
start_server "$THREADS" 16 4096 512
run_concurrent 16 "$PROMPT_SHORT"
cleanup_server
sleep 2

# --- Parallel: --parallel 32, concurrent requests ---
echo ""
echo "=== --parallel 32, 32 concurrent (AGGRESSIVE) ==="
echo "  Memory estimate: 32 slots x 4096 ctx x Q4_K_M ~ 2-4 GiB KV cache. Should fit in 16g."
if start_server "$THREADS" 32 4096 512; then
    run_concurrent 32 "$PROMPT_SHORT"
    cleanup_server
else
    echo "  CRASHED: --parallel 32 with ctx=4096 failed to start. Server log:"
    tail -20 "$WORKDIR/server.log" 2>/dev/null || true
    cleanup_server
fi
sleep 2

# --- For comparison: sequential on --parallel 4 server ---
echo ""
echo "=== --parallel 4 server, but sequential requests (isolation test) ==="
start_server "$THREADS" 4 4096 512
run_sequential "$N_REQUESTS" "$PROMPT_SHORT"
cleanup_server
sleep 2

# ============================================================================
echo ""
echo "========================================================"
echo "=== TEST 2: REALISTIC PROMPT (~16k tokens, 4096 gen) ==="
echo "========================================================"
echo ""
echo "This matches our production workload. Each request takes several minutes."
echo "Testing with fewer requests since each is slow."
echo ""

# --- Baseline: --parallel 1, sequential ---
echo "=== Baseline: --parallel 1, sequential (2 requests) ==="
start_server "$THREADS" 1 32768 4096
run_sequential 2 "$PROMPT_REALISTIC"
cleanup_server
sleep 2

# --- Parallel: --parallel 2, 2 concurrent ---
echo ""
echo "=== --parallel 2, 2 concurrent ==="
start_server "$THREADS" 2 32768 4096
run_concurrent 2 "$PROMPT_REALISTIC"
cleanup_server
sleep 2

# --- Parallel: --parallel 4, 4 concurrent ---
echo ""
echo "=== --parallel 4, 4 concurrent ==="
start_server "$THREADS" 4 32768 4096
run_concurrent 4 "$PROMPT_REALISTIC"
cleanup_server
sleep 2

# --- Parallel: --parallel 16, 16 concurrent (realistic) ---
echo ""
echo "=== --parallel 16, 16 concurrent (realistic prompt) ==="
echo "  Memory estimate: 16 slots x 32768 ctx x Q4_K_M ~ 8-16 GiB KV cache."
echo "  May be tight on 16g RAM workers but fine on a TPU node (440g)."
if start_server "$THREADS" 16 32768 4096; then
    run_concurrent 16 "$PROMPT_REALISTIC"
    cleanup_server
else
    echo "  CRASHED: --parallel 16 with ctx=32768 failed to start. Server log:"
    tail -20 "$WORKDIR/server.log" 2>/dev/null || true
    cleanup_server
fi
sleep 2

# --- Parallel: --parallel 32, 32 concurrent (realistic) ---
# This is the moon-shot: if it works, inference finishes in ~1 day instead of ~30.
# ctx=32768 x 32 slots is ~32 GiB KV cache. Needs a beefy node, not a 16g worker.
# We test it here to see if it even works, and measure any throughput gain.
echo ""
echo "=== --parallel 32, 32 concurrent (realistic prompt — MOON SHOT) ==="
echo "  Memory estimate: 32 slots x 32768 ctx x Q4_K_M ~ 16-32 GiB KV cache."
echo "  This may OOM on machines with <64g RAM. If it crashes, that's expected."
if start_server "$THREADS" 32 32768 4096; then
    run_concurrent 32 "$PROMPT_REALISTIC"
    cleanup_server
else
    echo "  CRASHED: --parallel 32 with ctx=32768 failed to start."
    echo "  This likely means we need more RAM per worker or smaller ctx for high parallelism."
    echo "  Server log:"
    tail -30 "$WORKDIR/server.log" 2>/dev/null || true
    cleanup_server
    echo ""
    echo "  Trying fallback: --parallel 32 with ctx=8192 (shorter context, less memory) ==="
    if start_server "$THREADS" 32 8192 4096; then
        run_concurrent 32 "$PROMPT_REALISTIC"
        cleanup_server
    else
        echo "  CRASHED again at ctx=8192. --parallel 32 is infeasible at this memory budget."
        tail -20 "$WORKDIR/server.log" 2>/dev/null || true
        cleanup_server
    fi
fi
sleep 2

# ============================================================================
echo ""
echo "========================================================"
echo "=== SUMMARY ==="
echo "========================================================"
echo ""
echo "Compare gen tok/s across modes:"
echo "  - Sequential baseline:  N requests * (1/single_request_time)"
echo "  - Concurrent --parallel: if higher, batching helps"
echo ""
echo "Key questions answered:"
echo "  1. Does --parallel N give >1x throughput? (batching wins)"
echo "  2. Does throughput scale linearly with N? (or diminishing returns)"
echo "  3. At what N does it crash or degrade? (memory/CPU limit)"
echo "  4. Does --parallel 32 work at all? (moon shot for 30x speedup)"
echo ""
echo "If --parallel N works, implement in inference_llamacpp.py by:"
echo "  1. Adding --parallel N to server launch cmd in _get_or_start_server"
echo "  2. Using concurrent.futures.ThreadPoolExecutor(N) in _process_shard"
echo "  3. Collecting results and yielding in order"
echo ""
echo "Memory per worker: --parallel N * ctx_size * ~0.5-1 MiB (KV cache per slot)."
echo "  --parallel 4,  ctx=32768 → ~64-128 MiB  → fits in 16g easily"
echo "  --parallel 16, ctx=32768 → ~256-512 MiB  → fits in 16g"
echo "  --parallel 32, ctx=32768 → ~512 MiB-1 GiB → fits in 16g but tight with model"
echo ""
echo "If --parallel 32 works but needs more RAM, consider:"
echo "  - Bumping RAM_PER_WORKER from 16g to 32g (fewer workers per node)"
echo "  - Reducing ctx_size for --parallel 32 (e.g. 16384 instead of 32768)"
