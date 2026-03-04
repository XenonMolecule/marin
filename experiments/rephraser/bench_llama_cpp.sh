#!/bin/bash
# Benchmark llama.cpp CPU inference on a TPU worker node.
#
# Usage (from SSH session on TPU node):
#   bash /tmp/bench_llama_cpp.sh
#
# Or via dev_tpu.py:
#   gcloud compute tpus tpu-vm scp experiments/rephraser/bench_llama_cpp.sh <TPU_NAME>:/tmp/ --zone=<ZONE>
#   gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> -- "bash /tmp/bench_llama_cpp.sh"
#
# Stage 1: Single-instance llama-bench (prefill + generation at various thread counts)
# Stage 2: Parallel llama-server throughput via /v1/chat/completions (~1k prompt tokens)
# Stage 3: Realistic workload — 16k/28k prompt tokens + 4096 generation tokens
#
# Stage 1 results (skip with SKIP_STAGE1=1):
#   threads | pp512  | pp2048 | pp8192 | tg256
#   1       | 44.6   | 36.5   | 20.5   | 6.3
#   2       | 77.6   | 65.7   | 39.1   | 12.5
#   4       | 153.8  | 124.6  | 73.8   | 23.7
#   8       | 227.1  | 185.1  | 117.5  | 35.4
#   16      | 333.5  | 286.4  | 198.5  | 54.3
#   32      | 422.2  | 390.1  | 287.8  | 73.5
#
# Stage 2 results (~931 prompt tokens, 512 gen tokens; skip with SKIP_STAGE2=1):
#   config   | gen tok/s | total tok/s
#   8×1      | 20.2     | 25.6
#   8×4      | 89.9     | 113.1
#   8×8      | 152.7    | 194.3
#   8×13     | 190.9    | 242.1
#   16×1     | 28.4     | 36.0
#   16×4     | 128.5    | 162.6
#   16×6     | 167.0    | 211.7
#   32×1     | 39.3     | 50.2
#   32×3     | 126.9    | 158.7
#   32×6     | 187.2    | 237.1

set -euo pipefail

WORKDIR="/tmp/llama-cpp-bench"
MODEL_REPO="MichaelR207/qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M-GGUF"
MODEL_FILE="qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"
SKIP_STAGE1="${SKIP_STAGE1:-1}"
SKIP_STAGE2="${SKIP_STAGE2:-1}"

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

if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    fi
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build build --config Release -j"$(nproc)" --target llama-bench llama-server
    cd "$WORKDIR"
else
    echo "llama.cpp already built, skipping."
fi

BENCH="$WORKDIR/llama.cpp/build/bin/llama-bench"
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

# ---------- Stage 1: Single-instance benchmarks ----------
if [ "$SKIP_STAGE1" = "0" ]; then
    echo ""
    echo "========================================"
    echo "=== STAGE 1: SINGLE-INSTANCE BENCH   ==="
    echo "========================================"
    echo ""

    echo "=== Prefill speed ==="
    "$BENCH" -m "$MODEL_PATH" -t 1,2,4,8,16,32 -p 512,2048,8192 -n 0 -r 2

    echo ""
    echo "=== Generation speed ==="
    "$BENCH" -m "$MODEL_PATH" -t 1,2,4,8,16,32 -p 0 -n 256 -r 2
else
    echo ""
    echo "=== Skipping Stage 1 (SKIP_STAGE1=1). Known results in script header. ==="
fi

# ---------- Shared server management functions ----------
# Cleanup function to kill all servers
cleanup_servers() {
    for pid_file in "$WORKDIR"/server_*.pid; do
        if [ -f "$pid_file" ]; then
            kill "$(cat "$pid_file")" 2>/dev/null || true
            rm -f "$pid_file"
        fi
    done
}
trap cleanup_servers EXIT

start_servers() {
    local threads=$1
    local count=$2
    local ctx_size=${3:-4096}
    local n_predict=${4:-512}
    local base_port=8080

    for i in $(seq 0 $((count - 1))); do
        local port=$((base_port + i))
        "$SERVER" \
            --model "$MODEL_PATH" \
            --host 127.0.0.1 \
            --port "$port" \
            --threads "$threads" \
            --ctx-size "$ctx_size" \
            --n-predict "$n_predict" \
            --log-disable \
            > "$WORKDIR/server_${port}.log" 2>&1 &
        echo $! > "$WORKDIR/server_${port}.pid"
    done

    # Wait for all servers to be healthy (model fully loaded, not just HTTP up)
    local all_ready=0
    for attempt in $(seq 1 120); do
        all_ready=1
        for i in $(seq 0 $((count - 1))); do
            local port=$((base_port + i))
            local health_status
            health_status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null)
            # /health returns 503 while model is loading, 200 when ready
            if [ "$health_status" != "200" ]; then
                all_ready=0
                break
            fi
            # Double-check: try a tiny completion to verify model is actually loaded
            local test_status
            test_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null)
            if [ "$test_status" = "503" ]; then
                all_ready=0
                break
            fi
        done
        if [ "$all_ready" = "1" ]; then
            break
        fi
        if [ $((attempt % 10)) -eq 0 ]; then
            echo "  Waiting for model load... (${attempt}s)"
        fi
        sleep 1
    done

    if [ "$all_ready" != "1" ]; then
        echo "  ERROR: Not all servers became healthy within 120s"
        cleanup_servers
        return 1
    fi
}

run_server_test() {
    local threads=$1
    local instances=$2
    local ctx_size=${3:-4096}
    local n_predict=${4:-512}
    local prompt_file=${5:-$PROMPT_JSON}
    local total_cpu=$((threads * instances))

    echo "--- ${instances} servers x ${threads} threads (ctx=${ctx_size}, predict=${n_predict}, total CPU: ${total_cpu}) ---"

    # Start servers
    start_servers "$threads" "$instances" "$ctx_size" "$n_predict"
    if [ $? -ne 0 ]; then
        return
    fi

    echo "  All ${instances} servers healthy. Sending requests..."

    local start_time
    start_time=$(date +%s%N)

    # Send one request to each server in parallel
    local pids=()
    local result_dir="$WORKDIR/results_${threads}_${instances}_ctx${ctx_size}_gen${n_predict}"
    mkdir -p "$result_dir"

    for i in $(seq 0 $((instances - 1))); do
        local port=$((8080 + i))
        curl -s -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$prompt_file" \
            -o "$result_dir/response_${i}.json" \
            -w "" \
            --max-time 1200 &
        pids+=($!)
    done

    # Wait for all requests
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++)) || true
        fi
    done

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Count total generated tokens from responses
    local total_gen_tokens=0
    local total_prompt_tokens=0
    for i in $(seq 0 $((instances - 1))); do
        local resp="$result_dir/response_${i}.json"
        if [ -f "$resp" ]; then
            local gen_tok
            gen_tok=$(python3 -c "
import json, sys
try:
    d = json.load(open('$resp'))
    u = d.get('usage', {})
    print(u.get('completion_tokens', 0))
except: print(0)
" 2>/dev/null)
            local prompt_tok
            prompt_tok=$(python3 -c "
import json, sys
try:
    d = json.load(open('$resp'))
    u = d.get('usage', {})
    print(u.get('prompt_tokens', 0))
except: print(0)
" 2>/dev/null)
            total_gen_tokens=$((total_gen_tokens + gen_tok))
            total_prompt_tokens=$((total_prompt_tokens + prompt_tok))
        fi
    done

    local gen_tok_per_sec
    gen_tok_per_sec=$(echo "scale=1; $total_gen_tokens * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "N/A")
    local total_tok_per_sec
    total_tok_per_sec=$(echo "scale=1; ($total_gen_tokens + $total_prompt_tokens) * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "N/A")

    echo "  Time: ${elapsed_ms}ms | Prompt tokens: ${total_prompt_tokens} | Gen tokens: ${total_gen_tokens}"
    echo "  Aggregate gen: ${gen_tok_per_sec} tok/s | Total throughput: ${total_tok_per_sec} tok/s | Failed: ${failed}"

    # Show first response if no tokens were generated (debugging)
    if [ "$total_gen_tokens" -eq 0 ]; then
        local first_resp="$result_dir/response_0.json"
        if [ -f "$first_resp" ]; then
            echo "  DEBUG response_0.json (first 300 chars):"
            head -c 300 "$first_resp"
            echo ""
        fi
    fi
    echo ""

    # Cleanup
    cleanup_servers
    sleep 2  # Let ports free up
}

# ---------- Stage 2: Parallel llama-server throughput (~1k prompt tokens) ----------
if [ "$SKIP_STAGE2" = "0" ]; then
    echo ""
    echo "========================================"
    echo "=== STAGE 2: PARALLEL SERVER TEST    ==="
    echo "========================================"
    echo ""

    # Create Stage 2 test prompt: system message + sample HTML (~1k tokens)
    PROMPT_JSON="$WORKDIR/test_request.json"
    python3 -c '
import json
system_message = (
    "Your input fields are:\n"
    "1. \`html\` (str): \n"
    "2. \`extraction_spec\` (str):\n"
    "Your output fields are:\n"
    "1. \`text\` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)
sample_html = """<!DOCTYPE html><html><head><title>Introduction to Machine Learning</title></head>
<body><nav><a href="/">Home</a> | <a href="/blog">Blog</a> | <a href="/about">About</a></nav>
<main><article><h1>Introduction to Machine Learning</h1>
<p>Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Instead of being explicitly programmed, these systems improve their
performance on a specific task through experience.</p>
<h2>Types of Machine Learning</h2>
<h3>Supervised Learning</h3>
<p>In supervised learning, the algorithm learns from labeled training data. Each example
in the training set consists of an input and a desired output. The algorithm learns a
mapping from inputs to outputs. Common algorithms include linear regression, decision
trees, random forests, and neural networks.</p>
<h3>Unsupervised Learning</h3>
<p>Unsupervised learning involves finding patterns in data without pre-existing labels.
The algorithm tries to find hidden structure in unlabeled data. Common techniques include
clustering (k-means, DBSCAN), dimensionality reduction (PCA, t-SNE), and autoencoders.</p>
<h3>Reinforcement Learning</h3>
<p>In reinforcement learning, an agent learns to make decisions by taking actions in an
environment to maximize cumulative reward. The agent learns from trial and error, receiving
feedback in the form of rewards or penalties.</p>
<h2>Key Concepts</h2>
<p>Some fundamental concepts in machine learning include:</p>
<ul>
<li><strong>Features</strong>: The input variables used to make predictions</li>
<li><strong>Labels</strong>: The output variable that the model is trying to predict</li>
<li><strong>Training</strong>: The process of learning from data</li>
<li><strong>Overfitting</strong>: When a model learns noise in the training data</li>
<li><strong>Regularization</strong>: Techniques to prevent overfitting</li>
<li><strong>Cross-validation</strong>: A method to assess model generalization</li>
</ul>
<h2>Deep Learning</h2>
<p>Deep learning is a subset of machine learning based on artificial neural networks with
multiple layers. These deep neural networks can learn complex patterns and representations
from large amounts of data. Key architectures include convolutional neural networks (CNNs)
for image processing, recurrent neural networks (RNNs) for sequential data, and transformers
for natural language processing.</p>
<p>The transformer architecture, introduced in the paper "Attention Is All You Need" by
Vaswani et al. (2017), has revolutionized NLP. It uses self-attention mechanisms to process
input sequences in parallel, enabling efficient training on large datasets. Models like
BERT, GPT, and T5 are all based on the transformer architecture.</p>
</article></main>
<footer><p>Copyright 2024 ML Blog. All rights reserved.</p>
<a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms of Service</a></footer>
</body></html>"""
spec = """Extract the main content from the provided HTML into clean Markdown.
First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if not useful.
If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads.
- Preserve all technical content exactly."""
user_content = (
    f"[[ ## html ## ]]\n{sample_html}\n\n"
    f"[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field \`[[ ## text ## ]]\`, "
    "and then ending with the marker for \`[[ ## completed ## ]]\`."
)
request = {
    "model": "qwen3",
    "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ],
    "max_tokens": 512,
    "temperature": 0.0,
}
print(json.dumps(request))
' > "$PROMPT_JSON"

    echo "Test prompt created. Estimated ~1k input tokens."
    echo ""
    echo "=== Testing different server configurations ==="
    echo "Each test: start N llama-servers, send 1 request each, measure aggregate throughput"
    echo ""

    # 8-thread configs
    run_server_test 8 1 4096 512 "$PROMPT_JSON"
    run_server_test 8 4 4096 512 "$PROMPT_JSON"
    run_server_test 8 8 4096 512 "$PROMPT_JSON"
    run_server_test 8 13 4096 512 "$PROMPT_JSON"

    # 16-thread configs
    run_server_test 16 1 4096 512 "$PROMPT_JSON"
    run_server_test 16 4 4096 512 "$PROMPT_JSON"
    run_server_test 16 6 4096 512 "$PROMPT_JSON"

    # 32-thread configs
    run_server_test 32 1 4096 512 "$PROMPT_JSON"
    run_server_test 32 3 4096 512 "$PROMPT_JSON"
    run_server_test 32 6 4096 512 "$PROMPT_JSON"

else
    echo ""
    echo "=== Skipping Stage 2 (SKIP_STAGE2=1). Known results in script header. ==="
fi

# ---------- Stage 3: Realistic workload (16k/28k prompt + 4096 gen) ----------
echo ""
echo "========================================"
echo "=== STAGE 3: REALISTIC WORKLOAD TEST ==="
echo "========================================"
echo ""
echo "Testing with production-realistic prompt lengths and 4096 generation tokens."
echo "Each test will take several minutes — running only a few configs."
echo ""

# Generate long HTML prompts (~16k and ~28k tokens) using Python
# The base HTML article is ~300 tokens. We repeat varied content to reach targets.
export WORKDIR
python3 -c '
import json

system_message = (
    "Your input fields are:\n"
    "1. \`html\` (str): \n"
    "2. \`extraction_spec\` (str):\n"
    "Your output fields are:\n"
    "1. \`text\` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

spec = """Extract the main content from the provided HTML into clean Markdown.
First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if not useful.
If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads.
- Preserve all technical content exactly."""

# Topics to generate varied, realistic long-form HTML content
topics = [
    ("Machine Learning Fundamentals", [
        ("Supervised Learning", "In supervised learning, the algorithm learns from labeled training data. The training set consists of input-output pairs, and the algorithm learns a mapping function. Common approaches include linear regression for continuous outputs, logistic regression for classification, decision trees that partition the feature space, random forests that ensemble multiple trees, support vector machines that find optimal hyperplanes, and neural networks that learn hierarchical representations. The choice of algorithm depends on the nature of the data, the size of the dataset, and the computational resources available."),
        ("Unsupervised Learning", "Unsupervised learning finds patterns in data without labeled examples. Clustering algorithms like k-means, hierarchical clustering, and DBSCAN group similar data points together. Dimensionality reduction techniques such as PCA, t-SNE, and UMAP project high-dimensional data into lower dimensions while preserving structure. Autoencoders learn compressed representations through neural network architectures. Generative models like GANs and VAEs learn to generate new data samples that resemble the training distribution."),
        ("Reinforcement Learning", "Reinforcement learning involves an agent that learns to make sequential decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns a policy that maximizes cumulative reward. Key concepts include the Markov Decision Process formalization, value functions that estimate expected returns, policy gradient methods that directly optimize the policy, and actor-critic architectures that combine both approaches. Applications range from game playing to robotics to resource management."),
    ]),
    ("Deep Learning Architectures", [
        ("Convolutional Neural Networks", "CNNs are designed for processing grid-like data such as images. They use convolutional layers that apply learnable filters to detect local patterns, pooling layers that reduce spatial dimensions, and fully connected layers for final classification. Key innovations include residual connections (ResNet), inception modules (GoogLeNet), dense connections (DenseNet), and attention mechanisms (Squeeze-and-Excitation networks). Modern architectures achieve superhuman performance on image classification benchmarks like ImageNet."),
        ("Recurrent Neural Networks", "RNNs process sequential data by maintaining hidden states that capture temporal dependencies. Long Short-Term Memory (LSTM) networks address the vanishing gradient problem with gating mechanisms including input, forget, and output gates. Gated Recurrent Units (GRUs) simplify the LSTM architecture with fewer parameters. Bidirectional RNNs process sequences in both directions. Sequence-to-sequence models with attention revolutionized machine translation before being superseded by transformers."),
        ("Transformer Architecture", "The transformer architecture relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. Multi-head attention allows the model to jointly attend to information from different representation subspaces. Positional encodings inject sequence order information. The encoder-decoder structure enables tasks like translation, while encoder-only (BERT) and decoder-only (GPT) variants excel at understanding and generation respectively. Scaling laws show predictable performance improvements with model size."),
        ("Generative Models", "Generative adversarial networks consist of a generator and discriminator trained in competition. Variational autoencoders combine probabilistic inference with neural networks. Diffusion models generate data by learning to reverse a noise process. Flow-based models use invertible transformations for exact likelihood computation. Large language models like GPT-4 demonstrate remarkable few-shot learning abilities across diverse tasks."),
    ]),
    ("Natural Language Processing", [
        ("Text Representation", "Words can be represented as dense vectors through techniques like Word2Vec, GloVe, and FastText. Contextual embeddings from models like ELMo, BERT, and GPT capture word meaning that varies with context. Subword tokenization methods like BPE, WordPiece, and SentencePiece handle rare words and morphological variations. Sentence and document embeddings aggregate token representations for downstream tasks."),
        ("Language Understanding", "Named entity recognition identifies and classifies entities in text. Sentiment analysis determines the emotional tone of text. Question answering systems extract or generate answers from context. Relation extraction identifies semantic relationships between entities. Coreference resolution links mentions that refer to the same entity. Semantic parsing converts natural language into formal representations."),
        ("Text Generation", "Autoregressive models generate text token by token, conditioning on previously generated tokens. Beam search, nucleus sampling, and temperature scaling control the diversity-quality tradeoff. Retrieval-augmented generation combines parametric knowledge with retrieved documents. Instruction tuning and RLHF align model outputs with human preferences. Chain-of-thought prompting elicits step-by-step reasoning."),
    ]),
    ("Computer Vision", [
        ("Image Classification", "Modern image classifiers use deep convolutional networks pretrained on large datasets. Transfer learning adapts pretrained features to new domains with limited data. Data augmentation techniques like random cropping, flipping, color jittering, and mixup improve generalization. Vision transformers (ViT) apply the transformer architecture directly to image patches, achieving competitive results with CNNs."),
        ("Object Detection", "Two-stage detectors like Faster R-CNN first propose regions then classify them. Single-stage detectors like YOLO and SSD directly predict bounding boxes and classes. Anchor-free methods like CenterNet detect objects as keypoints. Feature pyramid networks handle objects at multiple scales. Non-maximum suppression filters redundant detections. Modern detectors achieve real-time performance with high accuracy."),
        ("Image Segmentation", "Semantic segmentation assigns a class label to every pixel. Instance segmentation additionally distinguishes individual object instances. Panoptic segmentation unifies both tasks. U-Net and its variants use skip connections for precise localization. DeepLab employs atrous convolutions and conditional random fields. Mask R-CNN extends Faster R-CNN with a segmentation branch."),
    ]),
    ("Optimization and Training", [
        ("Gradient Descent Variants", "Stochastic gradient descent updates parameters using mini-batches of data. Momentum accumulates past gradients to accelerate convergence. Adam combines momentum with adaptive learning rates per parameter. Learning rate schedules like cosine annealing and warm restarts improve training dynamics. Gradient clipping prevents exploding gradients in deep networks."),
        ("Regularization Techniques", "Dropout randomly zeros activations during training to prevent co-adaptation. Weight decay adds L2 penalty to the loss function. Batch normalization normalizes layer inputs to stabilize training. Layer normalization is preferred in transformers and recurrent networks. Label smoothing prevents overconfident predictions. Early stopping halts training when validation performance degrades."),
        ("Distributed Training", "Data parallelism replicates the model across devices and splits the data. Model parallelism partitions the model across devices for large models. Pipeline parallelism overlaps computation across micro-batches. ZeRO optimization partitions optimizer states, gradients, and parameters. Mixed precision training uses FP16 computation with FP32 accumulation to reduce memory and increase throughput."),
    ]),
]

def generate_html(target_tokens):
    """Generate a realistic HTML document with approximately target_tokens tokens.
    Rough heuristic: 1 token ~ 4 characters for English text with HTML markup."""
    target_chars = target_tokens * 4

    parts = []
    parts.append("<!DOCTYPE html><html><head><title>Comprehensive Guide to Modern AI</title>")
    parts.append("<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
    parts.append("<link rel=\"stylesheet\" href=\"/css/main.css\"></head>")
    parts.append("<body>")
    parts.append("<nav class=\"main-nav\"><div class=\"nav-container\">")
    parts.append("<a href=\"/\" class=\"logo\">AI Research Hub</a>")
    parts.append("<ul><li><a href=\"/tutorials\">Tutorials</a></li>")
    parts.append("<li><a href=\"/papers\">Papers</a></li>")
    parts.append("<li><a href=\"/blog\">Blog</a></li>")
    parts.append("<li><a href=\"/about\">About</a></li></ul></div></nav>")
    parts.append("<main class=\"content\"><article class=\"post\">")

    current_chars = sum(len(p) for p in parts)
    round_num = 0

    while current_chars < target_chars:
        for topic_name, sections in topics:
            if current_chars >= target_chars:
                break
            round_num += 1
            parts.append(f"<h2>{topic_name} (Part {round_num})</h2>")
            for section_title, section_text in sections:
                if current_chars >= target_chars:
                    break
                parts.append(f"<h3>{section_title}</h3>")
                parts.append(f"<p>{section_text}</p>")
                # Add some structural variety
                if round_num % 3 == 0:
                    parts.append("<div class=\"info-box\"><p><strong>Key Takeaway:</strong> "
                                f"Understanding {section_title.lower()} is crucial for modern AI applications.</p></div>")
                if round_num % 4 == 0:
                    nl = "\n"
                    class_name = section_title.replace(" ", "")
                    code_block = (
                        "<pre><code># Example: " + section_title + nl
                        + "import torch" + nl + "import torch.nn as nn" + nl + nl
                        + "class " + class_name + "Module(nn.Module):" + nl
                        + "    def __init__(self):" + nl + "        super().__init__()" + nl
                        + "    def forward(self, x):" + nl + "        return x" + nl
                        + "</code></pre>"
                    )
                    parts.append(code_block)
                current_chars = sum(len(p) for p in parts)

    parts.append("</article></main>")
    parts.append("<footer><p>Copyright 2024 AI Research Hub. All rights reserved.</p>")
    parts.append("<a href=\"/privacy\">Privacy</a> | <a href=\"/terms\">Terms</a></footer>")
    parts.append("</body></html>")

    return "".join(parts)

def make_request(html, max_tokens):
    user_content = (
        f"[[ ## html ## ]]\n{html}\n\n"
        f"[[ ## extraction_spec ## ]]\n{spec}\n\n"
        "Respond with the corresponding output fields, "
        "starting with the field \`[[ ## text ## ]]\`, "
        "and then ending with the marker for \`[[ ## completed ## ]]\`."
    )
    return {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

# Generate 16k and 28k token prompts
html_16k = generate_html(16000)
html_28k = generate_html(28000)

req_16k = make_request(html_16k, 4096)
req_28k = make_request(html_28k, 4096)

import os
workdir = os.environ.get("WORKDIR", "/tmp/llama-cpp-bench")
with open(os.path.join(workdir, "prompt_16k.json"), "w") as f:
    json.dump(req_16k, f)
with open(os.path.join(workdir, "prompt_28k.json"), "w") as f:
    json.dump(req_28k, f)

# Report approximate sizes
print(f"16k prompt: ~{len(json.dumps(req_16k)) // 4} estimated tokens ({len(json.dumps(req_16k))} chars)")
print(f"28k prompt: ~{len(json.dumps(req_28k)) // 4} estimated tokens ({len(json.dumps(req_28k))} chars)")
' 2>&1

PROMPT_16K="$WORKDIR/prompt_16k.json"
PROMPT_28K="$WORKDIR/prompt_28k.json"

echo ""

# Run a few configs at each prompt length.
# With 4096 gen tokens at ~20-70 tok/s per instance, each request takes 1-4 minutes.
echo "--- 16k prompt tests (ctx=32768, gen=4096) ---"
echo ""
run_server_test 16 1 32768 4096 "$PROMPT_16K"
run_server_test 16 4 32768 4096 "$PROMPT_16K"
run_server_test 8 4  32768 4096 "$PROMPT_16K"
run_server_test 8 8  32768 4096 "$PROMPT_16K"

echo ""
echo "--- 28k prompt tests (ctx=32768, gen=4096) ---"
echo ""
run_server_test 16 1 32768 4096 "$PROMPT_28K"
run_server_test 16 4 32768 4096 "$PROMPT_28K"
run_server_test 8 4  32768 4096 "$PROMPT_28K"
run_server_test 8 8  32768 4096 "$PROMPT_28K"

echo ""
echo "========================================"
echo "=== SUMMARY ==="
echo "========================================"
echo ""
echo "Compare aggregate gen tok/s against TPU baseline:"
echo "  TPU: ~500 tok/s per TPU, typically 1-2 allocated = 500-1000 tok/s"
echo ""
echo "To estimate multi-node throughput:"
echo "  cluster_tok_s = best_per_node_aggregate * num_nodes"
echo "  us-east5-a has ~23 nodes × 208 CPUs each"
