#!/bin/bash
# Auto-test script: waits for vLLM to become healthy, then runs test prompts
# and saves results to GCS. Designed to run unattended on the TPU VM.
#
# Usage: nohup bash auto_test_when_ready.sh > /tmp/auto_test.log 2>&1 &

set -uo pipefail
# NOTE: intentionally NOT using set -e. Individual command failures are handled
# explicitly. We don't want one flaky curl request to abort the entire test suite.

SERVER="http://localhost:8000"
GCS_RESULTS="gs://marin-us-central1/experiments/kimi-k2-instruct-fp8-test/$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/auto_test.log"

echo "============================================"
echo "Auto-test script started at $(date)"
echo "Server: $SERVER"
echo "Results will be saved to: $GCS_RESULTS"
echo "============================================"

# Step 1: Wait for vLLM to become healthy (poll every 60 seconds, max 3 hours)
echo ""
echo "Step 1: Waiting for vLLM server to become healthy..."
MAX_WAIT=180  # 3 hours in minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  HEALTH=$(docker exec ray-node curl -s -o /dev/null -w "%{http_code}" "$SERVER/health" 2>/dev/null) || HEALTH="000"
  if [ "$HEALTH" = "200" ]; then
    echo "$(date) - Server is HEALTHY! (waited ${WAITED} minutes)"
    break
  fi
  echo "$(date) - Not ready yet (HTTP $HEALTH), waiting... (${WAITED}/${MAX_WAIT} min)"
  sleep 60
  WAITED=$((WAITED + 1))
done

if [ "$HEALTH" != "200" ]; then
  echo "TIMEOUT: Server did not become healthy after ${MAX_WAIT} minutes"
  echo "Saving logs and exiting..."
  docker exec ray-node cat /tmp/vllm_serve.log > /tmp/vllm_serve_dump.log 2>/dev/null
  gcloud storage cp /tmp/vllm_serve_dump.log "$GCS_RESULTS/vllm_serve.log" 2>/dev/null
  gcloud storage cp "$LOG" "$GCS_RESULTS/auto_test.log" 2>/dev/null
  exit 1
fi

# Step 2: Get model info
echo ""
echo "Step 2: Checking model info..."
docker exec ray-node curl -s "$SERVER/v1/models" | python3 -m json.tool > /tmp/model_info.json 2>/dev/null
cat /tmp/model_info.json
gcloud storage cp /tmp/model_info.json "$GCS_RESULTS/model_info.json"

# Step 3: Run smoke test prompts
echo ""
echo "Step 3: Running smoke test prompts..."

run_prompt() {
  local NAME="$1"
  local PROMPT="$2"
  local MAX_TOKENS="${3:-1024}"

  echo ""
  echo "--- Test: $NAME ---"
  START=$(date +%s%N)

  RESPONSE=$(docker exec ray-node curl -s "$SERVER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"/mnt/gcs-models\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"max_tokens\": $MAX_TOKENS,
      \"temperature\": 0.0
    }" 2>/dev/null)

  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))

  echo "$RESPONSE" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    content = r['choices'][0]['message']['content']
    usage = r.get('usage', {})
    print(f'Response ({len(content)} chars):')
    print(content[:500])
    if len(content) > 500:
        print('...[truncated]...')
        print(content[-200:])
    print(f'Usage: {usage}')
    print(f'Latency: ${ELAPSED_MS}ms')
except Exception as e:
    print(f'Error parsing response: {e}')
    print(sys.stdin.read() if hasattr(sys.stdin, 'read') else 'no input')
" 2>/dev/null || echo "Response: $RESPONSE"

  echo "$RESPONSE" > "/tmp/test_${NAME}.json"
  gcloud storage cp "/tmp/test_${NAME}.json" "$GCS_RESULTS/test_${NAME}.json" 2>/dev/null
  echo "Saved to $GCS_RESULTS/test_${NAME}.json"
}

# Test 1: Simple math
run_prompt "simple_math" \
  "What is 7 * 13 + 42? Just give me the number." \
  256

# Test 2: Non-trivial math proof
run_prompt "sqrt2_proof" \
  "Prove that the square root of 2 is irrational. Be rigorous." \
  2048

# Test 3: Harder math problem
run_prompt "cubic_roots" \
  "Find all real roots of x^3 - 6x^2 + 11x - 6 = 0. Show your work step by step." \
  1024

# Test 4: Coding
run_prompt "fizzbuzz" \
  "Write a Python function that solves FizzBuzz for numbers 1 to 100. Include the output." \
  1024

# Test 5: Reasoning
run_prompt "reasoning" \
  "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your reasoning carefully." \
  512

# Step 4: Quick throughput test (10 concurrent requests)
echo ""
echo "Step 4: Throughput test (10 concurrent requests)..."
START=$(date +%s%N)

for i in $(seq 1 10); do
  docker exec ray-node curl -s "$SERVER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"/mnt/gcs-models\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is $i * $i?\"}],
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }" > "/tmp/throughput_${i}.json" 2>/dev/null &
done
wait

END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
echo "10 concurrent requests completed in ${ELAPSED_MS}ms"
echo "Approximate throughput: $(python3 -c "print(f'{10000/$ELAPSED_MS:.1f} req/s')")"

# Collect throughput results
python3 -c "
import json, glob
total_tokens = 0
for f in sorted(glob.glob('/tmp/throughput_*.json')):
    try:
        with open(f) as fh:
            r = json.load(fh)
        usage = r.get('usage', {})
        total_tokens += usage.get('completion_tokens', 0) + usage.get('prompt_tokens', 0)
    except: pass
print(f'Total tokens across 10 requests: {total_tokens}')
print(f'Tokens/sec estimate: {total_tokens / ($ELAPSED_MS / 1000):.1f}')
" 2>/dev/null

# Step 5: Run MATH-500 benchmark (Ahmed's full eval)
echo ""
echo "Step 5: Running MATH-500 benchmark..."
echo "Installing benchmark dependencies..."
pip install -q aiohttp datasets sympy 2>/dev/null

# Copy benchmark script from GCS or use local
if [ -f /tmp/benchmark_math500.py ]; then
  echo "Using existing benchmark script"
else
  # The script should have been copied during setup, but just in case:
  echo "benchmark_math500.py not found, skipping MATH-500"
fi

if [ -f /tmp/benchmark_math500.py ]; then
  echo "Running MATH-500 (concurrency=16, max_tokens=2048)..."
  python3 /tmp/benchmark_math500.py \
    --server "$SERVER" \
    --concurrency 16 \
    --max-tokens 2048 \
    --output /tmp/math500_results.jsonl \
    > /tmp/math500_output.txt 2>&1 || echo "MATH-500 benchmark had errors (check logs)"

  echo "MATH-500 output:"
  tail -30 /tmp/math500_output.txt

  # Save results
  gcloud storage cp /tmp/math500_results.jsonl "$GCS_RESULTS/math500_results.jsonl" 2>/dev/null || true
  gcloud storage cp /tmp/math500_output.txt "$GCS_RESULTS/math500_output.txt" 2>/dev/null || true
  echo "MATH-500 results saved."

  # Also run at higher concurrency for throughput measurement
  echo ""
  echo "Running MATH-500 throughput test (concurrency=64, limit=50)..."
  python3 /tmp/benchmark_math500.py \
    --server "$SERVER" \
    --concurrency 64 \
    --max-tokens 2048 \
    --limit 50 \
    --output /tmp/math500_throughput.jsonl \
    > /tmp/math500_throughput_output.txt 2>&1 || echo "Throughput test had errors"

  echo "Throughput test output:"
  tail -20 /tmp/math500_throughput_output.txt
  gcloud storage cp /tmp/math500_throughput.jsonl "$GCS_RESULTS/math500_throughput.jsonl" 2>/dev/null || true
  gcloud storage cp /tmp/math500_throughput_output.txt "$GCS_RESULTS/math500_throughput_output.txt" 2>/dev/null || true
fi

# Step 6: Save summary
echo ""
echo "Step 5: Saving summary..."

cat > /tmp/test_summary.txt << SUMMARY
============================================
Kimi K2-Instruct FP8 on TPU v5p-32
Test run: $(date)
============================================

Server: $SERVER
Model: moonshotai/Kimi-K2-Instruct (FP8, DeepseekV3ForCausalLM)
Hardware: v5p-32 (4 hosts x 4 chips, PP=4 TP=4)
Region: us-central1-a

Weight loading wait time: ${WAITED} minutes

Smoke tests: 5 prompts (simple math, proof, cubic roots, fizzbuzz, reasoning)
Throughput test: 10 concurrent requests in ${ELAPSED_MS}ms
MATH-500: Full benchmark with concurrency=16 and throughput test at concurrency=64

Results saved to: $GCS_RESULTS/
============================================
SUMMARY

cat /tmp/test_summary.txt
gcloud storage cp /tmp/test_summary.txt "$GCS_RESULTS/test_summary.txt"

# Save vLLM serve log
docker exec ray-node cat /tmp/vllm_serve.log > /tmp/vllm_serve_dump.log 2>/dev/null
gcloud storage cp /tmp/vllm_serve_dump.log "$GCS_RESULTS/vllm_serve.log" 2>/dev/null

echo ""
echo "============================================"
echo "ALL DONE! Results at: $GCS_RESULTS"
echo "============================================"
