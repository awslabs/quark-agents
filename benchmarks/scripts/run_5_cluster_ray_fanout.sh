#!/bin/bash
# run_cluster_ray_fanout.sh — provision EC2 cluster, run fanout pipeline
# benchmark as background job, return immediately.
#
# Pipeline per task: fetch_topic → summarize → [critique+factcheck+style] → edit
# = 4 LLM calls per task
#
# Usage:
#   bash benchmarks/scripts/run_5_cluster_ray_fanout.sh
#   bash benchmarks/scripts/run_5_cluster_ray_fanout.sh --batch-sizes 100,250,500,1000
#
# Check progress:
#   bash benchmarks/scripts/check_cluster_progress.sh
#
# Fetch results when done:
#   bash benchmarks/scripts/fetch_cluster_fanout_results.sh
#
# Tear down:
#   bash benchmarks/scripts/teardown_cluster.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"

RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then
    RAY="$(which ray 2>/dev/null || echo '')"
fi
if [ -z "$RAY" ] || [ ! -f "$RAY" ]; then
    echo "ERROR: ray CLI not found. Run: pip install 'ray[default]'"
    exit 1
fi

BATCH_SIZES="100,250,500,1000"
FRAMEWORKS="quark,langgraph,strands,crewai"
MAX_CONCURRENT="10"   # 4 workers × 10 slots = 40; fanout uses ~4 slots/pipeline
INTER_WAIT="60"

NO_SHUFFLE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-sizes)    BATCH_SIZES="$2";    shift 2 ;;
    --frameworks)     FRAMEWORKS="$2";     shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --inter-wait)     INTER_WAIT="$2";     shift 2 ;;
    --no-shuffle)     NO_SHUFFLE="--no-shuffle"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "Fanout Pipeline Benchmark — EC2 Ray Cluster"
echo "============================================================"
echo "Batch sizes   : $BATCH_SIZES"
echo "Frameworks    : $FRAMEWORKS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Inter-wait    : ${INTER_WAIT}s"
echo "Pipeline      : fetch → summarize → [critique+factcheck+style] → edit"
echo "LLM calls     : 5 per task"
echo "Cluster config: $CLUSTER_CONFIG"
echo ""

# ── Step 1: Provision cluster ─────────────────────────────────────────────
echo "[1/4] Provisioning Ray cluster..."
WAIT=15
for attempt in $(seq 1 5); do
    echo "  Attempt $attempt/5..."
    if "$RAY" up "$CLUSTER_CONFIG" --yes --no-config-cache 2>&1; then
        echo "  Cluster is up."
        break
    fi
    if [ "$attempt" -eq 5 ]; then
        echo "ERROR: Failed to provision cluster after 5 attempts."
        exit 1
    fi
    echo "  Retrying in ${WAIT}s..."
    sleep $WAIT
    WAIT=$((WAIT * 2))
done

# ── Step 2: Fix venv stubs ────────────────────────────────────────────────
echo "[2/4] Fixing venv stubs on head node..."
HEAD_IP=$("$RAY" get-head-ip "$CLUSTER_CONFIG" 2>/dev/null | tail -1)
KEY="$HOME/.ssh/ray-autoscaler_us-west-2.pem"

ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 ubuntu@"$HEAD_IP" '
  for dir in ~/quark-agents/quark ~/quark-agents/quark-agents; do
    mkdir -p "$dir/.venv/bin"
    ln -sf /usr/bin/python3 "$dir/.venv/bin/python"
    ln -sf /usr/bin/python3 "$dir/.venv/bin/python3"
  done
  ~/.local/bin/ray stop 2>/dev/null || true
  sleep 3
  ulimit -n 65536
  RAY_memory_usage_threshold=0.99 ~/.local/bin/ray start \
    --head --port=6379 --object-manager-port=8076 --num-cpus=0 \
    --autoscaling-config=~/ray_bootstrap_config.yaml 2>&1 | tail -2
  echo "Head restarted"
' 2>&1 | grep -v "^Fetch\|^Check\|^Shared"

# ── Step 3: Wait for all 4 workers ───────────────────────────────────────
echo "[3/4] Waiting for all 4 workers (up to 15 minutes)..."
for i in $(seq 1 90); do
    COUNT=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        ubuntu@"$HEAD_IP" \
        '~/.local/bin/ray status 2>/dev/null | grep -c "ray.worker.default"' \
        2>/dev/null || echo "0")
    echo "  Workers ready: ${COUNT:-0}/4"
    if [ "${COUNT:-0}" -ge "4" ] 2>/dev/null; then
        echo "  All 4 workers ready."
        break
    fi
    sleep 10
done

# ── Step 4: Fire benchmark ────────────────────────────────────────────────
echo "[4/4] Starting fanout benchmark on head node (background)..."
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  "cd ~/quark-agents && nohup python3 benchmarks/bench_fanout_ray.py \
    --mode cluster \
    --batch-sizes $BATCH_SIZES \
    --frameworks $FRAMEWORKS \
    --max-concurrent $MAX_CONCURRENT \
    --inter-wait $INTER_WAIT \
    $NO_SHUFFLE \
    --output-dir /tmp \
    > /tmp/bench_fanout.log 2>&1 & echo \"Benchmark PID: \$!\"" \
  2>&1 | grep -v "^Fetch\|^Check\|^Shared"

echo ""
echo "Benchmark is running on the head node."
echo "Head IP: $HEAD_IP"
echo ""
echo "Check progress:  bash benchmarks/scripts/check_cluster_progress.sh"
echo "  (looks at /tmp/bench_fanout.log)"
echo "Fetch results:   bash benchmarks/scripts/fetch_cluster_fanout_results.sh"
echo "Tear down:       bash benchmarks/scripts/teardown_cluster.sh"
