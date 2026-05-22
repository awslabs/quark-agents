#!/bin/bash
# run_cluster_ray.sh — provision EC2 cluster, run framework benchmark,
# return immediately. Benchmark runs as nohup on head node.
#
# Covers all 5 modes:
#   langgraph (ray_gather), strands (ray_gather), crewai (ray_gather),
#   quark (ray_gather), quark (ray_reactor)
#
# Usage:
#   bash benchmarks/scripts/run_4_cluster_ray.sh
#   bash benchmarks/scripts/run_4_cluster_ray.sh --n-stocks 150 --inter-wait 60
#
# Check progress:
#   bash benchmarks/scripts/check_cluster_progress.sh
#
# Fetch results when done:
#   bash benchmarks/scripts/fetch_cluster_results.sh
#
# Tear down:
#   bash benchmarks/scripts/teardown_cluster.sh
#
# Prerequisites (one-time):
#   bash benchmarks/scripts/setup_iam.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"

# Ray CLI lives in the project venv
RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then
    RAY="$(which ray 2>/dev/null || echo '')"
fi
if [ -z "$RAY" ] || [ ! -f "$RAY" ]; then
    echo "ERROR: ray CLI not found. Run: pip install 'ray[default]'"
    exit 1
fi

N_STOCKS="150"
INTER_WAIT="60"
MAX_CONCURRENT="10"   # 4 workers × 10 slots = 40; safe for 150 stocks

NO_SHUFFLE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --n-stocks)       N_STOCKS="$2";       shift 2 ;;
    --inter-wait)     INTER_WAIT="$2";     shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --no-shuffle)     NO_SHUFFLE="--no-shuffle"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "Framework Benchmark — EC2 Ray Cluster"
echo "============================================================"
echo "Stocks        : $N_STOCKS"
echo "Inter-wait    : ${INTER_WAIT}s between frameworks"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Modes         : langgraph/strands/crewai/quark (ray_gather) + quark (ray_reactor)"
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

# ── Step 2: Fix venv stubs (autoscaler hashes file_mounts) ───────────────
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

# ── Step 4: Fire benchmark as background job ──────────────────────────────
echo "[4/4] Starting benchmark on head node (background)..."
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  "cd ~/quark-agents && nohup python3 benchmarks/bench_frameworks_ray.py \
    --mode cluster \
    --n-stocks $N_STOCKS \
    --inter-wait $INTER_WAIT \
    --max-concurrent $MAX_CONCURRENT \
    $NO_SHUFFLE \
    --output-dir /tmp \
    > /tmp/bench_frameworks.log 2>&1 & echo \"Benchmark PID: \$!\"" \
  2>&1 | grep -v "^Fetch\|^Check\|^Shared"

echo ""
echo "Benchmark is running on the head node."
echo "Head IP: $HEAD_IP"
echo ""
echo "Check progress:  bash benchmarks/scripts/check_cluster_progress.sh"
echo "Fetch results:   bash benchmarks/scripts/fetch_cluster_results.sh"
echo "Tear down:       bash benchmarks/scripts/teardown_cluster.sh"
