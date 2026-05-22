#!/bin/bash
# check_cluster_progress.sh — check benchmark progress on the head node

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"
RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then RAY="$(which ray 2>/dev/null || echo ray)"; fi
KEY="$HOME/.ssh/ray-autoscaler_us-west-2.pem"

HEAD_IP=$("$RAY" get-head-ip "$CLUSTER_CONFIG" 2>/dev/null | tail -1)
echo "Head IP: $HEAD_IP"
echo ""

echo "=== Benchmark log (last 30 lines) ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  'echo "--- bench_frameworks ---"; tail -15 /tmp/bench_frameworks.log 2>/dev/null || echo "No log yet"
   echo "--- bench_fanout ---"; tail -15 /tmp/bench_fanout.log 2>/dev/null || echo "No log yet"' \
  2>/dev/null | grep -v "HOLD\|BUY\|SELL\|While\|Despite\|Buy\|Sell\|Hold\|strong\|modest\|elevated\|valuation\|momentum\|earnings\|analyst\|stock\|market\|price\|entry\|position\|multiple\|growth\|signal\|quality\|warrant\|suggest\|recent\|near-term\|upside\|downside\|trading\|ratio\|52-week\|P/E\|PE\|pct\|gain\|decline\|rally\|fundamental" || true

echo ""
echo "=== Is benchmark still running? ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  'pgrep -a python3 | grep -E "bench_frameworks_ray|bench_fanout_ray" || echo "Not running (finished or not started)"' \
  2>/dev/null

echo ""
echo "=== Output files on head node ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  'ls -lh /tmp/logs_*_ray_*_stocks.json /tmp/comparison_ray_*_stocks.json 2>/dev/null || echo "No output files yet"' \
  2>/dev/null

echo ""
echo "=== Cluster status ==="
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_IP" \
  '~/.local/bin/ray status 2>/dev/null | head -15' \
  2>/dev/null
