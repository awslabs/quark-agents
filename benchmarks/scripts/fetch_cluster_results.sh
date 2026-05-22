#!/bin/bash
# fetch_cluster_results.sh — download benchmark results from head node,
# then generate the comparison chart locally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"
RESULTS_DIR="$SCRIPT_DIR/../results/cluster_ray"
RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then RAY="$(which ray 2>/dev/null || echo ray)"; fi
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi
KEY="$HOME/.ssh/ray-autoscaler_us-west-2.pem"

mkdir -p "$RESULTS_DIR"

HEAD_IP=$("$RAY" get-head-ip "$CLUSTER_CONFIG" 2>/dev/null | tail -1)
echo "Head IP: $HEAD_IP"
echo ""

# ── Download per-framework Ray logs ──────────────────────────────────────
N_STOCKS="${1:-150}"
for fw in quark strands langgraph crewai; do
    REMOTE="/tmp/logs_${fw}_ray_${N_STOCKS}_stocks.json"
    LOCAL="$RESULTS_DIR/logs_${fw}_ray_${N_STOCKS}_stocks.json"
    if scp -i "$KEY" -o StrictHostKeyChecking=no \
        ubuntu@"$HEAD_IP":"$REMOTE" "$LOCAL" 2>/dev/null; then
        echo "Downloaded: logs_${fw}_ray_${N_STOCKS}_stocks.json"
    else
        echo "  Not found: $REMOTE"
    fi
done

# ── Download summary JSON ─────────────────────────────────────────────────
SUMMARY_REMOTE="/tmp/comparison_ray_${N_STOCKS}_stocks.json"
SUMMARY_LOCAL="$RESULTS_DIR/comparison_ray_${N_STOCKS}_stocks.json"
if scp -i "$KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$HEAD_IP":"$SUMMARY_REMOTE" "$SUMMARY_LOCAL" 2>/dev/null; then
    echo "Downloaded: comparison_ray_${N_STOCKS}_stocks.json"
fi

# ── Download log ──────────────────────────────────────────────────────────
if scp -i "$KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$HEAD_IP":/tmp/bench_frameworks.log \
    "$RESULTS_DIR/bench_frameworks_cluster.log" 2>/dev/null; then
    echo "Downloaded: bench_frameworks_cluster.log"
fi

echo ""
echo "Results saved to: $RESULTS_DIR/"

# ── Print summary ─────────────────────────────────────────────────────────
if [ -f "$SUMMARY_LOCAL" ]; then
    echo ""
    echo "=== Summary ==="
    "$PYTHON" - "$SUMMARY_LOCAL" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
summary = d.get("summary", {})
print(f"{'Mode':>28}  {'wall time':>10}  {'success':>9}  {'tasks/s':>8}")
print("-" * 60)
for fw, modes in summary.items():
    for mode, info in modes.items():
        if not info: continue
        total   = info.get("total", 0)
        success = info.get("success", 0)
        n       = d.get("n_stocks", 150)
        tput    = success / total if total > 0 else 0
        label   = f"{fw} ({mode})"
        print(f"  {label:>26}  {total:>8.2f}s  {success:>3}/{n}     {tput:>6.2f}/s")
PYEOF
fi

# ── Generate chart ────────────────────────────────────────────────────────
echo ""
echo "Generating chart..."
CHART="$RESULTS_DIR/comparison_ray_${N_STOCKS}_stocks.png"
"$PYTHON" "$BENCHMARKS_DIR/plot_frameworks_ray.py" \
  --n-stocks "$N_STOCKS" \
  --output "$CHART" 2>&1

if [ -f "$CHART" ]; then
    echo "Chart: $CHART"
    open "$CHART" 2>/dev/null || true
fi
