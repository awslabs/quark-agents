#!/bin/bash
# fetch_cluster_fanout_results.sh — download fanout benchmark results
# from head node and generate the comparison chart locally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"
RESULTS_DIR="$SCRIPT_DIR/../results/cluster_ray_fanout"
RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then RAY="$(which ray 2>/dev/null || echo ray)"; fi
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi
KEY="$HOME/.ssh/ray-autoscaler_us-west-2.pem"

mkdir -p "$RESULTS_DIR"

HEAD_IP=$("$RAY" get-head-ip "$CLUSTER_CONFIG" 2>/dev/null | tail -1)
echo "Head IP: $HEAD_IP"
echo ""

# ── Download per-framework logs ───────────────────────────────────────────
for fw in quark strands langgraph crewai; do
    REMOTE="/tmp/logs_fanout_${fw}_ray.json"
    LOCAL="$RESULTS_DIR/logs_fanout_${fw}_ray.json"
    if scp -i "$KEY" -o StrictHostKeyChecking=no \
        ubuntu@"$HEAD_IP":"$REMOTE" "$LOCAL" 2>/dev/null; then
        echo "Downloaded: logs_fanout_${fw}_ray.json"
    else
        echo "  Not found: $REMOTE"
    fi
done

# ── Download summary JSON ─────────────────────────────────────────────────
if scp -i "$KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$HEAD_IP":/tmp/comparison_fanout_ray.json \
    "$RESULTS_DIR/comparison_fanout_ray.json" 2>/dev/null; then
    echo "Downloaded: comparison_fanout_ray.json"
fi

# ── Download log ──────────────────────────────────────────────────────────
if scp -i "$KEY" -o StrictHostKeyChecking=no \
    ubuntu@"$HEAD_IP":/tmp/bench_fanout.log \
    "$RESULTS_DIR/bench_fanout_cluster.log" 2>/dev/null; then
    echo "Downloaded: bench_fanout_cluster.log"
fi

echo ""
echo "Results saved to: $RESULTS_DIR/"

# ── Print summary ─────────────────────────────────────────────────────────
SUMMARY="$RESULTS_DIR/comparison_fanout_ray.json"
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== Summary ==="
    "$PYTHON" - "$SUMMARY" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
frameworks  = d.get("frameworks", [])
batch_sizes = sorted(set(r["batch_size"] for fw in frameworks for r in d["results"].get(fw, [])))
header = f"{'Batch':>6}  " + "  ".join(f"{fw:>16}" for fw in frameworks)
print(header)
print("-" * len(header))
for bs in batch_sizes:
    row = f"{bs:>6}  "
    for fw in frameworks:
        r = next((x for x in d["results"].get(fw, []) if x["batch_size"] == bs), None)
        if r:
            row += f"{r['wall_time_s']:>5.1f}s {r['n_success']:>4}/{bs}  "
        else:
            row += f"{'n/a':>16}  "
    print(row)
PYEOF
fi

# ── Generate chart ────────────────────────────────────────────────────────
echo ""
echo "Generating chart..."
CHART="$RESULTS_DIR/comparison_fanout_ray.png"
"$PYTHON" "$BENCHMARKS_DIR/plot_fanout.py" \
  --input-dir "$RESULTS_DIR" \
  --output "$CHART" 2>&1

if [ -f "$CHART" ]; then
    echo "Chart: $CHART"
    open "$CHART" 2>/dev/null || true
fi
