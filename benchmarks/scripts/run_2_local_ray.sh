#!/bin/bash
# run_2_local_ray.sh — Framework comparison benchmark, local single-node Ray.
#
# Covers: langgraph/strands/crewai/quark (ray_gather) + quark (ray_reactor)
# Model:  Claude Haiku 4.5 via Bedrock, stream=False
# Output: benchmarks/results/local_ray/
#
# Usage (from quark-agents/ root):
#   bash benchmarks/scripts/run_2_local_ray.sh
#   bash benchmarks/scripts/run_2_local_ray.sh --n-stocks 30 --inter-wait 30
#   bash benchmarks/scripts/run_2_local_ray.sh --no-shuffle --frameworks langgraph,strands,crewai,quark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$BENCHMARKS_DIR/../.venv/bin/python"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi

N_STOCKS="150"
INTER_WAIT="60"
NO_SHUFFLE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --n-stocks)   N_STOCKS="$2";   shift 2 ;;
    --inter-wait) INTER_WAIT="$2"; shift 2 ;;
    --no-shuffle) NO_SHUFFLE="--no-shuffle"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$BENCHMARKS_DIR/results/local_ray"

echo "============================================================"
echo "Benchmark 2 — Local Ray (single-node)"
echo "============================================================"
echo "Python    : $PYTHON"
echo "Stocks    : $N_STOCKS"
echo "Inter-wait: ${INTER_WAIT}s between frameworks"
echo "Modes     : ray_gather (all) + ray_reactor (quark only)"
echo "Output    : benchmarks/results/local_ray/"
echo ""

"$PYTHON" "$BENCHMARKS_DIR/bench_frameworks_ray.py" \
  --mode local \
  --n-stocks "$N_STOCKS" \
  --inter-wait "$INTER_WAIT" \
  --output-dir "$BENCHMARKS_DIR/results/local_ray" \
  $NO_SHUFFLE

echo ""
echo "Generating chart..."
"$PYTHON" "$BENCHMARKS_DIR/plot_frameworks_ray.py" \
  --n-stocks "$N_STOCKS" \
  --output "$BENCHMARKS_DIR/results/local_ray/comparison_ray_${N_STOCKS}_stocks.png" 2>&1

echo ""
echo "Done. Results in: benchmarks/results/local_ray/"
ls -lh "$BENCHMARKS_DIR/results/local_ray/" 2>/dev/null
open "$BENCHMARKS_DIR/results/local_ray/comparison_ray_${N_STOCKS}_stocks.png" 2>/dev/null || true
