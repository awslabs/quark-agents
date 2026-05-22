#!/bin/bash
# run_1_local.sh — Framework comparison benchmark, local asyncio (no Ray).
#
# Covers: langgraph/strands/crewai/quark (gather) + quark (reactor)
# Model:  Claude Haiku 4.5 via Bedrock, stream=False
# Output: benchmarks/results/local/
#
# Usage (from quark-agents/ root):
#   bash benchmarks/scripts/run_1_local.sh
#   bash benchmarks/scripts/run_1_local.sh --n-stocks 30 --inter-wait 30
#   bash benchmarks/scripts/run_1_local.sh --no-shuffle --frameworks langgraph,strands,crewai,quark

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

mkdir -p "$BENCHMARKS_DIR/results/local"

echo "============================================================"
echo "Benchmark 1 — Local asyncio (no Ray)"
echo "============================================================"
echo "Python    : $PYTHON"
echo "Stocks    : $N_STOCKS"
echo "Inter-wait: ${INTER_WAIT}s between frameworks"
echo "Modes     : gather (all) + reactor (quark only)"
echo "Output    : benchmarks/results/local/"
echo ""

"$PYTHON" "$BENCHMARKS_DIR/bench_frameworks.py" \
  --n-stocks "$N_STOCKS" \
  --inter-wait "$INTER_WAIT" \
  $NO_SHUFFLE

# Move logs from benchmarks/ to results/local/
for fw in quark langgraph strands crewai; do
  SRC="$BENCHMARKS_DIR/logs_${fw}_${N_STOCKS}_stocks.json"
  test -f "$SRC" && mv "$SRC" "$BENCHMARKS_DIR/results/local/" && echo "Moved: logs_${fw}_${N_STOCKS}_stocks.json"
done

echo ""
echo "Generating chart..."
"$PYTHON" "$BENCHMARKS_DIR/plot_frameworks_ray.py" \
  --n-stocks "$N_STOCKS" \
  --output "$BENCHMARKS_DIR/results/local/comparison_local_${N_STOCKS}_stocks.png" 2>&1

echo ""
echo "Done. Results in: benchmarks/results/local/"
ls -lh "$BENCHMARKS_DIR/results/local/" 2>/dev/null
open "$BENCHMARKS_DIR/results/local/comparison_local_${N_STOCKS}_stocks.png" 2>/dev/null || true
