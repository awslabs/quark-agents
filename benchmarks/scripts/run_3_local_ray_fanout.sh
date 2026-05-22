#!/bin/bash
# run_3_local_ray_fanout.sh — Fanout pipeline benchmark, local single-node Ray.
#
# Pipeline: fetch_topic → summarize → [critique+factcheck+style] → edit
# = 5 LLM calls per task. Quark runs fan-out concurrently via asyncio.gather.
# Output: benchmarks/results/fanout_ray/
#
# Usage (from quark-agents/ root):
#   bash benchmarks/scripts/run_3_local_ray_fanout.sh
#   bash benchmarks/scripts/run_3_local_ray_fanout.sh --batch-sizes 10,50,100
#   bash benchmarks/scripts/run_3_local_ray_fanout.sh --no-shuffle --frameworks langgraph,strands,crewai,quark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$BENCHMARKS_DIR/../.venv/bin/python"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi

BATCH_SIZES="100,250,500,1000"
FRAMEWORKS="quark,langgraph,strands,crewai"
MAX_CONCURRENT="5"
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

mkdir -p "$BENCHMARKS_DIR/results/fanout_ray"

echo "============================================================"
echo "Benchmark 3 — Local Ray Fanout Pipeline"
echo "============================================================"
echo "Python        : $PYTHON"
echo "Batch sizes   : $BATCH_SIZES"
echo "Frameworks    : $FRAMEWORKS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Inter-wait    : ${INTER_WAIT}s"
echo "Pipeline      : fetch → summarize → [critique+factcheck+style] → edit"
echo "LLM calls     : 5 per task"
echo "Output        : benchmarks/results/fanout_ray/"
echo ""

"$PYTHON" "$BENCHMARKS_DIR/bench_fanout_ray.py" \
  --mode local \
  --batch-sizes "$BATCH_SIZES" \
  --frameworks "$FRAMEWORKS" \
  --max-concurrent "$MAX_CONCURRENT" \
  --inter-wait "$INTER_WAIT" \
  --output-dir "$BENCHMARKS_DIR/results/fanout_ray" \
  $NO_SHUFFLE

echo ""
echo "Generating chart..."
"$PYTHON" "$BENCHMARKS_DIR/plot_fanout.py" \
  --input-dir "$BENCHMARKS_DIR/results/fanout_ray" \
  --output "$BENCHMARKS_DIR/results/fanout_ray/comparison_fanout_ray.png" 2>&1

echo ""
echo "Done. Results in: benchmarks/results/fanout_ray/"
ls -lh "$BENCHMARKS_DIR/results/fanout_ray/" 2>/dev/null
open "$BENCHMARKS_DIR/results/fanout_ray/comparison_fanout_ray.png" 2>/dev/null || true
