#!/bin/bash
# teardown_cluster.sh — terminate the EC2 Ray cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARKS_DIR="$PROJECT_DIR/benchmarks"
CLUSTER_CONFIG="$BENCHMARKS_DIR/ray-cluster/ray-cluster-frameworks.yaml"
RAY="$PROJECT_DIR/.venv/bin/ray"
if [ ! -f "$RAY" ]; then RAY="$(which ray 2>/dev/null || echo ray)"; fi

echo "Tearing down Ray cluster..."
"$RAY" down "$CLUSTER_CONFIG" --yes
echo "Cluster terminated."
