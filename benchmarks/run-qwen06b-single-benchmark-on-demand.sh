#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_CONFIG=${MODEL_CONFIG:-"$SCRIPT_DIR/benchmarks/configs/qwen3-0.6b-jq4-on-demand.json"}

exec env BENCHMARK_SCRIPT_NAME=run-qwen06b-single-benchmark-on-demand MODEL_CONFIG="$MODEL_CONFIG" \
  sh "$SCRIPT_DIR/benchmarks/run-qwen06b-single-benchmark.sh" "$@"
