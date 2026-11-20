#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RECKONING_SPRING_CONFIG=jan-nano.yaml \
  exec sh "$SCRIPT_DIR/run-server-qwen3-4b-jq4.sh" "$@"
