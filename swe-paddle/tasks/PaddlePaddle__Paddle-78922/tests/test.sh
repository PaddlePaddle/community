#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -n "${PADDLE_ROOT:-}" ]; then
  REPO_ROOT="$PADDLE_ROOT"
else
  REPO_ROOT="$(git rev-parse --show-toplevel)"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" -m pytest -q \
  test/legacy_test/test_flatten_state_dict_lifetime.py
