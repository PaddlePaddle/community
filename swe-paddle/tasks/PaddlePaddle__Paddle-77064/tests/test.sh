#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a built PaddlePaddle/Paddle source checkout.
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pytest \
  test/legacy_test/test_allclose_op.py \
  test/legacy_test/test_compat_allclose.py \
  -q
