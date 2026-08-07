#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59383 (#59383 + #60835).
# Run from the root of a PaddlePaddle/Paddle checkout with Paddle importable.
python -m pytest \
  test/legacy_test/test_masked_scatter.py \
  test/legacy_test/test_inplace.py \
  -q
