#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59383 (#59383 + #60835).
# Run from the root of a PaddlePaddle/Paddle checkout with Paddle importable.
#
# test_masked_scatter.py imports op_test, and op_test imports white_list.
# Both live under the test tree, so PYTHONPATH must include:
#   - test/legacy_test  -> op_test
#   - test              -> white_list package
export PYTHONPATH="test/legacy_test:test${PYTHONPATH:+:${PYTHONPATH}}"

python -m pytest \
  test/legacy_test/test_masked_scatter.py \
  test/legacy_test/test_inplace.py \
  -q
