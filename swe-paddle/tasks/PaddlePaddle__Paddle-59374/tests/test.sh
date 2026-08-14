#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-59374.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest \
  test/legacy_test/test_apply.py \
  test/legacy_test/test_inplace.py::TestDygraphTensorApplyInplace \
  -q
