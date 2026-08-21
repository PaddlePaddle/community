#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57827.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest \
  test/dygraph_to_static/test_build_strategy.py \
  -q
