#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-52948 (#52948 + #53572).
# Run from the root of a PaddlePaddle/Paddle checkout with Paddle importable.
python -m pytest \
  test/dygraph_to_static/test_tensor_hook.py \
  python/paddle/fluid/tests/unittests/test_tensor_register_hook.py \
  -q
