#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57741.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py \
  -q

# Optional if CUDA is available:
# python -m pytest test/dygraph_to_static/test_tensor_memcpy_on_gpu.py -q
