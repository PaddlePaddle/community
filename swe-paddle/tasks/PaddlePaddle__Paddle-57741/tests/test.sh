#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57741.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# F2P: CPU dy2static memcpy via PIR/new-IR executor (no CUDA required).
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCpuOnDefaultCPU::test_tensor_cpu_on_default_cpu \
  -q
