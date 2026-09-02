#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57741.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# F2P: dy2static memcpy via PIR/new-IR executor.
# - CPU path always runs.
# - GPU paths use @unittest.skipIf when CUDA is unavailable (no bare-return pass).
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCpuOnDefaultCPU::test_tensor_cpu_on_default_cpu \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCUDAOnDefaultCPU::test_tensor_cuda_on_default_cpu \
  test/dygraph_to_static/test_tensor_memcpy_on_gpu.py::TestTensorCopyToCpuOnDefaultGPU::test_tensor_cpu_on_default_gpu \
  test/dygraph_to_static/test_tensor_memcpy_on_gpu.py::TestTensorCopyToCUDAOnDefaultGPU::test_tensor_cuda_on_default_gpu \
  -q
