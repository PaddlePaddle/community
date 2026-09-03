#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57741.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# P2P: existing CPU dy2static `.cpu()` path (no-op when already on CPU; old IR).
# F2P: static-graph PIR memcpy translation on CPU, plus CUDA dy2static
#      cases (skipped via @unittest.skipIf when CUDA is unavailable).

# P2P tests (pass-to-pass)
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCpuOnDefaultCPU::test_tensor_cpu_on_default_cpu \
  -q

# F2P tests (fail-to-pass)
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestStaticCpuMemcpyNewIr::test_translate_cpu_memcpy_to_new_ir \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCUDAOnDefaultCPU::test_tensor_cuda_on_default_cpu \
  test/dygraph_to_static/test_tensor_memcpy_on_gpu.py::TestTensorCopyToCpuOnDefaultGPU::test_tensor_cpu_on_default_gpu \
  test/dygraph_to_static/test_tensor_memcpy_on_gpu.py::TestTensorCopyToCUDAOnDefaultGPU::test_tensor_cuda_on_default_gpu \
  -q
