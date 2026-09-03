#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-57741.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# P2P: existing CPU dy2static `.cpu()` path (no-op when already on CPU; old IR).
# F2P: static-graph `.cpu()` always inserts memcpy and must translate under PIR.

# P2P tests (pass-to-pass)
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestTensorCopyToCpuOnDefaultCPU::test_tensor_cpu_on_default_cpu \
  -q

# F2P tests (fail-to-pass)
python -m pytest \
  test/dygraph_to_static/test_tensor_memcpy_on_cpu.py::TestStaticCpuMemcpyNewIr::test_translate_cpu_memcpy_to_new_ir \
  -q
