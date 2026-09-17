#!/bin/bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${repo_root}"

# Fail hard when CUDA is unavailable: this task targets a CUDA-only kernel.
"${PYTHON_BIN}" -c "
import paddle
assert paddle.is_compiled_with_cuda(), 'Paddle is not compiled with CUDA'
assert paddle.device.cuda.device_count() >= 1, 'No CUDA device available'
"

export PYTHONPATH="${repo_root}/python:${repo_root}/test/legacy_test:${repo_root}/test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P node first: pass-to-pass behavior (passes on base and gold).
# Dedicated CUDA gradient node plus existing forward/reduce and elementwise
# regression nodes for both min and max.
"${PYTHON_BIN}" -m pytest -xvs \
  test/legacy_test/test_compat_minmax.py::TestCompatMinMaxCUDAGrad::test_p2p_grad \
  test/legacy_test/test_compat_minmax.py::TestCompatMinMaxBase::test_case2_reduce_dim \
  test/legacy_test/test_compat_minmax.py::TestCompatMax::test_case2_reduce_dim \
  test/legacy_test/test_compat_minmax.py::TestCompatMinMaxBase::test_case3_elementwise \
  test/legacy_test/test_compat_minmax.py::TestCompatMax::test_case3_elementwise

# F2P node second: fail-to-pass behavior (fails on base, passes on gold).
"${PYTHON_BIN}" -m pytest -xvs test/legacy_test/test_compat_minmax.py::TestCompatMinMaxCUDAGrad::test_f2p_grad