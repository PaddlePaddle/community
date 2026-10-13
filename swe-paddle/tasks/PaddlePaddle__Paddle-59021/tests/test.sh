#!/usr/bin/env bash
set -uo pipefail

# Target tests for PaddlePaddle__Paddle-59021.
# Run from the root of a built PaddlePaddle/Paddle source checkout.
#
# Every block runs even if an earlier one fails; the script exits non-zero if
# any block fails.
export PYTHONPATH="test/legacy_test:test${PYTHONPATH:+:${PYTHONPATH}}"
status=0

# P2P: fusion pass module, matching CTest test_fuse_elewise_add_act_pass_pir.
# CTest runs it through tools/test_runner.py, which calls paddle.enable_static()
# first because the module is in static_mode_white_list; pytest bypasses that
# runner, so static mode is enabled explicitly before collection.
FLAGS_enable_pir_in_executor=true python -c '
import sys

import paddle
import pytest

paddle.enable_static()
sys.exit(pytest.main(["test/legacy_test/test_fuse_elewise_add_act_pass.py", "-q"]))
' || status=1

# P2P: test_len Tensor / TensorArray cases. Upstream lists test_len in
# DISABLE_PIR_PT_MODES (test/dygraph_to_static/CMakeLists.txt) and runs it with
# FLAGS_enable_pir_with_pt_in_dy2st=0; the flag defaults to true.
FLAGS_enable_pir_with_pt_in_dy2st=0 python -m pytest \
  test/dygraph_to_static/test_len.py::TestLen \
  test/dygraph_to_static/test_len.py::TestLenWithTensorArray \
  -q || status=1

# F2P: SelectedRows len / shape through the PIR executor.
FLAGS_enable_pir_with_pt_in_dy2st=0 FLAGS_enable_pir_in_executor=true python -m pytest \
  test/dygraph_to_static/test_len.py::TestLenWithSelectedRows \
  -q || status=1

exit "$status"
