#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# P2P (pass-to-pass): existing amin/amax API compatibility, out-tensor, and
# gradient behavior. The gold patch does not modify this test file.
PYTHONPATH="$repo_root/test/legacy_test:$repo_root/test${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m pytest -q \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAPI_Compatibility::test_dygraph_Compatibility \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAminAPI_Compatibility::test_dygraph_Compatibility \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAminOutAPI::test_amax_out_in_dygraph \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAminOutAPI::test_amin_out_in_dygraph

# F2P (fail-to-pass): run both target suites even when the first one fails,
# so Base logs contain the complete role matrix before the wrapper exits 1.
f2p_status=0

if ! PYTHONPATH="$repo_root/test/legacy_test:$repo_root/test${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m pytest test/legacy_test/test_aminmax_op.py -q; then
  f2p_status=1
fi

# The symbolic-shape suite needs a different utils.py. Keep the CINN/PIR
# directory ahead of the legacy test directory because both modules are named
# utils.py.
#
# It also needs the same FLAGS environment that upstream's CTest registration
# uses. check_infer_results reads the op attribute `sym_shape_str`, which is
# attached by the shape optimization pass that CheckInferSymbolicIfNeed adds,
# and that function returns early unless FLAGS_prim_forward,
# FLAGS_prim_backward and FLAGS_check_infer_symbolic are all set
# (FLAGS_prim_all sets the two prim flags). Without them the node raises
# KeyError: 'sym_shape_str' on the gold build as well. The flags below mirror
# test/ir/pir/cinn/symbolic/CMakeLists.txt.
if ! PYTHONPATH="$repo_root/test/ir/pir/cinn:$repo_root/test/ir/pir/cinn/symbolic${PYTHONPATH:+:$PYTHONPATH}" \
  FLAGS_check_infer_symbolic=1 \
  FLAGS_enable_pir_api=1 \
  FLAGS_prim_enable_dynamic=true \
  FLAGS_prim_all=True \
  FLAGS_cinn_new_group_scheduler=1 \
  "$PYTHON_BIN" -m pytest \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest \
  -q; then
  f2p_status=1
fi

exit "$f2p_status"
