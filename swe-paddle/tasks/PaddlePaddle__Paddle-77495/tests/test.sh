#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_FILE="test/legacy_test/test_max_pool_dilation_contract.py"

# P2P: established positional calls must remain valid.
"$PYTHON_BIN" -m pytest \
  "$TEST_FILE::TestMaxPoolDilationContract::test_existing_positional_calls_1d" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_existing_positional_calls_2d" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_existing_positional_calls_3d" \
  -q

# F2P: dilation semantics and the additional compatible call forms.
"$PYTHON_BIN" -m pytest \
  "test/legacy_test/test_pool_max_op.py::TestMaxPoolWithIndex_Op" \
  "test/legacy_test/test_pool_max_op.py::TestCase5" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_functional_1d_forward_and_backward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_functional_2d_forward_and_backward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_functional_3d_forward_and_backward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_layer_1d_forward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_layer_2d_forward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_layer_3d_forward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_asymmetric_dilation_2d" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_static_2d_forward" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_return_mask_and_ceil_mode" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_compatibility_trap_1d" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_compatibility_trap_2d" \
  "$TEST_FILE::TestMaxPoolDilationContract::test_compatibility_trap_3d" \
  "test/legacy_test/test_imperative_layers.py::TestLayerPrint::test_layer_str" \
  -q
