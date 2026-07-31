#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2API::test_static_api -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2API::test_dygraph_api -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2Broadcasting::test_api_with_dygraph_empty_tensor_input -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2EmptyTensorInput::test_api_with_dygraph_empty_tensor_input -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2_int32::test_check_output -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2_int64::test_check_output -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_atan2_op.py::TestAtan2Broadcasting::test_dygraph_broadcast_gradient_values -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_api_compatibility.py::TestAtan2API_Compatibility::test_dygraph_Compatibility -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_api_compatibility.py::TestAtan2API_Compatibility::test_static_Compatibility -q
"${PYTHON_BIN}" -m pytest test/legacy_test/test_api_compatibility.py::TestAtan2API_Compatibility::test_tensor_method -q
