#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a PaddlePaddle/Paddle source checkout.
export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# P2P: existing regression coverage, unchanged by the gold patch.
"${PYTHON_BIN}" -m pytest -v \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_less \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_greater \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_mod \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_matmul \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_floordiv \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_item \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_place \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_some_dim \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_math_exists

# F2P: target behavior added by the gold patch.
"${PYTHON_BIN}" -m pytest -v \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_pow \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_bitwise_not \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_bitwise_xor \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_bitwise_or \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_bitwise_and \
  test/legacy_test/test_math_op_patch_pir.py::TestMathOpPatchesPir::test_equal_and_nequal
