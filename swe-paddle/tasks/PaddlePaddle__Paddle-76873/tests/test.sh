#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass on base+test.patch and gold)
python -m pytest \
  test/legacy_test/test_rrelu_op.py::RReluTest \
  test/legacy_test/test_selu_op.py::SeluTest \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::SumOpInferSymbolicShapeTest::test_eval_symbolic \
  -q

# F2P tests (fail-to-pass on base+test.patch; pass on gold after rebuild)
python -m pytest \
  test/legacy_test/test_celu_op.py \
  test/legacy_test/test_hardsigmoid_op.py \
  test/legacy_test/test_mish_op.py \
  test/legacy_test/test_swish_op.py \
  test/legacy_test/test_rrelu_op.py::TestRRELUOpClass_Inplace \
  test/legacy_test/test_rrelu_op.py::TestRRELUAPI \
  test/legacy_test/test_selu_op.py::TestSELUOpClass_Inplace \
  test/legacy_test/test_selu_op.py::TestSELUAPI \
  test/legacy_test/test_imperative_layers.py::TestLayerPrint::test_layer_str \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_multinary_op.py::CELUOpInferSymbolicShapeTest::test_eval_symbolic \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::RRELUOpInferSymbolicShapeTest::test_eval_symbolic \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::SELUOpInferSymbolicShapeTest::test_eval_symbolic \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::HardSigmoidInferSymbolicShapeTest::test_eval_symbolic \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::MishOpInferSymbolicShapeTest::test_eval_symbolic \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::SwishOpInferSymbolicShapeTest::test_eval_symbolic \
  -q
