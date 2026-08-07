#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_aminmax_op.py -q
python -m pytest \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest \
  -q
