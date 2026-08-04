#!/usr/bin/env bash
set -euo pipefail

# Target tests for PaddlePaddle__Paddle-76873 (#76873 + #77103).
# Run from the root of a built PaddlePaddle/Paddle source checkout.
python -m pytest \
  test/legacy_test/test_celu_op.py \
  test/legacy_test/test_rrelu_op.py \
  test/legacy_test/test_swish_op.py \
  test/legacy_test/test_mish_op.py \
  test/legacy_test/test_hardsigmoid_op.py \
  test/legacy_test/test_selu_op.py \
  test/legacy_test/test_imperative_layers.py \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_multinary_op.py \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py \
  -q
