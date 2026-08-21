#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp_ZeroSize -q
python -m pytest test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp_ZeroSize2 -q
