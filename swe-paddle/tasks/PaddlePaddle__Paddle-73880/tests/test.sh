#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp_ZeroSize test/legacy_test/test_softmax_with_cross_entropy_op.py::TestSoftmaxWithCrossEntropyOp_ZeroSize2 -q
