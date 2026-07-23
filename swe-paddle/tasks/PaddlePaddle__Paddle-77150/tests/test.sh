#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  test/legacy_test/test_pylayer_op.py::TestPyLayer::test_simple_pylayer_multiple_output \
  test/legacy_test/test_pylayer_op.py::TestPyLayer::test_pylayer_with_partial_grad
