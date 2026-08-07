#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_multi_dot_op.py::TestMultiDotOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_multi_dot_op.py::TestMultiDotOp_ZeroSize1 -q
