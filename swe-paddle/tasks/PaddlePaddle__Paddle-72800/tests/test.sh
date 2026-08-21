#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_cummax_op.py::TestCummaxOp -q
python -m pytest test/legacy_test/test_cummin_op.py::TestCumminOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_cummax_op.py::TestCummaxOp_ZeroSize -q
python -m pytest test/legacy_test/test_cummin_op.py::TestCumminOp_ZeroSize -q
