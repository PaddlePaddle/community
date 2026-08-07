#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_cumsum_op.py::TestCumsumOp -q
python -m pytest test/legacy_test/test_logcumsumexp_op.py::TestLogcumsumexpOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_cumsum_op.py::cumsumfloat32_0SizeTest -q
python -m pytest test/legacy_test/test_logcumsumexp_op.py::logcumsumexpfloat32_0SizeTest -q
