#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_fold_op.py::TestFoldOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_fold_op.py::TestFoldOpError -q
