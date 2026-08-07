#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp -q
python -m pytest test/legacy_test/test_svdvals_op.py::TestSvdvalsOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp_ZeroSize -q
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp_ZeroSize2 -q
python -m pytest test/legacy_test/test_svdvals_op.py::TestSvdvalsOp_ZeroSize -q
