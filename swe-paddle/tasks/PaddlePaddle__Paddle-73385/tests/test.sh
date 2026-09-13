#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root directory
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Add repo root and test/legacy_test to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/test/legacy_test:${PYTHONPATH:-}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp -q
python -m pytest test/legacy_test/test_svdvals_op.py::TestSvdvalsOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp_ZeroSize -q
python -m pytest test/legacy_test/test_eigvals_op.py::TestEigvalsOp_ZeroSize2 -q
python -m pytest test/legacy_test/test_svdvals_op.py::TestSvdvalsOp_ZeroSize -q
