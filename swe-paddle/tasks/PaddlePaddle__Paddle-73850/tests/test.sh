#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_triangular_solve_op.py::TestTriangularSolveOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_triangular_solve_op.py::TestTriangularSolveOp_ZeroSize -q
