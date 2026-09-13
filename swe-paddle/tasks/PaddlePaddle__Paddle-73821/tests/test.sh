#!/bin/bash
set -e

# Resolve repository root directory
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Add repo root and test/legacy_test to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/test/legacy_test:${PYTHONPATH:-}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_pad_op.py::TestPadOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_pad_op.py::TestPadOp_ZeroSize2 -q
python -m pytest test/legacy_test/test_pad3d_op.py::TestPad3dOp_ZeroSize_Circular -q
python -m pytest test/legacy_test/test_pad3d_op.py::TestPad3dOp_ZeroSize_Replicate -q
