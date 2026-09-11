#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root directory
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Add repo root and test/legacy_test to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/test/legacy_test:${PYTHONPATH:-}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp_ZeroSize -q
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp_ZeroSize2 -q
