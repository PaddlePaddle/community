#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp_ZeroSize -q
python -m pytest test/legacy_test/test_gather_tree_op.py::TestGatherTreeOp_ZeroSize2 -q
