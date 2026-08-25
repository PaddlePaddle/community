#!/bin/bash
set -e

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_gather_nd_op.py::TestGatherNdOpWithEmptyIndex -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_gather_nd_op.py::TestGatherNdOp_ZeroSize -q
