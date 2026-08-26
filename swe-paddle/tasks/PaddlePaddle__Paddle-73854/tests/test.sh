#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/legacy_test${PYTHONPATH:+:${PYTHONPATH}}"

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_instance_norm_op_v2.py::TestInstanceNorm -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_instance_norm_op_v2.py::TestInstanceNormOp_ZeroSize -q
