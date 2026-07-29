#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_multiplex_op.py::TestMultiplexOp -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_multiplex_op.py::TestMultiplexOp_ZeroSize -q
