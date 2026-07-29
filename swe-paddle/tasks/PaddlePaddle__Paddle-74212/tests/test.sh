#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_multiplex_op.py::TestMultiplexOp_ZeroSize -q
