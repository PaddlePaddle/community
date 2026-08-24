#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_add_n_op.py::TestAddnOpZeroSizeAndNonZeroSize -q
