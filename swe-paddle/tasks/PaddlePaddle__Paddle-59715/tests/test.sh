#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_linalg_pinv_op.py -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_linalg_matrix_exp.py -q
