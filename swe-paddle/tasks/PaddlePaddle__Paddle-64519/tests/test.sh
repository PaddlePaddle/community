#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_cholesky_op.py -q -k "not test_with_pir_api"

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_linalg_cholesky_inverse.py -q -k "not test_with_pir_api"
