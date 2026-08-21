#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest \
  test/legacy_test/test_pool2d_api.py \
  test/legacy_test/test_pool3d_api.py \
  -q

# F2P tests (fail-to-pass)
python -m pytest \
  test/legacy_test/test_fractional_max_pool2d_api.py \
  test/legacy_test/test_fractional_max_pool2d_op.py \
  test/legacy_test/test_fractional_max_pool3d_api.py \
  test/legacy_test/test_fractional_max_pool3d_op.py \
  -q