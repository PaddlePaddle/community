#!/usr/bin/env bash
set -euo pipefail

python -m pytest \
  test/legacy_test/test_fractional_max_pool2d_api.py \
  test/legacy_test/test_fractional_max_pool2d_op.py \
  test/legacy_test/test_fractional_max_pool3d_api.py \
  test/legacy_test/test_fractional_max_pool3d_op.py \
  -q