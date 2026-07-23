#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/test/legacy_test"

python -m pytest -q \
  test_cdist.py::TestCdistZeroSizeGrad::test_stop_gradient_true \
  test_cdist.py::TestCdistZeroSizeBatch4D::test_dygraph_api \
  test_cdist.py::TestCdistZeroSizeGrad::test_stop_gradient_false
