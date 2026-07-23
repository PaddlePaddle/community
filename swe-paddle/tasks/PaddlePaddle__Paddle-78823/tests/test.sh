#!/usr/bin/env bash
set -euo pipefail

python -m pytest \
  test/legacy_test/test_to_pinned_place.py \
  test/legacy_test/test_eager_tensor.py \
  test/legacy_test/test_rand.py \
  test/legacy_test/test_randint_op.py \
  test/legacy_test/test_randperm_op.py \
  -q
