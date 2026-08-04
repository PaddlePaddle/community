#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_sparse_mask_as_op.py -q
