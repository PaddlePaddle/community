#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_sparse_elementwise_op.py -q
