#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr79310_sparse_initializer.py -q
