#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_linalg_cholesky_inverse.py -q
