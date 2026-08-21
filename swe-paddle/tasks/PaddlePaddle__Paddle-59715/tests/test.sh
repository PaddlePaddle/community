#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_linalg_matrix_exp.py -q
