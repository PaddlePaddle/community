#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74439_ravel.py -q
