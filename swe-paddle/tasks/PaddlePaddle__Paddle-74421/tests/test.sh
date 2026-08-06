#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74421_msort.py -q
