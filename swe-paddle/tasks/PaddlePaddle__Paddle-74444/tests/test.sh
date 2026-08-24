#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74444_dropout1d.py -q
