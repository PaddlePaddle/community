#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74586_scatter_add.py -q
