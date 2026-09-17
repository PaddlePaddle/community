#!/usr/bin/env bash
set -euo pipefail
python -m pytest test/swe_paddle/test_pr60346_lookahead_pir.py -q
