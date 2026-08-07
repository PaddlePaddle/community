#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr56470_upsampling_single_int.py -q
