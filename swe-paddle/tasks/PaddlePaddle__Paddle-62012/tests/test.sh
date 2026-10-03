#!/usr/bin/env bash

set -euo pipefail
python -m pytest \
  test/sot/test_11_jumps.py \
  test/sot/test_analysis_inputs.py \
  -q
