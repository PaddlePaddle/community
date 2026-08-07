#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr79275_flex_attention_masks.py -q
