#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr79161_set_rng_state_alias.py -q
