#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr79167_initial_seed_alias.py -q
