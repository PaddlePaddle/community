#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr56705_mp_ops_pylayer_lifecycle.py -q
