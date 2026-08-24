#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr76522_torch_proxy_compat_overrides.py -q
