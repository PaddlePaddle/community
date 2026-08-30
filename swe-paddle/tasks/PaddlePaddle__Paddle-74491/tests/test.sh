#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74491_requires_grad.py -q
