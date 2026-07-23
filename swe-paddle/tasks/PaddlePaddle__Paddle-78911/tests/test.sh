#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_recompute_context.py -q
