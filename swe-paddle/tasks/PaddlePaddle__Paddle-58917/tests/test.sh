#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_splits_api.py -q -k "not test_with_pir_api"
