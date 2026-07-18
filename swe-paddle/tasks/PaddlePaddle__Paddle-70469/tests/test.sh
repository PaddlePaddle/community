#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_fused_dropout_add_fallback.py -q
