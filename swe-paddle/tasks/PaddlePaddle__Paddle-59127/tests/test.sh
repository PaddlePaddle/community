#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_stack_extension_api.py -q
