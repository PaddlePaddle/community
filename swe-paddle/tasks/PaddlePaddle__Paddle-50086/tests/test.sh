#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_conditional_block_scope_lifecycle.py -q
