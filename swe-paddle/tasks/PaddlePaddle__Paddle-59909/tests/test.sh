#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_hybrid_parallel_mode_selection.py -q
