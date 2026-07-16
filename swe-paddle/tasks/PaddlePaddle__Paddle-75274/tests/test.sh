#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_normal_benchmark.py -q
