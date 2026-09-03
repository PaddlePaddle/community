#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr40111_profiler_ranges.py -q
