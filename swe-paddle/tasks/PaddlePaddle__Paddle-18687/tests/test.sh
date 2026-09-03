#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr18687_launch_ps.py -q
