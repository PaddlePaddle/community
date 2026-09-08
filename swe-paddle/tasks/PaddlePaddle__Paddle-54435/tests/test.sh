#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr54435_sort_ip.py -q
