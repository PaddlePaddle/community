#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr74594_broadcast_shapes.py -q
