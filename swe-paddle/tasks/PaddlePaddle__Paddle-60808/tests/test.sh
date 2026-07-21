#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_broadcast_to_zero_dim_shape.py -q
