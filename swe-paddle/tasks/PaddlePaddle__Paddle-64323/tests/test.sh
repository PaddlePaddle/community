#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/sot/test_dynamic_shape.py -q
