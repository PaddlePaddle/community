#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/sot/test_sot_paddle_size.py -q
