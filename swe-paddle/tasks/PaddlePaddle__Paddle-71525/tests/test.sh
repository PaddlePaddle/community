#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/sot/test_info_collect.py -q
