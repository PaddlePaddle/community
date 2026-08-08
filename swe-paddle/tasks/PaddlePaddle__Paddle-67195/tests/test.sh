#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_pp_nan_checker_before_send.py -q
