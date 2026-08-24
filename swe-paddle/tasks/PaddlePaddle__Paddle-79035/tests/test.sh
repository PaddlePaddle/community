#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr79035_lr_scheduler_alias.py -q
