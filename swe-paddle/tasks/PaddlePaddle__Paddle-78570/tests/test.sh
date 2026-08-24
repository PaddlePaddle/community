#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr78570_optimizer_step_closure.py -q
