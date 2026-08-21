#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_launch_main_kill.py -q
