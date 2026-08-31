#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_test/test_auto_tuner_resume.py -q
