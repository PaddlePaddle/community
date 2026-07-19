#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr54625_pipeline_output_release.py -q
