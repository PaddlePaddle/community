#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr36684_elastic_scale.py -q
