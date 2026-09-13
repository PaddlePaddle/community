#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/swe_paddle/test_pr33369_elastic_fault_tolerance.py -q
