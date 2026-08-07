#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr79268_distributed_sampler.py -q
