#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_bmm_dynamic_shape_contract.py -q
