#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_device_place_conversion_contract.py -q
