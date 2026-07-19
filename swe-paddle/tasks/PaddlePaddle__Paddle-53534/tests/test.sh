#!/usr/bin/env bash

set -euo pipefail
python -m pytest test/legacy_test/test_to_tensor_numpy124_contract.py -q
