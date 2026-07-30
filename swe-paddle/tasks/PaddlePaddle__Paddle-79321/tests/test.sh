#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  test/legacy_test/test_state_dict_convert.py::TestStateDictReturn::test_missing_keys_and_unexpected_keys_attr \
  test/legacy_test/test_state_dict_convert.py::TestStateDictReturn::test_missing_keys_and_unexpected_keys
