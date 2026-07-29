#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/legacy_test/test_unique.py::TestUniqueAPI_ZeroSize -q
