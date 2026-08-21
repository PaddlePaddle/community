#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  test/legacy_test/test_iinfo_and_finfo.py::TestIInfoUInt64Boundary
