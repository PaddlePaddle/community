#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# P2P tests (pass-to-pass)
"${PYTHON_BIN}" -m pytest test/legacy_test/test_assert_close.py::TestAssertClose -q

# F2P tests (fail-to-pass)
"${PYTHON_BIN}" -m pytest test/legacy_test/test_api_compatibility_part2.py::TestAssertAPI -q
