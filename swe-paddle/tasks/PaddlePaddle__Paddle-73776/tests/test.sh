#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_svd_lowrank.py::TestSvdLowrankAPI -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_svd_lowrank.py::TestSvdLowRankAPI_ZeroSize -q
