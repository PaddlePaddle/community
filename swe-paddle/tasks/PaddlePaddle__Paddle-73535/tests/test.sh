#!/usr/bin/env bash
set -euo pipefail

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_functional_conv1d.py::TestFunctionalConv1DError -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_functional_conv1d.py::TestFunctionalConv1D_CPU_FP16 -q
