#!/bin/bash
set -e

# P2P tests (pass-to-pass)
python -m pytest test/legacy_test/test_diag_v2.py::TestDiagV2Op -q
python -m pytest test/legacy_test/test_masked_fill.py::TestMaskedFillAPI -q

# F2P tests (fail-to-pass)
python -m pytest test/legacy_test/test_diag_v2.py::TestDiagV2Op_ZeroSize -q
python -m pytest test/legacy_test/test_masked_fill.py::TestMaskedFillAPI_ZeroSize2 -q
